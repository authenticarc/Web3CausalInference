from dataclasses import dataclass
import numpy as np, optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, CatBoostRegressor
from econml.dr import ForestDRLearner

class HTETester:
    """
    只负责 HTE 训练：
      - （可选）以 nAUUC 为目标做超参与带选择
      - 在选定 overlap 带内训练 ForestDRLearner
      - fit(...) 返回训练好的 ForestDRLearner 实例
      - report() 仅在 fit 之后调用；使用 fit 时缓存的数据生成可用性报告
    """
    def __init__(self,
                 regressor=None,
                 classifier=None,
                 n_splits=5,
                 trim=0.01,
                 nauuc_band=(0.3, 0.7),
                 nauuc_policy_ks=(0.1, 0.2, 0.3),
                 min_nauuc=0.35,          # 可用性门槛
                 tune_nauuc=False,
                 n_trials=200,
                 early_stop=0.60,
                 search_forest_head=True,
                 reg_loss="auto",          # 'RMSE'|'Tweedie'|'auto'
                 reg_tweedie_p=1.3,
                 random_state=42,
                 verbose=1):
        self.n_splits = int(n_splits)
        self.trim = float(trim)
        self.nauuc_band = tuple(nauuc_band)
        self.nauuc_policy_ks = tuple(nauuc_policy_ks)
        self.min_nauuc = float(min_nauuc)
        self.tune_nauuc = bool(tune_nauuc)
        self.n_trials = int(n_trials)
        self.early_stop = early_stop
        self.search_forest_head = bool(search_forest_head)
        self.reg_loss = reg_loss
        self.reg_tweedie_p = float(reg_tweedie_p)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

        # base learners（可被调参覆盖）
        self.regressor = regressor or CatBoostRegressor(
            depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
            subsample=0.9, verbose=0, random_seed=random_state,
            loss_function="RMSE"
        )
        self.classifier = classifier or CatBoostClassifier(
            depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            subsample=0.9, auto_class_weights="Balanced",
            verbose=0, random_state=random_state
        )

        # ForestDR 头部
        self.forest_head = dict(
            n_estimators=800, max_depth=12, min_samples_split=4, min_samples_leaf=10,
            max_features='sqrt', max_samples=0.4, min_balancedness_tol=0.5,
            honest=True, subforest_size=4, cv=3, min_propensity=1e-3,
            categories='auto', random_state=random_state, n_jobs=-1
        )

        # 输出/缓存
        self.study_ = None
        self._fitted = False
        self._dr = None
        self._cache = dict()  # 存放用于 report 的数据与指标

    # ---------- 工具 ----------
    @staticmethod
    def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
        e = np.clip(e, trim, 1 - trim)
        return (t*(y - mu1)/e) - ((1-t)*(y - mu0)/(1-e)) + (mu1 - mu0)

    @staticmethod
    def _area_cumgain_centered(psi, scores):
        order = np.argsort(-scores)
        psi_ord = psi[order]
        psi_c = psi_ord - psi_ord.mean()
        csum = np.cumsum(psi_c)
        x = np.arange(1, len(csum)+1) / len(csum)
        return float(np.trapz(csum, x))

    @staticmethod
    def _policy_values(psi, scores, ks=(0.1,0.2,0.3)):
        out = {}
        order = np.argsort(-scores)
        psi_ord = psi[order]
        n = len(psi_ord)
        for k in ks:
            m = max(1, int(np.floor(n * k)))
            out[f'policy@{int(k*100)}'] = float(np.sum(psi_ord[:m]))
        return out

    def _oof_propensity(self, X, T, clf_proto, trim=None):
        trim = self.trim if trim is None else trim
        e_oof = np.zeros(len(T), float)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in skf.split(X, T):
            clf = clone(clf_proto)
            clf.fit(X[tr], T[tr])
            e_oof[va] = clf.predict_proba(X[va])[:, 1]
        return np.clip(e_oof, trim, 1 - trim)

    # 更稳的回归器拟合（防 Tweedie 在小样本/常数目标时崩）
    def _safe_fit_predict_reg(self, reg, X_tr, y_tr, X_va,
                              min_n=30, min_std=1e-8, zero_share_cap=0.98):
        y_tr = np.asarray(y_tr, float)
        if (len(y_tr) < min_n) or (np.nanstd(y_tr) < min_std) or \
           (np.mean(np.isclose(y_tr, 0.0)) >= zero_share_cap):
            mu = float(np.nanmean(y_tr)) if len(y_tr) else 0.0
            return np.full(len(X_va), mu, dtype=float)
        reg.fit(X_tr, y_tr)
        return reg.predict(X_va)

    def _oof_nauuc_on_band(self, Xb, Tb, Yb, reg_proto, clf_proto, forest_kwargs, trim=None):
        trim = self.trim if trim is None else trim
        psi_oof = np.full(len(Xb), np.nan); tau_oof = np.full(len(Xb), np.nan)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(Xb):
            X_tr, X_va = Xb[tr], Xb[va]; T_tr, T_va = Tb[tr], Tb[va]; Y_tr, Y_va = Yb[tr], Yb[va]
            clf = clone(clf_proto).fit(X_tr, T_tr)
            e_va = np.clip(clf.predict_proba(X_va)[:,1], trim, 1-trim)

            reg1 = clone(reg_proto); reg0 = clone(reg_proto)
            mu1_va = np.zeros(len(va)); mu0_va = np.zeros(len(va))
            if (T_tr==1).any(): mu1_va = self._safe_fit_predict_reg(reg1, X_tr[T_tr==1], Y_tr[T_tr==1], X_va)
            if (T_tr==0).any(): mu0_va = self._safe_fit_predict_reg(reg0, X_tr[T_tr==0], Y_tr[T_tr==0], X_va)
            psi_oof[va] = self._aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=trim)

            if (T_tr==1).any() and (T_tr==0).any():
                dr = ForestDRLearner(model_regression=clone(reg_proto),
                                     model_propensity=clone(clf_proto),
                                     **forest_kwargs)
                dr.fit(Y_tr, T_tr, X=X_tr)
                tau_oof[va] = dr.effect(X_va)

        m = ~np.isnan(psi_oof) & ~np.isnan(tau_oof)
        if m.sum() < max(30, 2*self.n_splits):
            return 0.0, dict(note="too few valid oof points", n=int(m.sum()))
        psi_m, tau_m = psi_oof[m], tau_oof[m]
        area_model  = self._area_cumgain_centered(psi_m, tau_m)
        area_oracle = self._area_cumgain_centered(psi_m, psi_m)
        nauuc = float(np.clip(area_model/area_oracle, 0.0, 1.0)) if abs(area_oracle) > 1e-12 else 0.0
        pol = self._policy_values(psi_m, tau_m, ks=self.nauuc_policy_ks)
        return nauuc, dict(area_model=area_model, area_oracle=area_oracle, n=int(m.sum()), **pol)

    # ---------- 构建器 ----------
    def _ensure_divisible(self, frh: dict):
        frh = frh.copy()
        s = int(frh.get('subforest_size', 1) or 1)
        n = int(frh.get('n_estimators', 100))
        if n % s != 0:
            frh['n_estimators'] = int((n + s - 1) // s * s)
        return frh

    def _build_forest_head(self, trial):
        if not self.search_forest_head:
            return self._ensure_divisible(self.forest_head)
        s = trial.suggest_int("fr_subforest_size", 2, 8)
        m = trial.suggest_int("fr_num_subforests", 10, 30)
        frh = dict(
            n_estimators=s * m,
            subforest_size=s,
            max_depth=trial.suggest_int("fr_max_depth", 4, 8),
            min_samples_split=trial.suggest_int("fr_min_split", 2, 20),
            min_samples_leaf=trial.suggest_int("fr_min_leaf", 5, 50),
            max_features=trial.suggest_categorical("fr_max_features", ["auto","sqrt","log2", 0.5, 0.8]),
            max_samples=trial.suggest_float("fr_max_samples", 0.3, 0.5),
            min_balancedness_tol=trial.suggest_float("fr_min_bal_tol", 0.3, 0.5),
            honest=trial.suggest_categorical("fr_honest", [True, False]),
            cv=trial.suggest_int("fr_cv", 2, 5),
            categories='auto', random_state=self.random_state, n_jobs=-1
        )
        return frh

    def _build_reg(self, trial):
        if self.reg_loss in ("RMSE","Tweedie"):
            loss_choice = self.reg_loss
        else:
            loss_choice = trial.suggest_categorical("reg_loss", ["RMSE", "Tweedie"])
        params = dict(
            depth=trial.suggest_int("reg_depth", 4, 8),
            learning_rate=trial.suggest_float("reg_lr", 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("reg_l2", 1e-2, 10.0, log=True),
            random_seed=self.random_state, verbose=0,
        )
        if loss_choice == "RMSE":
            params.update(loss_function="RMSE", eval_metric="RMSE")
        else:
            vp = self.reg_tweedie_p if self.reg_loss=="Tweedie" else trial.suggest_float("reg_tweedie_p", 1.1, 1.8)
            params.update(loss_function=f"Tweedie:variance_power={vp}",
                          eval_metric=f"Tweedie:variance_power={vp}")
        return CatBoostRegressor(**params)

    def _build_clf(self, trial):
        params = dict(
            depth=trial.suggest_int("clf_depth", 3, 8),
            learning_rate=trial.suggest_float("clf_lr", 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("clf_l2", 1e-2, 10.0, log=True),
            auto_class_weights=trial.suggest_categorical("clf_class_wt", [None, "Balanced", "SqrtBalanced"]),
            random_seed=self.random_state, verbose=0,
        )
        base = CatBoostClassifier(**params)
        calib = trial.suggest_categorical("ps_calibration", ["none", "isotonic", "sigmoid"])
        if calib == "none":
            return base
        else:
            cv = trial.suggest_int("ps_calib_cv", 2, 5)
            return CalibratedClassifierCV(estimator=base, method=calib, cv=cv)

    def _tune_by_nauuc(self, X, T, Y, base_band):
        def objective(trial):
            reg = self._build_reg(trial)
            clf = self._build_clf(trial)
            frh = self._build_forest_head(trial)
            trim = trial.suggest_float("trim", 1e-3, 5e-2, log=True)

            e_oof = self._oof_propensity(X, T, clf, trim=trim)
            lo = float(self.nauuc_band[0])  # 固定下界
            band = (e_oof >= lo) & (e_oof <= 1 - lo)
            if band.sum() < max(100, 3*self.n_splits):
                band = base_band

            nauuc, stats = self._oof_nauuc_on_band(X[band], T[band], Y[band], reg, clf, frh, trim=trim)
            cov = float(band.mean()); effn = int(stats.get("n", 0))
            if effn < max(120, 3*self.n_splits): nauuc *= 0.6
            if cov < 0.25: nauuc *= 0.85

            trial.set_user_attr("cov", cov); trial.set_user_attr("n_eff", effn)
            if (self.early_stop is not None) and (nauuc >= self.early_stop):
                trial.study.stop()
            return float(nauuc)
        
        def stop_callback(study, trial):
            if study.best_value is not None and study.best_value >= 0.35:
                print(f"🎉 提前停止：best nAUUC={study.best_value:.3f} ≥ 0.35")
                study.stop()

        pruner = optuna.pruners.MedianPruner(n_startup_trials=20)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=-1,callbacks=[stop_callback])

        self.study_ = study
        best = study.best_trial

        # 用最佳超参重建模型与带
        best_reg = self._build_reg(best)
        best_clf = self._build_clf(best)
        best_frh = self._build_forest_head(best)
        best_trim = best.params.get("trim", self.trim)
        lo_best = float(self.nauuc_band[0])

        e_oof_best = self._oof_propensity(X, T, best_clf, trim=best_trim)
        best_band = (e_oof_best >= lo_best) & (e_oof_best <= 1 - lo_best)
        if best_band.sum() < max(100, 15): best_band = base_band

        # 覆盖实例属性
        self.regressor = best_reg
        self.classifier = best_clf
        self.forest_head = best_frh
        self.trim = best_trim
        self.nauuc_band = (float(lo_best), float(1 - lo_best))
        return best_band

    # ---------- 训练 ----------
    def fit(self, X, T, Y):
        """
        训练 ForestDRLearner 并返回训练好的 dr 模型（不做全量打分）
        同时在内部缓存用于 report() 的评估所需数据
        """
        X = np.asarray(X); T = np.asarray(T).astype(int); Y = np.asarray(Y).astype(float)

        # 保底带：用当前 classifier 的 OOF-PS
        e_base = self._oof_propensity(X, T, self.classifier, trim=self.trim)
        lo0, hi0 = self.nauuc_band
        base_band = (e_base >= lo0) & (e_base <= hi0)

        # 可选：nAUUC 调参
        if self.tune_nauuc:
            if self.verbose: print("[NAUUCTuner] running...")
            best_band = self._tune_by_nauuc(X, T, Y, base_band=base_band)
            if self.verbose and self.study_ is not None:
                print(f"[NAUUCTuner] best_params={self.study_.best_params}, best_value={self.study_.best_value:.3f}")
        else:
            best_band = base_band

        # 训练 DR 模型（仅带内）
        dr_learner = ForestDRLearner(
            model_regression=self.regressor,
            model_propensity=self.classifier,
            **self.forest_head
        )
        dr_learner.fit(Y[best_band], T[best_band], X=X[best_band])

        # === 缓存用于 report 的数据与指标（不做全量打分） ===
        # 1) 重新用当前 classifier 计算全量 OOF e，并确定固定评估带
        e_oof = self._oof_propensity(X, T, self.classifier, trim=self.trim)
        band = (e_oof >= self.nauuc_band[0]) & (e_oof <= self.nauuc_band[1])
        coverage = float(band.mean()); n_band = int(band.sum())

        # 2) 带内 OOF nAUUC + Policy@k
        fr_kwargs = self.forest_head# {k:v for k,v in self.forest_head.items() if k not in {'categories','random_state','n_jobs'}}
        nauuc, stats = self._oof_nauuc_on_band(X[band], T[band], Y[band],
                                               self.regressor, self.classifier, fr_kwargs, trim=self.trim)

        # 3) 缓存
        self._cache = dict(
            coverage=coverage,
            n=n_band,
            nauuc=nauuc,
            stats=stats,
            band=self.nauuc_band
        )

        self._fitted = True
        self._dr = dr_learner
        return dr_learner

    # ---------- 报告 ----------
    def report(self):
        """
        仅在 fit() 之后可调用。返回 (text, metrics_dict)
        """
        assert self._fitted, "请先调用 fit(X, T, Y) 再调用 report()。"
        cov = self._cache.get("coverage", float("nan"))
        n   = self._cache.get("n", 0)
        nauuc = self._cache.get("nauuc", float("nan"))
        stats = self._cache.get("stats", {})
        lo, hi = self._cache.get("band", self.nauuc_band)

        pol_keys = [f"policy@{int(k*100)}" for k in self.nauuc_policy_ks]
        pol_str = " / ".join([f"{k.split('@')[1]}%={stats.get(k, float('nan')):.2f}" for k in pol_keys])

        passed = bool(nauuc >= self.min_nauuc)
        txt = (f"【HTE 可用性报告】\n"
               f"- 评估带 e∈[{lo:.2f},{hi:.2f}] 覆盖率={cov:.2%}, n={n}\n"
               f"- nAUUC={nauuc:.3f}（阈值≥{self.min_nauuc:.2f}）=> {'✅ 可用' if passed else '❌ 暂不建议用于投放'}\n"
               f"- area_model={stats.get('area_model', float('nan')):.2f}, "
               f"area_oracle={stats.get('area_oracle', float('nan')):.2f}\n"
               f"- Policy@k：{pol_str}")

        # metrics = dict(enabled=True, coverage=cov, nauuc=nauuc, passed=passed, **stats)
        return txt

    # ---------- 取回训练好的 DR 模型（可选便捷方法） ----------
    def get_model(self):
        assert self._fitted and (self._dr is not None), "请先调用 fit(X, T, Y)。"
        return self._dr
