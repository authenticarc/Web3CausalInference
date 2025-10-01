from dataclasses import dataclass
import numpy as np, optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.calibration import CalibratedClassifierCV

import pandas as pd
import xgboost as xgb
from pandas.api.types import (
    is_numeric_dtype, is_integer_dtype, is_string_dtype,
    is_categorical_dtype, is_datetime64_any_dtype
)

# ============ 小工具 ============

def _area_cumgain_centered(psi, scores):
    order = np.argsort(-scores)
    psi_ord = psi[order]
    psi_c = psi_ord - psi_ord.mean()
    csum = np.cumsum(psi_c)
    x = np.arange(1, len(csum)+1) / len(csum)
    return float(np.trapz(csum, x))

def _policy_values(psi, scores, ks=(0.1,0.2,0.3)):
    out = {}
    order = np.argsort(-scores)
    psi_ord = psi[order]
    n = len(psi_ord)
    for k in ks:
        m = max(1, int(np.floor(n * k)))
        out[f'policy@{int(k*100)}'] = float(np.sum(psi_ord[:m]))
    return out

def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
    e = np.clip(e, trim, 1 - trim)
    return (t*(y - mu1)/e) - ((1-t)*(y - mu0)/(1-e)) + (mu1 - mu0)

# ============ 轻量 DR 模型包装 ============

class DRModel:
    """轻量封装：提供 effect(X)；X 可为 DataFrame（将使用 tester 的编码映射）"""
    def __init__(self, tester, ps_model, m1_model, m0_model, tau_model, trim=1e-3):
        self._tester = tester
        self.ps_model = ps_model
        self.m1_model = m1_model
        self.m0_model = m0_model
        self.tau_model = tau_model
        self.trim = float(trim)

    def effect(self, X):
        X_enc = self._tester._encode_X(X)
        return self.tau_model.predict(X_enc)

    def propensity(self, X):
        X_enc = self._tester._encode_X(X)
        return np.clip(self.ps_model.predict_proba(X_enc)[:,1], self.trim, 1-self.trim)

    def mu_hat(self, X):
        X_enc = self._tester._encode_X(X)
        return self.m1_model.predict(X_enc), self.m0_model.predict(X_enc)

# ============ 主类：HTETester（XGBoost + DR + 自动编码） ============

class HTETester:
    """
    只负责 HTE 训练（XGBoost + 手写 DR-learner）：
      - （可选）以 nAUUC 为目标做超参与带选择
      - 在选定 overlap 带内训练 DR：ps/m1/m0 + tau head（XGBRegressor）
      - fit(...) 返回 DRModel（.effect(X) 可直接对原始 DataFrame 打分）
      - report() 使用 fit 时缓存的数据生成可用性报告
    """
    def __init__(self,
                 regressor=None,
                 classifier=None,
                 n_splits=5,
                 trim=0.01,
                 nauuc_band=(0.3, 0.7),
                 nauuc_policy_ks=(0.1, 0.2, 0.3),
                 min_nauuc=0.35,
                 tune_nauuc=False,
                 n_trials=200,
                 early_stop=0.60,
                 search_tau_head=True,     # ← 原来的 search_forest_head 改名
                 reg_loss="auto",          # 'rmse'|'tweedie'|'auto'
                 reg_tweedie_p=1.3,
                 use_gpu=False,
                 random_state=42,
                 verbose=1,
                 # 自动编码配置
                 auto_encode_cats=True,
                 int_as_cat_unique_thresh=30,
                 unique_ratio_thresh=0.05,
                 rare_freq_ratio=0.001,
                 max_onehot_levels=200):
        self.n_splits = int(n_splits)
        self.trim = float(trim)
        self.nauuc_band = tuple(nauuc_band)
        self.nauuc_policy_ks = tuple(nauuc_policy_ks)
        self.min_nauuc = float(min_nauuc)
        self.tune_nauuc = bool(tune_nauuc)
        self.n_trials = int(n_trials)
        self.early_stop = early_stop
        self.search_tau_head = bool(search_tau_head)
        self.reg_loss = reg_loss.lower() if isinstance(reg_loss, str) else "auto"
        self.reg_tweedie_p = float(reg_tweedie_p)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

        # 设备参数（XGBoost 2.0+ 推荐）
        self._device_args = {"tree_method": "hist", "device": ("cuda" if use_gpu else "cpu")}

        # base learners（可被调参覆盖）
        self.regressor = regressor or xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=6, learning_rate=0.05, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0,
            random_state=self.random_state, n_jobs=1, **self._device_args
        )
        self.classifier = classifier or xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=6, learning_rate=0.05, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0,
            random_state=self.random_state, n_jobs=1, **self._device_args
        )

        # tau head（XGBRegressor 的默认头部，可被调参覆盖）
        self.tau_head = dict(
            max_depth=6, n_estimators=600, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
            min_child_weight=5, objective="reg:squarederror",
            random_state=self.random_state, n_jobs=-1, **self._device_args
        )

        # 输出/缓存
        self.study_ = None
        self._fitted = False
        self._dr = None
        self._cache = dict()  # 存放用于 report 的数据与指标

        # 自动编码映射
        self.auto_encode_cats = bool(auto_encode_cats)
        self.int_as_cat_unique_thresh = int(int_as_cat_unique_thresh)
        self.unique_ratio_thresh = float(unique_ratio_thresh)
        self.rare_freq_ratio = float(rare_freq_ratio)
        self.max_onehot_levels = int(max_onehot_levels)
        self._cat_levels_ = {}
        self._colnames_ = None

    # ---------- 自动识别 & 稳定编码 ----------
    def _infer_cats(self, X: pd.DataFrame):
        cats = []
        n = len(X)
        for c in X.columns:
            s = X[c]
            if is_datetime64_any_dtype(s):
                continue
            nunq = s.nunique(dropna=False)
            ur = nunq / max(n, 1)
            is_cat = (
                is_string_dtype(s) or is_categorical_dtype(s) or
                (is_integer_dtype(s) and nunq <= self.int_as_cat_unique_thresh) or
                (not is_numeric_dtype(s) and ur <= self.unique_ratio_thresh)
            )
            if (not is_cat) and is_numeric_dtype(s) and nunq <= self.int_as_cat_unique_thresh:
                is_cat = True
            if is_cat:
                cats.append(c)
        return cats

    def _fit_cat_maps(self, X: pd.DataFrame, cat_cols):
        self._cat_levels_.clear()
        n = len(X); rare_thresh = max(int(self.rare_freq_ratio*n), 1)
        for c in cat_cols:
            vc = X[c].astype("string").fillna("__NA__").value_counts(dropna=False)
            common = vc[vc >= rare_thresh].index.tolist()
            if len(common) > self.max_onehot_levels:
                common = list(vc.index[: self.max_onehot_levels])
            self._cat_levels_[c] = ["__RARE__"] + [str(v) for v in common]

    def _apply_cat_maps(self, X: pd.DataFrame, cat_cols):
        X2 = X.copy()
        for c in cat_cols:
            X2[c] = X2[c].astype("string").fillna("__NA__")
            levels = set(self._cat_levels_[c][1:])
            X2[c] = X2[c].apply(lambda v: v if v in levels else "__RARE__")
        dummies = []
        for c in cat_cols:
            cats = pd.Categorical(X2[c], categories=self._cat_levels_[c])
            dummies.append(pd.get_dummies(cats, prefix=c, dummy_na=False))
        D = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=X2.index)
        num_cols = [c for c in X.columns if c not in cat_cols and is_numeric_dtype(X[c])]
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce")
        out = pd.concat([X_num, D], axis=1).fillna(0.0)
        return out

    def _encode_X_fit(self, X):
        if not self.auto_encode_cats or not isinstance(X, pd.DataFrame):
            Xn = np.asarray(X, float)
            self._colnames_ = [f"x{j}" for j in range(Xn.shape[1])]
            return Xn
        cats = self._infer_cats(X)
        self._fit_cat_maps(X, cats)
        Xenc = self._apply_cat_maps(X, cats)
        self._colnames_ = list(Xenc.columns)
        return Xenc.to_numpy(float, copy=False)

    def _encode_X(self, X):
        if not self.auto_encode_cats or not isinstance(X, pd.DataFrame):
            Xn = np.asarray(X, float)
            if self._colnames_ is None:
                self._colnames_ = [f"x{j}" for j in range(Xn.shape[1])]
            return Xn
        cat_cols = list(self._cat_levels_.keys())
        Xenc = self._apply_cat_maps(X, cat_cols)
        for c in self._colnames_:
            if c not in Xenc.columns:
                Xenc[c] = 0.0
        Xenc = Xenc[self._colnames_]
        return Xenc.to_numpy(float, copy=False)

    # ---------- OOF 倾向 ----------
    def _oof_propensity(self, X, T, clf_proto, trim=None):
        trim = self.trim if trim is None else trim
        e_oof = np.zeros(len(T), float)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in skf.split(X, T):
            clf = clone(clf_proto)
            # 避免内部并行太多
            try: clf.set_params(n_jobs=1)
            except: pass
            clf.fit(X[tr], T[tr])
            e_oof[va] = clf.predict_proba(X[va])[:, 1]
        return np.clip(e_oof, trim, 1 - trim)

    # ---------- 稳健回归器小工具 ----------
    def _safe_fit_predict_reg(self, reg, X_tr, y_tr, X_va,
                              min_n=30, min_std=1e-8, zero_share_cap=0.98):
        y_tr = np.asarray(y_tr, float)
        if (len(y_tr) < min_n) or (np.nanstd(y_tr) < min_std) or \
           (np.mean(np.isclose(y_tr, 0.0)) >= zero_share_cap):
            mu = float(np.nanmean(y_tr)) if len(y_tr) else 0.0
            return np.full(len(X_va), mu, dtype=float)
        reg.fit(X_tr, y_tr)
        return reg.predict(X_va)

    # ---------- 在带内用 DR 流程做 OOF nAUUC ----------
    def _oof_nauuc_on_band(self, Xb, Tb, Yb, reg_proto, clf_proto, tau_params, trim=None):
        trim = self.trim if trim is None else trim
        psi_oof = np.full(len(Xb), np.nan); tau_oof = np.full(len(Xb), np.nan)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(Xb):
            X_tr, X_va = Xb[tr], Xb[va]; T_tr, T_va = Tb[tr], Tb[va]; Y_tr, Y_va = Yb[tr], Yb[va]

            clf = clone(clf_proto)
            try: clf.set_params(n_jobs=1)
            except: pass
            clf.fit(X_tr, T_tr)
            e_tr = np.clip(clf.predict_proba(X_tr)[:,1], trim, 1-trim)
            e_va = np.clip(clf.predict_proba(X_va)[:,1], trim, 1-trim)

            if (T_tr==1).any() and (T_tr==0).any():
                reg1 = clone(reg_proto); reg0 = clone(reg_proto)
                try: reg1.set_params(n_jobs=1); reg0.set_params(n_jobs=1)
                except: pass
                mu1_va = self._safe_fit_predict_reg(reg1, X_tr[T_tr==1], Y_tr[T_tr==1], X_va)
                mu0_va = self._safe_fit_predict_reg(reg0, X_tr[T_tr==0], Y_tr[T_tr==0], X_va)

                # 训练折上拟合 tau head
                mu1_tr = self._safe_fit_predict_reg(clone(reg_proto), X_tr[T_tr==1], Y_tr[T_tr==1], X_tr)
                mu0_tr = self._safe_fit_predict_reg(clone(reg_proto), X_tr[T_tr==0], Y_tr[T_tr==0], X_tr)
                m_tr = e_tr * mu1_tr + (1 - e_tr) * mu0_tr
                Z_tr = ((T_tr - e_tr) / (e_tr * (1 - e_tr))) * (Y_tr - m_tr)
                w_tr = e_tr * (1 - e_tr)

                tau_model = xgb.XGBRegressor(**tau_params)
                tau_model.fit(X_tr, Z_tr, sample_weight=w_tr)
                tau_oof[va] = tau_model.predict(X_va)

                psi_oof[va] = _aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=trim)

        m = ~np.isnan(psi_oof) & ~np.isnan(tau_oof)
        if m.sum() < max(30, 2*self.n_splits):
            return 0.0, dict(note="too few valid oof points", n=int(m.sum()))
        psi_m, tau_m = psi_oof[m], tau_oof[m]
        area_model  = _area_cumgain_centered(psi_m, tau_m)
        area_oracle = _area_cumgain_centered(psi_m, psi_m)
        nauuc = float(np.clip(area_model/area_oracle, 0.0, 1.0)) if abs(area_oracle) > 1e-12 else 0.0
        pol = _policy_values(psi_m, tau_m, ks=self.nauuc_policy_ks)
        return nauuc, dict(area_model=area_model, area_oracle=area_oracle, n=int(m.sum()), **pol)

    # ---------- 构建器 ----------
    def _build_reg(self, trial):
        # 选择损失
        if self.reg_loss in ("rmse", "tweedie"):
            loss = self.reg_loss
        else:
            loss = trial.suggest_categorical("reg_loss", ["rmse", "tweedie"])
        params = dict(
            max_depth=trial.suggest_int("reg_max_depth", 3, 8),
            learning_rate=trial.suggest_float("reg_lr", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("reg_n_estimators", 300, 1200),
            subsample=trial.suggest_float("reg_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("reg_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
            min_child_weight=trial.suggest_int("reg_min_child_weight", 1, 15),
            random_state=self.random_state, n_jobs=1, **self._device_args
        )
        if loss == "tweedie":
            p = self.reg_tweedie_p if self.reg_loss=="tweedie" else trial.suggest_float("reg_tweedie_p", 1.1, 1.9)
            params.update(objective="reg:tweedie", tweedie_variance_power=p)
        else:
            params.update(objective="reg:squarederror")
        return xgb.XGBRegressor(**params)

    def _build_clf(self, trial):
        params = dict(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=trial.suggest_int("clf_max_depth", 3, 8),
            learning_rate=trial.suggest_float("clf_lr", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("clf_n_estimators", 300, 1200),
            subsample=trial.suggest_float("clf_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("clf_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("clf_reg_lambda", 0.0, 10.0),
            min_child_weight=trial.suggest_int("clf_min_child_weight", 1, 15),
            random_state=self.random_state, n_jobs=1, **self._device_args
        )
        base = xgb.XGBClassifier(**params)
        calib = trial.suggest_categorical("ps_calibration", ["none", "isotonic", "sigmoid"])
        if calib == "none":
            return base
        cv = trial.suggest_int("ps_calib_cv", 2, 5)
        return CalibratedClassifierCV(estimator=base, method=calib, cv=cv)

    def _build_tau_head(self, trial):
        if not self.search_tau_head:
            return self.tau_head.copy()
        params = dict(
            max_depth=trial.suggest_int("tau_max_depth", 3, 8),
            learning_rate=trial.suggest_float("tau_lr", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("tau_n_estimators", 400, 1500),
            subsample=trial.suggest_float("tau_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("tau_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("tau_reg_lambda", 0.0, 10.0),
            min_child_weight=trial.suggest_int("tau_min_child_weight", 1, 15),
            objective="reg:squarederror",
            random_state=self.random_state, n_jobs=-1, **self._device_args
        )
        return params

    def _tune_by_nauuc(self, X, T, Y, base_band):
        def objective(trial):
            reg = self._build_reg(trial)
            clf = self._build_clf(trial)
            tau_params = self._build_tau_head(trial)
            trim = trial.suggest_float("trim", 1e-3, 5e-2, log=True)

            e_oof = self._oof_propensity(X, T, clf, trim=trim)
            lo = float(self.nauuc_band[0])
            band = (e_oof >= lo) & (e_oof <= 1 - lo)
            if band.sum() < max(100, 3*self.n_splits):
                band = base_band

            nauuc, stats = self._oof_nauuc_on_band(X[band], T[band], Y[band], reg, clf, tau_params, trim=trim)
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1, callbacks=[stop_callback])

        self.study_ = study
        best = study.best_trial

        # 用最佳超参重建模型与带
        best_reg = self._build_reg(best)
        best_clf = self._build_clf(best)
        best_tau = self._build_tau_head(best)
        best_trim = best.params.get("trim", self.trim)
        lo_best = float(self.nauuc_band[0])

        e_oof_best = self._oof_propensity(X, T, best_clf, trim=best_trim)
        best_band = (e_oof_best >= lo_best) & (e_oof_best <= 1 - lo_best)
        if best_band.sum() < max(100, 15): best_band = base_band

        # 覆盖实例属性
        self.regressor = best_reg
        self.classifier = best_clf
        self.tau_head = best_tau
        self.trim = best_trim
        self.nauuc_band = (float(lo_best), float(1 - lo_best))
        return best_band

    # ---------- 训练 ----------
    def fit(self, X, T, Y):
        """
        训练 DR（ps/m1/m0 + τ head）并返回 DRModel
        同时在内部缓存用于 report() 的评估所需数据
        """
        # 先确立编码映射
        X_enc_full = self._encode_X_fit(X)
        T = np.asarray(T).astype(int)
        Y = np.asarray(Y).astype(float)

        # 保底带：用当前 classifier 的 OOF-PS
        e_base = self._oof_propensity(X_enc_full, T, self.classifier, trim=self.trim)
        lo0, hi0 = self.nauuc_band
        base_band = (e_base >= lo0) & (e_base <= hi0)

        # 可选：nAUUC 调参
        if self.tune_nauuc:
            if self.verbose: print("[NAUUCTuner] running...")
            best_band = self._tune_by_nauuc(X_enc_full, T, Y, base_band=base_band)
            if self.verbose and self.study_ is not None:
                print(f"[NAUUCTuner] best_params={self.study_.best_params}, best_value={self.study_.best_value:.3f}")
        else:
            best_band = base_band

        # —— 带内训练 ps/m1/m0 —— #
        Xb, Tb, Yb = X_enc_full[best_band], T[best_band], Y[best_band]

        ps_model = clone(self.classifier)
        try: ps_model.set_params(n_jobs=-1)
        except: pass
        ps_model.fit(Xb, Tb)
        e_b = np.clip(ps_model.predict_proba(Xb)[:,1], self.trim, 1-self.trim)

        reg1 = clone(self.regressor); reg0 = clone(self.regressor)
        try: reg1.set_params(n_jobs=-1); reg0.set_params(n_jobs=-1)
        except: pass
        reg1.fit(Xb[Tb==1], Yb[Tb==1]) if (Tb==1).any() else None
        reg0.fit(Xb[Tb==0], Yb[Tb==0]) if (Tb==0).any() else None

        mu1_b = reg1.predict(Xb) if (Tb==1).any() else np.full(len(Xb), Yb.mean())
        mu0_b = reg0.predict(Xb) if (Tb==0).any() else np.full(len(Xb), Yb.mean())

        m_b = e_b * mu1_b + (1 - e_b) * mu0_b
        Z_b = ((Tb - e_b) / (e_b * (1 - e_b))) * (Yb - m_b)
        w_b = e_b * (1 - e_b)

        # —— 训练 tau head —— #
        tau_params = self.tau_head.copy()
        tau_model = xgb.XGBRegressor(**tau_params)
        tau_model.fit(Xb, Z_b, sample_weight=w_b)

        # === 缓存用于 report 的数据与指标（不做全量打分） ===
        # 1) 用当前 classifier 计算全量 OOF e，并确定固定评估带
        e_oof = self._oof_propensity(X_enc_full, T, self.classifier, trim=self.trim)
        band = (e_oof >= self.nauuc_band[0]) & (e_oof <= self.nauuc_band[1])
        coverage = float(band.mean()); n_band = int(band.sum())

        # 2) 带内 OOF nAUUC + Policy@k（用 DR 流程）
        nauuc, stats = self._oof_nauuc_on_band(X_enc_full[band], T[band], Y[band],
                                               self.regressor, self.classifier, tau_params, trim=self.trim)

        # 3) 缓存
        self._cache = dict(
            coverage=coverage, n=n_band, nauuc=nauuc, stats=stats, band=self.nauuc_band
        )

        self._fitted = True
        self._dr = DRModel(self, ps_model, reg1, reg0, tau_model, trim=self.trim)
        return self._dr

    # ---------- 报告 ----------
    def report(self):
        """
        仅在 fit() 之后可调用。返回文本
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
        txt = (f"【HTE 可用性报告（XGBoost+DR）】\n"
               f"- 评估带 e∈[{lo:.2f},{hi:.2f}] 覆盖率={cov:.2%}, n={n}\n"
               f"- nAUUC={nauuc:.3f}（阈值≥{self.min_nauuc:.2f}）=> {'✅ 可用' if passed else '❌ 暂不建议用于投放'}\n"
               f"- area_model={stats.get('area_model', float('nan')):.2f}, "
               f"area_oracle={stats.get('area_oracle', float('nan')):.2f}\n"
               f"- Policy@k：{pol_str}")
        return txt

    # ---------- 取回训练好的 DR 模型 ----------
    def get_model(self):
        assert self._fitted and (self._dr is not None), "请先调用 fit(X, T, Y)。"
        return self._dr
    
    # === 新增：从外部承接编码映射（可选）===
    def adopt_encoder(self, cat_levels: dict, colnames: list):
        """把外部调参器/编码器学到的类别映射与列顺序塞进来，后续 effect(X) 会与外部一致"""
        self._cat_levels_ = {k: list(v) for k, v in cat_levels.items()}
        self._colnames_ = list(colnames)

    # === 新增：免训练入口（复用已拟合好的模型）===
    def fit_prefit(self, X, T, Y,
                   prefit_ps, prefit_m1, prefit_m0, prefit_tau=None,
                   band=None):
        """
        X: 原始 DataFrame/ndarray（若想复用外部编码，请先调用 adopt_encoder(...)）
        prefit_ps / prefit_m1 / prefit_m0 / prefit_tau: 已经 fit 好的 sklearn/xgboost 模型
        band: 可选布尔掩码（不提供则按 PS 的 e(x) 和 nauuc_band 自动生成）
        """
        # 如果还没编码映射，先在本数据上拟合一个（与外部不完全一致，但能自洽）
        X_enc_full = self._encode_X_fit(X) if self._colnames_ is None else self._encode_X(X)
        T = np.asarray(T, int); Y = np.asarray(Y, float)

        # —— 直接拿 prefit 的模型打分 —— #
        e_full = np.clip(prefit_ps.predict_proba(X_enc_full)[:, 1], self.trim, 1 - self.trim)
        mu1_full = prefit_m1.predict(X_enc_full)
        mu0_full = prefit_m0.predict(X_enc_full)

        # 评估带
        if band is None:
            lo, hi = self.nauuc_band
            band = (e_full >= lo) & (e_full <= hi)

        # tau 的打分：优先用 prefit_tau，否则回退 m1 - m0
        if prefit_tau is not None:
            tau_full = prefit_tau.predict(X_enc_full)
        else:
            tau_full = mu1_full - mu0_full  # 回退为 T-learner 排序分

        # === 缓存报告指标（不训练）===
        psi = _aipw_pseudo(Y[band], T[band], mu1_full[band], mu0_full[band], e_full[band], trim=self.trim)
        tau_b = tau_full[band]

        # nAUUC（中心化累积增益）
        area_model  = _area_cumgain_centered(psi, tau_b)
        area_oracle = _area_cumgain_centered(psi, psi)
        nauuc = float(np.clip(area_model/area_oracle, 0.0, 1.0)) if abs(area_oracle) > 1e-12 else 0.0
        pol = _policy_values(psi, tau_b, ks=self.nauuc_policy_ks)

        self._cache = dict(
            coverage=float(band.mean()),
            n=int(band.sum()),
            nauuc=nauuc,
            stats=dict(area_model=area_model, area_oracle=area_oracle, **pol),
            band=self.nauuc_band
        )

        # —— 包装成 DRModel（effect/propensity/mu_hat 可直接用）—— #
        self._dr = DRModel(self, prefit_ps, prefit_m1, prefit_m0,
                           prefit_tau if prefit_tau is not None else _TauWrapperFromM10(prefit_m1, prefit_m0, self),
                           trim=self.trim)
        self._fitted = True
        return self._dr

