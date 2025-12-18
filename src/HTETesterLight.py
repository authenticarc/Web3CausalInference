# === HTEtester.py (light + optuna) ===
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from catboost import CatBoostClassifier, CatBoostRegressor
from econml.dr import ForestDRLearner
import optuna


class HTETesterLight:
    """
    轻量 + nAUUC 调参与带选择版 HTE Tester

    目标场景：
      - 需要比默认模型更强的 uplift 排序能力；
      - 希望用 nAUUC 做一个“模型好坏”的客观标尺，但不追求特别重的 placebo / neg-control。

    做的事情：
      1) （可选）用 nAUUC + Optuna 调 CatBoost reg/clf 超参（在 overlap 带内）
      2) cross-fitting 估计 μ1(x), μ0(x), e(x)
      3) 计算 AIPW 伪效应 ψ_i
      4) 在 overlap 带内训练一个 ForestDRLearner，得到 τ̂(x)
      5) 提供简单的 report() 看 uplift 分布、带覆盖情况、τ̂–ψ 相关

    主要属性（fit 后）：
      - e_hat_      : OOF 倾向得分 e(x)
      - mu1_hat_    : OOF 预测 μ1(x)
      - mu0_hat_    : OOF 预测 μ0(x)
      - psi_        : AIPW 伪效应 ψ_i
      - band_mask_  : overlap 带掩码（基于 e_hat_ 和 band）
      - tau_hat_    : ForestDRLearner 估计 τ̂(x)
      - dr_         : 训练好的 ForestDRLearner 实例
      - best_nauuc_ : 调参得到的最佳 nAUUC（如启用）
      - best_params_: Optuna 最佳超参（如启用）
    """

    def __init__(
        self,
        regressor=None,
        classifier=None,
        n_splits: int = 5,
        trim: float = 0.01,
        band=(0.3, 0.7),         # e(x) overlap 初始带
        random_state: int = 42,
        verbose: int = 1,
        # ==== 新增：nAUUC 调参相关 ====
        tune_nauuc: bool = False,
        n_trials: int = 40,
        early_stop: float = 0.45,   # nAUUC 达到这个就停
    ):
        self.n_splits = int(n_splits)
        self.trim = float(trim)
        self.band = tuple(band)
        self.random_state = int(random_state)
        self.verbose = int(verbose)
        self.tune_nauuc = bool(tune_nauuc)
        self.n_trials = int(n_trials)
        self.early_stop = early_stop

        # 基础 learner（可被调参覆盖）
        self.regressor = regressor or CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            subsample=0.9,
            verbose=0,
            random_seed=self.random_state,
            loss_function="RMSE",
        )
        self.classifier = classifier or CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            subsample=0.9,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=self.random_state,
        )

        # ForestDR 配置：给一个中等规模，既不太慢也有表达力
        self.forest_params = dict(
            n_estimators=600,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=10,
            max_features="sqrt",
            max_samples=0.4,
            min_balancedness_tol=0.5,
            honest=True,
            subforest_size=4,
            cv=3,
            min_propensity=1e-3,
            categories="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )

        # 输出 / 缓存
        self._fitted = False
        self.dr_ = None
        self.e_hat_ = None
        self.mu1_hat_ = None
        self.mu0_hat_ = None
        self.psi_ = None
        self.band_mask_ = None
        self.tau_hat_ = None

        # 调参结果
        self.best_nauuc_ = None
        self.best_params_ = None

    # ---------- 工具 ----------
    @staticmethod
    def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
        """AIPW 伪效应 ψ_i"""
        e = np.clip(e, trim, 1 - trim)
        return (t * (y - mu1) / e) - ((1 - t) * (y - mu0) / (1 - e)) + (mu1 - mu0)

    def _crossfit_mu_e(self, X, T, Y):
        """
        用同一套 StratifiedKFold 做 OOF:
          - e_hat_: 倾向得分
          - mu1_hat_: μ1(x)
          - mu0_hat_: μ0(x)
        """
        n = len(Y)
        e_hat = np.zeros(n, dtype=float)
        mu1_hat = np.zeros(n, dtype=float)
        mu0_hat = np.zeros(n, dtype=float)

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold_id, (tr, va) in enumerate(skf.split(X, T), 1):
            if self.verbose >= 2:
                print(f"[CF] fold {fold_id}/{self.n_splits}")

            X_tr, X_va = X[tr], X[va]
            T_tr, Y_tr = T[tr], Y[tr]

            # 倾向模型
            clf = clone(self.classifier)
            if hasattr(clf, "set_params"):
                try:
                    clf.set_params(thread_count=1)
                except Exception:
                    pass
            clf.fit(X_tr, T_tr)
            e_hat[va] = clf.predict_proba(X_va)[:, 1]

            # 结果模型：分别在 treated / control 上回归
            reg1 = clone(self.regressor)
            reg0 = clone(self.regressor)
            for mdl in (reg1, reg0):
                if hasattr(mdl, "set_params"):
                    try:
                        mdl.set_params(thread_count=1)
                    except Exception:
                        pass

            if (T_tr == 1).any():
                reg1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
                mu1_hat[va] = reg1.predict(X_va)
            else:
                mu1_hat[va] = Y_tr.mean()

            if (T_tr == 0).any():
                reg0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
                mu0_hat[va] = reg0.predict(X_va)
            else:
                mu0_hat[va] = Y_tr.mean()

        # 裁剪 e_hat 提升稳定性
        e_hat = np.clip(e_hat, self.trim, 1 - self.trim)
        return mu1_hat, mu0_hat, e_hat

    # === 调参专用：更轻量的 OOF e(x) ===
    def _oof_propensity(self, X, T, clf_proto):
        e_oof = np.zeros(len(T), float)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for tr, va in skf.split(X, T):
            clf = clone(clf_proto)
            if hasattr(clf, "set_params"):
                try:
                    clf.set_params(thread_count=1)
                except Exception:
                    pass
            clf.fit(X[tr], T[tr])
            e_oof[va] = clf.predict_proba(X[va])[:, 1]
        return np.clip(e_oof, self.trim, 1 - self.trim)

    # === 调参专用：在 band 内算 OOF nAUUC ===
    def _oof_nauuc_on_band(self, Xb, Tb, Yb, reg_proto, clf_proto):
        psi_oof = np.full(len(Xb), np.nan)
        tau_oof = np.full(len(Xb), np.nan)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for tr, va in kf.split(Xb):
            X_tr, X_va = Xb[tr], Xb[va]
            T_tr, T_va = Tb[tr], Tb[va]
            Y_tr, Y_va = Yb[tr], Yb[va]

            # 倾向
            clf = clone(clf_proto)
            if hasattr(clf, "set_params"):
                try:
                    clf.set_params(thread_count=1)
                except Exception:
                    pass
            clf.fit(X_tr, T_tr)
            e_va = np.clip(clf.predict_proba(X_va)[:, 1], self.trim, 1 - self.trim)

            # 结果回归 μ1, μ0
            reg1 = clone(reg_proto)
            reg0 = clone(reg_proto)
            for mdl in (reg1, reg0):
                if hasattr(mdl, "set_params"):
                    try:
                        mdl.set_params(thread_count=1)
                    except Exception:
                        pass

            mu1_va = np.zeros(len(va))
            mu0_va = np.zeros(len(va))
            if (T_tr == 1).any():
                reg1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
                mu1_va = reg1.predict(X_va)
            if (T_tr == 0).any():
                reg0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
                mu0_va = reg0.predict(X_va)

            psi_oof[va] = self._aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)

            # DR uplift τ̂_oof
            if (T_tr == 1).any() and (T_tr == 0).any():
                dr = ForestDRLearner(
                    model_regression=clone(reg_proto),
                    model_propensity=clone(clf_proto),
                    **self.forest_params,
                )
                dr.fit(Y_tr, T_tr, X=X_tr)
                tau_oof[va] = dr.effect(X_va)

        m = ~np.isnan(psi_oof) & ~np.isnan(tau_oof)
        if m.sum() < max(30, 2 * self.n_splits):
            return 0.0, dict(note="too few valid oof points", n=int(m.sum()))

        psi_m = psi_oof[m]
        tau_m = tau_oof[m]

        # 计算 centered cumulative gain 的面积
        def _area_cumgain_centered(psi, scores):
            order = np.argsort(-scores)
            psi_ord = psi[order]
            psi_c = psi_ord - psi_ord.mean()
            csum = np.cumsum(psi_c)
            x = np.arange(1, len(csum) + 1) / len(csum)
            return float(np.trapz(csum, x))

        area_model = _area_cumgain_centered(psi_m, tau_m)
        area_oracle = _area_cumgain_centered(psi_m, psi_m)
        if abs(area_oracle) < 1e-12:
            nauuc = 0.0
        else:
            nauuc = float(np.clip(area_model / area_oracle, 0.0, 1.0))

        return nauuc, dict(area_model=area_model, area_oracle=area_oracle, n=int(m.sum()))

    # === 用 nAUUC 调参 CatBoost 超参 ===
    def _tune_by_nauuc(self, X, T, Y):
        if self.verbose:
            print("[HTETesterLight] nAUUC tuning with Optuna...")

        # 基于当前 classifier 的保底 band
        e_base = self._oof_propensity(X, T, self.classifier)
        lo0, hi0 = self.band
        base_band = (e_base >= lo0) & (e_base <= hi0)
        if base_band.sum() < max(100, 3 * self.n_splits):
            base_band = np.ones_like(e_base, dtype=bool)

        def objective(trial):
            # 简化版搜索空间（避免太重）
            reg = CatBoostRegressor(
                depth=trial.suggest_int("reg_depth", 4, 8),
                learning_rate=trial.suggest_float("reg_lr", 1e-3, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("reg_l2", 1e-2, 10.0, log=True),
                subsample=trial.suggest_float("reg_subsample", 0.7, 1.0),
                loss_function="RMSE",
                random_seed=self.random_state,
                verbose=0,
            )
            clf = CatBoostClassifier(
                depth=trial.suggest_int("clf_depth", 3, 8),
                learning_rate=trial.suggest_float("clf_lr", 1e-3, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("clf_l2", 1e-2, 10.0, log=True),
                subsample=trial.suggest_float("clf_subsample", 0.7, 1.0),
                auto_class_weights=trial.suggest_categorical(
                    "clf_class_wt", [None, "Balanced", "SqrtBalanced"]
                ),
                random_seed=self.random_state,
                verbose=0,
            )

            # 用这个 clf 算 OOF e(x)，重新定义 band
            e_oof = self._oof_propensity(X, T, clf)
            lo, hi = self.band
            band = (e_oof >= lo) & (e_oof <= hi)
            if band.sum() < max(100, 3 * self.n_splits):
                band = base_band

            Xb, Tb, Yb = X[band], T[band], Y[band]
            nauuc, stats = self._oof_nauuc_on_band(Xb, Tb, Yb, reg, clf)

            # 轻微惩罚过窄的 band
            cov = float(band.mean())
            if cov < 0.25:
                nauuc *= 0.9

            trial.set_user_attr("coverage", cov)
            trial.set_user_attr("n_eff", stats.get("n", 0))

            # 提前停止
            if (self.early_stop is not None) and (nauuc >= self.early_stop):
                trial.study.stop()
            return float(nauuc)

        def stop_callback(study, trial):
            if study.best_value is not None and study.best_value >= 0.35:
                print(
                    f"🎉 提前停止：best nAUUC={study.best_value:.3f} ≥ 0.35"
                )
                study.stop()

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=(self.verbose > 0),
            n_jobs=1,
            callbacks=[stop_callback],
        )

        best = study.best_trial
        self.best_nauuc_ = float(study.best_value)
        self.best_params_ = best.params

        if self.verbose:
            cov = best.user_attrs.get("coverage", float("nan"))
            neff = best.user_attrs.get("n_eff", 0)
            print(
                f"[HTETesterLight] tuning done. best nAUUC={self.best_nauuc_:.3f}, "
                f"coverage={cov:.2%}, n_eff={neff}"
            )
            print(f"[HTETesterLight] best params: {self.best_params_}")

        # 用最佳超参重建 regressor / classifier
        self.regressor = CatBoostRegressor(
            depth=best.params["reg_depth"],
            learning_rate=best.params["reg_lr"],
            l2_leaf_reg=best.params["reg_l2"],
            subsample=best.params["reg_subsample"],
            loss_function="RMSE",
            random_seed=self.random_state,
            verbose=0,
        )
        self.classifier = CatBoostClassifier(
            depth=best.params["clf_depth"],
            learning_rate=best.params["clf_lr"],
            l2_leaf_reg=best.params["clf_l2"],
            subsample=best.params["clf_subsample"],
            auto_class_weights=best.params["clf_class_wt"],
            random_seed=self.random_state,
            verbose=0,
        )

    # ---------- 训练 ----------
    def fit(self, X, T, Y):
        """
        训练 ForestDRLearner 并返回训练好的 dr 模型。
        同时缓存：
          - e_hat_, mu1_hat_, mu0_hat_, psi_, band_mask_, tau_hat_
        """
        X = np.asarray(X)
        T = np.asarray(T).astype(int)
        Y = np.asarray(Y).astype(float)

        if self.tune_nauuc:
            # 先调参再正式 cross-fit
            self._tune_by_nauuc(X, T, Y)

        if self.verbose >= 1:
            print(f"[HTETester] n={len(Y)}, n_splits={self.n_splits}, band={self.band}")

        # 1) cross-fit μ1, μ0, e
        mu1_hat, mu0_hat, e_hat = self._crossfit_mu_e(X, T, Y)

        # 2) AIPW 伪效应 ψ_i
        psi = self._aipw_pseudo(Y, T, mu1_hat, mu0_hat, e_hat, trim=self.trim)

        # 3) 根据 e_hat 定义 overlap 带
        lo, hi = self.band
        band_mask = (e_hat >= lo) & (e_hat <= hi)
        if band_mask.sum() < max(50, 3 * self.n_splits):
            if self.verbose:
                print(f"[HTETester] band too narrow, keeping all samples.")
            band_mask = np.ones_like(e_hat, dtype=bool)

        if self.verbose >= 1:
            print(
                f"[HTETester] overlap band e∈[{lo:.2f},{hi:.2f}] 覆盖率={band_mask.mean():.2%}, "
                f"n_band={band_mask.sum()}"
            )

        # 4) 在带内训练 DR-Learner
        dr = ForestDRLearner(
            model_regression=clone(self.regressor),
            model_propensity=clone(self.classifier),
            **self.forest_params,
        )
        dr.fit(Y[band_mask], T[band_mask], X=X[band_mask])

        # 5) 对全体样本打 τ̂(x)
        tau_hat = dr.effect(X)

        # 6) 缓存
        self.mu1_hat_ = mu1_hat
        self.mu0_hat_ = mu0_hat
        self.e_hat_ = e_hat
        self.psi_ = psi
        self.band_mask_ = band_mask
        self.tau_hat_ = tau_hat
        self.dr_ = dr
        self._fitted = True

        return dr

    # ---------- 报告 ----------
    def report(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)) -> str:
        """
        fit() 之后调用，输出一个简易 HTE 报告：
          - overlap 带覆盖情况
          - τ̂(x) 分布（全体 & 带内）
          - ψ 与 τ̂ 的相关性（带内）
          - 如调参，附带 best nAUUC
        """
        assert self._fitted, "请先调用 fit(X, T, Y)。"

        e = self.e_hat_
        psi = self.psi_
        tau = self.tau_hat_
        band = self.band_mask_
        lo, hi = self.band

        lines = []
        lines.append("【HTE 诊断报告（轻量 + 调参版）】")
        lines.append(
            f"- overlap 带 e∈[{lo:.2f},{hi:.2f}] 覆盖率={band.mean():.2%}, "
            f"n_band={int(band.sum())}"
        )

        if self.best_nauuc_ is not None:
            lines.append(f"- 调参 best nAUUC = {self.best_nauuc_:.3f}")

        # uplift 分布
        def _fmt_q(arr, name):
            arr = np.asarray(arr, float)
            qs = np.quantile(arr, quantiles)
            q_str = " / ".join(
                [f"{int(q*100)}%={v:.3f}" for q, v in zip(quantiles, qs)]
            )
            return f"  · {name} quantiles: {q_str}"

        lines.append(f"- τ̂(x) 分布：")
        lines.append(_fmt_q(tau, "全体"))
        lines.append(_fmt_q(tau[band], "带内"))

        # ψ 分布（更多是 sanity check）
        lines.append(f"- ψ (AIPW 伪效应) 分布（带内）：")
        lines.append(_fmt_q(psi[band], "ψ 带内"))

        # τ̂ 与 ψ 的相关性（带内）
        try:
            corr = np.corrcoef(tau[band], psi[band])[0, 1]
            lines.append(f"- τ̂ 与 ψ 在带内的相关系数：corr(τ̂, ψ) = {corr:.3f}")
        except Exception:
            lines.append("- τ̂ 与 ψ 相关性：计算失败（样本过少或全常数）")

        return "\n".join(lines)

    # ---------- 取回训练好的 DR 模型（便捷方法） ----------
    def get_model(self):
        assert self._fitted and (self.dr_ is not None), "请先调用 fit(X, T, Y)。"
        return self.dr_
