import numpy as np
import optuna
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, CatBoostClassifier
from econml.dr import ForestDRLearner
from typing import Optional, Tuple


class NAUUCCatBoostTunerV2:
    """
    目标：max  ( nAUUC  - lambda_ci * normalized_CI_width )
    - nAUUC: 用 AIPW 伪效应 ψ 作为“oracle”基准；
    - CI 宽度：基于验证折的 AIPW 影响函数 φ 估计标准误，计算 95% CI 宽度；
      再用 Y 的整体标准差做归一化，确保量纲无关。
    - API 不变：fit_return_models -> (best_reg, best_clf)

    你可以按需调参：
      - lambda_ci：对 CI 宽度的惩罚系数（默认 0.15，越大越偏好更紧的 CI）
      - min_nauuc / max_ci_width：可选的软门槛（默认 None，不启用）
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        random_seed: int = 42,
        categories: Optional[str] = "auto",
        trim: float = 1e-3,             # e(x) 裁剪，提升稳定性
        lambda_ci: float = 0.15,        # CI 惩罚强度（0~0.5常用）
        min_nauuc: Optional[float] = None,   # 例如 0.35；None 表示不设门槛
        max_ci_width: Optional[float] = None, # 例如 2.5（原尺度）；None 表示不设门槛
        verbose: bool = False,
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.categories = categories
        self.trim = trim
        self.lambda_ci = float(lambda_ci)
        self.min_nauuc = min_nauuc
        self.max_ci_width = max_ci_width
        self.verbose = verbose

        self.best_params_: Optional[dict] = None
        self.best_value_: Optional[float] = None
        self.best_reg_: Optional[CatBoostRegressor] = None
        self.best_clf_: Optional[CatBoostClassifier] = None

    # ---------- AIPW ψ / 影响函数 φ / nAUUC ---------- #
    @staticmethod
    def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
        e = np.clip(e, trim, 1 - trim)
        return (t * (y - mu1) / e) - ((1 - t) * (y - mu0) / (1 - e)) + (mu1 - mu0)

    @staticmethod
    def _phi(y, t, mu1, mu0, e, trim=1e-3):
        e = np.clip(e, trim, 1 - trim)
        return (mu1 - mu0) + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)

    @staticmethod
    def _area_from_scores(psi, score):
        order = np.argsort(-score)
        psi_ord = psi[order]
        csum = np.cumsum(psi_ord)
        x = np.arange(1, len(psi_ord) + 1) / len(psi_ord)
        return float(np.trapz(csum, x))

    @staticmethod
    def _nauuc_from(psi, score):
        a_model = NAUUCCatBoostTunerV2._area_from_scores(psi, score)
        a_oracle = NAUUCCatBoostTunerV2._area_from_scores(psi, psi)
        if a_oracle <= 0:
            return 0.0
        return float(np.clip(a_model / a_oracle, 0.0, 1.0))

    # ---------- 构建回归损失字符串 ---------- #
    @staticmethod
    def _build_reg_loss(loss_name: str, tweedie_p: Optional[float]) -> Tuple[str, str]:
        if loss_name == "RMSE":
            return "RMSE", "RMSE"
        p = 1.3 if tweedie_p is None else float(tweedie_p)
        lf = f"Tweedie:variance_power={p}"
        return lf, lf

    # ---------- 目标函数：nAUUC 为主，惩罚 CI 宽度 ---------- #
    def _objective(self, trial, X, T, Y):
        # 共用 & 早停参数
        iterations = trial.suggest_int("iterations", 300, 2000, log=True)
        od_wait = trial.suggest_int("od_wait", 50, 200)
        depth_reg = trial.suggest_int("reg_depth", 4, 10)
        depth_clf = trial.suggest_int("clf_depth", 3, 8)

        # 回归损失（含 Tweedie）
        reg_loss = trial.suggest_categorical("reg_loss", ["RMSE", "Tweedie"])
        reg_tweedie_p = None
        if reg_loss == "Tweedie":
            reg_tweedie_p = trial.suggest_float("reg_tweedie_p", 1.1, 1.9)
        reg_lf, _ = self._build_reg_loss(reg_loss, reg_tweedie_p)

        reg_params = dict(
            iterations=iterations,
            depth=depth_reg,
            learning_rate=trial.suggest_float("reg_lr", 1e-3, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("reg_l2", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("reg_subsample", 0.7, 1.0),
            random_seed=self.random_seed,
            verbose=False,
            loss_function=reg_lf,
            od_type="Iter",
            od_wait=od_wait,
        )
        clf_params = dict(
            iterations=iterations,
            depth=depth_clf,
            learning_rate=trial.suggest_float("clf_lr", 1e-3, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("clf_l2", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("clf_subsample", 0.7, 1.0),
            auto_class_weights=trial.suggest_categorical("clf_class_wt", ["Balanced", "SqrtBalanced", None]),
            random_seed=self.random_seed,
            verbose=False,
            loss_function="Logloss",
            od_type="Iter",
            od_wait=od_wait,
        )

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        nauuc_list = []
        phi_all = []  # 汇总所有验证折的 φ，用于整体 SE / CI 估计

        X_np = np.asarray(X)
        T_np = np.asarray(T, dtype=int)
        Y_np = np.asarray(Y, dtype=float)

        for tr_idx, va_idx in kf.split(X_np):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            T_tr, T_va = T_np[tr_idx], T_np[va_idx]
            Y_tr, Y_va = Y_np[tr_idx], Y_np[va_idx]

            # 倾向与结果（训练折拟合，验证折预测）
            clf = CatBoostClassifier(**clf_params)
            clf.fit(X_tr, T_tr)
            e_va = clf.predict_proba(X_va)[:, 1]

            reg1 = CatBoostRegressor(**reg_params)
            reg0 = CatBoostRegressor(**reg_params)
            reg1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            reg0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
            mu1_va = reg1.predict(X_va)
            mu0_va = reg0.predict(X_va)

            # AIPW 伪效应 ψ（oracle）
            psi_va = self._aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)

            # 用同一组超参训练 DR-Learner（训练折），在验证折上得到 taû
            dr = ForestDRLearner(
                model_regression=CatBoostRegressor(**reg_params),
                model_propensity=CatBoostClassifier(**clf_params),
                random_state=self.random_seed,
                categories=self.categories,
            )
            dr.fit(Y_tr, T_tr, X=X_tr)
            tau_va = dr.effect(X_va)

            # nAUUC
            nauuc = self._nauuc_from(psi_va, tau_va)
            nauuc_list.append(nauuc)

            # 收集 φ 用于 CI 估计
            phi_va = self._phi(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)
            phi_all.append(phi_va)

        nauuc_mean = float(np.mean(nauuc_list))

        # —— 用合并的 φ 估计总体 SE 与 95% CI 宽度 —— #
        phi_all = np.concatenate(phi_all, axis=0)
        se = phi_all.std(ddof=1) / np.sqrt(len(phi_all))
        ci_width = 2.0 * 1.96 * se  # 95% CI 宽度（原尺度）

        # 归一化 CI 宽度，避免量纲影响（用 Y 的 std 做尺度）
        y_scale = max(np.std(Y_np, ddof=1), 1e-8)
        ci_width_norm = ci_width / y_scale

        # 复合目标：nAUUC − λ·(归一化CI宽度)
        obj = nauuc_mean - self.lambda_ci * ci_width_norm

        # 可选软门槛：仅当你需要最基本过滤时启用（默认 None 不启用）
        if self.min_nauuc is not None and nauuc_mean < self.min_nauuc:
            raise optuna.TrialPruned()
        if self.max_ci_width is not None and ci_width > self.max_ci_width:
            raise optuna.TrialPruned()

        if self.verbose:
            print(f"[trial] nAUUC={nauuc_mean:.3f} | CIw={ci_width:.3f} (norm={ci_width_norm:.3f}) | obj={obj:.3f}")
        return obj

    # ---------- 对外接口：保持不变，返回“未拟合”的最佳 reg / clf ---------- #
    def fit_return_models(self, X, T, Y) -> Tuple[CatBoostRegressor, CatBoostClassifier]:
        # 早停：当 obj 足够高时直接停（这里给一个经验阈值，你也可以去掉）
        def stop_cb(study, trial):
            if study.best_value is not None and study.best_value >= 0.50:
                print(f"🎉 提前停止：objective={study.best_value:.3f}")
                study.stop()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda tr: self._objective(tr, X, T, Y),
                       n_trials=self.n_trials,
                       show_progress_bar=self.verbose,
                       n_jobs=1,
                       callbacks=[stop_cb],
                       gc_after_trial=True)

        self.best_params_ = study.best_params
        self.best_value_ = study.best_value

        # 还原回归损失
        reg_loss = self.best_params_.get("reg_loss", "RMSE")
        reg_tweedie_p = self.best_params_.get("reg_tweedie_p", None)
        reg_lf, _ = self._build_reg_loss(reg_loss, reg_tweedie_p)

        # 构造“未拟合”的最佳模型实例（下游直接使用）
        self.best_reg_ = CatBoostRegressor(
            iterations=self.best_params_["iterations"],
            depth=self.best_params_["reg_depth"],
            learning_rate=self.best_params_["reg_lr"],
            l2_leaf_reg=self.best_params_["reg_l2"],
            subsample=self.best_params_["reg_subsample"],
            random_seed=self.random_seed,
            verbose=False,
            loss_function=reg_lf,
        )
        self.best_clf_ = CatBoostClassifier(
            iterations=self.best_params_["iterations"],
            depth=self.best_params_["clf_depth"],
            learning_rate=self.best_params_["clf_lr"],
            l2_leaf_reg=self.best_params_["clf_l2"],
            subsample=self.best_params_["clf_subsample"],
            auto_class_weights=self.best_params_.get("clf_class_wt", None),
            random_seed=self.random_seed,
            verbose=False,
            loss_function="Logloss",
        )
        return self.best_reg_, self.best_clf_
