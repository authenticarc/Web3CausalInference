import numpy as np
import optuna
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, CatBoostClassifier
from econml.dr import ForestDRLearner
from typing import Optional, Tuple


class NAUUCCatBoostTuner:
    """
    用 nAUUC (normalized AUUC, 0~1) 作为 Optuna 目标，搜索 CatBoost 的
    回归器/分类器超参。返回两个“最佳参数配置”的 CatBoost 模型实例
    （未拟合，直接可传入 econml 的 DR/ForestDRLearner 使用）。

    用法：
        tuner = NAUUCCatBoostTuner(n_trials=50, n_splits=5, random_seed=42)
        best_reg, best_clf = tuner.fit_return_models(X, T, Y)
        # 然后：
        dr = ForestDRLearner(model_regression=best_reg,
                             model_propensity=best_clf,
                             random_state=42, categories='auto')
        dr.fit(Y, T, X=X)
        tau = dr.effect(X)
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42,
        categories: Optional[str] = "auto",
        trim: float = 1e-3,            # e(x) 裁剪，提升稳定性
        verbose: bool = False,
        reg_lf = None
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_seed = random_state
        self.categories = categories
        self.trim = trim
        self.verbose = verbose

        self.best_params_: Optional[dict] = None
        self.best_value_: Optional[float] = None
        self.best_reg_: Optional[CatBoostRegressor] = None
        self.best_clf_: Optional[CatBoostClassifier] = None
        self.reg_lf = reg_lf

    # ---------- AIPW 伪效应 ψ：一致估计真效应，作标准化基准 ---------- #
    @staticmethod
    def _aipw_pseudo_outcome(y, t, mu1, mu0, e, trim=1e-3):
        e = np.clip(e, trim, 1 - trim)
        return (t * (y - mu1) / e) - ((1 - t) * (y - mu0) / (1 - e)) + (mu1 - mu0)

    # ---------- 给定“排序分数”时，对 ψ 的累计积分面积 ---------- #
    @staticmethod
    def _area_from_scores(psi, order_score):
        order = np.argsort(-order_score)
        psi_ord = psi[order]
        csum = np.cumsum(psi_ord)
        x = np.arange(1, len(psi_ord) + 1) / len(psi_ord)  # 覆盖率 [0,1]
        return float(np.trapz(csum, x))
    
     # ---------- 构建回归损失字符串 ---------- #
    @staticmethod
    def _build_reg_loss(loss_name: str, tweedie_p: Optional[float]) -> Tuple[str, str]:
        """
        返回 (loss_function, eval_metric) 字符串
        """
        if loss_name == "RMSE":
            return "RMSE", "RMSE"
        # Tweedie
        p = 1.3 if tweedie_p is None else float(tweedie_p)
        lf = f"Tweedie:variance_power={p}"
        return lf, lf

    # ---------- Optuna 的目标：交叉验证上的 AIPW-nAUUC ---------- #
    def _objective(self, trial, X, T, Y):
        # ---- 共用 & 早停参数 ---- #
        iterations = trial.suggest_int("iterations", 300, 2000, log=True)
        od_wait = trial.suggest_int("od_wait", 50, 200)
        depth_reg = trial.suggest_int("reg_depth", 4, 10)
        depth_clf = trial.suggest_int("clf_depth", 3, 8)

        # 回归损失选择（包含 Tweedie 的 variance_power）
        if self.reg_lf == 'RMSE':
            reg_loss = trial.suggest_categorical("reg_loss", ["RMSE"])
        else:
            reg_loss = trial.suggest_categorical("reg_loss", ["RMSE", "Tweedie"])
        reg_tweedie_p = None
        if reg_loss == "Tweedie":
            reg_tweedie_p = trial.suggest_float("reg_tweedie_p", 1.1, 1.9)

        reg_lf, reg_eval = self._build_reg_loss(reg_loss, reg_tweedie_p)

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
        scores = []

        X_np = np.asarray(X)
        T_np = np.asarray(T, dtype=int)
        Y_np = np.asarray(Y, dtype=float)

        for tr_idx, va_idx in kf.split(X_np):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            T_tr, T_va = T_np[tr_idx], T_np[va_idx]
            Y_tr, Y_va = Y_np[tr_idx], Y_np[va_idx]

            # 1) 倾向与结果模型（在训练折上拟合，验证折上预测）
            clf = CatBoostClassifier(**clf_params)
            clf.fit(X_tr, T_tr)
            e_va = clf.predict_proba(X_va)[:, 1]

            reg1 = CatBoostRegressor(**reg_params)
            reg0 = CatBoostRegressor(**reg_params)
            # 注意：只用对应组的数据拟合 m1/m0
            reg1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            reg0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
            mu1_va = reg1.predict(X_va)
            mu0_va = reg0.predict(X_va)

            # 2) AIPW 伪效应 ψ（标准化基准）
            psi_va = self._aipw_pseudo_outcome(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)

            # 3) 用同一超参在训练折上拟合 DR-Learner，得到 τ̂ 作为模型排序分数
            dr = ForestDRLearner(
                model_regression=CatBoostRegressor(**reg_params),
                model_propensity=CatBoostClassifier(**clf_params),
                random_state=self.random_seed,
                categories=self.categories,
            )
            dr.fit(Y_tr, T_tr, X=X_tr)
            tau_va = dr.effect(X_va)

            # 4) 面积：模型按 τ̂ 排序 vs Oracle 按 ψ 排序
            area_model = self._area_from_scores(psi_va, tau_va)
            area_oracle = self._area_from_scores(psi_va, psi_va)

            # 5) 标准化：nAUUC ∈ [0,1]
            if area_oracle <= 0:
                nauuc = 0.0
            else:
                nauuc = max(0.0, min(1.0, area_model / area_oracle))

            scores.append(nauuc)

        return float(np.mean(scores))

    # ---------- 对外接口：运行调参并返回两个“最佳参数”的模型实例 ---------- #
    def fit_return_models(self, X, T, Y) -> Tuple[CatBoostRegressor, CatBoostClassifier]:
        def stop_callback(study, trial):
            if study.best_value is not None and study.best_value >= 0.45:
                print(f"🎉 提前停止：best nAUUC={study.best_value:.3f} ≥ 0.45")
                study.stop()
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda tr: self._objective(tr, X, T, Y),
                       n_trials=self.n_trials,
                       show_progress_bar=self.verbose,n_jobs=1,callbacks=[stop_callback],
                       gc_after_trial=True)

        self.best_params_ = study.best_params
        self.best_value_ = study.best_value
        
        # 还原最优回归损失
        reg_loss = self.best_params_.get("reg_loss", "RMSE")
        reg_tweedie_p = self.best_params_.get("reg_tweedie_p", None)
        reg_lf, reg_eval = self._build_reg_loss(reg_loss, reg_tweedie_p)

         # 构造“未拟合”的最佳模型实例（供 econml 复用）
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

        # 返回“未拟合”的最佳模型（用于传给 econml learner）
        return self.best_reg_, self.best_clf_
    
