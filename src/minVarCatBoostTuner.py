import numpy as np
import optuna
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, CatBoostClassifier
from typing import Optional, Tuple


class MinVarCatBoostTuner:
    """
    【改版说明】
    - 目标：最小化 AIPW 方差（基于折外 influence function ψ 的 Var(ψ)/n）
    - 回归损失：Optuna 搜索 RMSE / Tweedie(variance_power 可调)
    - 训练早停：CatBoost 内置早停 + Optuna MedianPruner
    - 外部接口保持不变：fit_return_models(X, T, Y) → (best_reg, best_clf)
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42,
        categories: Optional[str] = "auto",
        trim: float = 1e-3,            # e(x) 裁剪，提升稳定性
        verbose: bool = False,
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_seed = random_state
        self.categories = categories
        self.trim = trim
        self.verbose = verbose

        self.best_params_: Optional[dict] = None
        self.best_value_: Optional[float] = None  # 最小方差
        self.best_reg_: Optional[CatBoostRegressor] = None
        self.best_clf_: Optional[CatBoostClassifier] = None

    # ---------- AIPW 伪效应 ψ：一致估计真效应 ---------- #
    @staticmethod
    def _aipw_pseudo_outcome(y, t, mu1, mu0, e, trim=1e-3):
        e = np.clip(e, trim, 1 - trim)
        return (t * (y - mu1) / e) - ((1 - t) * (y - mu0) / (1 - e)) + (mu1 - mu0)

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

    # ---------- Optuna 的目标：交叉验证上的 AIPW 方差最小化 ---------- #
    def _objective(self, trial, X, T, Y):
        # ---- 共用 & 早停参数 ---- #
        iterations = trial.suggest_int("iterations", 300, 2000, log=True)
        od_wait = trial.suggest_int("od_wait", 50, 200)
        depth_reg = trial.suggest_int("reg_depth", 4, 10)
        depth_clf = trial.suggest_int("clf_depth", 3, 8)

        # 回归损失选择（包含 Tweedie 的 variance_power）
        reg_loss = trial.suggest_categorical("reg_loss", ["RMSE", "Tweedie"])
        reg_tweedie_p = None
        if reg_loss == "Tweedie":
            reg_tweedie_p = trial.suggest_float("reg_tweedie_p", 1.1, 1.9)

        reg_lf, reg_eval = self._build_reg_loss(reg_loss, reg_tweedie_p)

        reg_params_base = dict(
            iterations=iterations,
            depth=depth_reg,
            learning_rate=trial.suggest_float("reg_lr", 1e-3, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("reg_l2", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("reg_subsample", 0.7, 1.0),
            random_seed=self.random_seed,
            verbose=False,
            loss_function=reg_lf,
            eval_metric=reg_eval,
            use_best_model=True,
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
            eval_metric="AUC",
            use_best_model=True,
            od_type="Iter",
            od_wait=od_wait,
        )

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        fold_vars = []

        X_np = np.asarray(X)
        T_np = np.asarray(T, dtype=int)
        Y_np = np.asarray(Y, dtype=float)

        for tr_idx, va_idx in kf.split(X_np):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            T_tr, T_va = T_np[tr_idx], T_np[va_idx]
            Y_tr, Y_va = Y_np[tr_idx], Y_np[va_idx]

            # ------- 倾向得分模型（在训练折拟合，在验证折评估 + 早停） ------- #
            clf = CatBoostClassifier(**{k: v for k, v in clf_params.items() if v is not None})
            clf.fit(
                X_tr, T_tr,
                eval_set=(X_va, T_va),
                verbose=False,
            )
            e_va = clf.predict_proba(X_va)[:, 1]

            # ------- 结果模型：分别在处理/对照子样本上拟合，eval_set 也对应子集 ------- #
            reg1 = CatBoostRegressor(**reg_params_base)
            reg0 = CatBoostRegressor(**reg_params_base)

            tr_t1 = T_tr == 1
            tr_t0 = ~tr_t1
            va_t1 = T_va == 1
            va_t0 = ~va_t1

            # 如果某一组在该折样本过少，跳过早停的 eval_set，避免 CatBoost 报错
            if np.sum(va_t1) > 0:
                reg1.fit(X_tr[tr_t1], Y_tr[tr_t1], eval_set=(X_va[va_t1], Y_va[va_t1]), verbose=False)
            else:
                reg1.fit(X_tr[tr_t1], Y_tr[tr_t1], verbose=False)

            if np.sum(va_t0) > 0:
                reg0.fit(X_tr[tr_t0], Y_tr[tr_t0], eval_set=(X_va[va_t0], Y_va[va_t0]), verbose=False)
            else:
                reg0.fit(X_tr[tr_t0], Y_tr[tr_t0], verbose=False)

            mu1_va = reg1.predict(X_va)
            mu0_va = reg0.predict(X_va)

            # ------- 折外 ψ 与方差估计 ------- #
            psi_va = self._aipw_pseudo_outcome(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)

            # AIPW 估计量的方差 ~ Var(ψ)/n
            fold_var = float(np.var(psi_va, ddof=1) / max(len(psi_va), 1))
            fold_vars.append(fold_var)

        # 目标是“越小越好”
        return float(np.mean(fold_vars))

    # ---------- 对外接口：运行调参并返回两个“最佳参数”的模型实例 ---------- #
    def fit_return_models(self, X, T, Y) -> Tuple[CatBoostRegressor, CatBoostClassifier]:
        # Optuna：方差最小化 + 中位数剪枝 + 进度条
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, self.n_trials // 10))
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(
            lambda tr: self._objective(tr, X, T, Y),
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            gc_after_trial=True,
            n_jobs=1,  # CatBoost 线程安全；多个 trial 并行容易抢核导致波动
        )

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

        return self.best_reg_, self.best_clf_
