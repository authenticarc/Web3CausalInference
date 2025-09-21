# -*- coding: utf-8 -*-
"""
NAUUCCatBoostTunerV3
- 目标：在不改变对外 API 的前提下（仍返回 reg / clf），
  利用 DR 作为因果主体，同时引入 Ranker(ψ) 学排序，在调参阶段
  用 “score = (1-eta)*tau_DR + eta*score_ranker” 计算 nAUUC 作为评分指标。
- 你可以把本文件命名为 nauuctunerv2.py，替换/并存于现有工程中。
"""

from __future__ import annotations
import numpy as np
import optuna
from typing import Optional, Tuple
from sklearn.model_selection import KFold, StratifiedKFold
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoostRanker, Pool
from econml.dr import ForestDRLearner


class NAUUCCatBoostTunerV3:
    """
    用 nAUUC 进行调参，但评分用 “DR 的 taû 与 Ranker(ψ) 的融合分数”：
        score = (1 - eta) * tau_DR + eta * score_ranker,  eta∈[0,1]

    对外 API：
        tuner = NAUUCCatBoostTunerV3(n_trials=60, n_splits=5, random_state=42, eta=0.3, verbose=True)
        best_reg, best_clf = tuner.fit_return_models(X, T, Y)

    注意：
    - 返回的 best_reg / best_clf 仍是“未拟合”实例；
    - 实际线上估计与下游使用依旧用 DR（ForestDRLearner），保证因果一致性；
    - Ranker 仅用于“调参评分阶段”的排序学习，不参与对外推理。
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42,
        categories: Optional[str] = "auto",
        trim: float = 1e-3,       # e(x) 裁剪用于 ψ/φ 计算的稳定化
        eta: float = 0.30,        # 排序融合权重：0=纯DR，1=纯Ranker(ψ)
        lambda_ci: float = 0.1,   # 可选：对 CI 宽度的轻惩罚；默认 0 不启用
        verbose: bool = False,
        reg_lf: str = None  # 仅用于返回的 best_reg 的默认损失函数
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_seed = random_state
        self.categories = categories
        self.trim = float(trim)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.lambda_ci = float(max(0.0, lambda_ci))
        self.verbose = verbose
        self.reg_lf = reg_lf

        # 结果缓存
        self.best_params_: Optional[dict] = None
        self.best_value_: Optional[float] = None
        self.best_reg_: Optional[CatBoostRegressor] = None
        self.best_clf_: Optional[CatBoostClassifier] = None

    # ---------- AIPW 伪效应 ψ / 影响函数 φ / nAUUC ---------- #
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
        a_model = NAUUCCatBoostTunerV3._area_from_scores(psi, score)
        a_oracle = NAUUCCatBoostTunerV3._area_from_scores(psi, psi)
        if a_oracle <= 0:
            return 0.0
        return float(np.clip(a_model / a_oracle, 0.0, 1.0))

    # ---------- 构建回归损失（含 Tweedie） ---------- #
    @staticmethod
    def _build_reg_loss(loss_name: str, tweedie_p: Optional[float]) -> Tuple[str, str]:
        if loss_name == "RMSE":
            return "RMSE", "RMSE"
        p = 1.3 if tweedie_p is None else float(tweedie_p)
        lf = f"Tweedie:variance_power={p}"
        return lf, lf

    # ---------- Optuna 目标：nAUUC(fused score) - λ·CIwidth(norm) ---------- #
    def _objective(self, trial, X, T, Y):
        # 共用 & 早停参数
        iterations = trial.suggest_int("iterations", 300, 2000, log=True)
        od_wait = trial.suggest_int("od_wait", 50, 200)
        depth_reg = trial.suggest_int("reg_depth", 4, 10)
        depth_clf = trial.suggest_int("clf_depth", 3, 8)

        # 回归损失（含 Tweedie）
        if self.reg_lf == 'RMSE':
            reg_loss = trial.suggest_categorical("reg_loss", ["RMSE"])
        else:
            reg_loss = trial.suggest_categorical("reg_loss", ["RMSE", "Tweedie"])
        
        reg_tweedie_p = None
        if reg_loss == "Tweedie":
            reg_tweedie_p = trial.suggest_float("reg_tweedie_p", 1.1, 1.9)
        reg_lf, _ = self._build_reg_loss(reg_loss, reg_tweedie_p)

        # CatBoost 参数
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

        # 使用分层 K 折，保证各折 T 比例接近
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        # 收集指标
        nauuc_folds = []
        phi_all = []

        X = np.asarray(X)
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y, dtype=float)

        for tr_idx, va_idx in skf.split(X, T):
            X_tr, X_va = X[tr_idx], X[va_idx]
            T_tr, T_va = T[tr_idx], T[va_idx]
            Y_tr, Y_va = Y[tr_idx], Y[va_idx]

            # ========== 1) ê/μ̂：训练折拟合，训练折/验证折各自预测 ==========
            clf = CatBoostClassifier(**clf_params)
            clf.fit(X_tr, T_tr)
            e_tr = clf.predict_proba(X_tr)[:, 1]
            e_va = clf.predict_proba(X_va)[:, 1]

            reg1 = CatBoostRegressor(**reg_params)
            reg0 = CatBoostRegressor(**reg_params)
            reg1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            reg0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])

            mu1_tr = reg1.predict(X_tr)
            mu0_tr = reg0.predict(X_tr)
            mu1_va = reg1.predict(X_va)
            mu0_va = reg0.predict(X_va)

            # ========== 2) ψ（AIPW 伪效应）：训练折用于 ranker 监督，验证折用于 nAUUC ==========
            psi_tr = self._aipw_pseudo(Y_tr, T_tr, mu1_tr, mu0_tr, e_tr, trim=self.trim)
            psi_va = self._aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)

            # ========== 3) DR-Learner：因果主体，给出 taû（用于融合打分） ==========
            dr = ForestDRLearner(
                model_regression=CatBoostRegressor(**reg_params),
                model_propensity=CatBoostClassifier(**clf_params),
                random_state=self.random_seed,
                categories=self.categories,
            )
            dr.fit(Y_tr, T_tr, X=X_tr)
            tau_va = dr.effect(X_va).ravel()

            # ========== 4) Ranker(ψ)：学习排序，只用于评分，不替换 DR ==========
            group_tr = np.ones(len(X_tr), dtype=int)
            group_va = np.ones(len(X_va), dtype=int)
            tr_pool = Pool(X_tr, label=psi_tr, group_id=group_tr)
            va_pool = Pool(X_va, label=psi_va, group_id=group_va)

            ranker = CatBoostRanker(
                loss_function="YetiRank",
                iterations=min(300, iterations),
                depth=min(6, reg_params["depth"]),
                learning_rate=0.1,
                random_seed=self.random_seed,
                verbose=False,
                od_type="Iter",
                od_wait=min(50, od_wait),
                sampling_frequency="PerTree"  # 每棵树重新采样 pair，进一步减轻压力
            )
            ranker.fit(tr_pool, eval_set=va_pool)
            score_rank = np.asarray(ranker.predict(va_pool)).ravel()

            # ========== 5) 融合排序得分：score = (1-eta)*taû + eta*ranker ==========
            score_fused = (1.0 - self.eta) * tau_va + self.eta * score_rank

            # ========== 6) nAUUC ==========
            nauuc = self._nauuc_from(psi_va, score_fused)
            nauuc_folds.append(nauuc)

            # （可选）CI 惩罚：收集 φ
            if self.lambda_ci > 0.0:
                phi_va = self._phi(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)
                phi_all.append(phi_va)

        nauuc_mean = float(np.mean(nauuc_folds))

        # —— 可选：对 CI 宽度做轻惩罚（默认 lambda_ci=0 关闭）——
        if self.lambda_ci > 0.0 and len(phi_all) > 0:
            phi_all = np.concatenate(phi_all, axis=0)
            se = phi_all.std(ddof=1) / np.sqrt(len(phi_all))
            ciw = 2.0 * 1.96 * se
            y_scale = max(np.std(Y, ddof=1), 1e-8)
            ciw_norm = ciw / y_scale
            obj = nauuc_mean - self.lambda_ci * ciw_norm
        else:
            obj = nauuc_mean

        if self.verbose:
            msg = f"[trial] nAUUC(fused)={nauuc_mean:.3f}"
            if self.lambda_ci > 0.0 and len(phi_all) > 0:
                msg += f" | obj={obj:.3f}"
            print(msg)
        return obj

    # ---------- 对外接口：返回“未拟合”的最佳 reg / clf ---------- #
    def fit_return_models(self, X, T, Y) -> Tuple[CatBoostRegressor, CatBoostClassifier]:
        # 早停：当目标足够高时可以停（经验阈值，可按需调整/删除）
        def stop_cb(study, trial):
            if study.best_value is not None and study.best_value >= 0.55:
                print(f"🎉 提前停止：objective≈{study.best_value:.3f}")
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

        # 构造“未拟合”的最佳模型实例
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
