# -*- coding: utf-8 -*-
import numpy as np
import optuna
from typing import Optional, Tuple, Dict, Any, List
from sklearn.model_selection import KFold
from pandas.api.types import (
    is_numeric_dtype, is_integer_dtype,
    is_string_dtype, is_categorical_dtype, is_datetime64_any_dtype
)
import pandas as pd
import xgboost as xgb


class NAUUCCatBoostTunerV2:
    """
    XGBoost + DR-learner 版本（无 econml / dowhy）
    目标：maximize  nAUUC( taû vs ψ ) − λ * (CI_width / sd(Y))

    —— 流程（每个外层fold）——
      1) 估计 e(x)、m1(x)、m0(x)（用训练折）
      2) 得到 m(x)=e*m1+(1-e)*m0；构造 DR 伪结果 Z=((T-e)/(e(1-e)))*(Y-m)
      3) 用权重 w=e(1-e) 在训练折上拟合 tau-model g:  Z ~ g(X)
      4) 验证折得 taû=g(X_val)
      5) 用 AIPW ψ 作为“oracle”，与 taû 计算 nAUUC；φ 合并估计 CI 宽度
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        random_seed: int = 42,
        # AIPW/目标
        trim: float = 1e-3,
        lambda_ci: float = 0.15,
        min_nauuc: Optional[float] = None,
        max_ci_width: Optional[float] = None,
        verbose: bool = True,
        # 特征自动识别/编码
        auto_encode_cats: bool = True,
        int_as_cat_unique_thresh: int = 30,
        unique_ratio_thresh: float = 0.05,
        rare_freq_ratio: float = 0.001,
        max_onehot_levels: int = 200,
        # XGBoost/GPU
        use_gpu: bool = False,
        # 回归器损失（'rmse' 或 'tweedie'）
        reg_loss: str = "rmse",
        tweedie_p_range: Tuple[float, float] = (1.1, 1.9),
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_seed = int(random_seed)
        self.trim = float(trim)
        self.lambda_ci = float(lambda_ci)
        self.min_nauuc = min_nauuc
        self.max_ci_width = max_ci_width
        self.verbose = verbose

        # FE
        self.auto_encode_cats = auto_encode_cats
        self.int_as_cat_unique_thresh = int(int_as_cat_unique_thresh)
        self.unique_ratio_thresh = float(unique_ratio_thresh)
        self.rare_freq_ratio = float(rare_freq_ratio)
        self.max_onehot_levels = int(max_onehot_levels)

        # XGB
        self.use_gpu = bool(use_gpu)
        self.reg_loss = reg_loss.lower()
        self.tweedie_p_range = tweedie_p_range

        # 输出
        self.best_params_: Optional[dict] = None
        self.best_value_: Optional[float] = None
        self.best_tau_model_: Optional[xgb.XGBRegressor] = None     # g(x)
        self.best_ps_model_: Optional[xgb.XGBClassifier] = None     # e(x)
        self.best_m1_model_: Optional[xgb.XGBRegressor] = None      # m1(x)
        self.best_m0_model_: Optional[xgb.XGBRegressor] = None      # m0(x)

        # 训练期保存的编码映射
        self._cat_levels_: Dict[str, List[Any]] = {}

    # ---------- AIPW / 影响函数 / nAUUC ---------- #
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

    # ---------- 自动识别 & 稳定编码 ---------- #
    def _infer_cats(self, X: pd.DataFrame) -> List[str]:
        from pandas.api.types import (
            is_numeric_dtype, is_integer_dtype,
            is_string_dtype, is_categorical_dtype, is_datetime64_any_dtype
        )
        cats = []
        n = len(X)
        for c in X.columns:
            s = X[c]
            if is_datetime64_any_dtype(s):  # 时间列跳过
                continue
            nunq = s.nunique(dropna=False)
            ur = nunq / max(n, 1)
            is_cat = (
                is_string_dtype(s) or is_categorical_dtype(s) or
                (is_integer_dtype(s) and nunq <= self.int_as_cat_unique_thresh) or
                (not is_numeric_dtype(s) and ur <= self.unique_ratio_thresh)
            )
            if not is_cat and is_numeric_dtype(s) and nunq <= self.int_as_cat_unique_thresh:
                is_cat = True
            if is_cat:
                cats.append(c)
        return cats

    def _fit_categorical_maps(self, X: pd.DataFrame, cat_cols: List[str]) -> None:
        self._cat_levels_.clear()
        n = len(X)
        rare_thresh = max(int(self.rare_freq_ratio * n), 1)
        for c in cat_cols:
            vc = X[c].astype("string").fillna("__NA__").value_counts(dropna=False)
            common = vc[vc >= rare_thresh].index.tolist()
            if len(common) > self.max_onehot_levels:
                common = list(vc.index[: self.max_onehot_levels])
            self._cat_levels_[c] = ["__RARE__"] + [str(v) for v in common]

    def _apply_categorical_maps(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        X2 = X.copy()
        for c in cat_cols:
            X2[c] = X2[c].astype("string").fillna("__NA__")
            levels = set(self._cat_levels_[c][1:])
            X2[c] = X2[c].apply(lambda v: v if v in levels else "__RARE__")
        # one-hot（固定顺序）
        dummies = []
        for c in cat_cols:
            cats = pd.Categorical(X2[c], categories=self._cat_levels_[c])
            dummies.append(pd.get_dummies(cats, prefix=c, dummy_na=False))
        D = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=X2.index)
        # 数值
        num_cols = [c for c in X.columns if c not in cat_cols and is_numeric_dtype(X[c])]
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce")
        return pd.concat([X_num, D], axis=1).fillna(0.0)

    def _encode_X_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.auto_encode_cats:
            return X.copy()
        cat_cols = self._infer_cats(X)
        self._fit_categorical_maps(X, cat_cols)
        return self._apply_categorical_maps(X, cat_cols)

    def _encode_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.auto_encode_cats:
            return X.copy()
        cat_cols = list(self._cat_levels_.keys())
        return self._apply_categorical_maps(X, cat_cols)

    # ---------- XGB 参数空间 ---------- #
    def _device_args(self) -> Dict[str, Any]:
        # XGBoost 2.0+ 推荐写法
        return {"tree_method": "hist", "device": ("cuda" if self.use_gpu else "cpu")}

    def _reg_objective_space(self, trial) -> Dict[str, Any]:
        params = dict(
            max_depth=trial.suggest_int("reg_max_depth", 3, 8),
            learning_rate=trial.suggest_float("reg_lr", 1e-3, 0.2, log=True),
            n_estimators=trial.suggest_int("reg_n_estimators", 400, 1500),
            subsample=trial.suggest_float("reg_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("reg_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
            min_child_weight=trial.suggest_int("reg_min_child_weight", 1, 15),
            random_state=self.random_seed, n_jobs=-1, **self._device_args(),
        )
        if self.reg_loss == "tweedie":
            p = trial.suggest_float("tweedie_variance_power", self.tweedie_p_range[0], self.tweedie_p_range[1])
            params.update(objective="reg:tweedie", tweedie_variance_power=p)
        else:
            params.update(objective="reg:squarederror")
        return params

    def _clf_objective_space(self, trial) -> Dict[str, Any]:
        return dict(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=trial.suggest_int("clf_max_depth", 3, 8),
            learning_rate=trial.suggest_float("clf_lr", 1e-3, 0.2, log=True),
            n_estimators=trial.suggest_int("clf_n_estimators", 300, 1200),
            subsample=trial.suggest_float("clf_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("clf_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("clf_reg_lambda", 0.0, 10.0),
            min_child_weight=trial.suggest_int("clf_min_child_weight", 1, 15),
            random_state=self.random_seed, n_jobs=-1, **self._device_args(),
        )

    # ---------- 目标函数（外层 KFold，DR-learner） ---------- #
    def _objective(self, trial, X_df: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        # 每个 trial 内部单独编码，避免稀有并桶差异引入偏差
        X_enc = self._encode_X_fit(X_df)
        X_np = X_enc.to_numpy(dtype=float, copy=False)
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y, dtype=float)

        reg_params = self._reg_objective_space(trial)
        clf_params = self._clf_objective_space(trial)
        tau_params = self._reg_objective_space(trial)  # tau 模型同一搜索空间

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        nauuc_list, phi_all = [], []

        for tr_idx, va_idx in kf.split(X_np):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            T_tr, T_va = T[tr_idx], T[va_idx]
            Y_tr, Y_va = Y[tr_idx], Y[va_idx]

            # ---- 倾向 e(x) ----
            ps = xgb.XGBClassifier(**clf_params)
            ps.fit(X_tr, T_tr, eval_set=[(X_va, T_va)], verbose=False)
            e_tr = np.clip(ps.predict_proba(X_tr)[:, 1], self.trim, 1 - self.trim)
            e_va = np.clip(ps.predict_proba(X_va)[:, 1], self.trim, 1 - self.trim)

            # ---- 结果 m1/m0 ----
            if (T_tr == 1).sum() == 0 or (T_tr == 0).sum() == 0:
                return -1e9
            m1 = xgb.XGBRegressor(**reg_params)
            m0 = xgb.XGBRegressor(**reg_params)
            m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1], eval_set=[(X_va[T_va == 1], Y_va[T_va == 1])], verbose=False)
            m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0], eval_set=[(X_va[T_va == 0], Y_va[T_va == 0])], verbose=False)

            mu1_tr = m1.predict(X_tr)
            mu0_tr = m0.predict(X_tr)
            mu1_va = m1.predict(X_va)
            mu0_va = m0.predict(X_va)

            # ---- DR 伪结果（训练折） ----
            m_tr = e_tr * mu1_tr + (1 - e_tr) * mu0_tr
            Z_tr = ((T_tr - e_tr) / (e_tr * (1 - e_tr))) * (Y_tr - m_tr)
            w_tr = e_tr * (1 - e_tr)

            # ---- tau 模型 g(x): Z ~ X （权重 w）----
            tau_model = xgb.XGBRegressor(**tau_params)
            tau_model.fit(X_tr, Z_tr, sample_weight=w_tr, verbose=False)

            # 验证 uplift
            tau_va = tau_model.predict(X_va)

            # AIPW ψ（oracle）& nAUUC
            psi_va = self._aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)
            nauuc = self._nauuc_from(psi_va, tau_va)
            nauuc_list.append(nauuc)

            phi_va = self._phi(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)
            phi_all.append(phi_va)

        nauuc_mean = float(np.mean(nauuc_list))
        phi_all = np.concatenate(phi_all, axis=0)
        se = phi_all.std(ddof=1) / np.sqrt(len(phi_all))
        ci_width = 2.0 * 1.96 * se
        y_scale = max(np.std(Y, ddof=1), 1e-8)
        ci_width_norm = ci_width / y_scale

        obj = nauuc_mean - self.lambda_ci * ci_width_norm
        if self.min_nauuc is not None and nauuc_mean < self.min_nauuc:
            raise optuna.TrialPruned()
        if self.max_ci_width is not None and ci_width > self.max_ci_width:
            raise optuna.TrialPruned()

        if self.verbose:
            print(f"[trial] nAUUC={nauuc_mean:.3f} | CIw={ci_width:.3f} (norm={ci_width_norm:.3f}) | obj={obj:.3f}")
        return obj

    # ---------- 外部接口：返回“未拟合”的最优模型们 ---------- #
    def fit_return_models(
        self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray
    ) -> Tuple[xgb.XGBRegressor, xgb.XGBClassifier, xgb.XGBRegressor, xgb.XGBRegressor]:
        # 固定编码映射
        _ = self._encode_X_fit(X)

        def stop_cb(study, trial):
            if study.best_value is not None and study.best_value >= 0.50:
                if self.verbose:
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

        # 组装“未拟合”的最优参数模型
        base_reg = dict(
            max_depth=self.best_params_["reg_max_depth"],
            learning_rate=self.best_params_["reg_lr"],
            n_estimators=self.best_params_["reg_n_estimators"],
            subsample=self.best_params_["reg_subsample"],
            colsample_bytree=self.best_params_["reg_colsample_bytree"],
            reg_lambda=self.best_params_["reg_lambda"],
            min_child_weight=self.best_params_["reg_min_child_weight"],
            random_state=self.random_seed, n_jobs=-1, **self._device_args(),
        )
        if self.reg_loss == "tweedie":
            base_reg.update(objective="reg:tweedie",
                            tweedie_variance_power=self.best_params_.get("tweedie_variance_power",
                                                                         np.mean(self.tweedie_p_range)))
        else:
            base_reg.update(objective="reg:squarederror")

        ps_params = dict(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=self.best_params_["clf_max_depth"],
            learning_rate=self.best_params_["clf_lr"],
            n_estimators=self.best_params_["clf_n_estimators"],
            subsample=self.best_params_["clf_subsample"],
            colsample_bytree=self.best_params_["clf_colsample_bytree"],
            reg_lambda=self.best_params_["clf_reg_lambda"],
            min_child_weight=self.best_params_["clf_min_child_weight"],
            random_state=self.random_seed, n_jobs=-1, **self._device_args(),
        )

        # tau 模型参数与 reg 相同空间（可独立拆分；这里复用）
        tau_params = base_reg.copy()

        self.best_tau_model_ = xgb.XGBRegressor(**tau_params)     # g(x)
        self.best_ps_model_  = xgb.XGBClassifier(**ps_params)     # e(x)
        self.best_m1_model_  = xgb.XGBRegressor(**base_reg)       # m1(x)
        self.best_m0_model_  = xgb.XGBRegressor(**base_reg)       # m0(x)

        return self.best_tau_model_, self.best_ps_model_, self.best_m1_model_, self.best_m0_model_

# ================== DEMO（nhefs 数据） ================== #
if __name__ == "__main__":
    from causaldata import nhefs_complete
    from sklearn.metrics import roc_auc_score, mean_squared_error
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    # 1) 数据
    try:
        df_raw = nhefs_complete.load_pandas().data.copy()
    except Exception:
        df_raw = nhefs_complete.load_pandas().copy()

    t_col = "qsmk"
    y_col = "wt82_71"
    base_covs = [
        "sex","race","age","education","smokeintensity","smokeyrs",
        "exercise","active","wt71","ht","bmix","alcohol","marital"
    ]
    covs = [c for c in base_covs if c in df_raw.columns]
    if "bmi" in df_raw.columns and "bmix" not in df_raw.columns:
        covs.append("bmi")

    df = df_raw[covs + [t_col, y_col]].dropna(subset=[t_col, y_col]).copy()
    df[t_col] = (df[t_col] > 0).astype(int)

    X_df = df[covs]
    T = df[t_col].to_numpy(int)
    Y = df[y_col].to_numpy(float)

    print(f"Data: X={X_df.shape}, T={T.shape}, Y={Y.shape}; treated rate={T.mean():.3f}")

    # 2) 实例化（DR-learner + XGBoost）
    tuner = NAUUCCatBoostTunerV2(
        n_trials=15,
        n_splits=5,
        random_seed=2025,      # ← 统一用 random_seed
        trim=1e-3,
        lambda_ci=0.15,
        auto_encode_cats=True,
        max_onehot_levels=200,
        use_gpu=True,          # CUDA 可用的话设 True；XGB≥2.0 内部会设置 device="cuda"
        reg_loss="rmse"        # 若 Y 非负且右偏可改 "tweedie"
    )

    # 3) 超参搜索并返回“未拟合”的最优模型（tau / ps / m1 / m0）
    tau_m, ps_m, m1_m, m0_m = tuner.fit_return_models(X_df, T, Y)
    print("\nBest objective:", tuner.best_value_)
    print("Best params:")
    for k, v in tuner.best_params_.items():
        print(f"  {k}: {v}")

    # 4) 全量拟合与打分（保持与调参阶段一致的编码）
    X_enc = tuner._encode_X(X_df).to_numpy(float, copy=False)

    # 倾向 e(x)
    ps_m.fit(X_enc, T, verbose=False)
    e_hat = np.clip(ps_m.predict_proba(X_enc)[:, 1], 1e-3, 1 - 1e-3)
    print(f"\nPS AUC (full-data): {roc_auc_score(T, e_hat):.4f}")

    # 结果 m1/m0
    idx1, idx0 = (T == 1), (T == 0)
    m1_m.fit(X_enc[idx1], Y[idx1], verbose=False)
    m0_m.fit(X_enc[idx0], Y[idx0], verbose=False)

    mu1 = m1_m.predict(X_enc)
    mu0 = m0_m.predict(X_enc)

    # DR 伪结果（用来训练 tau 模型）
    m_all = e_hat * mu1 + (1.0 - e_hat) * mu0
    Z = ((T - e_hat) / (e_hat * (1.0 - e_hat))) * (Y - m_all)
    w = e_hat * (1.0 - e_hat)

    tau_m.fit(X_enc, Z, sample_weight=w, verbose=False)
    tau_hat = tau_m.predict(X_enc)

    # 5) 指标与预览
    if idx1.sum() > 5:
        rmse1 = np.sqrt(mean_squared_error(Y[idx1], m1_m.predict(X_enc[idx1])))
        print(f"RMSE(m1, treated):  {rmse1:.4f}")
    if idx0.sum() > 5:
        rmse0 = np.sqrt(mean_squared_error(Y[idx0], m0_m.predict(X_enc[idx0])))
        print(f"RMSE(m0, control):  {rmse0:.4f}")

    psi = tuner._aipw_pseudo(Y, T, mu1, mu0, e_hat, trim=tuner.trim)
    nauuc = tuner._nauuc_from(psi, tau_hat)
    print(f"\nnAUUC (taû vs ψ): {nauuc:.4f}")

    out = pd.DataFrame({
        "tau_hat": tau_hat,
        "psi": psi,
        "e_hat": e_hat,
        t_col: T,
        y_col: Y
    })
    print("\nUplift preview:")
    print(out.head(10))
