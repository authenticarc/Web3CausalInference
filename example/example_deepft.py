# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src")

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score
from catboost import CatBoostClassifier, CatBoostRegressor

# 注意：这里导入的是你无 featuretools 的版本，如果你的类名不同，改成对应名字
from DeepFeatureFactory import DeepFeatureFactory  # or DeepFeatureFactoryNoFT

# ---------- 工具函数：把类别列规范成 string，并找出 df 中的类别列 ----------
def normalize_cats(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")  # 关键：转为 string，避免 2.0 这样的 float
            X[c] = X[c].fillna("__NA__")
    return X

def infer_cat_features(df: pd.DataFrame, preferred_cols: List[str] = None) -> List[str]:
    """返回 df 中可作为 CatBoost 类别特征的列名（dtype 为 object/string 的列）。
       如果传了 preferred_cols，会取二者交集，避免把数值列误传为 cat_features。"""
    obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    if preferred_cols is not None:
        obj_cols = [c for c in obj_cols if c in preferred_cols]
    return obj_cols

# ========== 1) 数据加载 ==========
def load_nhefs() -> pd.DataFrame:
    try:
        return pd.read_csv("nhefs.csv")
    except Exception:
        try:
            from causaldata import nhefs as nhefs_pkg
            return nhefs_pkg.load_pandas().data
        except Exception:
            raise RuntimeError("请提供 nhefs.csv 或安装 causaldata: pip install causaldata")

# ========== 2) AIPW 伪效应（交叉拟合 OOF，稳健版） ==========
def _is_finite(a):
    return np.isfinite(a)

def aipw_pseudo_oof(
    X: pd.DataFrame, T: np.ndarray, Y: np.ndarray,
    n_splits: int = 5, seed: int = 42, trim: float = 1e-3,
    cat_features: List[str] = None,
    min_group_samples: int = 10
) -> Tuple[np.ndarray, dict]:
    """
    返回：psi_oof（长度 n 的 OOF AIPW 伪效应）、调试信息字典
    处理要点：
      - 回归目标 Y 不能有 NaN → 训练时按组过滤为 finite
      - 若某组样本太少，回退到“全体（Y 有效）”训练
    """
    T = np.asarray(T, int)
    Y = np.asarray(Y, float)
    n = len(X)

    e_hat = np.zeros(n, float)
    mu1_hat = np.zeros(n, float)
    mu0_hat = np.zeros(n, float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, T):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        Ttr, Tva = T[tr], T[va]
        Ytr, Yva = Y[tr], Y[va]

        cat_feats_tr = [c for c in (cat_features or []) if c in Xtr.columns and (pd.api.types.is_object_dtype(Xtr[c]) or pd.api.types.is_string_dtype(Xtr[c]))]

        # 倾向（分类目标 T 必须无 NaN，这里是 0/1 没问题）
        clf = CatBoostClassifier(
            iterations=1200, depth=6, learning_rate=0.05,
            loss_function="Logloss", random_seed=seed, verbose=False
        )
        clf.fit(Xtr, Ttr, cat_features=cat_feats_tr)
        e_hat[va] = clf.predict_proba(Xva)[:, 1]

        # 结局回归（目标 Y 不能 NaN）
        finite_all = _is_finite(Ytr)
        Xtr_all, Ytr_all = Xtr.iloc[finite_all], Ytr[finite_all]

        # 组1
        mask1 = (Ttr == 1) & finite_all
        Xtr1, Ytr1 = Xtr.iloc[mask1], Ytr[mask1]
        reg1 = CatBoostRegressor(
            iterations=1000, depth=6, learning_rate=0.05,
            loss_function="RMSE", random_seed=seed, verbose=False
        )
        if mask1.sum() >= min_group_samples:
            reg1.fit(Xtr1, Ytr1, cat_features=cat_feats_tr)
        else:
            # 回退：用所有 finite 的样本拟合一个“总体”模型兜底
            reg1.fit(Xtr_all, Ytr_all, cat_features=cat_feats_tr)

        # 组0
        mask0 = (Ttr == 0) & finite_all
        Xtr0, Ytr0 = Xtr.iloc[mask0], Ytr[mask0]
        reg0 = CatBoostRegressor(
            iterations=1000, depth=6, learning_rate=0.05,
            loss_function="RMSE", random_seed=seed, verbose=False
        )
        if mask0.sum() >= min_group_samples:
            reg0.fit(Xtr0, Ytr0, cat_features=cat_feats_tr)
        else:
            reg0.fit(Xtr_all, Ytr_all, cat_features=cat_feats_tr)

        mu1_fold = reg1.predict(Xva)
        mu0_fold = reg0.predict(Xva)

        # 兜底：若预测里还有 NaN/Inf，用折内均值替代，避免 ψ 里传播 NaN
        mu1_mean = float(np.nanmean(Ytr1)) if mask1.sum() > 0 else float(np.nanmean(Ytr_all))
        mu0_mean = float(np.nanmean(Ytr0)) if mask0.sum() > 0 else float(np.nanmean(Ytr_all))
        mu1_fold = np.where(np.isfinite(mu1_fold), mu1_fold, mu1_mean)
        mu0_fold = np.where(np.isfinite(mu0_fold), mu0_fold, mu0_mean)

        mu1_hat[va] = mu1_fold
        mu0_hat[va] = mu0_fold

    # AIPW ψ
    e_clip = np.clip(e_hat, trim, 1 - trim)
    psi = (T * (Y - mu1_hat) / e_clip) - ((1 - T) * (Y - mu0_hat) / (1 - e_clip)) + (mu1_hat - mu0_hat)

    # 清洗 ψ：把非有限值置为 NaN，留给下游筛选
    psi = np.where(np.isfinite(psi), psi, np.nan)

    debug = dict(e_mean=float(np.nanmean(e_hat)), e_min=float(np.nanmin(e_hat)), e_max=float(np.nanmax(e_hat)))
    return psi, debug

# ========== 3) 主流程：生成深特征（含树叶子特征） ==========
if __name__ == "__main__":
    df = load_nhefs().copy()

    required = ["qsmk", "wt71", "wt82"]
    missing = [k for k in required if k not in df.columns]
    if missing:
        raise ValueError(f"缺少关键列：{missing}；请检查 nhefs 列名。")

    df["Y"] = df["wt82"] - df["wt71"]
    df["T"] = df["qsmk"].astype(int)

    # 基础列
    cat_cols = [c for c in ["sex","race","education","exercise","active","marital"] if c in df.columns]
    num_cols = [c for c in ["age","wt71","wt82","smokeintensity","smokeyrs","ht","birthyr","Y"] if c in df.columns]

    # 统一把类别列转成 string，填充缺失，避免 2.0 这类浮点类别触发错误
    base_X = df[cat_cols + num_cols].copy()
    base_X = normalize_cats(base_X, cat_cols)

    # 3.1 先用交叉拟合得到 ψ（监督信号）
    psi, dbg = aipw_pseudo_oof(
        base_X, df["T"].values, df["Y"].values,
        n_splits=5, seed=42, cat_features=cat_cols
    )
    print(f"[AIPW ψ] mean={np.nanmean(psi):.4f}, std={np.nanstd(psi, ddof=1):.4f} | e(x): {dbg}")

    # —— 关键：为工厂准备 y（填充 ψ 中的 NaN，避免 CatBoost/TE 报错）——
    psi_median = float(np.nanmedian(psi))
    psi_for_factory = np.where(np.isfinite(psi), psi, psi_median)

    # 3.2 用 DeepFeatureFactory 生成深特征（含树叶子特征）
    factory = DeepFeatureFactory(
        cat_cols=cat_cols,
        num_cols=num_cols,
        group_keys=[["sex"], ["race"], ["education"]],
        enable_target_encoding=True,
        enable_count_freq=True,
        enable_groupby_agg=True,
        add_poly2=True,
        add_leaf_embeddings=True,
        add_pca=12,
        verbose=2,   # << 看详细日志
    )

    # 注意：fit_transform 输入要和我们传给 CatBoost 一致的类型；y 用填充后的 psi
    X_deep = factory.fit_transform(base_X, y=psi_for_factory)

    print("\n=== 深特征矩阵统计（含树特征） ===")
    print(f"原始：cat={len(cat_cols)}, num={len(num_cols)}")
    print(f"生成后的形状：{X_deep.shape}")
    print("示例列（前 30）：", list(X_deep.columns[:30]))

    # 3.3 简单 sanity-check：只在 ψ 有效（finite）的样本上做 5 折回归
    valid_idx = np.where(np.isfinite(psi))[0]
    Xv = X_deep.iloc[valid_idx].copy()
    yv = psi[valid_idx].astype(float)

    # 深特征通常都是数值列，这里自动探测（若仍有 string 列会被识别为类别）
    deep_cat_feats = infer_cat_features(Xv)  # 多数情况下为空列表

    oof = np.zeros(len(Xv))
    kf = KFold(5, shuffle=True, random_state=42)
    for tr, va in kf.split(Xv):
        reg = CatBoostRegressor(
            iterations=800, depth=6, learning_rate=0.05,
            loss_function="RMSE", random_seed=42, verbose=False
        )
        reg.fit(Xv.iloc[tr], yv[tr], cat_features=deep_cat_feats)
        oof[va] = reg.predict(Xv.iloc[va])

    r2 = r2_score(yv, oof)
    print(f"\n[Sanity] 用深特征 5 折回归 ψ 的 OOF R^2 = {r2:.3f} | 有效样本={len(yv)}")
