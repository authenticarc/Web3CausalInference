# example_att_est.py
import numpy as np

from ..src.UnifiedCausal import UnifiedCausalTester, CausalRules
from ..src.minVarCatBoostTuner import MinVarCatBoostTuner
from causaldata import nhefs_complete


import pandas as pd
import os.path
import os

# ===== 1) 数据地址 =====
try:
    df = nhefs_complete.load_pandas().data.copy()
except Exception:
    df = nhefs_complete.load_pandas().copy()

t_col = "qsmk"
y_col = "wt82_71"
y_nc_col = None

# 协变量（可按需增减；以下为常见一组）
base_covs = [
    "sex", "race", "age", "education",
    "smokeintensity", "smokeyrs",
    "exercise", "active",
    "wt71", "ht", "bmix",  # bmix 若不存在，可用 bmi
    "alcohol", "marital"
]
# 兼容：若某些列不存在，自动剔除
covs = [c for c in base_covs if c in df.columns]
if "bmi" in df.columns and "bmix" not in df.columns:
    covs.append("bmi")

# --- 构造 X / T / Y ---
df = df.dropna()
X_df = pd.get_dummies(df[covs], drop_first=True)   # one-hot → 全数值
X = X_df.to_numpy()
feature_names = list(X_df.columns)

T = df[t_col].to_numpy().ravel().astype(int)
Y = df[y_col].to_numpy().ravel().astype(float)
y_nc = None

# ===== 2) 公共规则 & 估计器配置（按需调快/调严）=====
rules = CausalRules(
    smd_max=0.3, ovl_min=0.50, ks_max=0.60, ess_min=0.70,
    placebo_alpha=0.09, nc_alpha=0.10, top_k_smd=8
)

reg, clf = MinVarCatBoostTuner(verbose=0,n_trials=300).fit_return_models(X,T,Y)

common_kwargs = dict(
    n_splits=5,
    trim=0.01,
    ps_clip=(0.05, 0.95),     # PS 裁剪，稳健
    weight_clip=10.0,         # 权重裁剪，稳健
    n_jobs=-1,                 # 单测/示例下设 1；真跑可以开大
    n_jobs_placebo=-1,
    random_state=2025,
    verbose=0,
    rules=rules,
    regressor=reg,
    classifier=clf
)

# ===== 3) ATE =====
print("\n=== ATE on full sample ===")
ate = UnifiedCausalTester(estimand="ATE", **common_kwargs)
ate.fit(X, T, Y, X_names=covs, y_nc=None, placebo_runs=10)
print(ate.report())

# 取得 OOF 倾向分数，后面做重叠带
e_oof = ate.result_["diag"]["e"]

# ===== 6) “重叠带” ATT（可选，更稳的口径）=====
print("\n=== ATT on overlap band e∈[0.3,0.7] (recommended when overlap is weak) ===")
band = (e_oof >= 0.3) & (e_oof <= 0.7)
print(f"Overlap-band coverage (treated kept): "
    f"{int((T[band]==1).sum())}/{int((T==1).sum())} "
    f"= {(T[band]==1).sum()/(T==1).sum():.1%}")

att_band = UnifiedCausalTester(estimand="ATT", **common_kwargs)
att_band.fit(X[band], T[band], Y[band], X_names=X_names, y_nc=y_nc, placebo_runs=10)
print(att_band.report())
