import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src")

import numpy as np

from UnifiedCausal import UnifiedCausalTester, CausalRules
from minVarCatBoostTuner import MinVarCatBoostTuner
from NAUUCCatBoostTunerV2 import NAUUCCatBoostTunerV2
from causaldata import nhefs_complete
import pandas as pd
from DistilledPolicy import  compute_teacher_oof_tau_psi_aligned, fit_rule_catboost_single_tree, evaluate_rule_model,evaluate_rules_per_leaf, extract_rules_from_catboost_single_tree
from HTETester import HTETester

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

reg, clf = NAUUCCatBoostTunerV2(verbose=0,n_trials=50,reg_lf='RMSE').fit_return_models(X,T,Y)

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
att_band.fit(X[band], T[band], Y[band], X_names=feature_names, y_nc=y_nc, placebo_runs=10)
print(att_band.report())


# ===== 7) HTE & Policy（可选）=====
hte = HTETester(
    tune_nauuc=True,
    n_trials=50,
    reg_loss="RMSE",
    reg_tweedie_p=1.3,
    nauuc_band=(0.3, 0.7),
    n_splits=5,
    trim=0.008,
    min_nauuc=0.35,      # 门槛
    random_state=42,
    verbose=1
)
dr = hte.fit(X, T, Y)      # 训练并返回 DR 模型
report = hte.report()
print(report)

tau_hat = dr.effect(X)
df['htt'] = tau_hat

try:
    lb_90, ub_90 = dr.effect_interval(X, alpha=0.1)
    df['lb_90'] = lb_90; df['ub_90'] = ub_90
    lb_99, ub_99 = dr.effect_interval(X, alpha=0.01)
    df['lb_99'] = lb_99; df['ub_99'] = ub_99
except Exception as e:
    print(f"[Warn] effect_interval 计算失败：{e}")
    for c in ('lb_90','ub_90','lb_99','ub_99'):
        df[c] = np.nan

df['label'] = df.apply(lambda row: 'strong' if row['lb_99'] > 0 else 'medium' if row['lb_90']> 0 else 'none',axis=1)

# 统计各类用户数量与占比
treated_df = df[df['qsmk'] == 1]

stats = treated_df['label'].value_counts(normalize=True).rename("proportion").to_frame()
stats['count'] = treated_df['label'].value_counts()
stats['proportion'] = stats['proportion'].round(4)

print("=== HTT 实验组用户分布 ===")
stat_res = []
for lbl, row in stats.iterrows():
    stat_res.append(f"{lbl:7s}: {row['count']} ({row['proportion']:.4f})")
    print(f"{lbl:7s}: {row['count']} ({row['proportion']:.4f})")
stat_res = "\n".join(stat_res)

# ---------- 与 HTETester/HTE 模型对接 ----------

# 1) 划定 overlap 带（用 HTE 搜出的分类器原型，做 OOF + Sigmoid 校准）
clf_proto = hte.classifier
reg_proto = hte.regressor
e_oof = hte._oof_propensity(X, T, clf_proto, trim=hte.trim) #_oof_propensity_with_calib(X, T, clf_proto, trim=0.02, cv=5, seed=42)
band_lo, band_hi = hte.nauuc_band
band = (e_oof >= band_lo) & (e_oof <= band_hi)
print(f"[Distill] 带覆盖率={band.mean():.2%}，n={int(band.sum())}，带=({band_lo:.2f},{band_hi:.2f})")

# 2) 老师 OOF τ 与 ψ（只在带内）
Xb, Tb, Yb = X[band], T[band], Y[band]
# tau_oof, psi_oof = compute_teacher_oof_tau_psi(
#     Xb, Tb, Yb, reg_proto=reg_proto, clf_proto=clf_proto, n_splits=5, trim=0.02, seed=2025
# )
tau_oof, psi_oof = compute_teacher_oof_tau_psi_aligned(Xb, Tb, Yb, hte)


# 3) 用单棵 CatBoost 树蒸馏（也可改成 Optuna 版搜索深度/正则）
rule_model, info = fit_rule_catboost_single_tree(
    Xb, tau_oof, psi_oof,
    seed=2025, n_trials=150, n_splits=5, ks=(0.1,0.2,0.3),
    )

print("\n=== 蒸馏单棵树模型 ===")
print(info)

# 4) 评估蒸馏质量
_ = evaluate_rule_model(rule_model, Xb, tau_oof, psi_oof, ks=(0.1,0.2,0.3))

# 5) 导出叶子规则 & 逐条业务价值
rules = extract_rules_from_catboost_single_tree(rule_model, feature_names)
df_rules = evaluate_rules_per_leaf(rules, Xb, tau_oof, psi_oof, feature_names)
print("\n=== Top Rules (overlap band) ===")
print(df_rules.head(10).to_string(index=False))
