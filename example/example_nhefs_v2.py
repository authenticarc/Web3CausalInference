# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src")

import numpy as np
import pandas as pd

from UnifiedCausal import UnifiedCausalTester, CausalRules
from NAUUCCatBoostTunerV2 import NAUUCCatBoostTunerV2
from causaldata import nhefs_complete
from DistilledPolicy import compute_teacher_oof_tau_psi_aligned, fit_rule_catboost_single_tree, evaluate_rule_model, evaluate_rules_per_leaf, extract_rules_from_catboost_single_tree
from HTETester import HTETester

# 新增：深特征工厂
from DeepFeatureFactory import DeepFeatureFactory

# ===================== 1) 数据加载 =====================
try:
    df = nhefs_complete.load_pandas().data.copy()
except Exception:
    df = nhefs_complete.load_pandas().copy()

t_col = "qsmk"
y_col = "wt82_71"   # 连续结局（体重变化）

# 协变量（可按需增减；以下为常见一组）
base_covs = [
    "sex", "race", "age", "education",
    "smokeintensity", "smokeyrs",
    "exercise", "active",
    "wt71", "ht", "bmix",  # bmix 若不存在，可用 bmi
    "alcohol", "marital"
]
covs = [c for c in base_covs if c in df.columns]
if "bmi" in df.columns and "bmix" not in df.columns:
    covs.append("bmi")

# ===================== 2) 先用 DeepFeatureFactory 生成特征 =====================
# 拆分类别与数值列（这里按“常识列名”粗分；也可以用 df.dtypes 自适应）
cat_cols = [c for c in covs if c in {"sex","race","education","exercise","active","marital"} and c in df.columns]
num_cols = [c for c in covs if c not in cat_cols and c in df.columns]

# 组内聚合的 key（可按需增减）
group_keys = [["sex"], ["race"], ["education"]]
group_keys = [g for g in group_keys if all(k in df.columns for k in g)]

# 准备工厂输入表
base_X = df[cat_cols + num_cols].copy()
# 目标用于 OOF-TE/叶子嵌入：用连续结局 y（也可以换成 ψ，更贴因果排序，但此处用 y 更简单）
Y_series = df[y_col].astype(float)
# y 中的 NaN 用中位数兜底，避免目标编码/叶子训练报错
y_for_factory = Y_series.fillna(Y_series.median()).to_numpy()

# 建立工厂
factory = DeepFeatureFactory(
    cat_cols=cat_cols,
    num_cols=num_cols,
    group_keys=group_keys,
    # 编码/统计
    enable_target_encoding=True,   # OOF-TE，用 y 监督
    enable_count_freq=True,
    enable_woe=False,              # y 为连续，WoE 关闭
    enable_groupby_agg=True,
    agg_funcs=("mean","median","std","count","nunique"),
    # 数值派生
    add_log1p=True,
    add_ratios=True,
    add_interactions=True,
    max_interactions=120,
    quantile_bins=12,
    # 多项式/AutoFeat
    add_poly2=True,
    autofeat_steps=0,              # 如需更深可设 1（更慢）
    # 树叶子特征
    add_leaf_embeddings=True,
    leaf_task="reg",               # y 连续 → 回归叶子
    leaf_rounds=600,
    leaf_depth=6,
    leaf_lr=0.05,
    # 降维
    add_pca=12,
    scale_before_pca=True,
    # 控规模
    max_total_cols=30,
    # 打印
    verbose=1
)

print("\n[STEP] 生成深特征矩阵 …")
X_deep_df = factory.fit_transform(base_X, y=y_for_factory)
feature_names = list(X_deep_df.columns)
X = X_deep_df.to_numpy()

# 处理 T / Y
T = df[t_col].to_numpy().ravel().astype(int)
Y = Y_series.to_numpy().ravel().astype(float)
y_nc = None  # 没有负控，这里保持 None

print(f"[INFO] 深特征完成 | X shape = {X.shape}, 原始 cov 数 = {len(covs)}, 输出特征数 = {len(feature_names)}")

# ===================== 3) Causal 估计器与规则配置 =====================
rules = CausalRules(
    smd_max=0.3, ovl_min=0.50, ks_max=0.60, ess_min=0.70,
    placebo_alpha=0.09, nc_alpha=0.10, top_k_smd=8
)

# 这里的 tuner 仍然只返回 reg / clf（不改对外 API）
print("\n[STEP] NAUUCCatBoostTunerV2 超参调优（用于 DR 的底模） …")
reg, clf = NAUUCCatBoostTunerV2(verbose=0, n_trials=20, reg_lf='RMSE').fit_return_models(X, T, Y)

common_kwargs = dict(
    n_splits=5,
    trim=0.01,
    ps_clip=(0.05, 0.95),
    weight_clip=10.0,
    n_jobs=-1,
    n_jobs_placebo=-1,
    random_state=2025,
    verbose=0,
    rules=rules,
    regressor=reg,
    classifier=clf
)

# ===================== 4) ATE on full sample =====================
print("\n=== ATE on full sample (with DeepFeatureFactory features) ===")
ate = UnifiedCausalTester(estimand="ATE", **common_kwargs)
# X_names 应该传入“深特征名”以便报告里显示
ate.fit(X, T, Y, X_names=feature_names, y_nc=None, placebo_runs=10)
print(ate.report())

# 取得 OOF 倾向分数，后面做重叠带
e_oof = ate.result_["diag"]["e"]

# ===================== 5) ATT on overlap band e∈[0.3,0.7] =====================
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
treated_df = df[df['t'] == 1]

stats = treated_df['label'].value_counts(normalize=True).rename("proportion").to_frame()
stats['count'] = treated_df['label'].value_counts()
stats['proportion'] = stats['proportion'].round(4)

print("=== HTT 实验组用户分布 ===")
stat_res = []
for lbl, row in stats.iterrows():
    stat_res.append(f"{lbl:7s}: {row['count']} ({row['proportion']:.4f})")
    print(f"{lbl:7s}: {row['count']} ({row['proportion']:.4f})")
stat_res = "\n".join(stat_res)
print(stat_res)

# # ---------- 与 HTETester/HTE 模型对接 ----------

# # 1) 划定 overlap 带（用 HTE 搜出的分类器原型，做 OOF + Sigmoid 校准）
# clf_proto = hte.classifier
# reg_proto = hte.regressor
# e_oof = hte._oof_propensity(X, T, clf_proto, trim=hte.trim) #_oof_propensity_with_calib(X, T, clf_proto, trim=0.02, cv=5, seed=42)
# band_lo, band_hi = hte.nauuc_band
# band = (e_oof >= band_lo) & (e_oof <= band_hi)
# print(f"[Distill] 带覆盖率={band.mean():.2%}，n={int(band.sum())}，带=({band_lo:.2f},{band_hi:.2f})")

# # 2) 老师 OOF τ 与 ψ（只在带内）
# Xb, Tb, Yb = X[band], T[band], Y[band]
# # tau_oof, psi_oof = compute_teacher_oof_tau_psi(
# #     Xb, Tb, Yb, reg_proto=reg_proto, clf_proto=clf_proto, n_splits=5, trim=0.02, seed=2025
# # )
# tau_oof, psi_oof = compute_teacher_oof_tau_psi_aligned(Xb, Tb, Yb, hte)


# # 3) 用单棵 CatBoost 树蒸馏（也可改成 Optuna 版搜索深度/正则）
# rule_model, info = fit_rule_catboost_single_tree(
#     Xb, tau_oof, psi_oof,
#     seed=2025, n_trials=150, n_splits=5, ks=(0.1,0.2,0.3),
#     )

# print("\n=== 蒸馏单棵树模型 ===")
# print(info)

# # 4) 评估蒸馏质量
# _ = evaluate_rule_model(rule_model, Xb, tau_oof, psi_oof, ks=(0.1,0.2,0.3))

# # 5) 导出叶子规则 & 逐条业务价值
# rules = extract_rules_from_catboost_single_tree(rule_model, feature_names)
# df_rules = evaluate_rules_per_leaf(rules, Xb, tau_oof, psi_oof, feature_names)
# print("\n=== Top Rules (overlap band) ===")
# print(df_rules.head(10).to_string(index=False))
