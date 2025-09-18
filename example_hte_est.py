# example_htt_est.py
import numpy as np

import sys
sys.path.append('/Users/clyde.ren/Documents/causalinference/src')  # 目录，而不是文件
import pandas as pd
from DistilledPolicy import  compute_teacher_oof_tau_psi_aligned, fit_rule_catboost_single_tree, evaluate_rule_model,evaluate_rules_per_leaf, extract_rules_from_catboost_single_tree

import os
import os.path

# import multiprocessing as mp, os, gc, tempfile
# mp.set_start_method("spawn", force=True)  # 关键：macOS 上禁止 fork
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ===== 1) 数据地址 =====
raw_path = '/Users/clyde.ren/Documents/causalinference/data' # 修改
res_path = '/Users/clyde.ren/Documents/causalinference/result' # 修改
res_name = 'id89.txt'

files = list(os.listdir(raw_path))
files = ['/Users/clyde.ren/Documents/causalinference/data/tmp_20250916_clyde_ren_casual_inference_fomo_review_10u_active_df_fomo_id89_watermarked.xlsx'] # 修改
#[i for i in files if 'convert' in i]

for file in files:
    FILE_PATH = os.path.join(raw_path, file)
    print(FILE_PATH)

    # 修改特征预处理逻辑
    df = pd.read_excel(FILE_PATH)
    df['before_amt'] = np.log1p(df['before_amt'])
    df['token_trade_success_amt_usd'] = np.log1p(df['token_trade_success_amt_usd'])
    df['btc_price'] = np.log1p(df['btc_price'])

    num_features = [
        'token_trade_success_amt_usd', 'btc_price', 'btc_ptr', 'btc_std', 'before_act_days','create_diff','act_cnts','before_amt'
        ]
    # 类别特征
    df = pd.get_dummies(df, columns=['region_name'], drop_first=True)
    cat_features = [c for c in df.columns if c.startswith("region_name_")]

    features = num_features + cat_features

    # --- 构造 X / T / Y ---
    df = df.dropna()
    X = df[features].values
    T = df['t'].astype(int).values
    Y = df['after_act_days'].astype(float).values
    X_names = features

    y_nc = None

    # ====== HTE with Optuna tuning on nAUUC (reg + clf + ForestDR head + dynamic band) ======
    from HTETester import HTETester

    hte = HTETester(
        tune_nauuc=True,
        n_trials=300,
        reg_loss="auto",
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
    rules = extract_rules_from_catboost_single_tree(rule_model, X_names)
    df_rules = evaluate_rules_per_leaf(rules, Xb, tau_oof, psi_oof, X_names)
    print("\n=== Top Rules (overlap band) ===")
    print(df_rules.head(10).to_string(index=False))

    with open(os.path.join(res_path,res_name),mode='a') as fobj:
        fobj.write("{}\n".format(file))
        fobj.write(report)
        fobj.write("\n")
        fobj.write(stat_res)
        fobj.write("\n")
        fobj.write(df_rules.head(10).to_string(index=False))
        fobj.write("\n")
