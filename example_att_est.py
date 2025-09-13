# example_att_est.py
import numpy as np

import sys
sys.path.append('/Users/clyde.ren/Documents/causalinference/src')  # 目录，而不是文件
from UnifiedCausal import UnifiedCausalTester, CausalRules
from nAUUCCatBoostTuner import NAUUCCatBoostTuner

import pandas as pd
import os.path
import os

# ===== 1) 数据地址 =====
raw_path = '/Users/clyde.ren/Documents/causalinference/data' # 修改
res_path = '/Users/clyde.ren/Documents/causalinference/result' # 修改
res_name = 'result_ate_convert.txt'

files = list(os.listdir(raw_path)) # 修改
files = ['/Users/clyde.ren/Documents/causalinference/data/tmp_20250909_clyde_ren_casual_inference_fomo_review_10u_convert_df_id82_watermarked.xlsx'] # 修改
#[i for i in files if 'convert' in i]

for file in files:
    FILE_PATH = os.path.join(raw_path, file)
    print(FILE_PATH)

    df = pd.read_excel(FILE_PATH)

    # 修改特征预处理逻辑
    df['before_amt'] = np.log1p(df['before_amt'])
    df['token_trade_success_amt_usd'] = np.log1p(df['token_trade_success_amt_usd'])
    df['btc_price'] = np.log1p(df['btc_price'])
    df['after_amt'] = np.log1p(df['after_amt'])

    num_features = ['token_trade_success_amt_usd', 'btc_price', 'btc_ptr', 'btc_std', 'last_7d_active_days', 'act_cnts','before_amt']
    # 类别特征
    df = pd.get_dummies(df, columns=['region_name'], drop_first=True)
    cat_features = [c for c in df.columns if c.startswith("region_name_")]

    features = num_features + cat_features

    # --- 构造 X / T / Y ---
    df = df.dropna()
    X = df[features].values
    T = df['t'].astype(int).values
    Y = df['after_amt'].astype(float).values
    X_names = features

    y_nc = None

    # ===== 2) 公共规则 & 估计器配置（按需调快/调严）=====
    rules = CausalRules(
        smd_max=0.3, ovl_min=0.50, ks_max=0.60, ess_min=0.70,
        placebo_alpha=0.09, nc_alpha=0.10, top_k_smd=8
    )

    reg, clf = NAUUCCatBoostTuner(verbose=0,n_trials=300).fit_return_models(X,T,Y)

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
    ate.fit(X, T, Y, X_names=X_names, y_nc=None, placebo_runs=10)
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

    att_band_result = att_band.report()

    with open(os.path.join(res_path,res_name),mode='a') as fobj:
        fobj.write("{}\n".format(file))
        fobj.write(att_band_result)
        fobj.write("\n")
