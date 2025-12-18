# -*- coding: utf-8 -*-
import uuid
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone

from UnifiedCausal import UnifiedCausalTester, CausalRules
from NAUUCCatBoostTunerV2 import NAUUCCatBoostTunerV2
from HTETesterLight import HTETesterLight
from DeepFeatureFactory import DeepFeatureFactory
from tools import get_dataframe, write_dataframe
import argparse

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import gc

def run_activity_causal_analysis(
    base_X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    cat_cols,
    num_cols,
    group_keys=None,
    t_col: str = "treatment",
    y_col: str = "outcome",
    activity_uuid: str | None = None,
    # 可选：一些关键超参暴露出来，方便以后调
    att_band: tuple[float, float] = (0.2, 0.8),
    hte_band: tuple[float, float] = (0.2, 0.8),
    identity: pd.Series | None = None
) -> dict:
    """
    从 DeepFeatureFactory 开始的一站式活动因果分析：
    输入预处理后的 base_X / T / Y 及列信息，输出结构化结果 dict。

    参数
    ----
    base_X : pd.DataFrame
        已选好的协变量特征（原始层面，cat + num），行为样本、列为特征。
    T : np.ndarray
        treatment 向量（0/1），shape = (n_samples,)
    Y : np.ndarray
        outcome 向量（连续），shape = (n_samples,)
    cat_cols : list[str]
        base_X 中的类别列名
    num_cols : list[str]
        base_X 中的数值列名
    group_keys : list[list[str]] | None
        DeepFeatureFactory 里 groupby 用的 key，形如 [["sex"], ["region", "device"]]。
    """

    # ========== 0) 元信息 & 输入清洗 ==========
    if activity_uuid is None:
        activity_uuid = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"

    cat_cols = list(cat_cols)
    num_cols = list(num_cols)
    group_keys = [list(g) for g in (group_keys or [])]

    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(float)

    # 用 Y 的中位数填 NaN，供工厂内部监督用
    y_for_factory = None

    n_samples = len(Y)
    n_treated = int((T == 1).sum())
    n_control = int((T == 0).sum())

    # ========== 1) DeepFeatureFactory ==========
    factory = DeepFeatureFactory(
        cat_cols=cat_cols,
        num_cols=num_cols,
        group_keys=group_keys,
        # 编码/统计
        enable_target_encoding=True,
        enable_count_freq=True,
        enable_woe=False,
        enable_groupby_agg=True,
        agg_funcs=("mean", "median", "std", "count", "nunique"),
        # 数值派生
        add_log1p=True,
        add_ratios=True,
        add_interactions=True,
        max_interactions=120,
        quantile_bins=12,
        # 多项式/AutoFeat
        add_poly2=True,
        autofeat_steps=0,
        # 树叶子特征
        add_leaf_embeddings=True,
        leaf_task="reg",
        leaf_rounds=100,
        leaf_depth=3,
        leaf_lr=0.05,
        # PCA + 列数控制
        add_pca=12,
        scale_before_pca=True,
        max_total_cols=30,
        verbose=0,
    )

    print("\n[STEP] 生成深特征矩阵 …")
    X_deep_df = factory.fit_transform(base_X, y=y_for_factory)
    feature_names = list(X_deep_df.columns)
    X = X_deep_df.to_numpy()

    print(f"[INFO] 深特征完成 | X shape = {X.shape}, feat = {len(feature_names)}")

    # ========== 2) 因果规则 & 底模调参 ==========
    rules = CausalRules(
        smd_max=0.3, ovl_min=0.50, ks_max=0.60, ess_min=0.70,
        placebo_alpha=0.09, nc_alpha=0.10, top_k_smd=8,
    )

    print("\n[STEP] NAUUCCatBoostTunerV2 超参调优（用于 DR 底模） …")
    reg, clf = NAUUCCatBoostTunerV2(
        verbose=0, n_trials=30, reg_lf="RMSE"
    ).fit_return_models(X, T, Y)

    common_kwargs = dict(
        n_splits=3,
        trim=0.01,
        ps_clip=(0.05, 0.95),
        weight_clip=10.0,
        n_jobs=4,
        n_jobs_placebo=4,
        random_state=2025,
        verbose=0,
        rules=rules,
        regressor=reg,
        classifier=clf,
    )

    # ========== 3) ATE ==========
    print("\n=== ATE on full sample (with DeepFeatureFactory features) ===")
    ate = UnifiedCausalTester(estimand="ATE", **common_kwargs)
    ate.fit(X, T, Y, X_names=feature_names, y_nc=None, placebo_runs=5)
    ate_report = ate.report()
    ate_res = ate.result_
    ate_point = float(ate_res["est"])
    ate_ci = [float(ate_res["ci"][0]), float(ate_res["ci"][1])]

    # 倾向得分用于后续 band
    e_oof = ate_res["diag"]["e"]

    # ========== 4) ATT on overlap band ==========
    print(f"\n=== ATT on overlap band e∈[{att_band[0]},{att_band[1]}] ===")
    band_low, band_high = att_band
    band = (e_oof >= band_low) & (e_oof <= band_high)
    treated_kept = int((T[band] == 1).sum())
    treated_total = int((T == 1).sum())

    print(
        f"Overlap-band coverage (treated kept): "
        f"{treated_kept}/{treated_total} = "
        f"{treated_kept / treated_total:.1%}" if treated_total > 0 else "N/A"
    )

    att_tester = UnifiedCausalTester(estimand="ATT", **common_kwargs)
    att_tester.fit(X[band], T[band], Y[band], X_names=feature_names, y_nc=None, placebo_runs=10)
    att_report = att_tester.report()
    att_res = att_tester.result_
    att_point = float(att_res["est"])
    att_ci = [float(att_res["ci"][0]), float(att_res["ci"][1])]

    # ========== 5) HTE ==========
    print("\n=== HTE (HTETesterLight) ===")
    hte_reg = clone(reg)
    hte_clf = clone(clf)

    hte = HTETesterLight(
        regressor=hte_reg,
        classifier=hte_clf,
        n_splits=3,
        band=hte_band,
        random_state=42,
        tune_nauuc=False,
        n_trials=3,
        early_stop=0.45,
        verbose=1,
    )
    _ = hte.fit(X, T, Y)
    hte_report = hte.report()
    tau_hat = hte.tau_hat_

    # ========== 6) τ̂ quantiles（全体 + 带内）==========
    qs = [10, 25, 50, 75, 90]

    # 全体
    tau_q_full = np.percentile(tau_hat, qs)
    tau_quantiles_full = {
        "p10": float(tau_q_full[0]),
        "p25": float(tau_q_full[1]),
        "p50": float(tau_q_full[2]),
        "p75": float(tau_q_full[3]),
        "p90": float(tau_q_full[4]),
    }

    # 带内（沿用 HTE 的 band ）
    band_low_hte, band_high_hte = hte_band
    band_hte = (e_oof >= band_low_hte) & (e_oof <= band_high_hte)
    if band_hte.any():
        tau_q_band = np.percentile(tau_hat[band_hte], qs)
        tau_quantiles_band = [float(v) for v in tau_q_band]
    else:
        tau_quantiles_band = []

    # ========== 7) 汇总结果 ==========
    result = {
        "activity_uuid": activity_uuid,
        "created_at": created_at,

        "t_col": t_col,
        "y_col": y_col,
        "n_samples": int(n_samples),
        "n_treated": n_treated,
        "n_control": n_control,

        "ate_point": ate_point,
        "ate_ci": ate_ci,
        "att_point": att_point,
        "att_ci": att_ci,

        "ate_report": ate_report,
        "att_report": att_report,
        "hte_report": hte_report,

        "tau_quantiles_full": tau_quantiles_full,   # json 对象
        "tau_quantiles_band": tau_quantiles_band,   # 数组

        "overlap_band_for_att": {
            "range": [float(band_low), float(band_high)],
            "treated_kept": treated_kept,
            "treated_total": treated_total,
            "treated_coverage": (
                treated_kept / treated_total if treated_total > 0 else None
            ),
        },
        "overlap_band_for_hte": {
            "range": [float(band_low_hte), float(band_high_hte)],
        },
    }
    if identity is not None:
        user_uplift = pd.DataFrame({
            "activity_uuid": activity_uuid,
            "identity": np.asarray(identity),
            "t_col": T,
            "y_col": y_col,
            "created_at": created_at,
            "tau_hat": tau_hat,
            "psi": hte.psi_
        })
        user_uplift = user_uplift[user_uplift['t_col'] == 1]
    else:
        user_uplift = pd.DataFrame({
            "activity_uuid":activity_uuid,
            "identity": '',
            "t_col": '',
            "y_col": '',
            "created_at": created_at,
            "tau_hat": '',
            "psi": ''
        }
        )

    return result, user_uplift