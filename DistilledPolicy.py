# ========== 基于现有 HTE 模型的“单树蒸馏 + 评估 + 规则导出” ==========
import json, os, tempfile
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from scipy.stats import spearmanr
from catboost import CatBoostRegressor, CatBoostClassifier
from econml.dr import ForestDRLearner
import numpy as np
import pandas as pd

import optuna
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
import numpy as np

# ---------- 工具 ----------
def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
    e = np.clip(e, trim, 1 - trim)
    return (t*(y - mu1)/e) - ((1-t)*(y - mu0)/(1-e)) + (mu1 - mu0)

def _area_cumgain_centered(psi, scores):
    order = np.argsort(-scores)
    psi_c = psi[order] - psi.mean()
    csum = np.cumsum(psi_c)
    x = np.arange(1, len(csum)+1) / len(csum)
    return float(np.trapz(csum, x))

def _policy_value(psi, scores, k):
    n = len(psi); m = max(1, int(np.floor(n*k)))
    order = np.argsort(-scores)
    return float(np.sum(psi[order][:m]))

def _oof_propensity_with_calib(X, T, base_clf, trim=0.02, cv=5, seed=42):
    """OOF + Sigmoid 校准得到更稳的 e_hat，用于划定 overlap 带"""
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    e = np.zeros(len(T), float)
    for tr, va in skf.split(X, T):
        clf = CalibratedClassifierCV(clone(base_clf), method="sigmoid", cv=3)
        clf.fit(X[tr], T[tr])
        e[va] = clf.predict_proba(X[va])[:, 1]
    return np.clip(e, trim, 1 - trim)

# ---------- 老师：在 overlap band 内得到 OOF τ 与 ψ ----------
def compute_teacher_oof_tau_psi(Xb, Tb, Yb, reg_proto, clf_proto, n_splits=5, trim=1e-3, seed=42,**params):
    """严格 OOF：每折 clone 你的 reg/clf，再训练 ForestDR 得 τ̂，同时用 AIPW 得 ψ"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(Xb)
    tau_oof = np.full(n, np.nan)
    psi_oof = np.full(n, np.nan)

    for tr, va in kf.split(Xb):
        X_tr, X_va = Xb[tr], Xb[va]
        T_tr, T_va = Tb[tr], Tb[va]
        Y_tr, Y_va = Yb[tr], Yb[va]

        # ψ 所需 e, μ1, μ0（e 用同一超参的 CatBoost + sigmoid 校准更稳）
        base_clf = clone(clf_proto)
        if not isinstance(base_clf, CalibratedClassifierCV):
            base_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        base_clf.fit(X_tr, T_tr)
        e_va = base_clf.predict_proba(X_va)[:, 1]

        reg1 = clone(reg_proto); reg0 = clone(reg_proto)
        # 退化保护：组内样本过少时直接用均值
        if (T_tr==1).sum() >= 30:
            reg1.fit(X_tr[T_tr==1], Y_tr[T_tr==1], verbose=0)
            mu1_va = reg1.predict(X_va)
        else:
            mu1_va = np.full(len(X_va), float(Y_tr[T_tr==1].mean() if (T_tr==1).any() else Y_tr.mean()))
        if (T_tr==0).sum() >= 30:
            reg0.fit(X_tr[T_tr==0], Y_tr[T_tr==0], verbose=0)
            mu0_va = reg0.predict(X_va)
        else:
            mu0_va = np.full(len(X_va), float(Y_tr[T_tr==0].mean() if (T_tr==0).any() else Y_tr.mean()))
        psi_oof[va] = _aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=trim)

        # ForestDR 老师 τ̂（折内重训，避免信息泄漏）
        dr_fold = ForestDRLearner(
            model_regression=clone(reg_proto),
            model_propensity=clone(clf_proto),
            random_state=seed, categories='auto',
            # 稳定头（与你HTE一致也可）
            n_estimators=800, max_depth=12, min_samples_split=4, min_samples_leaf=18,
            max_features='sqrt', max_samples=0.45, min_balancedness_tol=0.5,
            honest=True, subforest_size=8, cv=3, min_propensity=1e-3, n_jobs=-1
        )
        dr_fold.fit(Y_tr, T_tr, X=X_tr)
        tau_oof[va] = dr_fold.effect(X_va)

    return tau_oof, psi_oof

def compute_teacher_oof_tau_psi_aligned(Xb, Tb, Yb, hte):
    kf = KFold(n_splits=hte.n_splits, shuffle=True, random_state=hte.random_state)
    n = len(Xb)
    tau_oof = np.full(n, np.nan)
    psi_oof = np.full(n, np.nan)
    trim = hte.trim

    for tr, va in kf.split(Xb):
        X_tr, X_va = Xb[tr], Xb[va]
        T_tr, T_va = Tb[tr], Tb[va]
        Y_tr, Y_va = Yb[tr], Yb[va]

        # e, μ1, μ0 用 HTE 的原型（不强制再做 Sigmoid 校准，和 HTE 保持一致）
        clf = clone(hte.classifier); clf.fit(X_tr, T_tr)
        e_va = clf.predict_proba(X_va)[:, 1]
        reg1 = clone(hte.regressor); reg0 = clone(hte.regressor)
        if (T_tr==1).sum() >= 30:
            reg1.fit(X_tr[T_tr==1], Y_tr[T_tr==1]); mu1_va = reg1.predict(X_va)
        else:
            mu1_va = np.full(len(X_va), float(Y_tr[T_tr==1].mean() if (T_tr==1).any() else Y_tr.mean()))
        if (T_tr==0).sum() >= 30:
            reg0.fit(X_tr[T_tr==0], Y_tr[T_tr==0]); mu0_va = reg0.predict(X_va)
        else:
            mu0_va = np.full(len(X_va), float(Y_tr[T_tr==0].mean() if (T_tr==0).any() else Y_tr.mean()))
        # AIPW ψ（与 HTE 相同的 trim）
        e_clipped = np.clip(e_va, trim, 1-trim)
        psi_oof[va] = (T_va*(Y_va - mu1_va)/e_clipped
                      - (1-T_va)*(Y_va - mu0_va)/(1-e_clipped)
                      + (mu1_va - mu0_va))

        # 老师 τ̂：ForestDR 用 HTE 调好的 head
        frh = hte.forest_head
        dr = ForestDRLearner(model_regression=clone(hte.regressor),
                             model_propensity=clone(hte.classifier),
                             **frh)
        dr.fit(Y_tr, T_tr, X=X_tr)
        tau_oof[va] = dr.effect(X_va)
    return tau_oof, psi_oof

# ---------- 单树 CatBoost 训练（规则学生） ----------
# def fit_rule_catboost_single_tree(Xb, tau_oof, seed=42, depth=3, min_leaf_frac=0.02):
#     rule_model = CatBoostRegressor(
#         depth=depth, n_estimators=1, learning_rate=0.3,
#         l2_leaf_reg=5.0, loss_function='RMSE', random_seed=seed,
#         min_data_in_leaf=max(20, int(min_leaf_frac*len(Xb))), verbose=0
#     )
#     rule_model.fit(Xb, tau_oof)
#     return rule_model

def fit_rule_catboost_single_tree(
    Xb, tau_oof, psi_oof,
    seed: int = 2025,
    n_trials: int = 150,
    n_splits: int = 5,
    ks=(0.1, 0.2, 0.3),
    show_progress_bar: bool = True,
    early_stop_target: float = 0.75
):
    """
    用 Optuna 搜索“单棵 CatBoost 树”的超参（保持 n_estimators=1），
    目标：在 OOF 上最大化综合分（0.6*nAUUC_ratio + 0.3*Spearman + 0.1*Policy@k比）。
    返回：best_model(已在全量 Xb 上拟合) 与 info(最优 trial 信息)。
    依赖：tau_oof(老师 OOF τ̂)、psi_oof(带内 OOF AIPW ψ)。
    """
    Xb = np.asarray(Xb)
    tau_oof = np.asarray(tau_oof, float)
    psi_oof = np.asarray(psi_oof, float)
    n = len(Xb)

    def _policy_value(psi, scores, k):
        m = max(1, int(np.floor(len(scores)*k)))
        order = np.argsort(-scores)
        return float(np.sum(psi[order][:m]))

    def _area_cumgain_centered(psi, scores):
        order = np.argsort(-scores)
        psi_c = psi[order] - psi.mean()
        csum = np.cumsum(psi_c)
        x = np.arange(1, len(csum)+1) / len(csum)
        return float(np.trapz(csum, x))

    def _build_model(trial, n_samples):
        depth = trial.suggest_int("rule_depth", 2, 6)
        min_leaf_frac = trial.suggest_float("rule_min_leaf_frac", 0.01, 0.15)
        l2 = trial.suggest_float("rule_l2", 1e-2, 50.0, log=True)
        lr = trial.suggest_float("rule_lr", 0.05, 0.5, log=True)
        params = dict(
            depth=depth,
            n_estimators=1,              # 单棵树
            learning_rate=lr,
            l2_leaf_reg=l2,
            loss_function='RMSE',
            random_seed=seed,
            verbose=0,
        )
        # 有的版本支持 min_data_in_leaf
        try:
            params["min_data_in_leaf"] = max(20, int(min_leaf_frac * n_samples))
        except Exception:
            pass
        return CatBoostRegressor(**params)

    def _oof_pred_single_tree(trial):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = np.full(n, np.nan)
        for tr, va in kf.split(Xb):
            model = _build_model(trial, n_samples=len(tr))
            model.fit(Xb[tr], tau_oof[tr])
            pred[va] = model.predict(Xb[va])
        return pred

    def _objective(trial):
        pred_oof = _oof_pred_single_tree(trial)

        # nAUUC（以 ψ 为 oracle）
        area_teacher = _area_cumgain_centered(psi_oof, tau_oof)
        area_student = _area_cumgain_centered(psi_oof, pred_oof)
        area_oracle  = _area_cumgain_centered(psi_oof, psi_oof)
        nauuc_teacher = 0.0 if abs(area_oracle) < 1e-12 else float(np.clip(area_teacher/area_oracle, 0.0, 1.0))
        nauuc_student = 0.0 if abs(area_oracle) < 1e-12 else float(np.clip(area_student/area_oracle, 0.0, 1.0))
        nauuc_ratio   = 0.0 if nauuc_teacher <= 1e-12 else float(nauuc_student / nauuc_teacher)

        # 排序一致性
        sp = spearmanr(tau_oof, pred_oof, nan_policy='omit').correlation
        sp = 0.0 if np.isnan(sp) else float(sp)

        # Policy@k 比例（均值）
        ratios = []
        for k in ks:
            pt = _policy_value(psi_oof, tau_oof, k)
            ps = _policy_value(psi_oof, pred_oof, k)
            ratio_k = 0.0 if abs(pt) < 1e-12 else float(np.clip(ps / pt, 0.0, 1.2))
            ratios.append(ratio_k)
        pol_ratio = float(np.mean(ratios))

        score = 0.6 * nauuc_ratio + 0.3 * max(0.0, sp) + 0.1 * pol_ratio

        # 记录便于分析/调参
        trial.set_user_attr("nauuc_ratio", nauuc_ratio)
        trial.set_user_attr("spearman", sp)
        trial.set_user_attr("policy_ratio_mean", pol_ratio)
        trial.set_user_attr("nauuc_teacher", nauuc_teacher)
        trial.set_user_attr("nauuc_student", nauuc_student)

        if (early_stop_target is not None) and (score >= early_stop_target):
            trial.study.stop()
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best = study.best_trial
    best_model = _build_model(best, n_samples=len(Xb))
    best_model.fit(Xb, tau_oof)  # 用全量带内数据拟合最终“规则学生”

    info = {
        "best_score": float(best.value),
        "best_params": dict(best.params),
        "attrs": {k: best.user_attrs.get(k) for k in
                  ["nauuc_ratio", "spearman", "policy_ratio_mean",
                   "nauuc_teacher", "nauuc_student"]},
        "n_trials": n_trials
    }
    return best_model, info

# ---------- 从 CatBoost JSON 解析单棵树的规则 ----------
def extract_rules_from_catboost_single_tree(rule_model, feature_names):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json"); tmp.close()
    rule_model.save_model(tmp.name, format='json')
    data = json.load(open(tmp.name, 'r')); os.unlink(tmp.name)

    trees = data.get('oblivious_trees') or data.get('trees') or data
    if isinstance(trees, dict) and 'oblivious_trees' in trees:
        trees = trees['oblivious_trees']
    assert len(trees) >= 1, "No tree found in JSON."
    tree = trees[0]
    splits = tree['splits']
    leaf_vals = tree['leaf_values']

    rules = []
    for leaf_idx, leaf_val in enumerate(leaf_vals):
        conds = []
        for d, sp in enumerate(splits):
            bit = (leaf_idx >> d) & 1
            if 'float_feature_index' in sp:
                j = sp['float_feature_index']; thr = sp.get('border', sp.get('threshold', 0.5))
                op = '>=' if bit==1 else '<'
                conds.append((j, op, float(thr)))
            elif 'one_hot_feature_index' in sp:
                j = sp['one_hot_feature_index']; val = sp.get('value', 1)
                op = '==' if bit==1 else '!='
                conds.append((j, op, val))
            else:
                j = sp.get('feature_index', 0); thr = sp.get('border', 0.5)
                op = '>=' if bit==1 else '<'
                conds.append((j, op, float(thr)))
        parts = [f"{(feature_names[j] if j < len(feature_names) else f'f{j}')} {op} {thr:g}" for j,op,thr in conds]
        rules.append({'leaf': int(leaf_idx), 'value': float(leaf_val),
                      'conditions': conds, 'rule_text': " AND ".join(parts)})
    return rules

# ---------- 评估：全局蒸馏质量 + 逐条规则业务价值 ----------
def evaluate_rule_model(rule_model, Xb, tau_teacher, psi_oof, ks=(0.1,0.2,0.3)):
    pred = rule_model.predict(Xb)
    r2  = r2_score(tau_teacher, pred)
    rmse = mean_squared_error(tau_teacher, pred)
    sp  = spearmanr(tau_teacher, pred, nan_policy='omit').correlation

    area_teacher = _area_cumgain_centered(psi_oof, tau_teacher)
    area_student = _area_cumgain_centered(psi_oof, pred)
    area_oracle  = _area_cumgain_centered(psi_oof, psi_oof)
    nauuc_teacher = float(np.clip(area_teacher/area_oracle, 0.0, 1.0)) if abs(area_oracle)>1e-12 else 0.0
    nauuc_student = float(np.clip(area_student/area_oracle, 0.0, 1.0)) if abs(area_oracle)>1e-12 else 0.0
    nauuc_ratio   = 0.0 if nauuc_teacher==0 else float(nauuc_student/nauuc_teacher)

    pol = { f"policy_teacher@{int(k*100)}": _policy_value(psi_oof, tau_teacher, k) for k in ks }
    pol.update({ f"policy_student@{int(k*100)}": _policy_value(psi_oof, pred, k) for k in ks })

    print("\n=== Single-tree CatBoost Rule Model — Distillation Quality (overlap band) ===")
    print(f"Fit:   R2={r2:.3f}, RMSE={rmse:.4f}, Spearman={sp:.3f}")
    print(f"AUUC:  teacher nAUUC={nauuc_teacher:.3f}, rule nAUUC={nauuc_student:.3f}, "
          f"rule/teacher={nauuc_ratio:.3f}")
    ok = (nauuc_ratio >= 0.70 and (sp if not np.isnan(sp) else 0.0) >= 0.60 
        and pol['policy_student@10'] >= 0.85*pol['policy_teacher@10']
        and pol['policy_student@20'] >= 0.85*pol['policy_teacher@20']
        and pol['policy_student@30'] >= 0.85*pol['policy_teacher@30'])
    print(f"\nRule model PASS? {ok}  "
        f"(nAUUC_ratio={nauuc_ratio:.2f}, Spearman={(0.0 if np.isnan(sp) else sp):.2f}, "
        f"P@10/20/30={pol['policy_student@10']:.1f}/{pol['policy_student@20']:.1f}/{pol['policy_student@30']:.1f})")

    for k in ks:
        print(f"Policy@{int(k*100)}: teacher={pol[f'policy_teacher@{int(k*100)}']:.3f}, "
              f"rule={pol[f'policy_student@{int(k*100)}']:.3f}")

    return {"r2":r2, "rmse":rmse, "spearman":float(0.0 if np.isnan(sp) else sp),
            "nauuc_teacher":nauuc_teacher, "nauuc_rule":nauuc_student,
            "nauuc_ratio":nauuc_ratio, **pol}

def evaluate_rules_per_leaf(rules, Xb, tau_teacher, psi_oof, feature_names, seed=42):
    rows = []
    rng = np.random.RandomState(seed)
    n = len(Xb)
    for r in rules:
        mask = np.ones(n, dtype=bool)
        for j, op, thr in r['conditions']:
            if op == '<':   mask &= (Xb[:, j] <  thr)
            elif op == '>=':mask &= (Xb[:, j] >= thr)
            elif op == '==':mask &= (Xb[:, j] ==  thr)
            elif op == '!=':mask &= (Xb[:, j] !=  thr)
        cnt = int(mask.sum())
        if cnt == 0: 
            continue
        tau_mean = float(np.nanmean(tau_teacher[mask]))
        bs = rng.choice(tau_teacher[mask], size=(200, cnt), replace=True).mean(1)
        lo, hi = np.percentile(bs, [5, 95])
        policy_val = float(np.sum(psi_oof[mask]))
        coverage = cnt / n
        rows.append({
            "rule_text": r['rule_text'],
            "coverage": round(coverage, 4),
            "n": cnt,
            "uplift_mean_tau": round(tau_mean, 6),
            "uplift_lo90": round(float(lo), 6),
            "uplift_hi90": round(float(hi), 6),
            "policy_value": round(policy_val, 4)
        })
    df_rules = pd.DataFrame(rows).sort_values(
        ["policy_value", "uplift_mean_tau", "coverage"], ascending=False
    )
    return df_rules
