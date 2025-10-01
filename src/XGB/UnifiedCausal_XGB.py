# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import clone
from dataclasses import dataclass
from sklearn.model_selection import KFold
from scipy.stats import norm
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pandas as pd
import xgboost as xgb
from pandas.api.types import (
    is_numeric_dtype, is_integer_dtype, is_string_dtype,
    is_categorical_dtype, is_datetime64_any_dtype
)

# ===== 工具函数 =====
def _ks(ps_t, ps_c):
    a = np.sort(ps_t); b = np.sort(ps_c)
    n, m = len(a), len(b)
    ia = ib = 0; d = 0.0
    while ia < n and ib < m:
        if a[ia] <= b[ib]: ia += 1
        else: ib += 1
        d = max(d, abs(ia/n - ib/m))
    return d

def _ovl(ps_t, ps_c, bins=50):
    hist_t, edges = np.histogram(ps_t, bins=bins, range=(0,1), density=True)
    hist_c, _     = np.histogram(ps_c, bins=bins, range=(0,1), density=True)
    bw = edges[1] - edges[0]
    return float(np.sum(np.minimum(hist_t, hist_c)) * bw)

def _weighted_mean_var(x, w):
    w = np.asarray(w, float); x = np.asarray(x, float)
    wm = np.sum(w * x) / (np.sum(w) + 1e-12)
    wv = np.sum(w * (x - wm) ** 2) / (np.sum(w) + 1e-12)
    return wm, wv

def _ess_ratio(w, mask):
    w_g = np.asarray(w)[mask]
    n_g = int(mask.sum())
    ess = (w_g.sum() ** 2) / (np.sum(w_g ** 2) + 1e-12)
    return float(ess / (n_g + 1e-12))

def _smd_ate(x, t, w_t, w_c):
    x = np.asarray(x, float); t = np.asarray(t, int)
    mt, vt = _weighted_mean_var(x[t==1], w_t[t==1])
    mc, vc = _weighted_mean_var(x[t==0], w_c[t==0])
    pooled = np.sqrt((vt + vc)/2.0 + 1e-12)
    return float((mt - mc) / (pooled + 1e-12))

def _smd_att(x, t, w_c_odds):
    x = np.asarray(x, float); t = np.asarray(t, int)
    mt, vt = np.mean(x[t==1]), np.var(x[t==1])
    mc, vc = _weighted_mean_var(x[t==0], w_c_odds[t==0])
    pooled = np.sqrt((vt + vc)/2.0 + 1e-12)
    return float((mt - mc) / (pooled + 1e-12))

def _smd_ato(x, t, w1, w0):
    x = np.asarray(x, float); t = np.asarray(t, int)
    m1, v1 = _weighted_mean_var(x[t==1], w1[t==1])
    m0, v0 = _weighted_mean_var(x[t==0], w0[t==0])
    pooled = np.sqrt((v1 + v0)/2.0 + 1e-12)
    return float((m1 - m0) / (pooled + 1e-12))

# === nAUUC 专用工具 ===
def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
    e = np.clip(e, trim, 1 - trim)
    return (t*(y - mu1)/e) - ((1-t)*(y - mu0)/(1-e)) + (mu1 - mu0)

def _area_cumgain_centered(psi, scores):
    order = np.argsort(-scores)
    psi_ord = psi[order]
    psi_c = psi_ord - psi_ord.mean()
    csum = np.cumsum(psi_c)
    x = np.arange(1, len(csum)+1) / len(csum)
    return float(np.trapz(csum, x))

def _policy_values(psi, scores, ks=(0.1,0.2,0.3)):
    out = {}
    n = len(psi)
    order = np.argsort(-scores)
    psi_ord = psi[order]
    for k in ks:
        m = max(1, int(np.floor(n*k)))
        out[f'policy@{int(k*100)}'] = float(np.sum(psi_ord[:m]))
    return out

# ===== 规则阈值 =====
@dataclass
class CausalRules:
    smd_max: float = 0.10
    ovl_min: float = 0.50
    ks_max:  float = 0.40
    ess_min: float = 0.70
    placebo_alpha: float = 0.10
    nc_alpha: float = 0.10
    top_k_smd: int = 10

# ===== 单折函数（供并行）=====  —— 用 XGBoost
def _fit_fold_return_oof(va_idx, X_tr, X_va, T_tr, Y_tr,
                         regressor_proto, classifier_proto, trim, seed):
    clf = clone(classifier_proto)
    reg1 = clone(regressor_proto)
    reg0 = clone(regressor_proto)
    # 外层并行 → 内层单线程
    for mdl in (clf, reg1, reg0):
        if hasattr(mdl, "set_params"):
            try: mdl.set_params(n_jobs=1)
            except: pass

    clf.fit(X_tr, T_tr)
    e_hat = np.clip(clf.predict_proba(X_va)[:, 1], trim, 1 - trim)

    mu1_hat = np.zeros(len(va_idx))
    mu0_hat = np.zeros(len(va_idx))
    if (T_tr==1).any():
        reg1.fit(X_tr[T_tr==1], Y_tr[T_tr==1])
        mu1_hat = reg1.predict(X_va)
    if (T_tr==0).any():
        reg0.fit(X_tr[T_tr==0], Y_tr[T_tr==0])
        mu0_hat = reg0.predict(X_va)

    return va_idx, mu1_hat, mu0_hat, e_hat

# ===== 统一类（带 tqdm）=====  —— XGBoost + 自动特征识别/编码 + 手写 DR nAUUC
class UnifiedCausalTester:
    """
    estimand ∈ {"ATE","ATT","ATO"}
    - 基模：XGBoost
    - 自动识别类别 → 稀有并桶 + TopK + one-hot（固定列顺序确保 train/valid 对齐）
    - nAUUC：手写 DR-learner 的 OOF 版本（无 econml/dowhy）
    """
    def __init__(self,
                 estimand="ATE",
                 regressor=None,
                 classifier=None,
                 n_splits=5,
                 trim=0.01,
                 ps_clip=(0.01, 0.99),
                 weight_clip=None,
                 n_jobs=None,
                 n_jobs_placebo=1,
                 random_state=42,
                 rules: CausalRules = CausalRules(),
                 verbose: int = 1,
                 # === nAUUC 相关开关 ===
                 compute_nauuc: bool = False,
                 nauuc_band=(0.3, 0.7),
                 nauuc_policy_ks=(0.1, 0.2, 0.3),
                 # 自动编码参数
                 auto_encode_cats: bool = True,
                 int_as_cat_unique_thresh: int = 30,
                 unique_ratio_thresh: float = 0.05,
                 rare_freq_ratio: float = 0.001,
                 max_onehot_levels: int = 200,
                 # XGB 设备
                 use_gpu: bool = False):
        assert estimand in {"ATE","ATT","ATO"}
        self.estimand = estimand
        self.n_splits = int(n_splits)
        self.trim = float(trim)
        self.ps_clip = ps_clip
        self.weight_clip = weight_clip
        self.n_jobs = n_jobs
        self.n_jobs_placebo = n_jobs_placebo
        self.random_state = int(random_state)
        self.rules = rules
        self.verbose = int(verbose)

        # XGB 设备参数（2.0+ 推荐）
        self._device_args = {"tree_method": "hist", "device": ("cuda" if use_gpu else "cpu")}

        # 模型默认值（可外部注入）
        self.regressor = regressor or xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=6, learning_rate=0.05, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0,
            random_state=self.random_state, n_jobs=1, **self._device_args
        )
        self.classifier = classifier or xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=6, learning_rate=0.05, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0,
            random_state=self.random_state, n_jobs=1, **self._device_args
        )

        # nAUUC
        self.compute_nauuc = bool(compute_nauuc)
        self.nauuc_band = tuple(nauuc_band)
        self.nauuc_policy_ks = nauuc_policy_ks

        # 自动编码配置
        self.auto_encode_cats = bool(auto_encode_cats)
        self.int_as_cat_unique_thresh = int(int_as_cat_unique_thresh)
        self.unique_ratio_thresh = float(unique_ratio_thresh)
        self.rare_freq_ratio = float(rare_freq_ratio)
        self.max_onehot_levels = int(max_onehot_levels)
        self._cat_levels_: Dict[str, list] = {}

        self.result_ = None
        self._colnames_: list = None  # 编码后列名（可选，便于诊断输出）

    # ---------- 自动识别 & 稳定编码 ---------- #
    def _infer_cats(self, X: pd.DataFrame) -> list:
        cats = []
        n = len(X)
        for c in X.columns:
            s = X[c]
            if is_datetime64_any_dtype(s):
                continue
            nunq = s.nunique(dropna=False)
            ur = nunq / max(n, 1)
            is_cat = (
                is_string_dtype(s) or is_categorical_dtype(s) or
                (is_integer_dtype(s) and nunq <= self.int_as_cat_unique_thresh) or
                (not is_numeric_dtype(s) and ur <= self.unique_ratio_thresh)
            )
            if (not is_cat) and is_numeric_dtype(s) and nunq <= self.int_as_cat_unique_thresh:
                is_cat = True
            if is_cat:
                cats.append(c)
        return cats

    def _fit_cat_maps(self, X: pd.DataFrame, cat_cols: list):
        self._cat_levels_.clear()
        n = len(X)
        rare_thresh = max(int(self.rare_freq_ratio * n), 1)
        for c in cat_cols:
            vc = X[c].astype("string").fillna("__NA__").value_counts(dropna=False)
            common = vc[vc >= rare_thresh].index.tolist()
            if len(common) > self.max_onehot_levels:
                common = list(vc.index[: self.max_onehot_levels])
            self._cat_levels_[c] = ["__RARE__"] + [str(v) for v in common]

    def _apply_cat_maps(self, X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        X2 = X.copy()
        for c in cat_cols:
            X2[c] = X2[c].astype("string").fillna("__NA__")
            levels = set(self._cat_levels_[c][1:])
            X2[c] = X2[c].apply(lambda v: v if v in levels else "__RARE__")

        dummies = []
        for c in cat_cols:
            cats = pd.Categorical(X2[c], categories=self._cat_levels_[c])
            dummies.append(pd.get_dummies(cats, prefix=c, dummy_na=False))
        D = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=X2.index)

        num_cols = [c for c in X.columns if c not in cat_cols and is_numeric_dtype(X[c])]
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce")
        out = pd.concat([X_num, D], axis=1).fillna(0.0)
        return out

    def _encode_X_fit(self, X):
        if not self.auto_encode_cats or not isinstance(X, pd.DataFrame):
            # 直接返回 np 数组
            Xn = np.asarray(X, float)
            self._colnames_ = [f"x{j}" for j in range(Xn.shape[1])]
            return Xn
        cats = self._infer_cats(X)
        self._fit_cat_maps(X, cats)
        Xenc = self._apply_cat_maps(X, cats)
        self._colnames_ = list(Xenc.columns)
        return Xenc.to_numpy(float, copy=False)

    def _encode_X(self, X):
        if not self.auto_encode_cats or not isinstance(X, pd.DataFrame):
            Xn = np.asarray(X, float)
            # 列名数量保持
            if self._colnames_ is None:
                self._colnames_ = [f"x{j}" for j in range(Xn.shape[1])]
            return Xn
        cat_cols = list(self._cat_levels_.keys())
        Xenc = self._apply_cat_maps(X, cat_cols)
        # 对齐列（防止缺列）
        for c in self._colnames_:
            if c not in Xenc.columns:
                Xenc[c] = 0.0
        Xenc = Xenc[self._colnames_]
        return Xenc.to_numpy(float, copy=False)

    # --- 工具 ---
    def _apply_clip(self, arr, kind="ps"):
        if kind=="ps" and self.ps_clip is not None:
            lo, hi = self.ps_clip
            arr = np.clip(arr, lo, hi)
        if kind=="weight" and self.weight_clip is not None:
            arr = np.clip(arr, 0, self.weight_clip)
        return arr

    # --- cross-fit（支持 tqdm） ---
    def _crossfit_predictions_parallel(self, X, T, Y):
        n = len(Y)
        mu1_oof = np.zeros(n); mu0_oof = np.zeros(n); e_oof = np.zeros(n)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(X))

        if self.n_jobs in (None, 1):
            it = tqdm(enumerate(folds, 1), total=self.n_splits, disable=(self.verbose==0), desc="[CF]")
            for fold_id, (tr, va) in it:
                X_tr, X_va = X[tr], X[va]; T_tr, Y_tr = T[tr], Y[tr]
                va_idx, mu1_hat, mu0_hat, e_hat = _fit_fold_return_oof(
                    va, X_tr, X_va, T_tr, Y_tr, self.regressor, self.classifier, self.trim, self.random_state+fold_id
                )
                mu1_oof[va_idx] = mu1_hat; mu0_oof[va_idx] = mu0_hat; e_oof[va_idx] = e_hat
        else:
            tasks = []
            for fold_id, (tr, va) in enumerate(folds, 1):
                tasks.append(delayed(_fit_fold_return_oof)(
                    va, X[tr], X[va], T[tr], Y[tr],
                    self.regressor, self.classifier, self.trim, self.random_state+fold_id
                ))
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(tasks)
            for (va_idx, mu1_hat, mu0_hat, e_hat) in results:
                mu1_oof[va_idx] = mu1_hat; mu0_oof[va_idx] = mu0_hat; e_oof[va_idx] = e_hat

        return mu1_oof, mu0_oof, e_oof

    # --- 主估计 ---
    def _estimate_core(self, X, T, Y):
        X = np.asarray(X); T = np.asarray(T).astype(int); Y = np.asarray(Y).astype(float)
        n = len(Y)
        if self.verbose >= 1:
            print(f"[Main] Estimand={self.estimand}, n={n}, folds={self.n_splits}, n_jobs={self.n_jobs}")

        mu1, mu0, e = self._crossfit_predictions_parallel(X, T, Y)
        e = self._apply_clip(e, kind="ps")

        psi = ( T * (Y - mu1) / np.clip(e,1e-12,1-1e-12)
              - (1 - T) * (Y - mu0) / np.clip(1-e,1e-12,1-1e-12)
              + (mu1 - mu0) )

        if self.estimand == "ATE":
            est = float(np.mean(psi))
            IF  = psi - est
            se  = float(np.std(IF, ddof=1) / np.sqrt(n))
            w_t = self._apply_clip(T/np.clip(e,1e-12,1-1e-12), kind="weight")
            w_c = self._apply_clip((1-T)/np.clip(1-e,1e-12,1-1e-12), kind="weight")
            diag = dict(mu1=mu1, mu0=mu0, e=e, psi=psi, IF=IF, w_t=w_t, w_c=w_c)

        elif self.estimand == "ATT":
            p1 = T.mean()
            term = T*(Y - mu0) - (1 - T)*(e/(1-e))*(Y - mu0)
            est = float(np.mean(term) / (p1 + 1e-12))
            IF  = term/(p1+1e-12) - est*((T-p1)/(p1+1e-12))
            se  = float(np.std(IF, ddof=1) / np.sqrt(n))
            w_c_odds = self._apply_clip((1-T)*(e/(1-e)), kind="weight")
            diag = dict(mu0=mu0, mu1=mu1, e=e, psi=psi, IF=IF, w_c_odds=w_c_odds, p1=p1)

        else:  # ATO
            w = self._apply_clip(4.0 * e * (1 - e), kind="weight")
            est = float(np.sum(w * psi) / (np.sum(w) + 1e-12))
            mean_w = np.mean(w)
            IF  = (w/(mean_w+1e-12))*(psi - est)
            se  = float(np.std(IF, ddof=1) / np.sqrt(n))
            w1 = self._apply_clip(T*(1-e), kind="weight")
            w0 = self._apply_clip((1-T)*e, kind="weight")
            diag = dict(mu1=mu1, mu0=mu0, e=e, psi=psi, IF=IF, w=w, w1=w1, w0=w0)

        z = norm.ppf(0.975)
        ci = (est - z*se, est + z*se)
        return est, ci, diag

    # --- 诊断 ---
    def _diagnostics(self, X, T, diag, X_names=None):
        e = diag.get("e")
        ks  = _ks(e[T==1], e[T==0]) if (T==1).any() and (T==0).any() else np.nan
        ovl = _ovl(e[T==1], e[T==0]) if (T==1).any() and (T==0).any() else np.nan

        if self.estimand == "ATE":
            w_t, w_c = diag["w_t"], diag["w_c"]
            ess_t = _ess_ratio(w_t, T==1)
            ess_c = _ess_ratio(w_c, T==0)
            w_tail = dict(
                w1_p99=float(np.quantile(w_t[T==1], 0.99)) if (T==1).any() else np.nan,
                w0_p99=float(np.quantile(w_c[T==0], 0.99)) if (T==0).any() else np.nan
            )
            smd_fun = lambda x: _smd_ate(x, T, w_t, w_c)

        elif self.estimand == "ATT":
            w_c_odds = diag["w_c_odds"]
            ess_t = np.nan
            ess_c = _ess_ratio(w_c_odds, T==0)
            w_tail = dict(
                w_control_p99=float(np.quantile(w_c_odds[T==0], 0.99)) if (T==0).any() else np.nan,
                w_control_max=float(np.max(w_c_odds[T==0])) if (T==0).any() else np.nan
            )
            smd_fun = lambda x: _smd_att(x, T, w_c_odds)

        else:  # ATO
            w1, w0 = diag["w1"], diag["w0"]
            ess_t = _ess_ratio(w1, T==1)
            ess_c = _ess_ratio(w0, T==0)
            w_tail = dict(
                w1_p99=float(np.quantile(w1[T==1], 0.99)) if (T==1).any() else np.nan,
                w0_p99=float(np.quantile(w0[T==0], 0.99)) if (T==0).any() else np.nan
            )
            smd_fun = lambda x: _smd_ato(x, T, w1, w0)

        X = np.asarray(X)
        if X_names is None:
            X_names = self._colnames_ if self._colnames_ is not None else [f"x{j}" for j in range(X.shape[1])]
        smd = {}
        for j, name in enumerate(X_names):
            try:
                smd[name] = float(smd_fun(X[:, j]))
            except Exception:
                continue
        smd_max = float(np.nanmax(np.abs(list(smd.values()))) if smd else np.nan)

        return dict(ks=ks, ovl=ovl, ess_t=ess_t, ess_c=ess_c, w_tail=w_tail, smd=smd, smd_max=smd_max)

    # --- nAUUC（OOF，overlap 带内，纯 DR 实现） ---
    def _compute_nauuc_oof(self, X_raw, T, Y, e_full, ks=None):
        if not self.compute_nauuc:
            return None
        lo, hi = self.nauuc_band
        band = (e_full >= lo) & (e_full <= hi)
        if band.sum() < max(50, self.n_splits*3) or (T[band]==1).sum()==0 or (T[band]==0).sum()==0:
            return dict(enabled=True, note="band too small or mono treatment",
                        band=(float(lo),float(hi)), coverage=float(band.mean()), n=int(band.sum()))

        # 只在 band 内做 OOF
        Xb_raw = X_raw[band] if not isinstance(X_raw, pd.DataFrame) else X_raw.loc[band]
        Tb, Yb = T[band], Y[band]

        # 用与主流程一致的编码映射（不再改变 self._cat_levels_）
        Xb = self._encode_X(Xb_raw)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        psi_oof = np.full(len(Tb), np.nan)
        tau_oof = np.full(len(Tb), np.nan)

        for tr, va in kf.split(Xb):
            X_tr, X_va = Xb[tr], Xb[va]
            T_tr, T_va = Tb[tr], Tb[va]
            Y_tr, Y_va = Yb[tr], Yb[va]

            # 倾向
            ps = clone(self.classifier)
            ps.set_params(n_jobs=1)
            ps.fit(X_tr, T_tr)
            e_tr = np.clip(ps.predict_proba(X_tr)[:,1], self.trim, 1-self.trim)
            e_va = np.clip(ps.predict_proba(X_va)[:,1], self.trim, 1-self.trim)

            # m1/m0
            if (T_tr==1).sum()==0 or (T_tr==0).sum()==0:
                continue
            m1 = clone(self.regressor); m0 = clone(self.regressor)
            m1.set_params(n_jobs=1); m0.set_params(n_jobs=1)
            m1.fit(X_tr[T_tr==1], Y_tr[T_tr==1])
            m0.fit(X_tr[T_tr==0], Y_tr[T_tr==0])

            mu1_tr = m1.predict(X_tr); mu0_tr = m0.predict(X_tr)
            mu1_va = m1.predict(X_va); mu0_va = m0.predict(X_va)

            # DR 伪结果
            m_tr = e_tr * mu1_tr + (1 - e_tr) * mu0_tr
            Z_tr = ((T_tr - e_tr) / (e_tr * (1 - e_tr))) * (Y_tr - m_tr)
            w_tr = e_tr * (1 - e_tr)

            # τ 模型
            tau_model = clone(self.regressor); tau_model.set_params(n_jobs=1)
            tau_model.fit(X_tr, Z_tr, sample_weight=w_tr)
            tau_va = tau_model.predict(X_va)

            tau_oof[va] = tau_va
            psi_oof[va] = _aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)

        mask = ~np.isnan(psi_oof) & ~np.isnan(tau_oof)
        psi_oof, tau_oof = psi_oof[mask], tau_oof[mask]
        if len(psi_oof) < max(30, self.n_splits*2):
            return dict(enabled=True, note="too few valid oof points",
                        band=(float(lo),float(hi)), coverage=float(band.mean()), n=int(band.sum()))

        area_model  = _area_cumgain_centered(psi_oof, tau_oof)
        area_oracle = _area_cumgain_centered(psi_oof, psi_oof)
        nauuc = float(np.clip(area_model/area_oracle, 0.0, 1.0)) if abs(area_oracle) > 1e-12 else np.nan
        pol = _policy_values(psi_oof, tau_oof, ks=(ks or self.nauuc_policy_ks))

        return dict(enabled=True, band=(float(lo),float(hi)), coverage=float(band.mean()),
                    n=int(band.sum()), nauuc=nauuc,
                    area_model=area_model, area_oracle=area_oracle, **pol)

    # --- 拟合（placebo/负控 + tqdm） ---
    def fit(self, X, T, Y, X_names=None, y_nc=None, placebo_runs=300):
        # 按需编码
        if isinstance(X, pd.DataFrame):
            X_names = list(X.columns)
        X_enc_full = self._encode_X_fit(X)

        # 主估计 & 诊断
        est, ci, diag = self._estimate_core(X_enc_full, T, Y)
        diag2 = self._diagnostics(X_enc_full, T, diag, X_names=(X_names or self._colnames_))

        # nAUUC（OOF，overlap 带内，DR 实现）
        auuc = self._compute_nauuc_oof(X, np.asarray(T).astype(int), np.asarray(Y).astype(float), diag["e"])

        # Placebo
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(1e9, size=placebo_runs)
        if self.n_jobs_placebo in (None, 1):
            placebo_vals = []
            for s in tqdm(seeds, total=placebo_runs, disable=(self.verbose==0), desc="[Placebo]"):
                rng_local = np.random.default_rng(int(s))
                T_fake = rng_local.permutation(T)
                v, _, _ = self._estimate_core(X_enc_full, T_fake, Y)
                placebo_vals.append(v)
        else:
            def _one_placebo(seed):
                rng_local = np.random.default_rng(int(seed))
                T_fake = rng_local.permutation(T)
                v, _, _ = self._estimate_core(X_enc_full, T_fake, Y)
                return v
            placebo_vals = Parallel(n_jobs=self.n_jobs_placebo, backend="loky")(
                delayed(_one_placebo)(int(s)) for s in seeds
            )
        placebo_vals = np.asarray(placebo_vals)
        p_placebo = float((np.sum(np.abs(placebo_vals) >= np.abs(est)) + 1) / (placebo_runs + 1))

        # 负控
        p_nc = None; est_nc = None
        if y_nc is not None:
            y_nc = np.asarray(y_nc).astype(float)
            if self.n_jobs_placebo in (None, 1):
                vals = []
                for s in tqdm(seeds, total=placebo_runs, disable=(self.verbose==0), desc="[NegCtrl]"):
                    rng_local = np.random.default_rng(int(s))
                    T_fake = rng_local.permutation(T)
                    v, _, _ = self._estimate_core(X_enc_full, T_fake, y_nc)
                    vals.append(v)
            else:
                def _one_nc(seed):
                    rng_local = np.random.default_rng(int(seed))
                    T_fake = rng_local.permutation(T)
                    v, _, _ = self._estimate_core(X_enc_full, T_fake, y_nc)
                    return v
                vals = Parallel(n_jobs=self.n_jobs_placebo, backend="loky")(
                    delayed(_one_nc)(int(s)) for s in seeds
                )
            est_nc = float(self._estimate_core(X_enc_full, T, y_nc)[0])
            p_nc = float((np.sum(np.abs(np.asarray(vals)) >= np.abs(est_nc)) + 1) / (placebo_runs + 1))

        # 规则判定
        r = self.rules
        flags = dict(
            overlap_pass = ( (np.isnan(diag2['ovl']) or np.isnan(diag2['ks'])) and False )
                           or ( (diag2['ovl'] >= r.ovl_min) and (diag2['ks'] <= r.ks_max)
                                and ( (np.isnan(diag2['ess_t']) or diag2['ess_t']>=r.ess_min) )
                                and ( (np.isnan(diag2['ess_c']) or diag2['ess_c']>=r.ess_min) ) ),
            balance_pass = (diag2['smd_max'] <= r.smd_max) if not np.isnan(diag2['smd_max']) else False,
            placebo_pass = (p_placebo >= r.placebo_alpha),
            negctrl_pass = (p_nc is None) or (p_nc >= r.nc_alpha)
        )

        self.result_ = dict(
            estimand=self.estimand, est=est, ci=ci, diag=diag, **diag2,
            placebo_p=p_placebo, placebo_vals=placebo_vals.tolist(),
            negctrl_est=est_nc, negctrl_p=p_nc,
            auuc=auuc,
            flags=flags, rules=r.__dict__
        )
        return self

    # --- 报告 ---
    def report(self):
        assert self.result_ is not None, "请先调用 fit(...)"
        R = self.result_
        lines = []
        lines.append(f"【{R['estimand']} 可信度诊断报告（XGBoost，自动编码，DR nAUUC）】")
        lines.append(f"- 点估计 = {R['est']:.6f}, 95%CI = [{R['ci'][0]:.6f}, {R['ci'][1]:.6f}]")
        lines.append(
            f"- 重叠性: OVL={R['ovl']:.3f} (阈值≥{self.rules.ovl_min}) {'✅' if (R['ovl']>=self.rules.ovl_min) else '❌'}; "
            f"KS={R['ks']:.3f} (阈值≤{self.rules.ks_max}) {'✅' if (R['ks']<=self.rules.ks_max) else '❌'}; "
            f"权重尾部: {R['w_tail']}"
        )
        lines.append(
            f"- 平衡性: |SMD|_max={R['smd_max']:.3f} (阈值≤{self.rules.smd_max}) "
            f"{'✅' if (R['smd_max']<=self.rules.smd_max) else '❌'}；Top{self.rules.top_k_smd} 失衡特征："
        )
        if R['smd']:
            bad = sorted(((k, abs(v)) for k,v in R['smd'].items()), key=lambda x: -x[1])[:self.rules.top_k_smd]
            lines += [f"    · {k}: {R['smd'][k]:.3f}" for k,_ in bad]
        else:
            lines.append("    · （无可计算的数值型特征或全部缺失）")
        lines.append(
            f"- 安慰剂置换: p={R['placebo_p']:.3f} (应≥{self.rules.placebo_alpha}) "
            f"{'✅' if (R['placebo_p']>=self.rules.placebo_alpha) else '❌'}"
        )
        if R['negctrl_p'] is not None:
            lines.append(
                f"- 负控结局: 估计={R['negctrl_est']:.6f}, p={R['negctrl_p']:.3f} (应≥{self.rules.nc_alpha}) "
                f"{'✅' if (R['negctrl_p']>=self.rules.nc_alpha) else '❌'}"
            )

        if isinstance(R.get("auuc"), dict) and R["auuc"] is not None:
            A = R["auuc"]
            if A.get("enabled"):
                lines.append(
                    f"- DR τ̂ 排序能力（overlap e∈[{self.nauuc_band[0]:.2f},{self.nauuc_band[1]:.2f}]，OOF）"
                    f" 覆盖率={A.get('coverage', float('nan')):.2%}，n={A.get('n', 0)}"
                )
                if "nAUUC" in {k.lower() for k in A.keys()} or "nauuc" in A:
                    p10 = A.get('policy@10'); p20 = A.get('policy@20'); p30 = A.get('policy@30')
                    lines.append(
                        f"    · nAUUC={A['nauuc']:.3f} "
                        f"(area_model={A['area_model']:.2f}, area_oracle={A['area_oracle']:.2f})"
                    )
                    if (p10 is not None) and (p20 is not None) and (p30 is not None):
                        lines.append(
                            f"    · Policy@10/20/30 = {p10:.2f} / {p20:.2f} / {p30:.2f}"
                        )
                else:
                    lines.append(f"    · nAUUC 未计算：{A.get('note','')}")
            else:
                lines.append("- nAUUC 未启用（compute_nauuc=False）")

        passed = sum(bool(v) for v in R['flags'].values())
        verdict = "可信（通过）" if passed==4 else ("存疑（需谨慎解释/加强稳健性）" if passed>=2 else "不通过（建议改口径/方法）")
        lines.append(f"- 各项通过: {R['flags']}  => 结论：{verdict}")
        return "\n".join(lines)
