# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone

import xgboost as xgb
# XGB>=2.0 建议: tree_method="hist", device="cuda"|"cpu"

# ====================== 小工具 ======================

def _aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
    e = np.clip(e, trim, 1 - trim)
    return (t * (y - mu1) / e) - ((1 - t) * (y - mu0) / (1 - e)) + (mu1 - mu0)

def _area_cumgain_centered(psi, scores):
    order = np.argsort(-scores)
    psi_ord = psi[order]
    psi_c = psi_ord - np.mean(psi_ord)
    csum = np.cumsum(psi_c)
    x = np.arange(1, len(csum) + 1) / len(csum)
    return float(np.trapz(csum, x))

def _policy_values(psi, scores, ks=(0.1, 0.2, 0.3)):
    out = {}
    order = np.argsort(-scores)
    psi_ord = psi[order]
    n = len(psi_ord)
    for k in ks:
        m = max(1, int(np.floor(n * k)))
        out[f"policy@{int(k*100)}"] = float(np.sum(psi_ord[:m]))
    return out

# ================== 诊断常量（可扩展） ==================

@dataclass
class NAUUCConfig:
    enabled: bool = True
    band: Tuple[float, float] = (0.3, 0.7)   # 只在 overlap 带内评估排序
    policy_ks: Tuple[float, float, float] = (0.1, 0.2, 0.3)
    min_points: int = 120    # 计算 nAUUC 至少需要的样本数（带内 & 有效）

# ====================== 主类 ======================

class HTETester:
    """
    作用：
      1) 从零训练：基于 DR-learner 流程（e(x), m1(x), m0(x) → 伪结果 Z & 权重 w → tau(x)）
      2) 免训练复用：直接接收外部已拟合好的 ps/m1/m0/tau（例如来自 NAUUCCatBoostTunerV2）
      3) 自动识别类别/数值列，稳定 one-hot（稀有并桶 + Top-K）
      4) 在 overlap 带内评估 nAUUC 与 Policy@k（以 AIPW ψ 为“oracle”）

    重要接口：
      - fit(X, T, Y): 从零训练并生成报告指标（含 nAUUC）
      - fit_prefit(...): 免训练复用外部已拟合模型并生成报告指标
      - effect(X): 产出 taû(X)
      - propensity(X): 产出 ê(X)
      - mu_hat(X): 返回 (m1_hat(X), m0_hat(X))
      - report(): 返回评估文本
      - adopt_encoder(cat_levels, colnames): 承接外部编码映射/列顺序（强烈建议连同 NAUUC 调参器一起使用）
    """

    # ------------------- 初始化 ------------------- #
    def __init__(
        self,
        # XGB device &默认模型
        use_gpu: bool = False,
        random_seed: int = 42,

        # 自动编码配置
        auto_encode_cats: bool = True,
        int_as_cat_unique_thresh: int = 30,
        unique_ratio_thresh: float = 0.05,
        rare_freq_ratio: float = 0.001,
        max_onehot_levels: int = 200,

        # 训练/评估配置
        trim: float = 1e-3,                 # e(x) 裁剪
        n_splits: int = 5,                  # 仅用于 OOF/交叉评估
        nauuc_cfg: NAUUCConfig = NAUUCConfig(),

        # 若未提供外部模型，内部默认的 base 模型超参（可在创建实例时覆盖）
        reg_params: Optional[Dict[str, Any]] = None,
        clf_params: Optional[Dict[str, Any]] = None,
        tau_params: Optional[Dict[str, Any]] = None,
    ):
        self.use_gpu = bool(use_gpu)
        self.random_seed = int(random_seed)

        # 编码参数
        self.auto_encode_cats = bool(auto_encode_cats)
        self.int_as_cat_unique_thresh = int(int_as_cat_unique_thresh)
        self.unique_ratio_thresh = float(unique_ratio_thresh)
        self.rare_freq_ratio = float(rare_freq_ratio)
        self.max_onehot_levels = int(max_onehot_levels)

        # 评估
        self.trim = float(trim)
        self.n_splits = int(n_splits)
        self.nauuc_cfg = nauuc_cfg

        # XGB 通用 device 参数
        self._device_args = {"tree_method": "hist", "device": ("cuda" if self.use_gpu else "cpu")}

        # 内部默认 base learners（用户可传入自定义超参）
        self._reg_params_default = dict(
            objective="reg:squarederror",
            max_depth=6, learning_rate=0.05, n_estimators=600,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=5, random_state=self.random_seed, n_jobs=-1, **self._device_args
        )
        self._clf_params_default = dict(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=6, learning_rate=0.05, n_estimators=600,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=5, random_state=self.random_seed, n_jobs=-1, **self._device_args
        )
        self._tau_params_default = dict(
            objective="reg:squarederror",
            max_depth=6, learning_rate=0.05, n_estimators=800,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=3, random_state=self.random_seed, n_jobs=-1, **self._device_args
        )

        self.reg_params = (reg_params or self._reg_params_default).copy()
        self.clf_params = (clf_params or self._clf_params_default).copy()
        self.tau_params = (tau_params or self._tau_params_default).copy()

        # 状态：编码器 & 模型
        self._cat_levels_: Dict[str, List[Any]] = {}
        self._colnames_: Optional[List[str]] = None

        self.ps_model_: Optional[xgb.XGBClassifier] = None
        self.m1_model_: Optional[xgb.XGBRegressor] = None
        self.m0_model_: Optional[xgb.XGBRegressor] = None
        self.tau_model_: Optional[xgb.XGBRegressor] = None

        # 报告缓存
        self._cache: Dict[str, Any] = {}
        self._fitted: bool = False

    # ------------------- 编码相关 ------------------- #
    def adopt_encoder(self, cat_levels: Dict[str, Iterable[str]], colnames: Iterable[str]):
        """承接外部（如 NAUUCCatBoostTunerV2）已学好的类别映射与列顺序。"""
        self._cat_levels_ = {k: list(v) for k, v in cat_levels.items()}
        self._colnames_ = list(colnames)

    def _infer_cats(self, X: pd.DataFrame) -> List[str]:
        from pandas.api.types import (
            is_numeric_dtype, is_integer_dtype,
            is_string_dtype, is_categorical_dtype, is_datetime64_any_dtype
        )
        cats = []
        n = len(X)
        for c in X.columns:
            s = X[c]
            if is_datetime64_any_dtype(s):  # 跳过时间列
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
            levels = set(self._cat_levels_[c][1:])  # 去掉 __RARE__
            X2[c] = X2[c].apply(lambda v: v if v in levels else "__RARE__")

        # one-hot（固定顺序）
        dummies = []
        for c in cat_cols:
            cats = pd.Categorical(X2[c], categories=self._cat_levels_[c])
            dummies.append(pd.get_dummies(cats, prefix=c, dummy_na=False))
        D = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=X2.index)

        # 数值列
        from pandas.api.types import is_numeric_dtype
        num_cols = [c for c in X.columns if c not in cat_cols and is_numeric_dtype(X[c])]
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce")

        out = pd.concat([X_num, D], axis=1).fillna(0.0)
        if self._colnames_ is None:
            self._colnames_ = list(out.columns)
        else:
            # 对齐外部列顺序（漏列补零，多列保留）
            for col in self._colnames_:
                if col not in out:
                    out[col] = 0.0
            out = out[self._colnames_]
        return out

    def _encode_X_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.auto_encode_cats:
            out = X.copy()
            if self._colnames_ is None:
                self._colnames_ = list(out.columns)
            return out
        cat_cols = self._infer_cats(X)
        self._fit_categorical_maps(X, cat_cols)
        return self._apply_categorical_maps(X, cat_cols)

    def _encode_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.auto_encode_cats:
            out = X.copy()
            # 对齐列
            if self._colnames_ is not None:
                for col in self._colnames_:
                    if col not in out:
                        out[col] = 0.0
                out = out[self._colnames_]
            return out
        cat_cols = list(self._cat_levels_.keys())
        return self._apply_categorical_maps(X, cat_cols)

    # ------------------- 训练 DR 头 ------------------- #
    def _dr_train(self, X_enc: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """从零训练 ps/m1/m0/tau，并返回训练后的模型与打分。"""
        # 1) ps
        ps = xgb.XGBClassifier(**self.clf_params)
        ps.fit(X_enc, T, verbose=False)
        e_hat = np.clip(ps.predict_proba(X_enc)[:, 1], self.trim, 1 - self.trim)

        # 2) m1/m0
        idx1, idx0 = (T == 1), (T == 0)
        if idx1.sum() == 0 or idx0.sum() == 0:
            raise RuntimeError("训练数据中 treatment 或 control 为空，无法训练 m1/m0。")
        m1 = xgb.XGBRegressor(**self.reg_params)
        m0 = xgb.XGBRegressor(**self.reg_params)
        m1.fit(X_enc[idx1], Y[idx1], verbose=False)
        m0.fit(X_enc[idx0], Y[idx0], verbose=False)
        mu1 = m1.predict(X_enc)
        mu0 = m0.predict(X_enc)

        # 3) DR 伪结果 + 权重
        m_all = e_hat * mu1 + (1 - e_hat) * mu0
        Z = ((T - e_hat) / (e_hat * (1 - e_hat))) * (Y - m_all)
        w = e_hat * (1 - e_hat)

        # 4) tau 头
        tau = xgb.XGBRegressor(**self.tau_params)
        tau.fit(X_enc, Z, sample_weight=w, verbose=False)

        return ps, m1, m0, tau, e_hat, mu1, mu0

    # ------------------- 公开：从零训练 ------------------- #
    def fit(self, X: pd.DataFrame, T: Iterable[int], Y: Iterable[float]):
        """
        训练 DR-learner 头，并生成报告（含 nAUUC）。
        """
        X_df = pd.DataFrame(X).copy()
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y, dtype=float)

        # 编码
        X_enc = self._encode_X_fit(X_df).to_numpy(dtype=float, copy=False)

        # 训练 ps/m1/m0/tau
        ps, m1, m0, tau, e_hat, mu1, mu0 = self._dr_train(X_enc, T, Y)

        # 保存
        self.ps_model_, self.m1_model_, self.m0_model_, self.tau_model_ = ps, m1, m0, tau

        # nAUUC 评估（带内）
        self._eval_and_cache(X_enc, T, Y, e_hat=e_hat, mu1=mu1, mu0=mu0)

        self._fitted = True
        return self

    # ------------------- 公开：免训练复用 ------------------- #
    def fit_prefit(
        self,
        X: pd.DataFrame,
        T: Iterable[int],
        Y: Iterable[float],
        prefit_ps: xgb.XGBClassifier,
        prefit_m1: xgb.XGBRegressor,
        prefit_m0: xgb.XGBRegressor,
        prefit_tau: Optional[xgb.XGBRegressor] = None,
        band_mask: Optional[np.ndarray] = None,
    ):
        """
        直接复用外部**已拟合**好的模型，进行打分与报告（不再训练）。
        如需与外部编码严格一致，请先调用 adopt_encoder(...).
        """
        X_df = pd.DataFrame(X).copy()
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y, dtype=float)

        # 编码（若 adopt_encoder 已注入映射，则与外部严格对齐）
        X_enc = (self._encode_X(X_df) if self._cat_levels_ else self._encode_X_fit(X_df)).to_numpy(float, copy=False)

        # 直接打分
        e_hat = np.clip(prefit_ps.predict_proba(X_enc)[:, 1], self.trim, 1 - self.trim)
        mu1 = prefit_m1.predict(X_enc)
        mu0 = prefit_m0.predict(X_enc)

        # 保存引用
        self.ps_model_, self.m1_model_, self.m0_model_ = prefit_ps, prefit_m1, prefit_m0
        self.tau_model_ = prefit_tau  # 可能为 None

        # nAUUC 评估（带内；tau 使用 prefit_tau 或回退 m1-m0）
        self._eval_and_cache(X_enc, T, Y, e_hat=e_hat, mu1=mu1, mu0=mu0, band_mask=band_mask)

        self._fitted = True
        return self

    # ------------------- 内部：nAUUC 评估缓存 ------------------- #
    def _eval_and_cache(
        self,
        X_enc: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        e_hat: np.ndarray,
        mu1: np.ndarray,
        mu0: np.ndarray,
        band_mask: Optional[np.ndarray] = None,
    ):
        if not self.nauuc_cfg.enabled:
            self._cache = dict(enabled=False)
            return

        lo, hi = self.nauuc_cfg.band
        if band_mask is None:
            band = (e_hat >= lo) & (e_hat <= hi)
        else:
            band = np.asarray(band_mask, dtype=bool)

        coverage = float(np.mean(band))
        n_band = int(band.sum())

        if n_band < self.nauuc_cfg.min_points or (T[band] == 1).sum() == 0 or (T[band] == 0).sum() == 0:
            self._cache = dict(
                enabled=True, coverage=coverage, n=n_band, band=(float(lo), float(hi)),
                note="band too small or mono treatment", nauuc=np.nan
            )
            return

        psi = _aipw_pseudo(Y[band], T[band], mu1[band], mu0[band], e_hat[band], trim=self.trim)

        # tau 排序分：优先 tau_model；否则回退 m1-m0
        if self.tau_model_ is not None:
            tau_scores = self.tau_model_.predict(X_enc[band])
        else:
            tau_scores = (mu1 - mu0)[band]

        area_model = _area_cumgain_centered(psi, tau_scores)
        area_oracle = _area_cumgain_centered(psi, psi)
        nauuc = float(np.clip(area_model / area_oracle, 0.0, 1.0)) if abs(area_oracle) > 1e-12 else np.nan
        pol = _policy_values(psi, tau_scores, ks=self.nauuc_cfg.policy_ks)

        self._cache = dict(
            enabled=True, coverage=coverage, n=n_band, band=(float(lo), float(hi)),
            nauuc=nauuc, area_model=area_model, area_oracle=area_oracle, **pol
        )

    # ------------------- 推断接口 ------------------- #
    def _ensure_ready(self):
        if not self._fitted:
            raise RuntimeError("请先调用 fit(...) 或 fit_prefit(...) 完成初始化。")

    def propensity(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_ready()
        if self.ps_model_ is None:
            raise RuntimeError("ps_model_ 不存在。")
        X_enc = self._encode_X(pd.DataFrame(X)).to_numpy(float, copy=False)
        return np.clip(self.ps_model_.predict_proba(X_enc)[:, 1], self.trim, 1 - self.trim)

    def mu_hat(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_ready()
        if (self.m1_model_ is None) or (self.m0_model_ is None):
            raise RuntimeError("m1/m0 模型不存在。")
        X_enc = self._encode_X(pd.DataFrame(X)).to_numpy(float, copy=False)
        return self.m1_model_.predict(X_enc), self.m0_model_.predict(X_enc)

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_ready()
        X_enc = self._encode_X(pd.DataFrame(X)).to_numpy(float, copy=False)
        if self.tau_model_ is not None:
            return self.tau_model_.predict(X_enc)
        # 回退：T-learner 差分
        return self.m1_model_.predict(X_enc) - self.m0_model_.predict(X_enc)

    # ------------------- 报告 ------------------- #
    def report(self) -> str:
        self._ensure_ready()
        C = self._cache
        if not C.get("enabled", True):
            return "【HTE 可用性报告】\n- nAUUC 评估未启用。"
        lo, hi = C.get("band", self.nauuc_cfg.band)
        pol_keys = [k for k in C.keys() if k.startswith("policy@")]
        pol_str = " / ".join([f"{k.split('@')[1]}%={C.get(k, float('nan')):.2f}" for k in sorted(pol_keys)])
        lines = [
            "【HTE 可用性报告（DR 头）】",
            f"- 评估带 e∈[{lo:.2f},{hi:.2f}] 覆盖率={C.get('coverage', float('nan')):.2%}, n={C.get('n', 0)}",
            f"- nAUUC={C.get('nauuc', float('nan')):.3f} "
            f"(area_model={C.get('area_model', float('nan')):.2f}, area_oracle={C.get('area_oracle', float('nan')):.2f})",
            f"- Policy@k：{pol_str}" if pol_keys else "- Policy@k：N/A"
        ]
        if "note" in C:
            lines.append(f"- NOTE: {C['note']}")
        return "\n".join(lines)
