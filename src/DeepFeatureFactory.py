# -*- coding: utf-8 -*-
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Sequence, Optional, Dict, Any, Union, Tuple, List

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import WoEEncoder
import category_encoders as ce

try:
    from autofeat import AutoFeatRegressor
    _HAS_AUTOF = True
except Exception:
    _HAS_AUTOF = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool   # ← 加了 Pool
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


class DeepFeatureFactory:
    """
    一站式深特征工厂（无 Featuretools 版本）
    -------------------------------------------------
    - 数值扩展：AutoFeat（可选）、PolynomialFeatures(二阶)
    - 类别增强：K 折 OOF 目标编码、Count/Freq、可选 WoE
    - 统计组合：GroupBy 聚合回填、log1p、比率、乘除交互、分位分箱
    - 深层表征：CatBoost 叶子索引（leaf embeddings，可选）
    - 降维：PCA（可选）
    - 列数约束：当超出 max_total_cols 时，用 DSBAN 主成分裁剪为精确列数
    """

    def __init__(
        self,
        # 基础列
        cat_cols: Sequence[str],
        num_cols: Sequence[str],
        group_keys: Sequence[Sequence[str]] = (),
        # OOF target encoding
        enable_target_encoding: bool = True,
        te_smoothing: float = 20.0,
        n_splits: int = 5,
        random_state: int = 42,
        # 频数编码 & WoE
        enable_count_freq: bool = True,
        enable_woe: bool = False,   # 二分类时可用
        # GroupBy 统计
        enable_groupby_agg: bool = True,
        agg_funcs: Sequence[str] = ("mean", "median", "std", "count", "nunique"),
        # 数值派生
        add_log1p: bool = True,
        add_ratios: bool = True,
        add_interactions: bool = True,
        max_interactions: int = 40,
        quantile_bins: int = 10,     # 0 关闭
        # 多项式/AutoFeat
        add_poly2: bool = True,
        autofeat_steps: int = 0,     # >0 开启 AutoFeat（开销较大，建议 1~2）
        # 叶子嵌入 & PCA
        add_leaf_embeddings: bool = True,
        leaf_task: Optional[str] = None,  # "reg"|"clf"|None 自动
        leaf_rounds: int = 500,
        leaf_depth: int = 6,
        leaf_lr: float = 0.05,
        add_pca: int = 0,            # >0 输出 PC 数量
        scale_before_pca: bool = True,
        # 产出控制
        max_total_cols: Optional[int] = None,
        # DSBAN 主成分设置（当触发 max_total_cols 限制时）
        dsban_alpha: float = 0.7,        # 监督权重：相关性(α) vs. 方差(1-α)
        dsban_anchor_frac: float = 0.2,  # 锚特征占比
        dsban_min_anchors: int = 20,     # 最少锚数量
        dsban_whiten: bool = False,      # 是否 whiten（仍会先按 scale_before_pca 标准化）
        # 日志
        verbose: Union[int, bool] = 1
    ):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.group_keys = [list(g) for g in group_keys]

        self.enable_target_encoding = enable_target_encoding
        self.te_smoothing = float(te_smoothing)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

        self.enable_count_freq = enable_count_freq
        self.enable_woe = enable_woe

        self.enable_groupby_agg = enable_groupby_agg
        self.agg_funcs = tuple(agg_funcs)

        self.add_log1p = add_log1p
        self.add_ratios = add_ratios
        self.add_interactions = add_interactions
        self.max_interactions = int(max_interactions)
        self.quantile_bins = int(quantile_bins)

        self.add_poly2 = add_poly2
        self.autofeat_steps = int(autofeat_steps)

        self.add_leaf_embeddings = add_leaf_embeddings
        self.leaf_task = leaf_task
        self.leaf_rounds = int(leaf_rounds)
        self.leaf_depth = int(leaf_depth)
        self.leaf_lr = float(leaf_lr)

        self.add_pca = int(add_pca)
        self.scale_before_pca = bool(scale_before_pca)

        self.max_total_cols = max_total_cols

        # DSBAN
        self.dsban_alpha = float(np.clip(dsban_alpha, 0.0, 1.0))
        self.dsban_anchor_frac = float(np.clip(dsban_anchor_frac, 0.05, 1.0))
        self.dsban_min_anchors = int(max(1, dsban_min_anchors))
        self.dsban_whiten = bool(dsban_whiten)

        # 状态
        self._is_binary_target: Optional[bool] = None
        self._global_mean: Optional[float] = None
        self._te_maps: Dict[str, Dict[Any, float]] = {}
        self._qbin_models: Dict[str, EqualFrequencyDiscretiser] = {}
        self._woe_enc: Optional[WoEEncoder] = None
        self._leaf_model: Optional[Union[CatBoostRegressor, CatBoostClassifier]] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._autofeat_model: Optional[AutoFeatRegressor] = None
        self._poly: Optional[PolynomialFeatures] = None
        self._fitted = False

        # DSBAN 状态
        self._dsban_anchor_idx: Optional[np.ndarray] = None
        self._dsban_scaler: Optional[StandardScaler] = None
        self._dsban_pca: Optional[PCA] = None

        # 日志
        self.verbose = int(verbose)

    # ---------------- logging utils ---------------- #
    def _log(self, msg: str, level: int = 1):
        if self.verbose >= level:
            print(msg, flush=True)

    @staticmethod
    def _elapsed(t0: float) -> str:
        return f"{time.time() - t0:.2f}s"

    @staticmethod
    def _delta_cols(before: int, after: int) -> str:
        return f"+{after - before}" if after >= before else f"{after - before}"

    # ---------------- utils ---------------- #
    @staticmethod
    def _detect_binary(y: np.ndarray) -> bool:
        uy = np.unique(y[~pd.isna(y)])
        return len(uy) <= 2 and set(uy).issubset({0, 1})

    @staticmethod
    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)

    @staticmethod
    def _to_category(df: pd.DataFrame, cols: Sequence[str]) -> None:
        for c in cols:
            if c in df:
                df[c] = df[c].astype("string").fillna("__NA__")

    # ---------------- encodings ---------------- #
    def _oof_target_encode(self, X_cat: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        if not self.enable_target_encoding or not self.cat_cols:
            return pd.DataFrame(index=X_cat.index)

        t0 = time.time()
        self._log(f"[TE] Start OOF target encoding on {len(self.cat_cols)} cat cols …", 1)

        oof = pd.DataFrame(index=X_cat.index)
        self._is_binary_target = self._detect_binary(y)
        gm = float(np.nanmean(y)) if not self._is_binary_target else float(np.nanmean(y == 1))
        self._global_mean = gm

        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        ).split(X_cat, (y if self._is_binary_target else (y > np.nanmedian(y)).astype(int)))

        te_maps_fold: Dict[str, list] = {c: [] for c in self.cat_cols}

        fold_id = 0
        for tr, va in splitter:
            fold_id += 1
            self._log(f"[TE]  Fold {fold_id}/{self.n_splits}", 2)
            Xtr, Xva = X_cat.iloc[tr], X_cat.iloc[va]
            ytr = y[tr]
            for c in self.cat_cols:
                stats = (
                    pd.DataFrame({"y": ytr}, index=Xtr.index)
                    .join(Xtr[[c]])
                    .groupby(c)["y"]
                    .agg(["mean", "count"])
                )
                stats["te"] = (stats["mean"] * stats["count"] + gm * self.te_smoothing) / (stats["count"] + self.te_smoothing)
                m = stats["te"]
                te_maps_fold[c].append(m)
                oof.loc[Xva.index, f"{c}__te"] = Xva[c].map(m).fillna(gm).values

        for c in self.cat_cols:
            if te_maps_fold[c]:
                cat_values = pd.concat(te_maps_fold[c], axis=1)
                self._te_maps[c] = cat_values.mean(axis=1).to_dict()
            else:
                self._te_maps[c] = {}

        self._log(f"[TE] Done in {self._elapsed(t0)} | out_cols={oof.shape[1]}", 1)
        return oof.fillna(gm)

    def _count_freq(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enable_count_freq or not self.cat_cols:
            return pd.DataFrame(index=X.index)
        t0 = time.time()
        self._log(f"[CNT] Start count/freq on {len(self.cat_cols)} cat cols …", 1)
        out = pd.DataFrame(index=X.index)
        N = max(len(X), 1)
        for c in self.cat_cols:
            vc = X[c].value_counts(dropna=False)
            out[f"{c}__count"] = X[c].map(vc).fillna(0).astype(float)
            out[f"{c}__freq"] = out[f"{c}__count"] / N
        self._log(f"[CNT] Done in {self._elapsed(t0)} | out_cols={out.shape[1]}", 1)
        return out

    def _fit_woe(self, X_cat: pd.DataFrame, y: np.ndarray):
        t0 = time.time()
        if not self.enable_woe or not self.cat_cols or not self._detect_binary(y):
            self._woe_enc = None
            self._log(f"[WoE] Skip (enabled={self.enable_woe}, cats={len(self.cat_cols)}, binary={self._detect_binary(y)})", 1)
            return pd.DataFrame(index=X_cat.index)
        self._log(f"[WoE] Start WoE fit …", 1)
        enc = WoEEncoder(variables=self.cat_cols)
        enc.fit(X_cat, y)
        self._woe_enc = enc
        tr = enc.transform(X_cat)
        self._log(f"[WoE] Done in {self._elapsed(t0)} | out_cols={tr.shape[1]}", 1)
        return tr

    def _woe_transform(self, X_cat: pd.DataFrame):
        if self._woe_enc is None:
            return pd.DataFrame(index=X_cat.index)
        return self._woe_enc.transform(X_cat)

    # ---------------- groupby & numeric ---------------- #
    def _groupby_agg(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enable_groupby_agg or not self.group_keys or not self.num_cols:
            return pd.DataFrame(index=X.index)
        t0 = time.time()
        self._log(f"[GRP] Start groupby agg on {len(self.group_keys)} key sets × {len(self.num_cols)} nums …", 1)
        out = pd.DataFrame(index=X.index)
        for keys in self.group_keys:
            keyname = "__".join(keys)
            g = X.groupby(keys, dropna=False)
            for num in self.num_cols:
                if num not in X: continue
                for f in self.agg_funcs:
                    col = f"grp_{keyname}__{num}__{f}"
                    out[col] = X[keys].merge(
                        g[num].agg(f).rename("val"),
                        left_on=keys, right_index=True, how="left"
                    )["val"].values
        self._log(f"[GRP] Done in {self._elapsed(t0)} | out_cols={out.shape[1]}", 1)
        return out

    def _numeric_derivatives(self, X: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        self._log(f"[NUM] Start numeric derivatives …", 1)
        out = pd.DataFrame(index=X.index)
        nums = [c for c in self.num_cols if c in X]

        before = out.shape[1]
        if self.add_log1p:
            for c in nums:
                out[f"{c}__log1p"] = np.log1p(pd.to_numeric(X[c], errors="coerce").clip(lower=0))
        self._log(f"[NUM]  log1p {self._delta_cols(before, out.shape[1])}", 2); before = out.shape[1]

        if self.add_ratios and len(nums) >= 2:
            denom_like = [c for c in nums if any(s in c.lower() for s in ["cnt", "count", "num", "hour", "day", "n_"])]
            numer_like = [c for c in nums if c not in denom_like]
            made = 0
            for a in numer_like:
                for b in denom_like:
                    if a == b: continue
                    out[f"ratio__{a}__per__{b}"] = self._safe_div(pd.to_numeric(X[a], errors="coerce"),
                                                                  pd.to_numeric(X[b], errors="coerce"))
                    made += 1
                    if made >= 60: break
        self._log(f"[NUM]  ratios {self._delta_cols(before, out.shape[1])}", 2); before = out.shape[1]

        if self.add_interactions and len(nums) >= 2:
            rng = np.random.RandomState(self.random_state)
            pairs = []
            base = nums[:min(len(nums), 14)]
            for i in range(len(base)):
                for j in range(i+1, len(base)):
                    pairs.append((base[i], base[j]))
            rng.shuffle(pairs)
            for k, (a, b) in enumerate(pairs[: self.max_interactions]):
                A = pd.to_numeric(X[a], errors="coerce")
                B = pd.to_numeric(X[b], errors="coerce")
                out[f"inter__{a}__x__{b}"] = A * B
                out[f"inter__{a}__div__{b}"] = self._safe_div(A, B)
        self._log(f"[NUM]  interactions {self._delta_cols(before, out.shape[1])}", 2); before = out.shape[1]

        if self.quantile_bins and self.quantile_bins > 1:
            for c in nums:
                try:
                    disc = EqualFrequencyDiscretiser(q=self.quantile_bins, variables=[c], return_object=True)
                    disc.fit(X[[c]])
                    qb = disc.transform(X[[c]])
                    dummies = pd.get_dummies(qb[c], prefix=f"{c}__qbin", dummy_na=True)
                    keep = [col for col in dummies.columns][: min(6, dummies.shape[1])]
                    out = pd.concat([out, dummies[keep]], axis=1)
                    self._qbin_models[c] = disc
                except Exception:
                    pass
        self._log(f"[NUM]  qbins {self._delta_cols(before, out.shape[1])}", 2)

        self._log(f"[NUM] Done in {self._elapsed(t0)} | out_cols={out.shape[1]}", 1)
        return out

    # ---------------- poly & autofeat ---------------- #
    def _fit_poly2(self, feat: pd.DataFrame) -> pd.DataFrame:
        if not self.add_poly2:
            return pd.DataFrame(index=feat.index)
        t0 = time.time()
        self._log("[POLY] Fit PolynomialFeatures(degree=2)…", 1)
        Xn = feat.select_dtypes(include=[np.number]).fillna(0.0).values
        self._poly = PolynomialFeatures(degree=2, include_bias=False)
        Z = self._poly.fit_transform(Xn)
        cols = [f"poly2_{i}" for i in range(Z.shape[1])]
        out = pd.DataFrame(Z, index=feat.index, columns=cols)
        self._log(f"[POLY] Done in {self._elapsed(t0)} | out_cols={out.shape[1]}", 1)
        return out

    def _poly2_transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        if self._poly is None:
            return pd.DataFrame(index=feat.index)
        Xn = feat.select_dtypes(include=[np.number]).fillna(0.0).values
        Z = self._poly.transform(Xn)
        cols = [f"poly2_{i}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, index=feat.index, columns=cols)

    def _fit_autofeat(self, feat: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        if self.autofeat_steps <= 0 or not _HAS_AUTOF:
            return pd.DataFrame(index=feat.index)
        t0 = time.time()
        self._log(f"[AF] Fit AutoFeatRegressor(steps={self.autofeat_steps}) …", 1)
        afr = AutoFeatRegressor(feateng_steps=self.autofeat_steps, verbose=0)
        Z = afr.fit_transform(feat.select_dtypes(include=[np.number]).fillna(0.0), y)
        self._autofeat_model = afr
        Z = pd.DataFrame(Z, index=feat.index)
        Z.columns = [f"autofeat_{i}" for i in range(Z.shape[1])]
        self._log(f"[AF] Done in {self._elapsed(t0)} | out_cols={Z.shape[1]}", 1)
        return Z

    def _autofeat_transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        if self._autofeat_model is None:
            return pd.DataFrame(index=feat.index)
        Z = self._autofeat_model.transform(feat.select_dtypes(include=[np.number]).fillna(0.0))
        Z = pd.DataFrame(Z, index=feat.index)
        Z.columns = [f"autofeat_{i}" for i in range(Z.shape[1])]
        return Z

    # ---------------- leaf embeddings & PCA ---------------- #
    def _fit_leaf_model(self, feat: pd.DataFrame, y: np.ndarray):
        if not self.add_leaf_embeddings or not _HAS_CATBOOST:
            return None
        t0 = time.time()
        task = self.leaf_task or ("clf" if self._detect_binary(y) else "reg")
        self._log(f"[LEAF] Fit CatBoost ({task}) rounds={self.leaf_rounds}, depth={self.leaf_depth} …", 1)
        Xcb = feat.select_dtypes(include=[np.number]).fillna(0.0).values
        params = dict(iterations=self.leaf_rounds, depth=self.leaf_depth,
                      learning_rate=self.leaf_lr, random_seed=self.random_state, verbose=False)
        if task == "clf":
            model = CatBoostClassifier(loss_function="Logloss", **params)
        else:
            model = CatBoostRegressor(loss_function="RMSE", **params)
        model.fit(Xcb, y)
        self._leaf_model = model
        try:
            # 优先：calc_leaf_indexes（更通用稳定）
            pool = Pool(Xcb)
            leaf_idx = model.calc_leaf_indexes(pool)
            leaf_idx = np.array(leaf_idx)
            if leaf_idx.ndim == 1: leaf_idx = leaf_idx.reshape(-1, 1)
            self._log(f"[LEAF] Done in {self._elapsed(t0)} | trees={leaf_idx.shape[1] if leaf_idx.ndim==2 else 1}", 1)
            return leaf_idx
        except Exception as e1:
            # 备选回退
            try:
                leaf_idx = model.predict(Xcb, prediction_type="LeafIndex")
                leaf_idx = np.array(leaf_idx)
                if leaf_idx.ndim == 1: leaf_idx = leaf_idx.reshape(-1, 1)
                self._log(f"[LEAF] Done (fallback) in {self._elapsed(t0)} | trees={leaf_idx.shape[1] if leaf_idx.ndim==2 else 1}", 1)
                return leaf_idx
            except Exception as e2:
                self._log(f"[LEAF] WARN: cannot extract leaf indexes: {e1} | {e2}", 1)
                return None

    def _leaf_features(self, feat: pd.DataFrame):
        if self._leaf_model is None or not _HAS_CATBOOST:
            return None
        Xcb = feat.select_dtypes(include=[np.number]).fillna(0.0).values
        try:
            pool = Pool(Xcb)
            leaf_idx = self._leaf_model.calc_leaf_indexes(pool)
            leaf_idx = np.array(leaf_idx)
            if leaf_idx.ndim == 1: leaf_idx = leaf_idx.reshape(-1, 1)
            return leaf_idx
        except Exception:
            try:
                leaf_idx = self._leaf_model.predict(Xcb, prediction_type="LeafIndex")
                leaf_idx = np.array(leaf_idx)
                if leaf_idx.ndim == 1: leaf_idx = leaf_idx.reshape(-1, 1)
                return leaf_idx
            except Exception:
                return None

    def _fit_pca(self, feat_num: np.ndarray):
        if self.add_pca <= 0:
            return None
        t0 = time.time()
        self._log(f"[PCA] Fit PCA(n_components={self.add_pca}) …", 1)
        Z = feat_num
        if self.scale_before_pca:
            self._scaler = StandardScaler()
            Z = self._scaler.fit_transform(Z)
        self._pca = PCA(n_components=self.add_pca, random_state=self.random_state)
        comps = self._pca.fit_transform(Z)
        self._log(f"[PCA] Done in {self._elapsed(t0)}", 1)
        return comps

    def _pca_transform(self, feat_num: np.ndarray):
        if self._pca is None:
            return None
        Z = feat_num
        if self._scaler is not None:
            Z = self._scaler.transform(Z)
        return self._pca.transform(Z)

    # ---------------- DSBAN: 监督式主成分裁剪 ---------------- #
    def _score_features_for_dsban(self, Xnum: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """返回每个数值列的监督分数：alpha*|corr(y,x)| + (1-alpha)*std(x)"""
        p = Xnum.shape[1]
        if y is None:
            # 无监督回退：仅按标准差
            return Xnum.std(axis=0, ddof=1)
        yv = np.asarray(y).astype(float).ravel()
        yv = yv - np.nanmean(yv)
        yv /= (np.nanstd(yv, ddof=1) + 1e-12)
        # 逐列相关性（数值稳定处理）
        Xc = Xnum - np.nanmean(Xnum, axis=0, keepdims=True)
        Xc_std = np.nanstd(Xnum, axis=0, ddof=1) + 1e-12
        corr = (Xc * yv[:, None]).mean(axis=0) / Xc_std
        corr = np.nan_to_num(corr, nan=0.0)
        stds = Xc_std
        alpha = self.dsban_alpha
        score = alpha * np.abs(corr) + (1.0 - alpha) * stds
        return score

    def _fit_dsban(self, feat: pd.DataFrame, y: Optional[np.ndarray], k_out: int) -> pd.DataFrame:
        """在数值特征上做监督打分→选锚→PCA→输出 k_out 个 dsban_pc_*"""
        t0 = time.time()
        self._log(f"[DSBAN] Fit supervised PCA to meet max_total_cols={k_out} …", 1)

        Xnum_df = feat.select_dtypes(include=[np.number]).fillna(0.0)
        Xnum = Xnum_df.values
        p = Xnum.shape[1]
        if p == 0:
            self._log("[DSBAN] WARN: no numeric features; skip.", 1)
            return pd.DataFrame(index=feat.index)

        # 1) 打分并选锚
        scores = self._score_features_for_dsban(Xnum, y)
        n_anchor = max(self.dsban_min_anchors, int(np.ceil(self.dsban_anchor_frac * p)))
        n_anchor = min(n_anchor, p)
        anchor_idx = np.argsort(-scores)[:n_anchor]
        self._dsban_anchor_idx = anchor_idx

        # 2) 标准化（可选）+ PCA(k_out)
        Xa = Xnum[:, anchor_idx]
        if self.scale_before_pca:
            scaler = StandardScaler()
            Xa = scaler.fit_transform(Xa)
            self._dsban_scaler = scaler
        else:
            self._dsban_scaler = None

        pca = PCA(n_components=min(k_out, Xa.shape[1]), whiten=self.dsban_whiten, random_state=self.random_state)
        comps = pca.fit_transform(Xa)
        self._dsban_pca = pca

        # 3) 命名输出
        cols = [f"dsban_pc_{i+1}" for i in range(comps.shape[1])]
        out = pd.DataFrame(comps, index=feat.index, columns=[f"dsban_pc_{i+1}" for i in range(comps.shape[1])])
        self._log(f"[DSBAN] Done in {self._elapsed(t0)} | anchors={n_anchor}, pcs={out.shape[1]}", 1)
        return out

    def _dsban_transform(self, feat: pd.DataFrame, k_out: int) -> pd.DataFrame:
        """使用已拟合的锚 + PCA，把任意新样本映射到 k_out 个 dsban_pc_*"""
        if self._dsban_anchor_idx is None or self._dsban_pca is None:
            # 未拟合，直接返回空
            return pd.DataFrame(index=feat.index)
        Xnum_df = feat.select_dtypes(include=[np.number]).fillna(0.0)
        if Xnum_df.shape[1] == 0:
            return pd.DataFrame(index=feat.index)

        anchor_idx = self._dsban_anchor_idx
        Xa = Xnum_df.values[:, anchor_idx]
        if self._dsban_scaler is not None:
            Xa = self._dsban_scaler.transform(Xa)
        comps = self._dsban_pca.transform(Xa)
        # 截断/填充到 k_out（一般不需要，组件数在 fit 时定死了）
        comps = comps[:, :min(k_out, comps.shape[1])]
        out = pd.DataFrame(comps, index=feat.index, columns=[f"dsban_pc_{i+1}" for i in range(comps.shape[1])])
        return out

    # ---------------- Public API ---------------- #
    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        t0 = time.time()
        self._log(f"[FDF] fit_transform start | rows={len(df)}, cat={len(self.cat_cols)}, num={len(self.num_cols)}", 1)

        X = df.copy()
        self._to_category(X, self.cat_cols)

        # enc/agg
        te_df = self._oof_target_encode(X[self.cat_cols], y) if (y is not None and self.enable_target_encoding) else pd.DataFrame(index=X.index)
        cf_df = self._count_freq(X)
        woe_df = self._fit_woe(X[self.cat_cols], y) if (y is not None and self.enable_woe) else pd.DataFrame(index=X.index)
        gb_df = self._groupby_agg(X)
        num_df = self._numeric_derivatives(X)

        feat = pd.concat([X[self.num_cols], te_df, cf_df, woe_df, gb_df, num_df], axis=1)
        self._log(f"[FDF] after enc/agg | shape={feat.shape}", 1)

        # poly & autofeat
        before = feat.shape[1]
        poly_df = self._fit_poly2(feat)
        feat = pd.concat([feat, poly_df], axis=1)
        self._log(f"[FDF] add POLY {self._delta_cols(before, feat.shape[1])} | shape={feat.shape}", 1)

        if (y is not None) and self.autofeat_steps > 0 and _HAS_AUTOF:
            before = feat.shape[1]
            af_df = self._fit_autofeat(feat, y)
            feat = pd.concat([feat, af_df], axis=1)
            self._log(f"[FDF] add AutoFeat {self._delta_cols(before, feat.shape[1])} | shape={feat.shape}", 1)

        # leaf embeddings
        if (y is not None) and self.add_leaf_embeddings:
            before = feat.shape[1]
            leaf_idx = self._fit_leaf_model(feat, y)
            if leaf_idx is not None:
                for i in range(leaf_idx.shape[1]):
                    feat[f"leaf_idx_{i}"] = leaf_idx[:, i]
            self._log(f"[FDF] add LEAF {self._delta_cols(before, feat.shape[1])} | shape={feat.shape}", 1)

        # PCA（常规无监督 PC，可作为额外补充）
        before = feat.shape[1]
        num_mat = feat.select_dtypes(include=[np.number]).fillna(0.0).values
        comps = self._fit_pca(num_mat)
        if comps is not None:
            for i in range(comps.shape[1]):
                feat[f"pca_{i+1}"] = comps[:, i]
        self._log(f"[FDF] add PCA {self._delta_cols(before, feat.shape[1])} | shape={feat.shape}", 1)

        # —— 核心修改：若超出上限，启用 DSBAN 主成分裁剪 ——
        if self.max_total_cols is not None and feat.shape[1] > self.max_total_cols:
            self._log(f"[FDF] exceed max_total_cols={self.max_total_cols} → use DSBAN PCs", 1)
            dsban_df = self._fit_dsban(feat, y, k_out=self.max_total_cols)
            # 用 DSBANPC 直接替代整体特征（保证列数精确）
            feat = dsban_df
            self._log(f"[FDF] after DSBAN | shape={feat.shape}", 1)

        self._fitted = True
        self._log(f"[FDF] fit_transform done in {self._elapsed(t0)} | final shape={feat.shape}", 1)
        return feat

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "请先 fit_transform"
        t0 = time.time()
        self._log(f"[FDF] transform start | rows={len(df)}", 1)

        X = df.copy()
        self._to_category(X, self.cat_cols)

        te_df = pd.DataFrame(index=X.index)
        if self.enable_target_encoding and self._te_maps:
            gm = self._global_mean if self._global_mean is not None else 0.0
            for c in self.cat_cols:
                m = self._te_maps.get(c, {})
                te_df[f"{c}__te"] = X[c].map(m).fillna(gm).values

        cf_df = self._count_freq(X)
        woe_df = self._woe_transform(X[self.cat_cols])
        gb_df = self._groupby_agg(X)
        num_df = self._numeric_derivatives(X)

        feat = pd.concat([X[self.num_cols], te_df, cf_df, woe_df, gb_df, num_df], axis=1)
        self._log(f"[FDF] after enc/agg | shape={feat.shape}", 1)

        poly_df = self._poly2_transform(feat)
        feat = pd.concat([feat, poly_df], axis=1)
        self._log(f"[FDF] add POLY | shape={feat.shape}", 1)

        if self._autofeat_model is not None and self.autofeat_steps > 0:
            af_df = self._autofeat_transform(feat)
            feat = pd.concat([feat, af_df], axis=1)
            self._log(f"[FDF] add AutoFeat | shape={feat.shape}", 1)

        leaf_idx = self._leaf_features(feat)
        if leaf_idx is not None:
            for i in range(leaf_idx.shape[1]):
                feat[f"leaf_idx_{i}"] = leaf_idx[:, i]
            self._log(f"[FDF] add LEAF | shape={feat.shape}", 1)

        num_mat = feat.select_dtypes(include=[np.number]).fillna(0.0).values
        comps = self._pca_transform(num_mat)
        if comps is not None:
            for i in range(comps.shape[1]):
                feat[f"pca_{i+1}"] = comps[:, i]
        self._log(f"[FDF] add PCA | shape={feat.shape}", 1)

        # 推理时若超出上限，则使用已拟合的 DSBAN 投影到精确列数
        if self.max_total_cols is not None and feat.shape[1] > self.max_total_cols:
            self._log(f"[FDF] exceed max_total_cols={self.max_total_cols} → use DSBAN PCs (transform)", 1)
            dsban_df = self._dsban_transform(feat, k_out=self.max_total_cols)
            feat = dsban_df
            self._log(f"[FDF] after DSBAN(transform) | shape={feat.shape}", 1)

        self._log(f"[FDF] transform done in {self._elapsed(t0)} | final shape={feat.shape}", 1)
        return feat
