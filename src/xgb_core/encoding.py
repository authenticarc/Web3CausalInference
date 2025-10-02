# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
import numpy as np, pandas as pd
from pandas.api.types import (is_numeric_dtype, is_integer_dtype, is_string_dtype,
                              is_categorical_dtype, is_datetime64_any_dtype)

@dataclass
class EncoderConfig:
    auto_encode_cats: bool = True
    int_as_cat_unique_thresh: int = 30
    unique_ratio_thresh: float = 0.05
    rare_freq_ratio: float = 0.001
    max_onehot_levels: int = 200

class CatEncoder:
    """稳定可复用：自动识别类目、稀有并桶+TopK、固定顺序 one-hot、列对齐"""
    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        self.cfg = cfg
        self._cat_levels_: Dict[str, List[str]] = {}
        self._colnames_: Optional[List[str]] = None

    @property
    def cat_levels_(self): return self._cat_levels_
    @property
    def colnames_(self): return self._colnames_

    def adopt(self, cat_levels: Dict[str, Iterable[str]], colnames: Iterable[str]):
        self._cat_levels_ = {k: list(v) for k, v in cat_levels.items()}
        self._colnames_ = list(colnames)

    def infer_cats(self, X: pd.DataFrame) -> List[str]:
        cats, n = [], len(X)
        for c in X.columns:
            s = X[c]
            if is_datetime64_any_dtype(s):  # 跳过时间
                continue
            nunq = s.nunique(dropna=False); ur = nunq / max(n, 1)
            is_cat = (is_string_dtype(s) or is_categorical_dtype(s) or
                      (is_integer_dtype(s) and nunq <= self.cfg.int_as_cat_unique_thresh) or
                      (not is_numeric_dtype(s) and ur <= self.cfg.unique_ratio_thresh))
            if (not is_cat) and is_numeric_dtype(s) and nunq <= self.cfg.int_as_cat_unique_thresh:
                is_cat = True
            if is_cat: cats.append(c)
        return cats

    def fit(self, X: pd.DataFrame):
        if not self.cfg.auto_encode_cats or not isinstance(X, pd.DataFrame):
            self._colnames_ = list(X.columns)
            return self
        cat_cols = self.infer_cats(X)
        self._cat_levels_.clear()
        n = len(X); rare_thresh = max(int(self.cfg.rare_freq_ratio * n), 1)
        for c in cat_cols:
            vc = X[c].astype("string").fillna("__NA__").value_counts(dropna=False)
            common = vc[vc >= rare_thresh].index.tolist()
            if len(common) > self.cfg.max_onehot_levels:
                common = list(vc.index[: self.cfg.max_onehot_levels])
            self._cat_levels_[c] = ["__RARE__"] + [str(v) for v in common]
        # 固定列顺序
        self._colnames_ = list(self.transform(X).columns)
        return self

    def _apply_levels(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        X2 = X.copy()
        for c in cat_cols:
            X2[c] = X2[c].astype("string").fillna("__NA__")
            keep = set(self._cat_levels_[c][1:])
            X2[c] = X2[c].apply(lambda v: v if v in keep else "__RARE__")
        dummies = []
        for c in cat_cols:
            cats = pd.Categorical(X2[c], categories=self._cat_levels_[c])
            dummies.append(pd.get_dummies(cats, prefix=c, dummy_na=False))
        D = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=X.index)
        num_cols = [c for c in X.columns if c not in cat_cols and is_numeric_dtype(X[c])]
        X_num = X[num_cols].apply(pd.to_numeric, errors="coerce")
        return pd.concat([X_num, D], axis=1).fillna(0.0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.auto_encode_cats or not isinstance(X, pd.DataFrame):
            out = X.copy()
            if self._colnames_ is None:
                self._colnames_ = list(out.columns)
            return out
        if not self._cat_levels_:  # 未 fit 时兜底
            self.fit(X)
        cat_cols = list(self._cat_levels_.keys())
        out = self._apply_levels(X, cat_cols)
        # 列对齐
        if self._colnames_ is not None:
            for c in self._colnames_:
                if c not in out:
                    out[c] = 0.0
            out = out[self._colnames_]
        return out

    def transform_np(self, X: pd.DataFrame):
        return self.transform(X).to_numpy(dtype=float, copy=False)
