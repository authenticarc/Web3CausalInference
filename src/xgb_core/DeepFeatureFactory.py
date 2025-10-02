# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src\xgb_core")

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np, pandas as pd
from encoding import CatEncoder, EncoderConfig

class DeepFeatureFactory:
    """把类目处理交给 CatEncoder，其他 FE 积木可按需叠加到 transform() 内。"""
    def __init__(self, cat_cols: Optional[Sequence[str]] = None,
                 num_cols: Optional[Sequence[str]] = None,
                 auto_detect_cols: bool = True,
                 encoder_cfg: EncoderConfig = EncoderConfig(),
                 verbose: int = 1):
        self.cat_cols = list(cat_cols) if cat_cols is not None else None
        self.num_cols = list(num_cols) if num_cols is not None else None
        self.auto_detect_cols = bool(auto_detect_cols)
        self.encoder = CatEncoder(encoder_cfg)
        self.verbose = int(verbose)
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray]=None):
        if self.cat_cols is None or self.num_cols is None:
            cats = self.encoder.infer_cats(X)
            nums = [c for c in X.columns if c not in cats and pd.api.types.is_numeric_dtype(X[c])]
            self.cat_cols, self.num_cols = cats, nums
        self.encoder.fit(X)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Call fit first."
        # 你可以在这里叠加：OOF-TE、count/freq、leaf-emb、交互项、log1p、PCA等
        return self.encoder.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray]=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def cat_levels_(self): return self.encoder.cat_levels_
    @property
    def colnames_(self): return self.encoder.colnames_
