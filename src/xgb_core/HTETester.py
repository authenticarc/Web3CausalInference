# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src\xgb_core")

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import numpy as np, pandas as pd
from encoding import CatEncoder, EncoderConfig
from xgb import make_xgb_classifier, make_xgb_regressor
from aipw import aipw_pseudo, area_cumgain_centered, policy_values

class HTETester:
    """
    DR-learner style HTE estimator with support for injected (pre-fitted) models.

    Use cases:
    - If you already tuned & trained models in NAUUCCatBoostTunerV2,
      pass them in via __init__(clf=..., reg1=..., reg0=..., tau_model=..., assume_fitted=True).
      Then call fit(X, T, Y, cat_levels=..., colnames=...) to compute psi/tau/nAUUC without any re-training.

    - If you only tuned hyperparameters (not trained final models),
      leave models as None and HTETester will train its own using the same recipe.

    Notes:
    - If tau_model is not provided, HTETester falls back to plugin score: tau = mu1 - mu0.
    - To keep the same feature columns as DeepFeatureFactory, pass cat_levels/colnames.
      Otherwise encoder will be (re)fit on X to derive encoding maps.
    """
    def __init__(self, use_gpu: bool=False, trim: float = 1e-3,
                 encoder_cfg: EncoderConfig = EncoderConfig(),
                 nauuc_band: Tuple[float,float]=(0.3,0.7),
                 nauuc_policy_ks=(0.1,0.2,0.3),
                 # injected models (pre-fitted)
                 clf=None, reg1=None, reg0=None, tau_model=None,
                 assume_fitted: bool=False):
        self.encoder = CatEncoder(encoder_cfg)
        self.trim = float(trim)
        self.nauuc_band = nauuc_band
        self.nauuc_policy_ks = nauuc_policy_ks

        # models (either injected or will be instantiated on demand)
        self.ps_model = clf
        self.m1_model = reg1
        self.m0_model = reg0
        self.tau_model = tau_model
        self._default_use_gpu = bool(use_gpu)
        self.assume_fitted = bool(assume_fitted)

        self.cache_: Dict[str, Any] = {}

    def _ensure_models(self):
        # create defaults if not injected
        if self.ps_model is None:
            self.ps_model = make_xgb_classifier(self._default_use_gpu)
        if self.m1_model is None:
            self.m1_model = make_xgb_regressor(self._default_use_gpu)
        if self.m0_model is None:
            self.m0_model = make_xgb_regressor(self._default_use_gpu)
        # tau_model is optional; can be None (then plugin tau is used)

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray,
            *, cat_levels: Optional[Dict[str, list]] = None,
               colnames: Optional[list] = None):
        # adopt encoder mapping if provided (to keep feature space identical to DFF)
        if cat_levels is not None and colnames is not None:
            self.encoder.adopt(cat_levels, colnames)
        else:
            # (Re)fit encoder on X — this does *not* train ML models, just builds stable one-hot maps
            self.encoder.fit(X)

        Xn = self.encoder.transform_np(X)
        T = np.asarray(T, int)
        Y = np.asarray(Y, float)

        self._ensure_models()

        if self.assume_fitted:
            # STRICT: do not fit any model; just predict with injected ones
            assert hasattr(self.ps_model, "predict_proba"), "Injected clf must implement predict_proba"
            e = np.clip(self.ps_model.predict_proba(Xn)[:, 1], self.trim, 1 - self.trim)

            assert hasattr(self.m1_model, "predict") and hasattr(self.m0_model, "predict"), \
                "Injected reg1/reg0 must implement predict"
            mu1 = self.m1_model.predict(Xn)
            mu0 = self.m0_model.predict(Xn)

            if self.tau_model is not None and hasattr(self.tau_model, "predict"):
                tau = self.tau_model.predict(Xn)
            else:
                tau = mu1 - mu0  # plugin fallback (no new training)
        else:
            # Fit models on the fly (previous default behavior)
            self.ps_model.fit(Xn, T)
            e = np.clip(self.ps_model.predict_proba(Xn)[:, 1], self.trim, 1 - self.trim)
            self.m1_model.fit(Xn[T==1], Y[T==1])
            self.m0_model.fit(Xn[T==0], Y[T==0])
            mu1 = self.m1_model.predict(Xn)
            mu0 = self.m0_model.predict(Xn)

            if self.tau_model is not None:
                # if user provided a (not-fitted) tau model, train it here
                try:
                    m = e*mu1 + (1-e)*mu0
                    Z = (T - e) * (Y - m) / np.clip(e*(1-e), 1e-9, None)
                    self.tau_model.fit(Xn, Z, sample_weight=e*(1-e))
                    tau = self.tau_model.predict(Xn)
                except Exception:
                    # fallback to plugin
                    tau = mu1 - mu0
            else:
                tau = mu1 - mu0

        psi = aipw_pseudo(Y, T, mu1, mu0, e, trim=self.trim)
        band = (e >= self.nauuc_band[0]) & (e <= self.nauuc_band[1])
        nauuc = area_cumgain_centered(psi[band], tau[band]) / max(area_cumgain_centered(psi[band], psi[band]), 1e-9)
        pol = policy_values(psi[band], tau[band], ks=self.nauuc_policy_ks)

        self.cache_ = dict(e=e, mu1=mu1, mu0=mu0, psi=psi, tau=tau, nauuc=nauuc, policy=pol,
                           cat_levels=self.encoder.cat_levels_, colnames=self.encoder.colnames_)
        return self

    # inference helpers
    def effect(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self.encoder.transform_np(X)
        if self.tau_model is not None and hasattr(self.tau_model, "predict"):
            return self.tau_model.predict(Xn)
        # plugin fallback
        mu1 = self.m1_model.predict(Xn); mu0 = self.m0_model.predict(Xn)
        return mu1 - mu0

    def propensity(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self.encoder.transform_np(X)
        return np.clip(self.ps_model.predict_proba(Xn)[:,1], self.trim, 1-self.trim)

    def mu_hat(self, X: pd.DataFrame):
        Xn = self.encoder.transform_np(X)
        return self.m1_model.predict(Xn), self.m0_model.predict(Xn)

    def report(self) -> str:
        r = self.cache_
        return f"nAUUC={r['nauuc']:.3f}; " + " | ".join([f"{k}={v:.3f}" for k,v in r['policy'].items()])
