# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src\xgb_core")

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np, pandas as pd
from encoding import CatEncoder, EncoderConfig
from xgb import make_xgb_classifier, make_xgb_regressor
from crossfit import crossfit_oof_mu_e
from aipw import aipw_pseudo, area_cumgain_centered, policy_values

@dataclass
class CausalRules:
    smd_max: float = 0.10
    ovl_min: float = 0.50
    ks_max:  float = 0.40
    ess_min: float = 0.70
    placebo_alpha: float = 0.10
    nc_alpha: float = 0.10
    top_k_smd: int = 10

class UnifiedCausalTester:
    def __init__(self, estimand: str = "ATE",
                 n_splits: int = 5, trim: float = 1e-3,
                 use_gpu: bool = False, verbose: int = 1,
                 encoder_cfg: EncoderConfig = EncoderConfig(),
                 compute_nauuc: bool = False,
                 nauuc_band: Tuple[float,float]=(0.3,0.7),
                 nauuc_policy_ks=(0.1,0.2,0.3),
                 regressor_kwargs: Optional[Dict[str,Any]]=None,
                 classifier_kwargs: Optional[Dict[str,Any]]=None):
        assert estimand in {"ATE","ATT","ATO"}
        self.estimand = estimand
        self.n_splits = n_splits; self.trim = trim
        self.encoder = CatEncoder(encoder_cfg)
        self.regressor = make_xgb_regressor(use_gpu, **(regressor_kwargs or {}))
        self.classifier = make_xgb_classifier(use_gpu, **(classifier_kwargs or {}))
        self.compute_nauuc = bool(compute_nauuc)
        self.nauuc_band = nauuc_band; self.nauuc_policy_ks = nauuc_policy_ks
        self.verbose = int(verbose)
        self.result_ = None

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        X_enc = self.encoder.fit(X).transform_np(X)
        mu1, mu0, e = crossfit_oof_mu_e(X_enc, T.astype(int), Y.astype(float),
                                        self.regressor, self.classifier,
                                        self.trim, self.n_splits, seed=42)
        e = np.clip(e, self.trim, 1-self.trim)
        psi = aipw_pseudo(Y, T, mu1, mu0, e, trim=self.trim)
        tau_hat = (mu1 - mu0)  # 若需要单独 τ-model，可另加
        band = (e >= self.nauuc_band[0]) & (e <= self.nauuc_band[1])
        nauuc = None; pol = {}
        if self.compute_nauuc and band.any():
            den = area_cumgain_centered(psi[band], psi[band])
            num = area_cumgain_centered(psi[band], tau_hat[band])
            nauuc = float(num / max(den, 1e-9))
            pol = policy_values(psi[band], tau_hat[band], ks=self.nauuc_policy_ks)
        self.result_ = dict(mu1=mu1, mu0=mu0, e=e, psi=psi, tau=tau_hat,
                            nauuc=nauuc, policy=pol,
                            cat_levels=self.encoder.cat_levels_, colnames=self.encoder.colnames_)
        return self

    def report(self) -> str:
        r = self.result_ or {}
        if r.get("nauuc") is None: return "nAUUC=NA"
        tail = " | ".join([f"{k}={v:.3f}" for k,v in (r.get("policy") or {}).items()])
        return f"nAUUC={r['nauuc']:.3f}" + (("; " + tail) if tail else "")
