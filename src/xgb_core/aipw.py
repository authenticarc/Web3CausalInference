# -*- coding: utf-8 -*-
import numpy as np

def aipw_pseudo(y, t, mu1, mu0, e, trim=1e-3):
    e = np.clip(e, trim, 1 - trim)
    return (t*(y-mu1)/e) - ((1-t)*(y-mu0)/(1-e)) + (mu1-mu0)

def aipw_phi(y, t, mu1, mu0, e, trim=1e-3):
    e = np.clip(e, trim, 1 - trim)
    return (mu1 - mu0) + t*(y-mu1)/e - (1-t)*(y-mu0)/(1-e)

def area_cumgain_centered(psi, scores):
    order = np.argsort(-scores)
    psi_ord = psi[order]
    psi_c = psi_ord - psi_ord.mean()
    csum = np.cumsum(psi_c)
    x = np.arange(1, len(csum)+1) / len(csum)
    return float(np.trapz(csum, x))

def _area_from_scores(psi, score):
    order = np.argsort(-score)
    psi_ord = psi[order]
    csum = np.cumsum(psi_ord)
    x = np.arange(1, len(psi_ord)+1) / len(psi_ord)
    return float(np.trapz(csum, x))

def nauuc_from_scores(psi, score):
    a_model = _area_from_scores(psi, score)
    a_oracle = _area_from_scores(psi, psi)
    if a_oracle <= 0: return 0.0
    return float(np.clip(a_model / a_oracle, 0.0, 1.0))

def policy_values(psi, scores, ks=(0.1,0.2,0.3)):
    out = {}
    n = len(psi)
    order = np.argsort(-scores)
    psi_ord = psi[order]
    for k in ks:
        m = max(1, int(np.floor(n*k)))
        out[f'policy@{int(k*100)}'] = float(np.sum(psi_ord[:m]))
    return out
