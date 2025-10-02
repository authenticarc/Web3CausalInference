# -*- coding: utf-8 -*-
import numpy as np

def ks(ps_t, ps_c):
    a = np.sort(ps_t); b = np.sort(ps_c)
    n, m = len(a), len(b); ia = ib = 0; d = 0.0
    while ia < n and ib < m:
        if a[ia] <= b[ib]: ia += 1
        else: ib += 1
        d = max(d, abs(ia/n - ib/m))
    return d

def ovl(ps_t, ps_c, bins=50):
    hist_t, edges = np.histogram(ps_t, bins=bins, range=(0,1), density=True)
    hist_c, _     = np.histogram(ps_c, bins=bins, range=(0,1), density=True)
    bw = edges[1] - edges[0]
    return float(np.sum(np.minimum(hist_t, hist_c)) * bw)

def _weighted_mean_var(x, w):
    w = np.asarray(w, float); x = np.asarray(x, float)
    wm = np.sum(w * x) / (np.sum(w) + 1e-12)
    wv = np.sum(w * (x - wm) ** 2) / (np.sum(w) + 1e-12)
    return wm, wv

def ess_ratio(w, mask):
    w_g = np.asarray(w)[mask]; n_g = int(mask.sum())
    ess = (w_g.sum() ** 2) / (np.sum(w_g ** 2) + 1e-12)
    return float(ess / (n_g + 1e-12))

def smd_ate(x, t, w_t, w_c):
    x = np.asarray(x, float); t = np.asarray(t, int)
    mt, vt = _weighted_mean_var(x[t==1], w_t[t==1])
    mc, vc = _weighted_mean_var(x[t==0], w_c[t==0])
    pooled = np.sqrt((vt + vc)/2.0 + 1e-12)
    return float((mt - mc) / (pooled + 1e-12))

def smd_att(x, t, w_c_odds):
    x = np.asarray(x, float); t = np.asarray(t, int)
    mt, vt = np.mean(x[t==1]), np.var(x[t==1])
    mc, vc = _weighted_mean_var(x[t==0], w_c_odds[t==0])
    pooled = np.sqrt((vt + vc)/2.0 + 1e-12)
    return float((mt - mc) / (pooled + 1e-12))

def smd_ato(x, t, w1, w0):
    x = np.asarray(x, float); t = np.asarray(t, int)
    m1, v1 = _weighted_mean_var(x[t==1], w1[t==1])
    m0, v0 = _weighted_mean_var(x[t==0], w0[t==0])
    pooled = np.sqrt((v1 + v0)/2.0 + 1e-12)
    return float((m1 - m0) / (pooled + 1e-12))
