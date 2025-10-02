# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone

def crossfit_oof_mu_e(X, T, Y, regressor, classifier, trim: float, n_splits: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(Y)
    mu1_oof = np.zeros(n); mu0_oof = np.zeros(n); e_oof = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (tr, va) in enumerate(kf.split(X), 1):
        clf = clone(classifier); reg1 = clone(regressor); reg0 = clone(regressor)
        for mdl in (clf, reg1, reg0):
            try: mdl.set_params(n_jobs=1)
            except: pass
        X_tr, X_va = X[tr], X[va]; T_tr, Y_tr = T[tr], Y[tr]
        clf.fit(X_tr, T_tr)
        e_hat = np.clip(clf.predict_proba(X_va)[:, 1], trim, 1 - trim)
        mu1_hat = np.zeros(len(va)); mu0_hat = np.zeros(len(va))
        if (T_tr==1).any():
            reg1.fit(X_tr[T_tr==1], Y_tr[T_tr==1]); mu1_hat = reg1.predict(X_va)
        if (T_tr==0).any():
            reg0.fit(X_tr[T_tr==0], Y_tr[T_tr==0]); mu0_hat = reg0.predict(X_va)
        mu1_oof[va] = mu1_hat; mu0_oof[va] = mu0_hat; e_oof[va] = e_hat
    return mu1_oof, mu0_oof, e_oof
