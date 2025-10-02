# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\ASUS\Documents\GitHub\Web3CausalInference\src\xgb_core")

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import KFold
from encoding import CatEncoder, EncoderConfig
from xgb import make_xgb_classifier, make_xgb_regressor
from aipw import aipw_pseudo, nauuc_from_scores

class NAUUCCatBoostTunerV2:
    def __init__(self, n_trials: int = 50, n_splits: int = 5, trim: float = 1e-3,
                 use_gpu: bool = False, encoder_cfg: EncoderConfig = EncoderConfig(),
                 random_state: int = 42):
        self.n_trials = n_trials; self.n_splits = n_splits; self.trim = trim
        self.random_state = random_state
        self.encoder = CatEncoder(encoder_cfg)
        self.use_gpu = use_gpu
        self.best_ = {}

    def _reg_space(self, tr: optuna.trial.Trial) -> Dict[str,Any]:
        return dict(max_depth=tr.suggest_int("reg_max_depth",3,8),
                    learning_rate=tr.suggest_float("reg_lr",1e-3,0.2,log=True),
                    n_estimators=tr.suggest_int("reg_n_estimators",400,1200),
                    subsample=tr.suggest_float("reg_subsample",0.6,1.0),
                    colsample_bytree=tr.suggest_float("reg_colsample_bytree",0.6,1.0),
                    reg_lambda=tr.suggest_float("reg_lambda",0.0,10.0),
                    min_child_weight=tr.suggest_int("reg_min_child_weight",1,15))

    def _clf_space(self, tr: optuna.trial.Trial) -> Dict[str,Any]:
        return dict(max_depth=tr.suggest_int("clf_max_depth",3,8),
                    learning_rate=tr.suggest_float("clf_lr",1e-3,0.2,log=True),
                    n_estimators=tr.suggest_int("clf_n_estimators",300,1200),
                    subsample=tr.suggest_float("clf_subsample",0.6,1.0),
                    colsample_bytree=tr.suggest_float("clf_colsample_bytree",0.6,1.0),
                    reg_lambda=tr.suggest_float("clf_reg_lambda",0.0,10.0),
                    min_child_weight=tr.suggest_int("clf_min_child_weight",1,15))

    def _objective(self, tr, X: pd.DataFrame, T, Y):
        enc = CatEncoder(self.encoder.cfg).fit(X)   # 每个 trial 独立 fit，避免泄漏
        X_np = enc.transform_np(X)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        reg_params = self._reg_space(tr); clf_params = self._clf_space(tr)

        vals = []
        for tr_idx, va_idx in kf.split(X_np):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            T_tr, T_va = T[tr_idx].astype(int), T[va_idx].astype(int)
            Y_tr, Y_va = Y[tr_idx].astype(float), Y[va_idx].astype(float)

            ps = make_xgb_classifier(self.use_gpu, **clf_params).fit(X_tr, T_tr)
            e_tr = np.clip(ps.predict_proba(X_tr)[:,1], self.trim, 1-self.trim)
            e_va = np.clip(ps.predict_proba(X_va)[:,1], self.trim, 1-self.trim)

            if (T_tr==1).sum()==0 or (T_tr==0).sum()==0: return -1e9
            m1 = make_xgb_regressor(self.use_gpu, **reg_params).fit(X_tr[T_tr==1], Y_tr[T_tr==1])
            m0 = make_xgb_regressor(self.use_gpu, **reg_params).fit(X_tr[T_tr==0], Y_tr[T_tr==0])

            mu1_va, mu0_va = m1.predict(X_va), m0.predict(X_va)
            psi_va = aipw_pseudo(Y_va, T_va, mu1_va, mu0_va, e_va, trim=self.trim)
            tau_score = (mu1_va - mu0_va)  # plugin 作为排序分
            vals.append(nauuc_from_scores(psi_va, tau_score))
        return float(np.mean(vals))

    def fit(self, X: pd.DataFrame, T, Y, study_name: str = "nauuc_tuner"):
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(lambda tr: self._objective(tr, X, T, Y), n_trials=self.n_trials, show_progress_bar=False)
        self.best_ = dict(value=study.best_value, params=study.best_trial.params)
        return self

    def best_params(self): return self.best_.get("params", {})
