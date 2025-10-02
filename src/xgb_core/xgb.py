# -*- coding: utf-8 -*-
from typing import Dict, Any
import xgboost as xgb

def get_device_kwargs(use_gpu: bool) -> Dict[str, Any]:
    return {"tree_method": "hist", "device": ("cuda" if use_gpu else "cpu")}

def make_xgb_regressor(use_gpu: bool, **overrides) -> xgb.XGBRegressor:
    base = dict(objective="reg:squarederror",
                max_depth=6, learning_rate=0.05, n_estimators=600,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                min_child_weight=5, n_jobs=-1, **get_device_kwargs(use_gpu))
    base.update(overrides)
    return xgb.XGBRegressor(**base)

def make_xgb_classifier(use_gpu: bool, **overrides) -> xgb.XGBClassifier:
    base = dict(objective="binary:logistic", eval_metric="logloss",
                max_depth=6, learning_rate=0.05, n_estimators=600,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                min_child_weight=5, n_jobs=-1, **get_device_kwargs(use_gpu))
    base.update(overrides)
    return xgb.XGBClassifier(**base)
