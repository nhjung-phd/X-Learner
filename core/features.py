# core/features.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def build_lag_features(series_dict: Dict[str, pd.Series], window: int, standardize: bool=True) -> pd.DataFrame:
    """
    series_dict: {"ret_T1": pd.Series, "ret_T0": pd.Series, ...}
    window: seq_window (ex: 60)
    Return: X DataFrame (aligned, dropna)
    """
    dfs = []
    for name, s in series_dict.items():
        s = s.astype(float).copy()
        for k in range(1, window+1):
            dfs.append(s.shift(k).rename(f"{name}_lag{k}"))
    X = pd.concat(dfs, axis=1)
    X = X.dropna()
    if standardize:
        X = (X - X.mean()) / (X.std().replace(0, np.nan))
    return X

def make_forward_target(r: pd.Series, k: int) -> pd.Series:
    """
    k일 forward log-return 합 (예: 다음 k일 수익): shift(-k)
    """
    y = r.rolling(k).sum().shift(-k)
    return y

def train_test_split_time(X: pd.DataFrame, y: pd.Series, test_frac: float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    Xy = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    n = len(Xy)
    n_tr = max(1, int(n*(1.0-test_frac)))
    X_tr, X_te = Xy.iloc[:n_tr, :-1], Xy.iloc[n_tr:, :-1]
    y_tr, y_te = Xy.iloc[:n_tr, -1], Xy.iloc[n_tr:, -1]
    return X_tr, X_te, y_tr, y_te
