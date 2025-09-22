from __future__ import annotations
import pandas as pd, numpy as np

def resample_to_daily(s: pd.Series) -> pd.Series:
    return s.resample("B").ffill()

def pct_or_z(s: pd.Series, mode="z", window=252):
    if mode == "pct":
        return s.pct_change().replace([np.inf,-np.inf], np.nan)
    m = s.rolling(window, min_periods=max(10, int(window*0.5))).mean()
    v = s.rolling(window, min_periods=max(10, int(window*0.5))).std()
    return (s - m) / v

def apply_publication_lag(s: pd.Series, lag_days=5):
    return s.shift(lag_days)

def prepare_macro(macro_raw: dict[str, pd.Series], daily=True, lag_days=5, norm="z") -> pd.DataFrame:
    out = {}
    for k, v in macro_raw.items():
        x = v.copy().dropna()
        if daily: x = resample_to_daily(x)
        x = apply_publication_lag(x, lag_days=lag_days)
        x = pct_or_z(x, mode=("pct" if k in ["DCOILWTICO","DTWEXBGS","DGS10"] else norm))
        out[k] = x
    return pd.DataFrame(out).dropna(how="all")
