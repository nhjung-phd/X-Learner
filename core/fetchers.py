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

# core/fetchers.py
import pandas as pd
from typing import Sequence, Union
import datetime as dt

def _to_datestr(x):
    if isinstance(x, (dt.date, dt.datetime)): return x.strftime("%Y-%m-%d")
    return str(x)

def fetch_prices_prefer_fdr(tickers: Union[str, Sequence[str]], start, end, auto_adjust=True):
    """
    가격계열 우선순위: FDR → (실패 시) yfinance
    반환: (DataFrame) columns=tickers, values=Close/Adj Close
    """
    start, end = _to_datestr(start), _to_datestr(end)
    if isinstance(tickers, str): tickers = [tickers]

    # 1) FDR 시도
    try:
        import FinanceDataReader as fdr
        frames = []
        for t in tickers:
            df = fdr.DataReader(t, start, end)
            # FDR은 'Close' 컬럼이 일반적
            s = df["Close"].rename(t)
            frames.append(s)
        px = pd.concat(frames, axis=1).sort_index()
        return px
    except Exception as e_fdr:
        # 2) yfinance 폴백
        import yfinance as yf
        df = yf.download(list(tickers), start=start, end=end, auto_adjust=auto_adjust, progress=False)
        # 멀티인덱스/단일 모두 대응
        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in set(df.columns.get_level_values(0)):
                px = df["Adj Close"]
            elif "Close" in set(df.columns.get_level_values(0)):
                px = df["Close"]
            else:
                px = df.xs("Close", level=0, axis=1, drop_level=True)
        else:
            # 단일 티커인 경우
            if "Adj Close" in df.columns: px = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
            elif "Close" in df.columns:   px = df[["Close"]].rename(columns={"Close": tickers[0]})
            else:                         px = df.iloc[:, [0]].rename(columns={df.columns[0]: tickers[0]})
        return px

