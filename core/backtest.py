# core/backtest.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _drawdown(cum: pd.Series) -> pd.Series:
    peak = cum.cummax()
    dd = cum/peak - 1.0
    return dd

def _sharpe(ret: pd.Series, ann_factor: int = 252) -> float:
    r = ret.dropna()
    if r.std() == 0 or r.empty: return np.nan
    return (r.mean() / r.std()) * np.sqrt(ann_factor)

def _sortino(ret: pd.Series, ann_factor: int = 252) -> float:
    r = ret.dropna()
    dr = r[r < 0]
    denom = dr.std()
    if denom == 0 or r.empty: return np.nan
    return (r.mean() / denom) * np.sqrt(ann_factor)

def _annual_returns(ret: pd.Series, ann_factor: int = 252) -> pd.Series:
    # 로그수익률이면 exp(sum)-1, 단순수익률이면 (1+r).prod()-1
    if (ret.abs() > 0.2).mean() < 0.01:
        # log-like로 가정
        yr = ret.groupby(ret.index.year).sum().apply(lambda x: np.expm1(x))
    else:
        yr = ret.groupby(ret.index.year).apply(lambda s: (1+s).prod()-1)
    return yr

def backtest_policy(
    ret_T1: pd.Series,
    ret_T0: pd.Series,
    tau: pd.Series,
    cost_bps: float = 5.0,
    min_hold: int = 1,
) -> dict:
    """
    전략 수익률 = pos*ret_T1 + (1-pos)*ret_T0, pos = 1(τ>0) else 0.
    거래비용: 포지션 변경 시 |Δpos| * cost_bps/10000 만큼 당일 수익률에서 차감.
    """
    # 정렬
    df = pd.concat({"r1": ret_T1, "r0": ret_T0, "tau": tau}, axis=1).dropna()
    if df.empty:
        return {"error": "No overlapping data for backtest."}

    # 포지션 & 최소 홀딩
    pos_raw = (df["tau"] > 0).astype(int)
    if min_hold > 1:
        pos = pos_raw.copy()
        last = pos.iloc[0]
        hold = 1
        for i in range(1, len(pos)):
            if pos_raw.iloc[i] == last:
                hold += 1
            else:
                if hold < min_hold:
                    pos.iloc[i] = last
                else:
                    last = pos_raw.iloc[i]
                    hold = 1
        # 한번 더 정리
        pos = pos.astype(int)
    else:
        pos = pos_raw

    # 거래비용
    dpos = pos.diff().abs().fillna(0.0)
    turnover = dpos.mean()  # 일평균 전환율(포지션 변경 비율)
    cost = dpos * (cost_bps/10000.0)

    # 전략 수익률(총/순)
    r_strat_gross = pos*df["r1"] + (1-pos)*df["r0"]
    r_strat_net   = r_strat_gross - cost

    # 벤치
    r_T1 = df["r1"]; r_T0 = df["r0"]

    # 성과
    ann = 252
    out = {
        "ret_strat_gross": r_strat_gross,
        "ret_strat_net":   r_strat_net,
        "ret_T1": r_T1,
        "ret_T0": r_T0,
        "pos": pos,
        "turnover": float(turnover),
        "sharpe_gross": _sharpe(r_strat_gross, ann),
        "sharpe_net":   _sharpe(r_strat_net, ann),
        "sortino_gross": _sortino(r_strat_gross, ann),
        "sortino_net":   _sortino(r_strat_net, ann),
        "sharpe_T1": _sharpe(r_T1, ann),
        "sharpe_T0": _sharpe(r_T0, ann),
    }

    # 누적 & 드로다운(로그/단순 자동판별)
    if (r_strat_net.abs() > 0.2).mean() < 0.01:
        cum_strat = (1.0 + r_strat_net).apply(np.log1p).cumsum().apply(np.expm1)  # 안정적
        cum_T1 = (1.0 + r_T1).apply(np.log1p).cumsum().apply(np.expm1)
        cum_T0 = (1.0 + r_T0).apply(np.log1p).cumsum().apply(np.expm1)
    else:
        cum_strat = (1.0 + r_strat_net).cumprod() - 1.0
        cum_T1 = (1.0 + r_T1).cumprod() - 1.0
        cum_T0 = (1.0 + r_T0).cumprod() - 1.0

    dd = _drawdown(1.0 + cum_strat)  # 누적 기준 드로다운
    out["cum_strat"] = cum_strat
    out["cum_T1"] = cum_T1
    out["cum_T0"] = cum_T0
    out["drawdown"] = dd
    out["mdd"] = float(dd.min())

    # 연도별 성과
    out["annual_strat"] = _annual_returns(r_strat_net)
    out["annual_T1"] = _annual_returns(r_T1)
    out["annual_T0"] = _annual_returns(r_T0)

    return out

def regime_breakdown(
    ret_strat: pd.Series,
    ret_T1: pd.Series,
    ret_T0: pd.Series,
    vix: pd.Series | None = None,
    dgs10: pd.Series | None = None,
) -> pd.DataFrame:
    """
    레짐 성과: VIX 저/고, 10Y 금리 상승/하락(rolling Δ>0/<0).
    인풋이 없으면 가능한 레짐만 계산.
    """
    base = pd.concat({"strat": ret_strat, "T1": ret_T1, "T0": ret_T0}, axis=1).dropna()

    tables = []

    if vix is not None:
        vix = vix.reindex(base.index).ffill()
        med = vix.median()
        lab = pd.Series(np.where(vix <= med, "Low VIX", "High VIX"), index=base.index)
        t = base.groupby(lab).mean().rename_axis("Regime").reset_index()
        t["metric"] = "MeanRet_by_VIX"
        tables.append(t)

    if dgs10 is not None:
        dgs10 = dgs10.reindex(base.index).ffill()
        slope = dgs10.diff(20)  # 약 1개월 추세
        lab = pd.Series(np.where(slope >= 0, "Rate Up", "Rate Down"), index=base.index)
        t = base.groupby(lab).mean().rename_axis("Regime").reset_index()
        t["metric"] = "MeanRet_by_RateTrend"
        tables.append(t)

    if tables:
        out = pd.concat(tables, axis=0, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["Regime","strat","T1","T0","metric"])
    return out
