# gui/tab_data.py  — REPLACE WHOLE FILE
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import os, time, pandas as pd, numpy as np
from matplotlib import pyplot as plt

# allow both package and standalone execution
try:
    from .widgets import ExportBar, FigureFrame
    from core.fetchers import prepare_macro
except Exception:  # running as a script
    import os as _os, sys as _sys
    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from gui.widgets import ExportBar, FigureFrame
    try:
        from core.fetchers import prepare_macro
    except Exception:
        prepare_macro = None  # optional

import yfinance as yf


def _eq_weight_logret(px: pd.DataFrame) -> pd.Series:
    px = px.ffill().dropna(how="all")
    r = np.log(px).diff()
    return r.mean(axis=1)


def _resample_business_daily(s: pd.Series) -> pd.Series:
    """월간/주간 지표를 영업일 단위로 확장(ffill)."""
    return s.resample("B").ffill()


class TabData(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=8)
        self.app = app
        self.key = key
        self.current_df = None
        self.last_fig_path = None

        # left control panel
        ctrl = ttk.Frame(self); ctrl.pack(side="left", fill="y")
        ttk.Button(ctrl, text="Step0 실행", command=self.run_auto).pack(pady=6)

        # right panel: export bar + figure + tables
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.export = ExportBar(
            right, self.copy_table, self.copy_md, self.copy_metrics,
            self.save_figure, self.open_folder, self.copy_path
        )
        self.export.pack(fill="x")

        # 상단: 수익률 피겨
        self.figf = FigureFrame(right); self.figf.pack(fill="both", expand=True, pady=4)

        # 중단: 리턴 테이블
        self.table = ttk.Treeview(right, columns=("c1","c2","c3","c4"), show="headings", height=8)
        for i,h in enumerate(["col1","col2","col3","col4"], start=1):
            self.table.heading(f"c{i}", text=h); self.table.column(f"c{i}", width=180, anchor="w")
        self.table.pack(fill="x", expand=False, pady=(2,8))

        # 하단: 매크로 프리뷰 섹션 (NEW)
        macro_frame = ttk.LabelFrame(right, text="Macro Preview (Use Macro가 켜진 경우)")
        macro_frame.pack(fill="both", expand=True)

        topRow = ttk.Frame(macro_frame); topRow.pack(fill="x", pady=(4,2))
        self.macro_status = ttk.Label(topRow, text="Use Macro: OFF", foreground="#666")
        self.macro_status.pack(side="left")

        self.table_macro = ttk.Treeview(
            macro_frame,
            columns=("name","start","end","coverage","last"),
            show="headings", height=8
        )
        for col, w in [("name",160), ("start",95), ("end",95), ("coverage",90), ("last",90)]:
            self.table_macro.heading(col, text=col)
            self.table_macro.column(col, width=w, anchor="w")
        self.table_macro.pack(fill="both", expand=True, pady=(4,6))

    # --------------------------
    # Export & Copy handlers
    # --------------------------
    def update_table(self, df: pd.DataFrame | None):
        self.current_df = df
        for r in self.table.get_children():
            self.table.delete(r)
        if df is None or df.empty:
            return
        heads = list(df.columns[:4])
        for i,h in enumerate(heads, start=1):
            self.table.heading(f"c{i}", text=h)
        for _, row in df.iloc[:150,:4].iterrows():
            self.table.insert("", "end", values=[*row.values])

    def copy_table(self):
        if self.current_df is None:
            self.app.emit_log("No table to copy"); return
        txt = self.current_df.to_csv(index=False)
        self.clipboard_clear(); self.clipboard_append(txt); self.update()
        self.app.emit_log("Copied table")

    def copy_md(self):
        rows = len(self.current_df) if self.current_df is not None else 0
        md = (
            f"### {self.key} Summary\n\n"
            f"Rows: {rows}\n"
            f"T1={self.app.ctx['T1']} | T0={self.app.ctx['T0']} | "
            f"{self.app.ctx['start']}→{self.app.ctx['end']}"
        )
        self.clipboard_clear(); self.clipboard_append(md); self.update()
        self.app.emit_log("Copied markdown")

    def copy_metrics(self):
        text = "metric,value\nexample,1.0"
        self.clipboard_clear(); self.clipboard_append(text); self.update()
        self.app.emit_log("Copied metrics")

    def save_figure(self):
        os.makedirs(f"outputs/{self.key}", exist_ok=True)
        path = f"outputs/{self.key}/{int(time.time())}_figure.png"
        self.figf.fig.savefig(path, dpi=150, bbox_inches="tight")
        self.last_fig_path = path
        self.app.emit_log(f"Saved figure: {path}")

    def open_folder(self):
        import platform, subprocess
        folder = os.path.abspath(f"outputs/{self.key}"); os.makedirs(folder, exist_ok=True)
        if platform.system()=="Darwin": subprocess.Popen(["open", folder])
        elif platform.system()=="Windows": subprocess.Popen(["explorer", folder])
        else: subprocess.Popen(["xdg-open", folder])

    def copy_path(self):
        if not self.last_fig_path:
            self.app.emit_log("No saved figure yet"); return
        self.clipboard_clear(); self.clipboard_append(self.last_fig_path); self.update()
        self.app.emit_log("Copied path")

    # --------------------------
    # Macro Preview helpers (NEW)
    # --------------------------
    def update_macro_preview(self, macro_df: pd.DataFrame | None):
        """Use Macro가 켜진 경우, 매크로 지표 목록/커버리지를 표로 출력."""
        enabled = bool(self.app.ctx.get("macro_enable", False))
        self.macro_status.configure(
            text=f"Use Macro: {'ON' if enabled else 'OFF'}",
            foreground=("#0a7" if enabled else "#666")
        )
        # clear rows
        for r in self.table_macro.get_children():
            self.table_macro.delete(r)
        if not enabled or macro_df is None or macro_df.empty:
            return
        # fill rows
        for col in macro_df.columns:
            s = macro_df[col].dropna()
            if s.empty:
                start = end = "—"; cov = "0.0%"; last = "—"
            else:
                start = s.index.min().strftime("%Y-%m-%d")
                end   = s.index.max().strftime("%Y-%m-%d")
                cov   = f"{100.0 * s.size / macro_df.index.size:.1f}%"
                try:
                    last = f"{float(s.iloc[-1]):.4g}"
                except Exception:
                    last = str(s.iloc[-1])
            self.table_macro.insert("", "end", values=[col, start, end, cov, last])

    # --------------------------
    # Step0
    # --------------------------
    def run_step0(self): self.run_auto()

    def run_auto(self):
        T1 = self.app.ctx["T1"]; T0 = self.app.ctx["T0"]
        start = self.app.ctx["start"]; end = self.app.ctx["end"]
        self.app.emit_log(f"Fetching: T1={T1}, T0={T0}, {start}→{end}")

        # --- robust yfinance extraction (handles auto_adjust and MultiIndex columns)
        def _get_price(df):
            if isinstance(df, pd.Series):
                return df.to_frame()
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                if "Adj Close" in set(lvl0):
                    px = df["Adj Close"]
                elif "Close" in set(lvl0):
                    px = df["Close"]
                else:
                    # try common fallback
                    try:
                        px = df.xs("Close", level=0, axis=1, drop_level=True)
                    except Exception:
                        # heuristic: take last field group
                        fields = list(dict.fromkeys(lvl0))
                        px = df[fields[-1]]
                return px
            else:
                if "Adj Close" in df.columns:
                    col = df["Adj Close"]
                    return col if isinstance(col, pd.DataFrame) else col.to_frame()
                if "Close" in df.columns:
                    col = df["Close"]
                    return col if isinstance(col, pd.DataFrame) else col.to_frame()
                return df

        raw1 = yf.download(T1, start=start, end=end, auto_adjust=True, progress=False)
        raw0 = yf.download(T0, start=start, end=end, auto_adjust=True, progress=False)
        px1 = _get_price(raw1)
        px0 = _get_price(raw0)

        ret_T1 = _eq_weight_logret(px1).rename("ret_T1")
        ret_T0 = _eq_weight_logret(px0).rename("ret_T0")
        df = pd.concat([ret_T1, ret_T0], axis=1).dropna()

        # store to context (returns)
        self.app.ctx["data"]["ret_T1"] = ret_T1
        self.app.ctx["data"]["ret_T0"] = ret_T0

        # --- (옵션) Macro 지표 가져오기/정리 ---
        if bool(self.app.ctx.get("macro_enable", False)):
            macro_raw = {}
            # 1) VIX (yfinance, level)
            try:
                vix_raw = yf.download("^VIX", start=start, end=end, progress=False)
                vix_px = _get_price(vix_raw).iloc[:, 0]  # first column if multi
                vix_px.name = "VIX"
                macro_raw["VIX"] = vix_px
            except Exception as e:
                self.app.emit_log(f"[tab0] VIX fetch failed: {e}")

            # 2) FRED (if available)
            dgs10_level = None
            try:
                from fredapi import Fred
                fred = Fred(api_key=os.environ.get("FRED_API_KEY", None))
                def F(code):
                    s = fred.get_series(code)
                    s.index = pd.to_datetime(s.index)
                    s.name = code
                    return s
                macro_raw.update({
                    "INDPRO":     F("INDPRO"),
                    "CPIAUCSL":   F("CPIAUCSL"),
                    "UNRATE":     F("UNRATE"),
                    "DGS10":      F("DGS10"),
                    "DTWEXBGS":   F("DTWEXBGS"),
                    "DCOILWTICO": F("DCOILWTICO"),
                    "NFCI":       F("NFCI"),
                })
                # regime용 DGS10 레벨(영업일 보간)
                dgs10_level = _resample_business_daily(macro_raw["DGS10"].copy())
            except Exception as e:
                self.app.emit_log(f"[tab0] FRED skipped: {e}")

            # 3) 표준화/발표지연/일간 정합
            macro_df = None
            try:
                if prepare_macro is None:
                    raise RuntimeError("core.fetchers.prepare_macro not found")
                macro_df = prepare_macro(
                    macro_raw,
                    daily=True,
                    lag_days=int(self.app.ctx.get("macro_lag_days", 5)),
                    norm=self.app.ctx.get("macro_norm", "z"),
                )
                # 공통 인덱스로 정렬
                common_idx = ret_T1.index.union(ret_T0.index).union(macro_df.index)
                macro_df = macro_df.reindex(common_idx).ffill()
                self.app.ctx["data"]["macro"] = macro_df
                self.app.emit_log(f"[tab0] macro cols: {list(macro_df.columns)}")
            except Exception as e:
                self.app.emit_log(f"[tab0] macro prepare skipped: {e}")

            # 4) 레짐 분석용 저장 (레벨, lag 없음)
            if "VIX" in macro_raw:
                self.app.ctx["data"]["vix"] = _resample_business_daily(macro_raw["VIX"].copy()).reindex(df.index).ffill()
            if dgs10_level is not None:
                self.app.ctx["data"]["dgs10"] = dgs10_level.reindex(df.index).ffill()

        else:
            # 비활성 시 macro 제거(실험 대비)
            self.app.ctx["data"].pop("macro", None)
            self.app.ctx["data"].pop("vix", None)
            self.app.ctx["data"].pop("dgs10", None)

        # --- figure: cumulative returns
        self.figf.fig.clear()
        ax = self.figf.fig.add_subplot(111)
        (ret_T1.cumsum()).plot(ax=ax, label="T1 cum logret")
        (ret_T0.cumsum()).plot(ax=ax, label="T0 cum logret")
        ax.set_title(f"T1/T0 cumulative returns ({start}→{end})")
        ax.legend(); ax.grid(True, alpha=0.3)
        self.figf.draw()

        # --- table: head
        self.update_table(df.tail(10).reset_index().rename(columns={"Date":"date"}))

        # --- (NEW) Macro Preview 갱신
        self.update_macro_preview(self.app.ctx["data"].get("macro"))

        self.app.emit_log("Step0 done."); self.app.emit_log("[tab0] Step0 done.")


# Standalone test
if __name__ == "__main__":
    import datetime as dt
    class _DummyApp:
        def __init__(self):
            self.ctx = {
                "T1": ["XLY","XLI","XLF"],
                "T0": ["BTC-USD","ETH-USD"],
                "AUX": ["^GSPC","^VIX"],
                "start": (dt.date.today() - dt.timedelta(days=365*3)).isoformat(),
                "end": dt.date.today().isoformat(),
                "k_days": 10,
                "wf_splits": 3,
                "macro_enable": True,
                "macro_lag_days": 5,
                "macro_norm": "z",
                "data": {}
            }
            class _DummyStatus:
                def set(self, *args, **kwargs): pass
            self.status = _DummyStatus()
        def emit_log(self, msg): print(msg)

    root = tk.Tk(); root.title("tab0 (standalone)")
    frm = TabData(root, _DummyApp(), "tab0")
    frm.pack(fill="both", expand=True)
    root.geometry("1100x780")
    root.after(100, frm.run_auto)
    root.mainloop()
