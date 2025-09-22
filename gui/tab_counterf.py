
import tkinter as tk
from tkinter import ttk
import os, time, pandas as pd, numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
# allow both package and standalone execution
try:
    from .widgets import ExportBar, FigureFrame
except Exception:  # running as a script
    import os as _os, sys as _sys
    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from gui.widgets import ExportBar, FigureFrame


class TabCounterf(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=8)
        self.app = app
        self.key = key
        self.current_df = None
        self.last_fig_path = None

        # left control panel
        ctrl = ttk.Frame(self); ctrl.pack(side="left", fill="y")
        ttk.Button(ctrl, text="Step2 실행", command=self.run_auto).pack(pady=6)

        # right panel: export bar + figure + table
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.export = ExportBar(right, self.copy_table, self.copy_md, self.copy_metrics, self.save_figure, self.open_folder, self.copy_path)
        self.export.pack(fill="x")

        self.figf = FigureFrame(right); self.figf.pack(fill="both", expand=True, pady=4)
        self.table = ttk.Treeview(right, columns=("c1","c2","c3","c4"), show="headings", height=10)
        for i,h in enumerate(["col1","col2","col3","col4"], start=1):
            self.table.heading(f"c{i}", text=h); self.table.column(f"c{i}", width=180, anchor="w")
        self.table.pack(fill="both", expand=True, pady=4)

        # log
        #self.logc = LogConsole(self); self.logc.pack(side="bottom", fill="x")

    def run_auto(self):
        # implemented in child
        self.app.emit_log("run_auto not implemented")

    def update_table(self, df):
        self.current_df = df
        for r in self.table.get_children(): self.table.delete(r)
        if df is None or df.empty: return
        heads = list(df.columns[:4])
        for i,h in enumerate(heads, start=1): self.table.heading(f"c{i}", text=h)
        for _, row in df.iloc[:150,:4].iterrows():
            self.table.insert("", "end", values=[*row.values])

    # Export & Copy handlers
    def copy_table(self):
        if self.current_df is None: self.app.emit_log("No table to copy"); return
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
        import platform, subprocess, os
        folder = os.path.abspath(f"outputs/{self.key}"); os.makedirs(folder, exist_ok=True)
        if platform.system()=="Darwin": subprocess.Popen(["open", folder])
        elif platform.system()=="Windows": subprocess.Popen(["explorer", folder])
        else: subprocess.Popen(["xdg-open", folder])

    def copy_path(self):
        if not self.last_fig_path: self.app.emit_log("No saved figure yet"); return
        self.clipboard_clear(); self.clipboard_append(self.last_fig_path); self.update()
        self.app.emit_log("Copied path")

class TabCounterf(TabCounterf):
    def run_auto(self):
        d = self.app.ctx["data"]
        for k in ["ret_T1","ret_T0","mu1","mu0"]:
            if k not in d: self.app.emit_log("Missing data/mu. Run Step0 & Step1."); return
        Y1, Y0, mu1, mu0 = d["ret_T1"], d["ret_T0"], d["mu1"], d["mu0"]
        D1 = (Y1 - mu0).rename("D1")
        D0 = (mu1 - Y0).rename("D0")
        d["D1"], d["D0"] = D1, D0

        self.figf.fig.clear()
        ax = self.figf.fig.add_subplot(111)
        D1.tail(300).plot(ax=ax, label="D1"); D0.tail(300).plot(ax=ax, label="D0")
        ax.axhline(0, color="k", lw=0.8); ax.legend(); ax.set_title("Counterfactual gaps (recent)"); ax.grid(True, alpha=0.3)
        self.figf.draw()

        desc = pd.DataFrame({
            "stat":["mean_D1","std_D1","mean_D0","std_D0"],
            "value":[D1.mean(), D1.std(), D0.mean(), D0.std()]
        })
        self.update_table(desc)
        self.app.emit_log("Counterfactuals computed."); self.app.emit_log("[tab2] Counterfactuals computed.")

