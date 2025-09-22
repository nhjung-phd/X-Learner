# gui/tab_mu.py  — REPLACE WHOLE FILE
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import os, time, pandas as pd, numpy as np

# allow both package and standalone execution
try:
    from .widgets import ExportBar, FigureFrame
    from core.features import build_lag_features, make_forward_target, train_test_split_time
    from core.model_selector import get_regressor
except Exception:  # running as a script
    import os as _os, sys as _sys
    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from gui.widgets import ExportBar, FigureFrame
    from core.features import build_lag_features, make_forward_target, train_test_split_time
    from core.model_selector import get_regressor


class TabMu(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=8)
        self.app = app
        self.key = key
        self.current_df = None
        self.last_fig_path = None

        # left control panel
        ctrl = ttk.Frame(self); ctrl.pack(side="left", fill="y")
        ttk.Button(ctrl, text="Step1 실행", command=self.run_auto).pack(pady=6)

        # right panel
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.export = ExportBar(
            right, self.copy_table, self.copy_md, self.copy_metrics,
            self.save_figure, self.open_folder, self.copy_path
        )
        self.export.pack(fill="x")

        self.figf = FigureFrame(right); self.figf.pack(fill="both", expand=True, pady=4)
        self.table = ttk.Treeview(right, columns=("c1","c2","c3","c4"), show="headings", height=10)
        for i,h in enumerate(["metric","mu1_corr","mu0_corr","rmse_avg"], start=1):
            self.table.heading(f"c{i}", text=h); self.table.column(f"c{i}", width=160, anchor="w")
        self.table.pack(fill="both", expand=True, pady=4)

    # =========================
    # Step1: μ 모델 학습
    # =========================
    def run_auto(self):
        d = self.app.ctx["data"]
        if "ret_T1" not in d or "ret_T0" not in d:
            self.app.emit_log("No returns. Run Step0 first."); return

        r1, r0 = d["ret_T1"].copy(), d["ret_T0"].copy()
        k   = int(self.app.ctx.get("k_days", 10))
        win = int(self.app.ctx.get("seq_window", 60))
        stdz = bool(self.app.ctx.get("standardize", True))
        mu_kind = self.app.ctx.get("mu_model", "LSTM")

        # ---- Feature set: T1/T0 + (optional) MACRO_* ----
        series_dict = {"ret_T1": r1, "ret_T0": r0}
        if bool(self.app.ctx.get("macro_enable", False)):
            macro = d.get("macro")
            if macro is not None and not macro.empty:
                for col in macro.columns:
                    series_dict[f"MACRO_{col}"] = macro[col]

        X = build_lag_features(series_dict, window=win, standardize=stdz)

        # ---- Targets: k일 forward return (동일 인덱스로 정렬)
        y1 = make_forward_target(r1, k).reindex(X.index)
        y0 = make_forward_target(r0, k).reindex(X.index)

        # ---- Train/Test split (time-based)
        X1_tr, X1_te, y1_tr, y1_te = train_test_split_time(X, y1)
        X0_tr, X0_te, y0_tr, y0_te = train_test_split_time(X, y0)

        # ---- Model selection (DL 가능, 미설치 시 폴백)
        model1 = get_regressor(mu_kind, self.app.ctx)
        model0 = get_regressor(mu_kind, self.app.ctx)
        for nm, m in [("mu1", model1), ("mu0", model0)]:
            if hasattr(m, "_fallback_from"):
                self.app.emit_log(f"[tab1] WARNING: requested {m._fallback_from} → fallback to MLPRegressor (TensorFlow not found)")
            else:
                self.app.emit_log(f"[tab1] model[{nm}] = {type(m).__name__}")

        # ---- Logging: shapes & params
        n_feats = X.shape[1] if hasattr(X, "shape") else 0
        self.app.emit_log(f"[μ] kind={mu_kind}, win={win}, k={k}, stdz={stdz}")
        self.app.emit_log(f"[μ] X.shape={getattr(X,'shape',None)}, y1={y1.notna().sum()}, y0={y0.notna().sum()}, n_feats={n_feats}")

        # ---- Fit
        model1.fit(X1_tr, y1_tr); model0.fit(X0_tr, y0_tr)

        # ---- Predict μ on full aligned index
        mu1_full = pd.Series(model1.predict(X), index=X.index, name="mu1")
        mu0_full = pd.Series(model0.predict(X), index=X.index, name="mu0")

        # ---- Save for downstream
        self.app.ctx["data"]["mu1"] = mu1_full
        self.app.ctx["data"]["mu0"] = mu0_full

        # ---- Metrics
        p1_te = pd.Series(model1.predict(X1_te), index=X1_te.index)
        p0_te = pd.Series(model0.predict(X0_te), index=X0_te.index)
        corr1 = float(p1_te.corr(y1_te))
        corr0 = float(p0_te.corr(y0_te))
        rmse1 = float(np.sqrt(((p1_te - y1_te)**2).mean()))
        rmse0 = float(np.sqrt(((p0_te - y0_te)**2).mean()))
        rmse_avg = (rmse1 + rmse0) / 2.0

        met = pd.DataFrame({
            "metric":["corr_mu1","corr_mu0","rmse_mu1","rmse_mu0","rmse_avg"],
            "mu1_corr":[corr1, None, None, None, None],
            "mu0_corr":[None, corr0, None, None, None],
            "rmse_avg":[None, None, rmse1, rmse0, rmse_avg]
        })
        self.update_table(met.fillna(""))

        # ---- Figure: last 200 points μ1 vs μ0
        self.figf.fig.clear()
        ax = self.figf.fig.add_subplot(111)
        mu1_full.tail(200).plot(ax=ax, label="μ1")
        mu0_full.tail(200).plot(ax=ax, label="μ0")
        ax.set_title(f"μ predictions (model={mu_kind}, k={k}, win={win}, feats={n_feats})")
        ax.legend(); ax.grid(True, alpha=0.3)
        self.figf.draw()

        self.app.emit_log(f"[μ] corr_mu1={corr1:.3f}, corr_mu0={corr0:.3f}, rmse_avg={rmse_avg:.4f}")
        self.app.emit_log("[tab1] μ-models trained.")

    # =========================
    # Export helpers
    # =========================
    def update_table(self, df: pd.DataFrame):
        self.current_df = df
        for r in self.table.get_children(): self.table.delete(r)
        if df is None or len(df)==0: return
        heads = list(df.columns[:4])
        for i,h in enumerate(heads, start=1): self.table.heading(f"c{i}", text=h)
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
