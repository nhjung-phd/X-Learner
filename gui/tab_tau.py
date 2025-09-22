# gui/tab_tau.py  — REPLACE WHOLE FILE
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import os, time, pandas as pd, numpy as np

# allow both package and standalone execution
try:
    from .widgets import ExportBar, FigureFrame
    from core.features import build_lag_features, train_test_split_time
    from core.model_selector import get_regressor, get_classifier
except Exception:  # running as a script
    import os as _os, sys as _sys
    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from gui.widgets import ExportBar, FigureFrame
    from core.features import build_lag_features, train_test_split_time
    from core.model_selector import get_regressor, get_classifier


class TabTau(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=8)
        self.app = app
        self.key = key
        self.current_df = None
        self.last_fig_path = None

        # left control panel
        ctrl = ttk.Frame(self); ctrl.pack(side="left", fill="y")
        ttk.Button(ctrl, text="Step3 실행", command=self.run_auto).pack(pady=6)

        # right panel: export bar + figure + table
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.export = ExportBar(
            right, self.copy_table, self.copy_md, self.copy_metrics,
            self.save_figure, self.open_folder, self.copy_path
        )
        self.export.pack(fill="x")

        self.figf = FigureFrame(right); self.figf.pack(fill="both", expand=True, pady=4)
        self.table = ttk.Treeview(right, columns=("c1","c2","c3","c4"), show="headings", height=10)
        for i,h in enumerate(["metric","val1","val2","val3"], start=1):
            self.table.heading(f"c{i}", text=h); self.table.column(f"c{i}", width=160, anchor="w")
        self.table.pack(fill="both", expand=True, pady=4)

    # =========================
    # Step3: τ / e(X) 학습
    # =========================
    def run_auto(self):
        """
        τ/e(X) 학습을 수행하고, 과정을 상세 로깅합니다.
        - 디버그 로그: outputs/logs/tab3_tau_debug_*.log
        - 예측 덤프:   outputs/tab3/tau_debug_*.csv
        """
        # 로그 파일 준비
        os.makedirs("outputs/logs", exist_ok=True)
        dbg_fp = f"outputs/logs/tab3_tau_debug_{int(time.time())}.log"

        def _dbg(msg: str):
            line = f"[tab3][{time.strftime('%H:%M:%S')}] {msg}"
            self.app.emit_log(line)
            try:
                with open(dbg_fp, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        # 컨텍스트 점검
        d = self.app.ctx.get("data", {})
        need = ["ret_T1", "ret_T0", "mu1", "mu0", "D1", "D0"]
        missing = [k for k in need if k not in d]
        if missing:
            msg = f"Missing: {missing}. Run Step1 & Step2 first."
            self.app.emit_log(f"[tab3] {msg}")
            try: self.app.status.set(msg, 0.0, warn=True)
            except Exception: pass
            return

        r1, r0 = d["ret_T1"], d["ret_T0"]
        mu1, mu0 = d["mu1"].align(d["mu0"], join="inner")
        D1,  D0  = d["D1"].align(d["D0"],  join="inner")
        idx = mu1.index.intersection(mu0.index).intersection(D1.index).intersection(D0.index)
        if len(idx) == 0:
            msg = "No overlapping index among mu1/mu0/D1/D0."
            self.app.emit_log(f"[tab3] {msg}")
            try: self.app.status.set(msg, 0.0, error=True)
            except Exception: pass
            return

        win   = int(self.app.ctx.get("seq_window", 60))
        stdz  = bool(self.app.ctx.get("standardize", True))
        kind  = self.app.ctx.get("tau_model", "LGBM")

        _dbg(f"params: kind={kind}, win={win}, stdz={stdz}")
        _dbg(f"indices: aligned={len(idx)}, period={idx.min()}→{idx.max()}")

        # ---- 피처: T1/T0 + (옵션) MACRO_* ----
        series_dict = {"ret_T1": r1, "ret_T0": r0}
        if bool(self.app.ctx.get("macro_enable", False)):
            macro = d.get("macro")
            if macro is not None and not macro.empty:
                for col in macro.columns:
                    series_dict[f"MACRO_{col}"] = macro[col]

        X_all = build_lag_features(series_dict, window=win, standardize=stdz)
        X = X_all.reindex(idx).dropna()
        if X.empty:
            msg = "Feature matrix X is empty after alignment/dropna."
            self.app.emit_log(f"[tab3] {msg}")
            try: self.app.status.set(msg, 0.0, error=True)
            except Exception: pass
            return
        D1 = D1.reindex(X.index)
        D0 = D0.reindex(X.index)
        mu_gap = (mu1 - mu0).reindex(X.index)
        _dbg(f"X shape={X.shape}, D1/D0 notna=({D1.notna().sum()}/{D0.notna().sum()})")

        # ---- 시계열 분할 ----
        X1_tr, X1_te, y1_tr, y1_te = train_test_split_time(X, D1)
        X0_tr, X0_te, y0_tr, y0_te = train_test_split_time(X, D0)
        _dbg(f"split: X1_tr={X1_tr.shape}, X1_te={X1_te.shape}, X0_tr={X0_tr.shape}, X0_te={X0_te.shape}")

        # ---- τ 모델 학습 ----
        tau1 = get_regressor(kind, self.app.ctx)
        tau0 = get_regressor(kind, self.app.ctx)
        for nm, m in [("tau1", tau1), ("tau0", tau0)]:
            if hasattr(m, "_fallback_from"):
                self.app.emit_log(f"[tab3] WARNING: requested {m._fallback_from} → fallback to MLPRegressor (TensorFlow not found)")
            else:
                self.app.emit_log(f"[tab3] model[{nm}] = {type(m).__name__}")
        _dbg(f"models: tau1={type(tau1).__name__}, tau0={type(tau0).__name__}")

        t0 = time.time(); tau1.fit(X1_tr, y1_tr); t1 = time.time()
        t2 = time.time(); tau0.fit(X0_tr, y0_tr); t3 = time.time()
        _dbg(f"fit_time: tau1={t1-t0:.3f}s, tau0={t3-t2:.3f}s")

        # ---- τ 예측 ----
        tau1_full = pd.Series(tau1.predict(X), index=X.index, name="tau1")
        tau0_full = pd.Series(tau0.predict(X), index=X.index, name="tau0")
        _dbg(f"pred std: tau1={tau1_full.std():.6f}, tau0={tau0_full.std():.6f}")
        _dbg(f"pred head: tau1={tau1_full.head(3).to_dict()} | tau0={tau0_full.head(3).to_dict()}")

        # ---- e(X) (경향점수) ----
        # 간단 프록시: label=1 if ret_T1 > ret_T0
        y_label = (r1.reindex(X.index) > r0.reindex(X.index)).astype(int)
        clf = get_classifier(kind, self.app.ctx)
        Xc_tr, Xc_te, yl_tr, yl_te = train_test_split_time(X, y_label)
        t4 = time.time(); clf.fit(Xc_tr, yl_tr); t5 = time.time()
        if hasattr(clf, "predict_proba"):
            proba = pd.Series(clf.predict_proba(X)[:, 1], index=X.index, name="e")
        else:
            proba = pd.Series(clf.predict(X), index=X.index, name="e")
        _dbg(f"classifier={type(clf).__name__}, fit_time={t5-t4:.3f}s, e-mean={float(proba.mean()):.4f}, e-std={float(proba.std()):.4f}")

        # ---- 최종 τ(x) = e τ0 + (1-e) τ1 ----
        tau_final = (proba * tau0_full + (1.0 - proba) * tau1_full).rename("tau")

        # ---- 저장 ----
        d["tau1"] = tau1_full
        d["tau0"] = tau0_full
        d["e"]    = proba
        d["tau"]  = tau_final

        # ---- 메트릭 & 테이블 ----
        corr_tau_gap = float(tau_final.corr(mu_gap.loc[tau_final.index])) if not mu_gap.empty else np.nan
        met = pd.DataFrame({
            "metric": ["mean_e", "corr_tau_muGap", "mean_tau"],
            "val1":   [float(proba.mean()), corr_tau_gap, float(tau_final.mean())],
            "val2":   [float(proba.std()),  "",            float(tau_final.std())]
        })
        self.update_table(met)
        corr_str = f"{corr_tau_gap:.4f}" if pd.notna(corr_tau_gap) else "nan"
        _dbg(f"metrics: mean_e={float(proba.mean()):.4f}, std_e={float(proba.std()):.4f}, "
             f"corr_tau_muGap={corr_str}, "
             f"mean_tau={float(tau_final.mean()):.5f}, std_tau={float(tau_final.std()):.5f}")

        # ---- 그림 ----
        self.figf.fig.clear()
        ax = self.figf.fig.add_subplot(111)
        proba.tail(200).plot(ax=ax, label="e(X)")
        tau_final.tail(200).plot(ax=ax, label="τ(x)")
        ax.set_title(f"τ & e(X) (model={kind})"); ax.legend(); ax.grid(True, alpha=0.3)
        self.figf.draw()

        # ---- 디버그 CSV 덤프 ----
        try:
            os.makedirs("outputs/tab3", exist_ok=True)
            debug_df = pd.DataFrame({"tau": tau_final, "tau1": tau1_full, "tau0": tau0_full, "e": proba}) \
                        .dropna().tail(300)
            csv_path = f"outputs/tab3/tau_debug_{int(time.time())}.csv"
            debug_df.to_csv(csv_path)
            _dbg(f"saved debug tau/e: {csv_path}, shape={debug_df.shape}")
        except Exception as e:
            _dbg(f"debug csv save failed: {e}")

        # 완료 로그/상태
        self.app.emit_log("[tab3] tau/e estimated with selected model.")
        try: self.app.status.set("Step3 완료", 1.0, ok=True)
        except Exception: pass

    # =========================
    # Export helpers
    # =========================
    def update_table(self, df):
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
