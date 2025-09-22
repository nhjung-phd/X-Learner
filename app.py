import tkinter as tk
from tkinter import ttk, messagebox
import threading, os, datetime as dt

from gui import TopBar, StatusBar, TabPortfolio, TabData, TabMu, TabCounterf, TabTau, TabPolicy, TabLogs

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("X-Learner GUI Lab")
        self.geometry("1320x900")
        os.makedirs("outputs/logs", exist_ok=True)
        os.makedirs("dist", exist_ok=True)

        # shared context (defaults)
        self.ctx = {
            "T1": ["XLY","XLI","XLF"],
            "T0": ["BTC-USD","ETH-USD"],
            "AUX": ["^GSPC","^VIX"],
            "start": (dt.date.today() - dt.timedelta(days=365*5)).isoformat(),
            "end": dt.date.today().isoformat(),
            "freq": "D",
            "rebal": "M",
            "k_days": 10,
            "wf_splits": 3,
            # model / hyperparams (UI 상단에서 설정)
            "mu_model": "LSTM",
            "tau_model": "LGBM",
            "seq_window": 60,
            "standardize": True,
            "lr": 1e-3,
            "epochs": 20,
            "batch": 64,
            "early_stopping": True,
            "data": {}
        }

        # global log listeners (탭 공통 로그용)
        self._log_listeners = []

        # TopBar (1행: 버튼, 2행: 설정)
        self.topbar = TopBar(self, self)
        self.topbar.pack(fill="x")

        # Notebook 탭 구성
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.tabs = {}
        order = [
            (TabPortfolio, "tabP", "Portfolio"),            # 맨앞
            (TabData, "tab0", "Step 0: Data"),
            (TabMu, "tab1", "Step 1: μ-models"),
            (TabCounterf, "tab2", "Step 2: Counterfactuals"),
            (TabTau, "tab3", "Step 3: τ & e(X)"),
            (TabPolicy, "tab4", "Step 4: CATE & Policy"),
            (TabLogs, "tabL", "로그")                       # 맨뒤
        ]
        for cls, key, text in order:
            tab = cls(self.nb, self, key)
            self.nb.add(tab, text=text)
            self.tabs[key] = tab

        # 상태바
        self.status = StatusBar(self)
        self.status.pack(fill="x", side="bottom")

        # Cancel token
        self.cancel_flag = threading.Event()

    # ---- 글로벌 로그 브로드캐스트 ----
    def add_log_listener(self, fn):
        self._log_listeners.append(fn)
    def emit_log(self, line: str):
        for fn in self._log_listeners:
            try: fn(line)
            except Exception: pass

    # ---- Topbar 콜백 ----
    def on_click_apply(self):
        self.status.set(
            f"Applied: T1={self.ctx['T1']} T0={self.ctx['T0']} | "
            f"{self.ctx['start']}→{self.ctx['end']} | freq={self.ctx['freq']} "
            f"rebal={self.ctx['rebal']} | k={self.ctx['k_days']} splits={self.ctx['wf_splits']}",
            0.0, ok=True
        )

    def on_click_fetch(self):
        self.tabs["tab0"].run_step0()

    def on_click_run_all(self):
        def chain():
            try:
                for k in ["tab0","tab1","tab2","tab3","tab4"]:
                    if self.cancel_flag.is_set(): break
                    self.status.set(f"Running {k} ...", 0.2)
                    self.tabs[k].run_auto()
                self.status.set("Run all completed", 1.0, ok=True)
            except Exception as e:
                self.status.set(f"Run all failed: {e}", 0.0, error=True)
        threading.Thread(target=chain, daemon=True).start()

    def on_click_stop(self):
        self.cancel_flag.set()
        self.status.set("User requested stop", 0.0, warn=True)

    def on_click_export_project(self):
        from core.packaging import create_zip
        try:
            path = create_zip(root=".", out_dir="dist", name_prefix="xlearner_gui")
            self.status.set(f"ZIP created: {path}", 1.0, ok=True)
        except Exception as e:
            self.status.set(f"Export failed: {e}", 0.0, error=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
