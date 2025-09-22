# gui/views_topbar.py  — REPLACE WHOLE FILE
import tkinter as tk
from tkinter import ttk

class TopBar(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app

        # =========================
        # Row 1: 버튼
        # =========================
        row1 = ttk.Frame(self); row1.pack(fill="x", pady=(0,4))
        ttk.Button(row1, text="적용", command=self.apply_changes).pack(side="left", padx=4)
        ttk.Button(row1, text="데이터 받기", command=self.app.on_click_fetch).pack(side="left", padx=4)
        ttk.Button(row1, text="모두 실행 ▶", command=self.app.on_click_run_all).pack(side="left", padx=4)
        ttk.Button(row1, text="중단 ■", command=self.app.on_click_stop).pack(side="left", padx=4)
        ttk.Button(row1, text="프로젝트 내보내기 ⬇", command=self.app.on_click_export_project).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(row1, mode="determinate", maximum=1.0, value=0.0, length=260)
        self.progress.pack(side="right", padx=6)

        # =========================
        # Row 2: 기간/주기/리밸/k/splits
        # =========================
        row2 = ttk.Frame(self); row2.pack(fill="x", pady=(0,4))
        # 기간
        ttk.Label(row2, text="기간 From:").pack(side="left")
        self.start_var = tk.StringVar(value=app.ctx["start"])
        ttk.Entry(row2, width=10, textvariable=self.start_var).pack(side="left", padx=2)

        ttk.Label(row2, text="To:").pack(side="left")
        self.end_var = tk.StringVar(value=app.ctx["end"])
        ttk.Entry(row2, width=10, textvariable=self.end_var).pack(side="left", padx=4)

        # 주기 / 리밸런싱
        ttk.Label(row2, text="주기").pack(side="left", padx=(8,2))
        self.freq_var = tk.StringVar(value=app.ctx["freq"])
        ttk.Combobox(row2, width=4, textvariable=self.freq_var, values=["D","W","M"]).pack(side="left")

        ttk.Label(row2, text="리밸").pack(side="left", padx=(8,2))
        self.rebal_var = tk.StringVar(value=app.ctx["rebal"])
        ttk.Combobox(row2, width=4, textvariable=self.rebal_var, values=["W","M"]).pack(side="left")

        # k-일 타깃 슬라이더
        ttk.Label(row2, text="k-일 타깃").pack(side="left", padx=(8,2))
        self.k_var = tk.IntVar(value=self.app.ctx["k_days"])
        ttk.Scale(row2, from_=1, to=60, orient="horizontal", variable=self.k_var, length=120).pack(side="left")

        # splits
        ttk.Label(row2, text="WF splits").pack(side="left", padx=(8,2))
        self.splits_var = tk.IntVar(value=self.app.ctx["wf_splits"])
        ttk.Spinbox(row2, from_=2, to=10, width=4, textvariable=self.splits_var).pack(side="left")

        # =========================
        # Row 3: 모델 선택 (μ, τ)
        # =========================
        row3 = ttk.Frame(self); row3.pack(fill="x", pady=(0,4))

        ttk.Label(row3, text="μ-model").pack(side="left", padx=(2,2))
        self.mu_var = tk.StringVar(value=self.app.ctx.get("mu_model", "LSTM"))
        self.mu_combo = ttk.Combobox(row3, width=10, textvariable=self.mu_var,
                                     values=["CNN","LSTM","GRU","RNN","LGBM","XGB"], state="readonly")
        self.mu_combo.pack(side="left")

        # τ 동일 사용 여부 + τ 전용 선택
        self.same_tau_var = tk.BooleanVar(
            value=(self.app.ctx.get("tau_model", self.app.ctx.get("mu_model","LSTM")) == self.app.ctx.get("mu_model","LSTM"))
        )
        ttk.Checkbutton(row3, text="τ도 동일", variable=self.same_tau_var, command=self._sync_tau_state).pack(side="left", padx=8)

        ttk.Label(row3, text="τ-model").pack(side="left", padx=(8,2))
        self.tau_var = tk.StringVar(value=self.app.ctx.get("tau_model", self.mu_var.get()))
        self.tau_combo = ttk.Combobox(row3, width=10, textvariable=self.tau_var,
                                      values=["CNN","LSTM","GRU","RNN","LGBM","XGB"], state="readonly")
        self.tau_combo.pack(side="left")

        # 초기 상태 동기화
        self._sync_tau_state(init=True)

        # =========================
        # Row 4: 하이퍼파라미터 (윈도우/표준화/lr/epochs/batch/조기종료)
        # =========================
        row4 = ttk.Frame(self); row4.pack(fill="x", pady=(0,4))
        ttk.Label(row4, text="win").pack(side="left", padx=(2,2))
        self.win_var = tk.IntVar(value=self.app.ctx.get("seq_window", 60))
        ttk.Spinbox(row4, from_=5, to=256, width=5, textvariable=self.win_var).pack(side="left")

        self.std_var = tk.BooleanVar(value=self.app.ctx.get("standardize", True))
        ttk.Checkbutton(row4, text="표준화", variable=self.std_var).pack(side="left", padx=6)

        ttk.Label(row4, text="lr").pack(side="left")
        self.lr_var = tk.DoubleVar(value=float(self.app.ctx.get("lr", 1e-3)))
        ttk.Spinbox(row4, from_=1e-5, to=1e-1, increment=1e-4, width=7, textvariable=self.lr_var).pack(side="left")

        ttk.Label(row4, text="epochs").pack(side="left")
        self.epochs_var = tk.IntVar(value=int(self.app.ctx.get("epochs", 20)))
        ttk.Spinbox(row4, from_=1, to=500, width=5, textvariable=self.epochs_var).pack(side="left")

        ttk.Label(row4, text="batch").pack(side="left")
        self.batch_var = tk.IntVar(value=int(self.app.ctx.get("batch", 64)))
        ttk.Spinbox(row4, from_=8, to=1024, width=5, textvariable=self.batch_var).pack(side="left")

        self.es_var = tk.BooleanVar(value=self.app.ctx.get("early_stopping", True))
        ttk.Checkbutton(row4, text="조기종료", variable=self.es_var).pack(side="left", padx=6)

        # =========================
        # Row 5: Macro 옵션 (Use Macro / Lag / Norm)
        # =========================
        row5 = ttk.Frame(self); row5.pack(fill="x", pady=(0,0))

        self.var_macro = tk.IntVar(value=int(self.app.ctx.get("macro_enable", 0)))
        self.var_macro_lag = tk.IntVar(value=int(self.app.ctx.get("macro_lag_days", 5)))
        self.var_macro_norm = tk.StringVar(value=self.app.ctx.get("macro_norm", "z"))  # 'z' or 'pct'

        ttk.Checkbutton(row5, text="Use Macro", variable=self.var_macro).pack(side="left", padx=(2,6))

        ttk.Label(row5, text="Macro Lag(d)").pack(side="left")
        ttk.Spinbox(row5, from_=0, to=20, width=5, textvariable=self.var_macro_lag).pack(side="left", padx=4)

        ttk.Label(row5, text="Norm").pack(side="left")
        ttk.Combobox(row5, width=6, state="readonly",
                     values=["z","pct"], textvariable=self.var_macro_norm).pack(side="left", padx=4)

    # ---- 내부: τ 동기화 및 상태 토글 ----
    def _sync_tau_state(self, init: bool=False):
        same = bool(self.same_tau_var.get())
        # 동일 사용이면 τ 콤보 비활성화 + 값 동기화
        state = "disabled" if same else "readonly"
        try:
            self.tau_combo.configure(state=state)
        except Exception:
            pass
        if same or init:
            self.tau_var.set(self.mu_var.get())
        # μ 변경 시 τ도 자동 반영 (동일 체크된 경우)
        if not hasattr(self, "_mu_trace"):
            self._mu_trace = self.mu_var.trace_add("write", lambda *args: self._on_mu_change())

    def _on_mu_change(self):
        if bool(self.same_tau_var.get()):
            self.tau_var.set(self.mu_var.get())

    # ---- 적용 ----
    def apply_changes(self):
        # Row2
        self.app.ctx["start"] = self.start_var.get().strip()
        self.app.ctx["end"] = self.end_var.get().strip()
        self.app.ctx["freq"] = self.freq_var.get().strip()
        self.app.ctx["rebal"] = self.rebal_var.get().strip()
        self.app.ctx["k_days"] = int(self.k_var.get())
        self.app.ctx["wf_splits"] = int(self.splits_var.get())

        # Row3
        self.app.ctx["mu_model"] = self.mu_var.get().strip()
        if bool(self.same_tau_var.get()):
            self.app.ctx["tau_model"] = self.app.ctx["mu_model"]
        else:
            self.app.ctx["tau_model"] = self.tau_var.get().strip()

        # Row4
        self.app.ctx["seq_window"] = int(self.win_var.get())
        self.app.ctx["standardize"] = bool(self.std_var.get())
        self.app.ctx["lr"] = float(self.lr_var.get())
        self.app.ctx["epochs"] = int(self.epochs_var.get())
        self.app.ctx["batch"] = int(self.batch_var.get())
        self.app.ctx["early_stopping"] = bool(self.es_var.get())

        # Row5 (Macro)
        self.app.ctx["macro_enable"] = bool(self.var_macro.get())
        self.app.ctx["macro_lag_days"] = int(self.var_macro_lag.get())
        self.app.ctx["macro_norm"] = self.var_macro_norm.get()

        # 상태 표시/콜백
        self.app.on_click_apply()
