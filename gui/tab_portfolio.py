import tkinter as tk
from tkinter import ttk
import json


class TabPortfolio(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=10)
        self.app = app; self.key = key

        left = ttk.Frame(self); left.pack(side="left", fill="y", padx=6)
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="T1 (경기민감) 티커, 콤마로 구분").pack(anchor="w")
        self.t1 = tk.Text(left, height=5, width=40)
        self.t1.insert("1.0", ",".join(app.ctx["T1"])); self.t1.pack(pady=4)

        ttk.Label(left, text="T0 (경기둔감) 티커, 콤마로 구분").pack(anchor="w")
        self.t0 = tk.Text(left, height=5, width=40)
        self.t0.insert("1.0", ",".join(app.ctx["T0"])); self.t0.pack(pady=4)

        btns = ttk.Frame(left); btns.pack(anchor="w", pady=6)
        ttk.Button(btns, text="적용", command=self.apply).pack(side="left", padx=4)
        ttk.Button(btns, text="JSON 저장", command=self.save_json).pack(side="left", padx=4)
        ttk.Button(btns, text="JSON 불러오기", command=self.load_json).pack(side="left", padx=4)

        self.preview = tk.Text(right, height=18)
        self.preview.pack(fill="both", expand=True)

        self.refresh_preview()

    def apply(self):
        T1 = [x.strip() for x in self.t1.get("1.0","end").split(",") if x.strip()]
        T0 = [x.strip() for x in self.t0.get("1.0","end").split(",") if x.strip()]
        self.app.ctx["T1"] = T1; self.app.ctx["T0"] = T0
        self.app.emit_log(f"[포트폴리오] T1={T1} T0={T0}")
        self.app.status.set("포트폴리오 적용 완료", 0.0, ok=True)
        self.refresh_preview()

    def refresh_preview(self):
        ctx = self.app.ctx
        txt = json.dumps(
            {
                "T1": ctx["T1"], "T0": ctx["T0"],
                "기간": [ctx["start"], ctx["end"]],
                "freq": ctx["freq"], "rebal": ctx["rebal"],
                "k": ctx["k_days"], "splits": ctx["wf_splits"],
                "mu": ctx["mu_model"], "tau": ctx["tau_model"]
            }, ensure_ascii=False, indent=2
        )
        self.preview.delete("1.0","end"); self.preview.insert("1.0", txt)

    def save_json(self):
        import os, time
        os.makedirs("configs", exist_ok=True)
        path = f"configs/baskets_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(path,"w",encoding="utf-8") as f:
            json.dump({"T1":self.app.ctx["T1"],"T0":self.app.ctx["T0"]}, f, ensure_ascii=False, indent=2)
        self.app.emit_log(f"[포트폴리오] 저장: {path}")

    def load_json(self):
        import tkinter.filedialog as fd, json
        fp = fd.askopenfilename(title="Select baskets json", filetypes=[("JSON","*.json")])
        if not fp: return
        try:
            with open(fp,"r",encoding="utf-8") as f:
                d = json.load(f)
            self.t1.delete("1.0","end"); self.t1.insert("1.0", ",".join(d.get("T1",[])))
            self.t0.delete("1.0","end"); self.t0.insert("1.0", ",".join(d.get("T0",[])))
            self.apply()
        except Exception as e:
             self.app.emit_log(f"[포트폴리오] 불러오기 실패: {e}")
             self.app.status.set(f"불러오기 실패: {e}", 0.0, error=True)
