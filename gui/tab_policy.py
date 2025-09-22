# gui/tab_policy.py  — REPLACE WHOLE FILE
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import os, time, pandas as pd, numpy as np

# allow both package and standalone execution
try:
    from .widgets import ExportBar, FigureFrame
    from core.backtest import backtest_policy, regime_breakdown
except Exception:  # running as a script
    import os as _os, sys as _sys
    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from gui.widgets import ExportBar, FigureFrame
    from core.backtest import backtest_policy, regime_breakdown


class TabPolicy(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=8)
        self.app = app
        self.key = key
        self.current_df = None
        self.last_fig_path = None

        # left control panel
        ctrl = ttk.Frame(self); ctrl.pack(side="left", fill="y", padx=4)
        ttk.Button(ctrl, text="Step4 실행", command=self.run_auto).pack(pady=6)

        # right panel: export bar + figure + table
        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.export = ExportBar(
            right, self.copy_table, self.copy_md, self.copy_metrics,
            self.save_figure, self.open_folder, self.copy_path
        )
        self.export.pack(fill="x")

        self.figf = FigureFrame(right); self.figf.pack(fill="both", expand=True, pady=4)

        # 넓은 표: metric/value + 연도별 성과/레짐 테이블까지 함께 표시될 수 있도록 가변 사용
        cols = [f"c{i}" for i in range(1, 9)]
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=12)
        for i, h in enumerate(["metric","value","Y2019","Y2020","Y2021","Y2022","Y2023","Y2024"], start=1):
            self.table.heading(f"c{i}", text=h); self.table.column(f"c{i}", width=130, anchor="w")
        self.table.pack(fill="both", expand=True, pady=4)

    # =========================
    # 핵심 실행
    # =========================
    def run_auto(self):
        d = self.app.ctx.get("data", {})
        need = ["ret_T1","ret_T0","tau"]
        missing = [k for k in need if k not in d]
        if missing:
            self.app.emit_log(f"[tab4] Need {missing}. Run Step1-3 first."); return

        r1, r0, tau = d["ret_T1"], d["ret_T0"], d["tau"]
        # 옵션(상단 TopBar에서 ctx로 내려오는 값)
        cost_bps = float(self.app.ctx.get("cost_bps", 5.0))
        min_hold = int(self.app.ctx.get("min_hold", 1))

        # 백테스트
        bt = backtest_policy(r1, r0, tau, cost_bps=cost_bps, min_hold=min_hold)
        if "error" in bt:
            self.app.emit_log("[tab4] " + bt["error"]); return

        # ===== 표: Turnover/Sharpe(net)/Sortino(net)/MDD/연도별 성과
        yr = bt["annual_strat"]  # Series: {year: return}
        # 가용 연도를 자동 확장
        year_cols = {f"Y{int(y)}": float(v) for y, v in yr.items()}
        metrics_rows = [
            {"metric":"Sharpe(strat,gross)",  "value": bt["sharpe_gross"]},
            {"metric":"Sharpe(strat,net)",    "value": bt["sharpe_net"]},
            {"metric":"Sortino(strat,gross)", "value": bt["sortino_gross"]},
            {"metric":"Sortino(strat,net)",   "value": bt["sortino_net"]},
            {"metric":"Sharpe(T1)",           "value": bt["sharpe_T1"]},
            {"metric":"Sharpe(T0)",           "value": bt["sharpe_T0"]},
            {"metric":"Turnover(daily avg)",  "value": bt["turnover"]},
            {"metric":"MDD(strat,net)",       "value": bt["mdd"]},
        ]
        row_year = {"metric":"Annual(strat,net)"}
        row_year.update(year_cols)
        metrics_rows.append(row_year)
        met_df = pd.DataFrame(metrics_rows)

        # ===== 레짐 테이블(VIX/금리) — ctx에 있으면 계산
        vix   = d.get("vix", None)    # Tab0에서 ^VIX를 저장했다면 사용
        dgs10 = d.get("dgs10", None)  # FRED DGS10 저장 시 사용
        reg = regime_breakdown(bt["ret_strat_net"], bt["ret_T1"], bt["ret_T0"], vix=vix, dgs10=dgs10)
        if not reg.empty:
            # 보기 좋게 컬럼 이름 정리
            reg_disp = reg.rename(columns={"strat":"strat_mean","T1":"T1_mean","T0":"T0_mean"})
            reg_disp = reg_disp[["metric","Regime","strat_mean","T1_mean","T0_mean"]]
            # 기존 메트릭 표 아래로 일부(최대 20행) 이어붙여 보여주기
            merged = pd.concat([met_df, reg_disp.head(20)], axis=0, ignore_index=True)
        else:
            merged = met_df

        self.update_table(merged.fillna(""))

        # ===== 그림: 3단 (1) 누적수익(넷) (2) 드로다운 (3) 포지션 히트맵
        self._draw_three_panel(bt)

        # 레짐 CSV 저장(있을 때만)
        if not reg.empty:
            try:
                os.makedirs("outputs/tab4", exist_ok=True)
                reg_path = f"outputs/tab4/regime_{int(time.time())}.csv"
                reg.to_csv(reg_path, index=False)
                self.app.emit_log(f"[tab4] Regime table saved: {reg_path}")
            except Exception as e:
                self.app.emit_log(f"[tab4] Regime save failed: {e}")

        self.app.emit_log("Policy backtest done.")
        self.app.status.set("Step4 완료", 1.0, ok=True)

    # =========================
    # 시각화
    # =========================
    def _draw_three_panel(self, bt: dict):
        import matplotlib.pyplot as plt  # lazy import for Tk backend stability
        self.figf.fig.clear()
        gs = self.figf.fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 0.8], hspace=0.25)

        # (1) 누적수익(넷)
        ax1 = self.figf.fig.add_subplot(gs[0,0])
        bt["cum_strat"].plot(ax=ax1, label="Strat (net)")
        bt["cum_T1"].plot(ax=ax1, label="All T1")
        bt["cum_T0"].plot(ax=ax1, label="All T0")
        ax1.set_title("Cumulative Returns (net)")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # (2) 드로다운
        ax2 = self.figf.fig.add_subplot(gs[1,0], sharex=ax1)
        bt["drawdown"].plot(ax=ax2, color="tab:red")
        ax2.set_title("Drawdown (Strat)")
        ax2.grid(True, alpha=0.3)

        # (3) 포지션 히트맵 (0/1)
        ax3 = self.figf.fig.add_subplot(gs[2,0], sharex=ax1)
        pos = bt["pos"].astype(int)
        ax3.imshow(pos.values.reshape(1,-1), aspect="auto", interpolation="nearest")
        ax3.set_yticks([0]); ax3.set_yticklabels(["pos (1=T1,0=T0)"])
        ax3.set_title("Position Heatmap")
        ax3.grid(False)

        self.figf.draw()

    # =========================
    # 표/Export 도우미
    # =========================
    def update_table(self, df: pd.DataFrame):
        self.current_df = df
        # 컬럼 헤더를 df에 맞추어 유연하게 갱신
        cols = list(df.columns)
        # Treeview를 재구성(간단하게는 기존 컬럼 수를 유지한 채 헤더/값만 맞춤)
        # 8개보다 많으면 앞부분만 보여주기(필요시 확장 가능)
        view_cols = cols[:8]
        self.table["columns"] = [f"c{i}" for i in range(1, len(view_cols)+1)]
        for i, h in enumerate(view_cols, start=1):
            if f"c{i}" not in self.table["columns"]:
                self.table["columns"] = (*self.table["columns"], f"c{i}")
            self.table.heading(f"c{i}", text=h)
            self.table.column(f"c{i}", width=140, anchor="w")
        # 기존 행 제거 후 삽입
        for r in self.table.get_children(): self.table.delete(r)
        for _, row in df[view_cols].iterrows():
            vals = [row.get(c, "") for c in view_cols]
            self.table.insert("", "end", values=vals)

    # Export & Copy handlers
    def copy_table(self):
        if self.current_df is None:
            self.app.emit_log("[tab4] No table to copy"); return
        txt = self.current_df.to_csv(index=False)
        self.clipboard_clear(); self.clipboard_append(txt); self.update()
        self.app.emit_log("[tab4] Copied table")

    def copy_md(self):
        rows = len(self.current_df) if self.current_df is not None else 0
        md = (
            f"### {self.key} Summary\n\n"
            f"Rows: {rows}\n"
            f"T1={self.app.ctx['T1']} | T0={self.app.ctx['T0']} | "
            f"{self.app.ctx['start']}→{self.app.ctx['end']}\n"
            f"cost_bps={self.app.ctx.get('cost_bps', 5.0)}, min_hold={self.app.ctx.get('min_hold',1)}"
        )
        self.clipboard_clear(); self.clipboard_append(md); self.update()
        self.app.emit_log("[tab4] Copied markdown")

    def copy_metrics(self):
        if self.current_df is None:
            self.app.emit_log("[tab4] No metrics to copy"); return
        # metric/value 쌍만 우선 추출
        cols = [c for c in self.current_df.columns if c in ("metric","value")]
        if not cols: cols = list(self.current_df.columns[:2])
        text = self.current_df[cols].to_csv(index=False)
        self.clipboard_clear(); self.clipboard_append(text); self.update()
        self.app.emit_log("[tab4] Copied metrics")

    def save_figure(self):
        os.makedirs(f"outputs/{self.key}", exist_ok=True)
        path = f"outputs/{self.key}/{int(time.time())}_figure.png"
        self.figf.fig.savefig(path, dpi=150, bbox_inches="tight")
        self.last_fig_path = path
        self.app.emit_log(f"[tab4] Saved figure: {path}")

    def open_folder(self):
        import platform, subprocess, os as _os
        folder = _os.path.abspath(f"outputs/{self.key}"); _os.makedirs(folder, exist_ok=True)
        cmd = ["open", folder] if platform.system()=="Darwin" else ["explorer", folder] if platform.system()=="Windows" else ["xdg-open", folder]
        try: subprocess.Popen(cmd)
        except Exception as e: self.app.emit_log(f"[tab4] open_folder failed: {e}")

    def copy_path(self):
        if not self.last_fig_path:
            self.app.emit_log("[tab4] No saved figure yet"); return
        self.clipboard_clear(); self.clipboard_append(self.last_fig_path); self.update()
        self.app.emit_log("[tab4] Copied path")
