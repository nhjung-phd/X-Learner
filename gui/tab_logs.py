import tkinter as tk
from tkinter import ttk
from .widgets import LogConsole

class TabLogs(ttk.Frame):
    def __init__(self, master, app, key):
        super().__init__(master, padding=8)
        self.app = app
        self.key = key

        # 공통 로그 콘솔
        self.logc = LogConsole(self)
        self.logc.pack(fill="both", expand=True)

        # App의 글로벌 로그 브로드캐스트 구독
        self.app.add_log_listener(self.logc.log)

        # 유틸 바
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Copy", command=self.copy).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left", padx=4)

    def copy(self):
        txt = self.logc.text.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.update()

    def clear(self):
        self.logc.text.delete("1.0", "end")
