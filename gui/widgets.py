
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class StatusBar(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.msg = tk.StringVar(value="Ready")
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=1.0, value=0.0)
        ttk.Label(self, textvariable=self.msg).pack(side="left", padx=6, pady=4)
        self.progress.pack(side="right", fill="x", expand=True, padx=6, pady=4)
    def set(self, text, pct=0.0, ok=False, warn=False, error=False):
        self.msg.set(text); self.progress["value"]=pct

class LogConsole(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.text = scrolledtext.ScrolledText(self, height=7, wrap="word")
        self.text.pack(fill="both", expand=True)
    def log(self, line):
        self.text.insert("end", line.rstrip()+"\n"); self.text.see("end")

class ExportBar(ttk.Frame):
    def __init__(self, master, on_copy_table, on_copy_md, on_copy_metrics, on_save_fig, on_open_folder, on_copy_path=None):
        super().__init__(master)
        ttk.Button(self, text="Copy Table", command=on_copy_table).pack(side="left", padx=2)
        ttk.Button(self, text="Copy Markdown", command=on_copy_md).pack(side="left", padx=2)
        ttk.Button(self, text="Copy Metrics", command=on_copy_metrics).pack(side="left", padx=2)
        ttk.Button(self, text="Save Figure", command=on_save_fig).pack(side="left", padx=2)
        ttk.Button(self, text="Open Folder", command=on_open_folder).pack(side="left", padx=2)
        if on_copy_path:
            ttk.Button(self, text="Copy Path", command=on_copy_path).pack(side="left", padx=2)

class FigureFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.fig = Figure(figsize=(6,3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
    def draw(self): self.canvas.draw()
