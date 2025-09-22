from __future__ import annotations
import importlib
from typing import Any, List

__version__ = "0.3.0"

__all__: List[str] = [
    "TopBar","StatusBar","LogConsole","ExportBar","FigureFrame",
    "TabPortfolio","TabData","TabMu","TabCounterf","TabTau","TabPolicy","TabLogs"
]

_alias = {
    "TopBar": ("gui.views_topbar", "TopBar"),
    "StatusBar": ("gui.widgets", "StatusBar"),
    "LogConsole": ("gui.widgets", "LogConsole"),
    "ExportBar": ("gui.widgets", "ExportBar"),
    "FigureFrame": ("gui.widgets", "FigureFrame"),
    "TabPortfolio": ("gui.tab_portfolio", "TabPortfolio"),
    "TabData": ("gui.tab_data", "TabData"),
    "TabMu": ("gui.tab_mu", "TabMu"),
    "TabCounterf": ("gui.tab_counterf", "TabCounterf"),
    "TabTau": ("gui.tab_tau", "TabTau"),
    "TabPolicy": ("gui.tab_policy", "TabPolicy"),
    "TabLogs": ("gui.tab_logs", "TabLogs"),
}

def __getattr__(name: str) -> Any:
    if name in _alias:
        mod, attr = _alias[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(name)

def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_alias.keys()))
