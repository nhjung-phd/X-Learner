
"""Core package for X-Learner Lab.

Provides packaging helpers and (future) ML components.
Generated: 2025-09-21 05:58:06
"""
from __future__ import annotations

__version__ = "0.2.1"

# Re-exports (extend as real implementations are added)
from .packaging import create_zip  # noqa: F401

__all__ = ["create_zip"]
