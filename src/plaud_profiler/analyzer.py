"""Backward-compatibility shim — use analyzers/ package directly."""

from .analyzers import ClaudeAnalyzer as Analyzer, get_analyzer  # noqa: F401
