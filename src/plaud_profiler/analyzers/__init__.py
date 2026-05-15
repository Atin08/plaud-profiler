"""Analyzer engines — choose based on your setup."""

from __future__ import annotations

from typing import Literal

from .base import AnalyzerBase
from .claude_analyzer import ClaudeAnalyzer
from .nlp_analyzer import NLPAnalyzer
from .ollama_analyzer import OllamaAnalyzer, check_ollama

Engine = Literal["nlp", "ollama", "claude"]


def get_analyzer(engine: Engine, api_key: str | None = None, model: str = "llama3") -> AnalyzerBase:
    if engine == "nlp":
        return NLPAnalyzer()
    if engine == "ollama":
        return OllamaAnalyzer(model=model)
    if engine == "claude":
        return ClaudeAnalyzer(api_key=api_key)
    raise ValueError(f"Unknown engine '{engine}'. Choose: nlp, ollama, claude")


__all__ = ["AnalyzerBase", "NLPAnalyzer", "OllamaAnalyzer", "ClaudeAnalyzer", "get_analyzer", "check_ollama", "Engine"]
