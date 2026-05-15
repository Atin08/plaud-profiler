"""Ollama local LLM analyzer — no API key, runs entirely on your machine.

Requires Ollama to be installed and running: https://ollama.ai
Recommended models: llama3, mistral, phi3 (balance of speed and quality).

Install a model:  ollama pull llama3
Start server:     ollama serve  (or the Ollama desktop app)
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from ..models import (
    AnalysisRequest,
    AnalysisResult,
    BigFiveScores,
    CollaborationTip,
    Trait,
    TraitAnalysis,
)
from .base import AnalyzerBase
from .claude_analyzer import _TRAIT_GUIDE, _build_prompt, _parse_result  # reuse prompt + parser

_OLLAMA_BASE = "http://localhost:11434"
_DEFAULT_MODEL = "llama3"


def check_ollama() -> list[str]:
    """Return list of locally available model names, or raise if Ollama isn't running."""
    try:
        resp = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except httpx.ConnectError:
        raise RuntimeError(
            "Ollama is not running. Start it with `ollama serve` or open the Ollama app."
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


class OllamaAnalyzer(AnalyzerBase):
    """Sends Big Five analysis prompts to a locally running Ollama model."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        prompt = _build_prompt(request)

        payload = {
            "model": self.model,
            "prompt": (
                "You are a scientific psycholinguistic analyst. "
                "Respond only with valid JSON. Never explain or add commentary.\n\n"
                + prompt
            ),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 4096,
            },
        }

        try:
            resp = httpx.post(
                f"{_OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                "Ollama is not running. Start it with `ollama serve` or open the Ollama app."
            )

        raw = resp.json().get("response", "")
        if not raw:
            raise RuntimeError(f"Ollama returned an empty response for model '{self.model}'.")

        # Strip accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Ollama returned invalid JSON: {e}\n\nRaw response:\n{raw[:500]}"
            )

        return _parse_result(data)
