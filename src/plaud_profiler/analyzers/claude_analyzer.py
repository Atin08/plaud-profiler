"""Claude (Anthropic API) analyzer — highest quality, requires ANTHROPIC_API_KEY."""

from __future__ import annotations

import json
from typing import Optional

import anthropic

from ..models import (
    AnalysisRequest,
    AnalysisResult,
    BigFiveScores,
    CollaborationTip,
    Trait,
    TraitAnalysis,
    TranscriptSegment,
)
from .base import AnalyzerBase

# Linguistic markers grounded in Mairesse et al. (2007) and Pennebaker & King (1999)
_TRAIT_GUIDE = """
OPENNESS (O): abstract/complex vocabulary, intellectual topics, hypotheticals, metaphors,
  references to arts/ideas, varied sentence structure, curiosity language.
  High O → creative, exploratory. Low O → concrete, conventional, practical.

CONSCIENTIOUSNESS (C): precise ordered language, plans/goals/deadlines, self-discipline markers
  ("I will", "I need to", "by Thursday"), cautiousness, thoroughness, checking/verifying.
  High C → reliable, structured. Low C → spontaneous, flexible, disorganised.

EXTRAVERSION (E): high word count, positive emotion words, social references ("people",
  "we", "everyone"), action verbs, enthusiasm, talking about experiences.
  High E → sociable, energetic. Low E → reserved, reflective, prefers depth over breadth.

AGREEABLENESS (A): cooperative language ("together", "let's", "what do you think"),
  hedging/softening ("maybe", "perhaps"), warmth markers, conflict avoidance, empathy signals.
  High A → warm, accommodating. Low A → direct, competitive, sceptical.

NEUROTICISM (N): negative emotion words, anxiety/worry language, self-referential + negative
  affect, uncertainty/doubt patterns, stress indicators, catastrophising.
  High N → emotionally reactive, stress-prone. Low N → stable, calm, resilient.

SCORING SCALE (0–100):
  0–20: Very low | 21–40: Below average | 41–60: Average | 61–80: Above average | 81–100: Very high
  50 = population mean. Most people score 35–65 on each trait.

CONFIDENCE:
  < 300 words → 0.2–0.3 (low)  |  300–800 words → 0.4–0.6 (moderate)  |  > 800 words → 0.7–0.9 (high)
"""


def _build_prompt(request: AnalysisRequest) -> str:
    transcript_text = " ".join(s.text for s in request.transcript_segments)
    word_count = len(transcript_text.split())

    prior_context = ""
    if request.existing_profile:
        p = request.existing_profile
        prior_context = f"""
PRIOR PROFILE (from {len(p.analyzed_recording_ids)} previous recording(s)):
  O={p.scores.openness:.0f}  C={p.scores.conscientiousness:.0f}  E={p.scores.extraversion:.0f}
  A={p.scores.agreeableness:.0f}  N={p.scores.neuroticism:.0f}
  Prior summary: {p.communication_style_summary}

Weight this new data alongside the prior proportional to word-count ratio.
"""

    return f"""You are a psycholinguistic analyst applying the Big Five (OCEAN) personality model.
Analyse the transcript below for speaker "{request.display_name}" and return a JSON object.

SCIENTIFIC FRAMEWORK:
{_TRAIT_GUIDE}

{prior_context}

TRANSCRIPT ({word_count} words from recording {request.recording_id}):
\"\"\"
{transcript_text}
\"\"\"

Return ONLY valid JSON — no markdown, no commentary:
{{
  "speaker_id": "{request.speaker_id}",
  "scores": {{"openness": 0-100, "conscientiousness": 0-100, "extraversion": 0-100, "agreeableness": 0-100, "neuroticism": 0-100}},
  "trait_analyses": [
    {{
      "trait": "openness",
      "score": 0-100,
      "confidence": 0.0-1.0,
      "evidence_quotes": ["verbatim quote"],
      "linguistic_markers": ["pattern observed"],
      "interpretation": "one sentence plain-English"
    }}
    ... repeat for all 5 traits
  ],
  "collaboration_tips": [
    {{
      "trait": "openness",
      "tip": "concise actionable suggestion",
      "rationale": "cite Big Five research finding",
      "approach_do": ["specific behavior to adopt"],
      "approach_avoid": ["behavior likely to create friction"]
    }}
    ... repeat for all 5 traits
  ],
  "communication_style_summary": "2-3 sentences describing how this person communicates"
}}"""


def _parse_result(data: dict) -> AnalysisResult:
    scores = BigFiveScores(**data["scores"])

    trait_analyses = [
        TraitAnalysis(
            trait=Trait(t["trait"]),
            score=t["score"],
            confidence=t["confidence"],
            evidence_quotes=t["evidence_quotes"],
            linguistic_markers=t.get("linguistic_markers", []),
            interpretation=t["interpretation"],
        )
        for t in data["trait_analyses"]
    ]

    collaboration_tips = [
        CollaborationTip(
            trait=Trait(c["trait"]),
            tip=c["tip"],
            rationale=c["rationale"],
            approach_do=c.get("approach_do", [c.get("tip", "")]),
            approach_avoid=c.get("approach_avoid", []),
        )
        for c in data["collaboration_tips"]
    ]

    return AnalysisResult(
        speaker_id=data["speaker_id"],
        scores=scores,
        trait_analyses=trait_analyses,
        collaboration_tips=collaboration_tips,
        communication_style_summary=data["communication_style_summary"],
    )


class ClaudeAnalyzer(AnalyzerBase):
    """Uses Claude claude-opus-4-7 via Anthropic API for highest-quality Big Five analysis."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        prompt = _build_prompt(request)

        message = self._client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=(
                "You are a scientific psycholinguistic analyst. "
                "Respond only with valid JSON. Never explain or add commentary."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return _parse_result(json.loads(raw))
