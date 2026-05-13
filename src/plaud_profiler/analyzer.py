"""Big Five personality analyzer — uses Claude to score speakers from transcripts."""

from __future__ import annotations

import json
from typing import Optional

import anthropic

from .models import (
    AnalysisRequest,
    AnalysisResult,
    BigFiveScores,
    CollaborationTip,
    SpeakerProfile,
    Trait,
    TraitAnalysis,
    TranscriptSegment,
)

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
  0–20: Very low (bottom quintile)
  21–40: Below average
  41–60: Average range
  61–80: Above average
  81–100: Very high (top quintile)
  50 represents the population mean. Most people score 35–65 on each trait.

CONFIDENCE:
  < 300 words → 0.2–0.3 (low — profile is tentative)
  300–800 words → 0.4–0.6 (moderate)
  > 800 words → 0.7–0.9 (high)
"""


def _build_transcript_text(segments: list[TranscriptSegment]) -> tuple[str, int]:
    lines = [s.text for s in segments]
    full_text = " ".join(lines)
    word_count = len(full_text.split())
    return full_text, word_count


def _build_prompt(request: AnalysisRequest) -> str:
    transcript_text, word_count = _build_transcript_text(request.transcript_segments)

    prior_context = ""
    if request.existing_profile:
        p = request.existing_profile
        prior_context = f"""
PRIOR PROFILE (from {len(p.analyzed_recording_ids)} previous recording(s)):
  Openness: {p.scores.openness:.0f}
  Conscientiousness: {p.scores.conscientiousness:.0f}
  Extraversion: {p.scores.extraversion:.0f}
  Agreeableness: {p.scores.agreeableness:.0f}
  Neuroticism: {p.scores.neuroticism:.0f}
  Prior summary: {p.communication_style_summary}

Weight this new data alongside the prior — update scores proportionally to the
ratio of new words to total cumulative words.
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

Return ONLY valid JSON matching this exact structure — no markdown, no commentary:
{{
  "speaker_id": "{request.speaker_id}",
  "scores": {{
    "openness": <0-100>,
    "conscientiousness": <0-100>,
    "extraversion": <0-100>,
    "agreeableness": <0-100>,
    "neuroticism": <0-100>
  }},
  "trait_analyses": [
    {{
      "trait": "openness",
      "score": <0-100>,
      "confidence": <0.0-1.0>,
      "evidence_quotes": ["<verbatim quote from transcript>"],
      "linguistic_markers": ["<pattern observed>"],
      "interpretation": "<one sentence plain-English>"
    }},
    ... (repeat for all 5 traits)
  ],
  "collaboration_tips": [
    {{
      "trait": "openness",
      "tip": "<concise actionable suggestion>",
      "rationale": "<cite the Big Five research finding>",
      "approach_do": ["<specific behavior to adopt>"],
      "approach_avoid": ["<behavior likely to create friction>"]
    }},
    ... (repeat for all 5 traits)
  ],
  "communication_style_summary": "<2-3 sentences describing how this person communicates>"
}}"""


class Analyzer:
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

        raw_json = message.content[0].text.strip()
        # Strip any accidental markdown fences
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]

        data = json.loads(raw_json)
        return _parse_result(data)

    def merge_with_existing(
        self, new_result: AnalysisResult, existing: SpeakerProfile
    ) -> BigFiveScores:
        """Weighted average: existing N recordings vs 1 new recording."""
        n = len(existing.analyzed_recording_ids)
        w_old, w_new = n / (n + 1), 1 / (n + 1)

        def blend(old: float, new: float) -> float:
            return round(old * w_old + new * w_new, 1)

        return BigFiveScores(
            openness=blend(existing.scores.openness, new_result.scores.openness),
            conscientiousness=blend(existing.scores.conscientiousness, new_result.scores.conscientiousness),
            extraversion=blend(existing.scores.extraversion, new_result.scores.extraversion),
            agreeableness=blend(existing.scores.agreeableness, new_result.scores.agreeableness),
            neuroticism=blend(existing.scores.neuroticism, new_result.scores.neuroticism),
        )


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
