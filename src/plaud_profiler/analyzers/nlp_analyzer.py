"""Rule-based Big Five analyzer — no API key required.

Scoring is grounded in Mairesse et al. (2007) LIWC feature correlates and
Pennebaker & King (1999) linguistic style research. Each trait has a set of
positive markers (raise score) and negative markers (lower score). Frequencies
are normalised to a 0-100 scale centred on 50 (population average).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import NamedTuple

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

# ── Word lists (Mairesse et al. 2007; Pennebaker & King 1999) ─────────────────

_MARKERS: dict[Trait, dict[str, list[str]]] = {
    Trait.openness: {
        "positive": [
            "abstract", "aesthetic", "artistic", "complex", "concept", "create",
            "creative", "curious", "diverse", "explore", "fantasy", "hypothesis",
            "idea", "imagine", "innovative", "insight", "intellectual", "introspect",
            "invent", "novel", "original", "philosophy", "poetic", "reflect",
            "sophisticated", "theoretical", "unique", "unusual", "wonder", "broad",
            "culture", "experience", "variety", "metaphor", "perspective", "vision",
            "creative", "imagine", "dream", "possibility", "potential", "discover",
            "experiment", "question", "challenge", "think", "understand", "meaning",
            "beauty", "art", "music", "literature", "history", "science",
        ],
        "negative": [
            "routine", "simple", "basic", "traditional", "conventional", "standard",
            "normal", "regular", "typical", "usual", "practical", "concrete",
            "straightforward", "obvious", "clear-cut",
        ],
    },
    Trait.conscientiousness: {
        "positive": [
            "accomplish", "accurate", "careful", "complete", "deadline", "detail",
            "diligent", "efficient", "finish", "goal", "methodical", "orderly",
            "organize", "plan", "precise", "prepare", "prioritize", "productive",
            "responsible", "schedule", "systematic", "thorough", "structured",
            "disciplined", "reliable", "consistent", "deliberate", "focused",
            "intentional", "task", "objective", "achieve", "track", "review",
            "verify", "checklist", "agenda", "committed", "follow", "through",
            "always", "ensure", "measure", "standard", "quality", "process",
            "step", "procedure", "document", "record",
        ],
        "negative": [
            "forgot", "late", "miss", "disorganized", "messy", "random",
            "spontaneous", "impulsive", "lazy", "procrastinate", "delay", "scatter",
        ],
    },
    Trait.extraversion: {
        "positive": [
            "active", "adventure", "assert", "communicate", "confident", "energetic",
            "enthusiastic", "exciting", "expressive", "fun", "gregarious", "interact",
            "lively", "party", "people", "positive", "social", "talkative", "vibrant",
            "bold", "cheerful", "outgoing", "engaging", "dynamic", "network",
            "connect", "share", "celebrate", "everyone", "together", "group",
            "team", "friends", "join", "meet", "invite", "gather", "discuss",
            "collaborate", "talk", "chat", "tell", "say", "speak", "mention",
            "announce", "present", "show", "demonstrate",
        ],
        "negative": [
            "alone", "quiet", "solitary", "private", "reserved", "introverted",
            "prefer", "independent", "individual", "solo", "myself",
        ],
    },
    Trait.agreeableness: {
        "positive": [
            "accommodate", "agree", "care", "collaborate", "compassion", "cooperate",
            "empathy", "flexible", "forgive", "gentle", "generous", "helpful",
            "kind", "patient", "supportive", "sympathetic", "trust", "understanding",
            "warm", "willing", "mutual", "respect", "appreciate", "consider",
            "listen", "nurture", "compromise", "together", "we", "us", "our",
            "certainly", "absolutely", "happy", "glad", "please", "welcome",
            "appreciate", "thank", "grateful",
        ],
        "hedging": [
            "maybe", "perhaps", "possibly", "might", "could", "think",
            "feel", "believe", "seems", "sort of", "kind of", "probably",
            "I guess", "I suppose", "not sure", "I wonder",
        ],
        "negative": [
            "no", "never", "refuse", "reject", "disagree", "wrong", "bad",
            "stupid", "ridiculous", "absolutely not", "impossible", "demand",
            "insist", "must", "force", "argue", "fight", "compete",
        ],
    },
    Trait.neuroticism: {
        "positive": [
            "afraid", "angry", "anxious", "bitter", "depressed", "desperate",
            "distressed", "fearful", "frustrated", "guilty", "hopeless", "hurt",
            "insecure", "irritable", "jealous", "lonely", "nervous", "overwhelmed",
            "panicked", "pessimistic", "stressed", "tense", "uncomfortable",
            "unhappy", "unstable", "upset", "vulnerable", "worried", "moody",
            "sensitive", "emotional", "reactive", "dread", "concern", "worry",
            "problem", "difficult", "hard", "impossible", "terrible", "awful",
            "hate", "can't", "cannot", "never work", "fail", "wrong",
        ],
        "negative": [
            "calm", "stable", "confident", "relaxed", "secure", "comfortable",
            "fine", "okay", "good", "great", "excellent", "happy", "pleased",
            "content", "settled", "steady", "balanced", "resilient",
        ],
    },
}

# Pre-written collaboration tips per score tier (low/mid/high) — grounded in research
_TIPS: dict[Trait, dict[str, dict]] = {
    Trait.openness: {
        "high": {
            "tip": "Engage their curiosity — bring ideas, not just answers",
            "rationale": "High-O individuals are motivated by novelty and intellectual stimulation (McCrae & Costa, 1997)",
            "do": ["Frame problems as interesting puzzles", "Invite speculation and 'what if' thinking", "Share unconventional approaches"],
            "avoid": ["Presenting only one rigid solution", "Dismissing ideas as impractical too early", "Over-relying on precedent"],
        },
        "mid": {
            "tip": "Balance new ideas with practical grounding",
            "rationale": "Mid-range O individuals appreciate innovation within familiar frameworks",
            "do": ["Connect new ideas to existing processes", "Test concepts before full commitment"],
            "avoid": ["Pure abstraction without clear application", "Resisting any new approach"],
        },
        "low": {
            "tip": "Lead with proven methods and clear precedent",
            "rationale": "Low-O individuals prefer familiar, tested approaches (Goldberg, 1992)",
            "do": ["Reference what has worked before", "Present step-by-step plans", "Emphasise reliability"],
            "avoid": ["Proposing radical changes", "Abstract theorising", "Frequent pivots"],
        },
    },
    Trait.conscientiousness: {
        "high": {
            "tip": "Respect their structure — be prepared and on time",
            "rationale": "High-C individuals value reliability and thoroughness (Barrick & Mount, 1991)",
            "do": ["Send agendas before meetings", "Follow up in writing", "Meet every deadline"],
            "avoid": ["Last-minute changes", "Vague commitments", "Skipping process steps"],
        },
        "mid": {
            "tip": "Offer structure but stay flexible on execution",
            "rationale": "Mid-C individuals appreciate organisation without rigidity",
            "do": ["Set clear goals with some flexibility in method", "Check in at milestones"],
            "avoid": ["Micromanaging the process", "Completely open-ended assignments"],
        },
        "low": {
            "tip": "Keep things loose and outcome-focused, not process-driven",
            "rationale": "Low-C individuals work best with autonomy and big-picture goals (Costa & McCrae, 1992)",
            "do": ["Define clear outcomes, not steps", "Allow creative problem-solving", "Avoid rigid timelines"],
            "avoid": ["Heavy process and paperwork", "Detailed status reports", "Rigid check-ins"],
        },
    },
    Trait.extraversion: {
        "high": {
            "tip": "Engage them verbally and in group settings",
            "rationale": "High-E individuals gain energy from social interaction (Eysenck, 1967)",
            "do": ["Think out loud together", "Involve them in group discussions early", "Give real-time verbal feedback"],
            "avoid": ["Long solo written tasks", "Isolation from team conversations", "Delayed feedback"],
        },
        "mid": {
            "tip": "Mix solo and social contexts based on the task",
            "rationale": "Moderate-E individuals are flexible across interaction styles",
            "do": ["Offer both group and independent work options", "Read their energy in the moment"],
            "avoid": ["Assuming they always want group work", "Assuming they prefer to work alone"],
        },
        "low": {
            "tip": "Give them space to think before responding",
            "rationale": "Low-E (introverted) individuals process internally and prefer depth (Cain, 2012)",
            "do": ["Share agendas before meetings", "Allow written responses as an option", "Respect quiet focus time"],
            "avoid": ["Hot-seat questions in group settings", "Expecting instant answers", "Over-scheduling social events"],
        },
    },
    Trait.agreeableness: {
        "high": {
            "tip": "Create psychological safety — they avoid conflict by default",
            "rationale": "High-A individuals suppress disagreement to preserve harmony (Jensen-Campbell et al., 2003)",
            "do": ["Explicitly ask for their honest opinion", "Make it safe to disagree", "Acknowledge their contributions warmly"],
            "avoid": ["Confrontational debate styles", "Assuming silence means agreement", "Pressuring quick decisions"],
        },
        "mid": {
            "tip": "Direct communication works — they can handle both warmth and candour",
            "rationale": "Mid-A individuals balance cooperation with assertiveness effectively",
            "do": ["Be straightforward", "Acknowledge their perspective before pivoting"],
            "avoid": ["Being unnecessarily blunt or dismissive"],
        },
        "low": {
            "tip": "Match their directness — they respect candour over politeness",
            "rationale": "Low-A individuals are sceptical and competitive; they interpret hedging as weakness (Graziano & Eisenberg, 1997)",
            "do": ["Be direct and confident in your position", "Use evidence and logic", "Welcome debate as productive"],
            "avoid": ["Over-apologising", "Seeking excessive consensus", "Vague or softened feedback"],
        },
    },
    Trait.neuroticism: {
        "high": {
            "tip": "Reduce ambiguity — uncertainty amplifies their stress response",
            "rationale": "High-N individuals are more reactive to stressors and threats (Clark & Watson, 1999)",
            "do": ["Communicate clearly and early about changes", "Acknowledge concerns before problem-solving", "Provide reassurance with facts"],
            "avoid": ["Leaving things open-ended", "Springing surprises", "Dismissing concerns as overreaction"],
        },
        "mid": {
            "tip": "Acknowledge concerns while maintaining a solution focus",
            "rationale": "Mid-N individuals are moderately emotionally reactive but manageable",
            "do": ["Validate feelings first, then redirect to action", "Check in during stressful periods"],
            "avoid": ["Pure logic with no emotional acknowledgement"],
        },
        "low": {
            "tip": "Stay matter-of-fact — they prefer direct problem-solving over emotional processing",
            "rationale": "Low-N individuals are emotionally stable and resilient (Costa & McCrae, 1992)",
            "do": ["Discuss problems objectively", "Move quickly to solutions", "Skip excessive reassurance"],
            "avoid": ["Over-emotional framing", "Dwelling on feelings before facts"],
        },
    },
}


def _tier(score: float) -> str:
    if score >= 65:
        return "high"
    if score >= 35:
        return "mid"
    return "low"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _find_evidence_sentences(segments: list[TranscriptSegment], markers: list[str], n: int = 2) -> list[str]:
    """Return the n sentences containing the most marker words."""
    sentences: list[tuple[int, str]] = []
    for seg in segments:
        for sent in re.split(r"[.!?]+", seg.text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            words = set(_tokenize(sent))
            hits = sum(1 for m in markers if m in words or any(m in w for w in words))
            if hits > 0:
                sentences.append((hits, sent))
    sentences.sort(reverse=True)
    return [s for _, s in sentences[:n]] or ["(insufficient evidence in transcript)"]


def _score_trait(
    words: list[str],
    positive: list[str],
    negative: list[str],
    hedging: list[str] | None = None,
) -> tuple[float, list[str]]:
    total = max(len(words), 1)
    pos_hits = [w for w in words if w in positive]
    neg_hits = [w for w in words if w in negative]
    hed_hits = []
    if hedging:
        text = " ".join(words)
        hed_hits = [h for h in hedging if h in text]

    pos_rate = (len(pos_hits) + len(hed_hits) * 0.5) / total * 100
    neg_rate = len(neg_hits) / total * 100

    # Sensitivity: empirically tuned so typical conversation maps to 35-65 range
    raw = pos_rate * 8 - neg_rate * 6
    score = max(10.0, min(90.0, 50.0 + raw))
    found_markers = list(set(pos_hits[:5] + hed_hits[:3]))
    return round(score, 1), found_markers


def _confidence(word_count: int) -> float:
    if word_count < 300:
        return round(0.2 + (word_count / 300) * 0.2, 2)
    if word_count < 800:
        return round(0.4 + ((word_count - 300) / 500) * 0.3, 2)
    return min(0.9, round(0.7 + (word_count - 800) / 5000, 2))


class NLPAnalyzer(AnalyzerBase):
    """Scores Big Five traits using LIWC-grounded word-frequency analysis. No API key needed."""

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        all_text = " ".join(s.text for s in request.transcript_segments)
        words = _tokenize(all_text)
        word_count = len(words)
        conf = _confidence(word_count)

        trait_analyses: list[TraitAnalysis] = []
        scores: dict[str, float] = {}

        for trait in Trait:
            markers = _MARKERS[trait]
            pos = markers["positive"]
            neg = markers["negative"]
            hed = markers.get("hedging")

            score, found = _score_trait(words, pos, neg, hed)
            scores[trait.value] = score

            evidence = _find_evidence_sentences(request.transcript_segments, pos + (hed or []))
            tier = _tier(score)
            tip_data = _TIPS[trait][tier]

            interpretation = _interpretation(trait, score)

            trait_analyses.append(TraitAnalysis(
                trait=trait,
                score=score,
                confidence=conf,
                evidence_quotes=evidence,
                linguistic_markers=found if found else ["(no strong markers detected)"],
                interpretation=interpretation,
            ))

        collaboration_tips: list[CollaborationTip] = []
        for trait in Trait:
            tier = _tier(scores[trait.value])
            td = _TIPS[trait][tier]
            collaboration_tips.append(CollaborationTip(
                trait=trait,
                tip=td["tip"],
                rationale=td["rationale"],
                approach_do=td["do"],
                approach_avoid=td["avoid"],
            ))

        summary = _build_summary(request.display_name, scores)

        return AnalysisResult(
            speaker_id=request.speaker_id,
            scores=BigFiveScores(**scores),
            trait_analyses=trait_analyses,
            collaboration_tips=collaboration_tips,
            communication_style_summary=summary,
        )


def _interpretation(trait: Trait, score: float) -> str:
    tier = _tier(score)
    descriptions = {
        Trait.openness: {
            "high": f"Highly open to ideas and experience (score {score:.0f}) — thinks broadly and values novelty.",
            "mid": f"Moderate openness (score {score:.0f}) — balances curiosity with practicality.",
            "low": f"Prefers concrete, familiar approaches (score {score:.0f}) — values proven methods over experimentation.",
        },
        Trait.conscientiousness: {
            "high": f"Highly conscientious (score {score:.0f}) — structured, reliable, and goal-oriented.",
            "mid": f"Moderate conscientiousness (score {score:.0f}) — organised but adaptable.",
            "low": f"Low conscientiousness (score {score:.0f}) — flexible and spontaneous, may struggle with rigid structure.",
        },
        Trait.extraversion: {
            "high": f"Highly extraverted (score {score:.0f}) — energised by social interaction and verbal thinking.",
            "mid": f"Ambiverted (score {score:.0f}) — comfortable in both social and independent settings.",
            "low": f"Introverted (score {score:.0f}) — prefers depth over breadth, needs time to process before responding.",
        },
        Trait.agreeableness: {
            "high": f"Highly agreeable (score {score:.0f}) — warm, cooperative, and conflict-averse.",
            "mid": f"Moderate agreeableness (score {score:.0f}) — balances warmth with assertiveness.",
            "low": f"Low agreeableness (score {score:.0f}) — direct, sceptical, and comfortable with debate.",
        },
        Trait.neuroticism: {
            "high": f"High neuroticism (score {score:.0f}) — emotionally reactive; stress and ambiguity feel amplified.",
            "mid": f"Moderate emotional reactivity (score {score:.0f}) — generally stable with occasional stress responses.",
            "low": f"Low neuroticism (score {score:.0f}) — emotionally stable and resilient under pressure.",
        },
    }
    return descriptions[trait][tier]


def _build_summary(name: str, scores: dict[str, float]) -> str:
    o, c, e, a, n = (
        scores["openness"], scores["conscientiousness"],
        scores["extraversion"], scores["agreeableness"], scores["neuroticism"],
    )
    parts = []
    parts.append(f"{name} communicates in a {'wide-ranging, idea-driven' if o >= 65 else 'grounded, practical' if o <= 35 else 'balanced'} way.")
    parts.append(
        f"They tend to be {'structured and deliberate' if c >= 65 else 'flexible and spontaneous' if c <= 35 else 'moderately organised'} "
        f"and {'energised by group interaction' if e >= 65 else 'more reflective and reserved' if e <= 35 else 'comfortable in both solo and group settings'}."
    )
    parts.append(
        f"Collaboration with them works best when you {'respect their need for harmony and explicitly invite candour' if a >= 65 else 'match their directness and back positions with evidence' if a <= 35 else 'balance warmth with clarity'}"
        f"{' and minimise ambiguity to reduce stress' if n >= 65 else ' — they handle pressure steadily' if n <= 35 else ''}."
    )
    return " ".join(parts)
