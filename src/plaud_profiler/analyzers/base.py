"""Abstract base class shared by all analyzer engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AnalysisRequest, AnalysisResult, BigFiveScores, SpeakerProfile


class AnalyzerBase(ABC):

    @abstractmethod
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyse a speaker's transcript segments and return a Big Five result."""
        ...

    def merge_with_existing(
        self, new_result: AnalysisResult, existing: SpeakerProfile
    ) -> BigFiveScores:
        """Weighted average of existing N recordings vs 1 new recording."""
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
