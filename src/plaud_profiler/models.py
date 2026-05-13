"""Pydantic models — source of truth is specs/openapi.yaml."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Trait(str, Enum):
    openness = "openness"
    conscientiousness = "conscientiousness"
    extraversion = "extraversion"
    agreeableness = "agreeableness"
    neuroticism = "neuroticism"


class Speaker(BaseModel):
    id: str
    display_name: str
    recording_count: int = 0
    total_word_count: int = 0


class Recording(BaseModel):
    id: str
    name: str
    created_at: datetime
    duration_seconds: float
    speaker_ids: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    speaker_id: str
    text: str
    start_seconds: float
    end_seconds: float = 0.0


class TraitAnalysis(BaseModel):
    trait: Trait
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quotes: list[str] = Field(min_length=1)
    linguistic_markers: list[str]
    interpretation: str


class BigFiveScores(BaseModel):
    openness: float = Field(ge=0, le=100)
    conscientiousness: float = Field(ge=0, le=100)
    extraversion: float = Field(ge=0, le=100)
    agreeableness: float = Field(ge=0, le=100)
    neuroticism: float = Field(ge=0, le=100)


class CollaborationTip(BaseModel):
    trait: Trait
    tip: str
    rationale: str
    approach_do: list[str] = Field(min_length=1)
    approach_avoid: list[str] = Field(min_length=1)


class SpeakerProfile(BaseModel):
    speaker: Speaker
    scores: BigFiveScores
    trait_analyses: list[TraitAnalysis] = Field(min_length=5, max_length=5)
    collaboration_tips: list[CollaborationTip] = Field(min_length=5, max_length=5)
    communication_style_summary: str
    last_updated: datetime
    analyzed_recording_ids: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    speaker_id: str
    display_name: str
    transcript_segments: list[TranscriptSegment]
    recording_id: str
    existing_profile: Optional[SpeakerProfile] = None


class AnalysisResult(BaseModel):
    speaker_id: str
    scores: BigFiveScores
    trait_analyses: list[TraitAnalysis]
    collaboration_tips: list[CollaborationTip]
    communication_style_summary: str
