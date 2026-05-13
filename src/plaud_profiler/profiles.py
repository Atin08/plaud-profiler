"""Profile storage — local JSON files. Abstracted so Supabase can be swapped in later."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import AnalysisResult, BigFiveScores, Speaker, SpeakerProfile

_PROFILES_DIR = Path.home() / ".plaud-profiler" / "profiles"


def _profile_path(speaker_id: str) -> Path:
    safe_id = speaker_id.replace("/", "_").replace("\\", "_")
    return _PROFILES_DIR / f"{safe_id}.json"


def load(speaker_id: str) -> Optional[SpeakerProfile]:
    path = _profile_path(speaker_id)
    if not path.exists():
        return None
    return SpeakerProfile.model_validate_json(path.read_text())


def save(profile: SpeakerProfile) -> None:
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.speaker.id)
    path.write_text(profile.model_dump_json(indent=2))


def list_all() -> list[SpeakerProfile]:
    if not _PROFILES_DIR.exists():
        return []
    profiles = []
    for p in _PROFILES_DIR.glob("*.json"):
        try:
            profiles.append(SpeakerProfile.model_validate_json(p.read_text()))
        except Exception:
            continue
    return sorted(profiles, key=lambda p: p.last_updated, reverse=True)


def delete(speaker_id: str) -> bool:
    path = _profile_path(speaker_id)
    if path.exists():
        path.unlink()
        return True
    return False


def upsert_from_result(
    result: AnalysisResult,
    display_name: str,
    recording_id: str,
    merged_scores: Optional[BigFiveScores] = None,
) -> SpeakerProfile:
    existing = load(result.speaker_id)

    recording_ids = list(existing.analyzed_recording_ids) if existing else []
    if recording_id not in recording_ids:
        recording_ids.append(recording_id)

    prior_word_count = existing.speaker.total_word_count if existing else 0
    prior_recordings = existing.speaker.recording_count if existing else 0

    speaker = Speaker(
        id=result.speaker_id,
        display_name=display_name,
        recording_count=prior_recordings + (0 if recording_id in (existing.analyzed_recording_ids if existing else []) else 1),
        total_word_count=prior_word_count,
    )

    scores = merged_scores if merged_scores else result.scores

    profile = SpeakerProfile(
        speaker=speaker,
        scores=scores,
        trait_analyses=result.trait_analyses,
        collaboration_tips=result.collaboration_tips,
        communication_style_summary=result.communication_style_summary,
        last_updated=datetime.now(timezone.utc),
        analyzed_recording_ids=recording_ids,
    )
    save(profile)
    return profile
