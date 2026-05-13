"""MCP client for plaud.ai — wraps the @plaud-ai/mcp Node.js server via stdio."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .models import Recording, TranscriptSegment


class PlaudClient:
    """Async context manager that talks to the local Plaud MCP server."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    # ── Auth ─────────────────────────────────────────────────────────────────

    async def login(self) -> str:
        result = await self._call("login", {})
        return _text(result)

    async def get_current_user(self) -> dict[str, Any]:
        result = await self._call("get_current_user", {})
        return _parse(result)

    # ── Recordings ───────────────────────────────────────────────────────────

    async def list_recordings(
        self,
        query: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[Recording]:
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        result = await self._call("list_files", params)
        raw = _parse(result)
        files = raw if isinstance(raw, list) else raw.get("files", raw.get("items", []))
        return [_parse_recording(f) for f in files]

    async def get_recording(self, file_id: str) -> Recording:
        result = await self._call("get_file", {"file_id": file_id})
        return _parse_recording(_parse(result))

    async def get_transcript(self, file_id: str) -> list[TranscriptSegment]:
        result = await self._call("get_transcript", {"file_id": file_id})
        raw = _parse(result)
        segments = raw if isinstance(raw, list) else raw.get("segments", raw.get("transcript", []))
        return [_parse_segment(s) for s in segments]

    async def get_note(self, file_id: str) -> dict[str, Any]:
        result = await self._call("get_note", {"file_id": file_id})
        return _parse(result)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _call(self, tool: str, params: dict[str, Any]) -> Any:
        return await self._session.call_tool(tool, params)


# ── Context manager factory ───────────────────────────────────────────────────

@asynccontextmanager
async def plaud_client() -> AsyncIterator[PlaudClient]:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@plaud-ai/mcp@latest"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield PlaudClient(session)


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _text(result: Any) -> str:
    if hasattr(result, "content"):
        for block in result.content:
            if hasattr(block, "text"):
                return block.text
    return str(result)


def _parse(result: Any) -> Any:
    text = _text(result)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _parse_recording(raw: dict[str, Any]) -> Recording:
    created_raw = raw.get("created_at") or raw.get("createdAt") or raw.get("create_time", "")
    try:
        created_at = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        created_at = datetime.utcnow()

    return Recording(
        id=str(raw.get("id", "")),
        name=str(raw.get("name") or raw.get("title", "Untitled")),
        created_at=created_at,
        duration_seconds=float(raw.get("duration") or raw.get("duration_seconds", 0)),
        speaker_ids=raw.get("speaker_ids", []),
    )


def _parse_segment(raw: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        speaker_id=str(raw.get("speaker") or raw.get("speaker_id") or raw.get("speakerId", "unknown")),
        text=str(raw.get("text") or raw.get("content", "")),
        start_seconds=float(raw.get("start") or raw.get("start_time") or raw.get("startSeconds", 0)),
        end_seconds=float(raw.get("end") or raw.get("end_time") or raw.get("endSeconds", 0)),
    )
