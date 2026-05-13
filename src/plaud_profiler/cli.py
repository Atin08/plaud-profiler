"""CLI entry point — `plaud-profiler` command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm

from . import analyzer as analyze_module
from . import profiles as profile_store
from . import reporter
from .models import AnalysisRequest
from .plaud_client import plaud_client

app = typer.Typer(
    name="plaud-profiler",
    help="Big Five personality profiles from your Plaud.ai recordings.",
    no_args_is_help=True,
)
console = Console()


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.command()
def login():
    """Log in to your Plaud.ai account."""
    async def _run():
        async with plaud_client() as client:
            msg = await client.login()
            console.print(f"[green]{msg}[/green]")
    asyncio.run(_run())


@app.command()
def whoami():
    """Show the currently logged-in Plaud.ai account."""
    async def _run():
        async with plaud_client() as client:
            user = await client.get_current_user()
            console.print(user)
    asyncio.run(_run())


# ── Recordings ────────────────────────────────────────────────────────────────

@app.command(name="recordings")
def list_recordings(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Keyword filter"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date YYYY-MM-DD"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date YYYY-MM-DD"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """List your Plaud.ai recordings."""
    async def _run():
        async with plaud_client() as client:
            recs = await client.list_recordings(
                query=query, date_from=from_date, date_to=to_date, limit=limit
            )
        if not recs:
            console.print("[dim]No recordings found.[/dim]")
            return
        reporter.print_recordings_table(recs)
    asyncio.run(_run())


# ── Analysis ──────────────────────────────────────────────────────────────────

@app.command()
def analyze(
    recording_id: str = typer.Argument(help="Plaud recording ID to analyse"),
    speaker: Optional[str] = typer.Option(
        None, "--speaker", "-s",
        help="Only analyse this speaker label (e.g. 'Speaker 1')",
    ),
    anthropic_api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key",
    ),
):
    """Analyse a recording and update speaker personality profiles."""
    async def _fetch():
        async with plaud_client() as client:
            segments = await client.get_transcript(recording_id)
            rec = await client.get_recording(recording_id)
        return segments, rec

    segments, rec = asyncio.run(_fetch())

    if not segments:
        console.print("[red]No transcript found for this recording.[/red]")
        raise typer.Exit(1)

    # Group segments by speaker
    by_speaker: dict[str, list] = {}
    for seg in segments:
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    if speaker:
        if speaker not in by_speaker:
            console.print(f"[red]Speaker '{speaker}' not found. Available: {list(by_speaker.keys())}[/red]")
            raise typer.Exit(1)
        by_speaker = {speaker: by_speaker[speaker]}

    analyzer = analyze_module.Analyzer(api_key=anthropic_api_key)

    for speaker_id, segs in by_speaker.items():
        word_count = sum(len(s.text.split()) for s in segs)
        console.print(f"\n[cyan]Analysing[/cyan] [bold]{speaker_id}[/bold] ({word_count} words)…")

        existing = profile_store.load(speaker_id)

        request = AnalysisRequest(
            speaker_id=speaker_id,
            display_name=speaker_id,
            transcript_segments=segs,
            recording_id=recording_id,
            existing_profile=existing,
        )

        result = analyzer.analyze(request)

        merged_scores = None
        if existing:
            merged_scores = analyzer.merge_with_existing(result, existing)

        profile = profile_store.upsert_from_result(
            result=result,
            display_name=speaker_id,
            recording_id=recording_id,
            merged_scores=merged_scores,
        )

        reporter.print_profile_summary(profile)
        console.print(f"[green]Profile saved.[/green]")

    console.print(f"\n[dim]Run 'plaud-profiler profile <speaker-id>' for full details.[/dim]")


# ── Profiles ──────────────────────────────────────────────────────────────────

@app.command(name="profiles")
def list_profiles():
    """List all saved speaker profiles."""
    all_profiles = profile_store.list_all()
    reporter.print_profiles_list(all_profiles)


@app.command()
def profile(
    speaker_id: str = typer.Argument(help="Speaker ID to display"),
    tips: bool = typer.Option(True, "--tips/--no-tips", help="Show collaboration tips"),
):
    """Show the full Big Five profile for a speaker."""
    p = profile_store.load(speaker_id)
    if not p:
        console.print(f"[red]No profile found for '{speaker_id}'.[/red]")
        console.print("[dim]Run 'plaud-profiler analyze <recording-id>' first.[/dim]")
        raise typer.Exit(1)

    reporter.print_profile_summary(p)
    if tips:
        reporter.print_collaboration_tips(p)


@app.command()
def report(
    speaker_id: str = typer.Argument(help="Speaker ID to export"),
    output: Path = typer.Option(Path("."), "--output", "-o", help="Output directory"),
):
    """Export a speaker's profile as a markdown report."""
    p = profile_store.load(speaker_id)
    if not p:
        console.print(f"[red]No profile found for '{speaker_id}'.[/red]")
        raise typer.Exit(1)

    path = reporter.export_markdown(p, output_dir=output)
    console.print(f"[green]Report written to[/green] {path}")


@app.command()
def delete(
    speaker_id: str = typer.Argument(help="Speaker ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a speaker's profile."""
    if not yes:
        confirmed = Confirm.ask(f"Delete profile for '{speaker_id}'?")
        if not confirmed:
            raise typer.Abort()

    if profile_store.delete(speaker_id):
        console.print(f"[green]Deleted profile for '{speaker_id}'.[/green]")
    else:
        console.print(f"[red]No profile found for '{speaker_id}'.[/red]")
        raise typer.Exit(1)
