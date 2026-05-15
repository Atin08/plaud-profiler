"""CLI entry point — `plaud-profiler` command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm

from . import profiles as profile_store
from . import reporter
from .analyzers import Engine, check_ollama, get_analyzer
from .models import AnalysisRequest
from .plaud_client import plaud_client

app = typer.Typer(
    name="plaud-profiler",
    help="Big Five personality profiles from your Plaud.ai recordings.",
    no_args_is_help=True,
)
console = Console()

# Engine option reused across commands
_ENGINE_OPTION = typer.Option(
    "nlp",
    "--engine", "-e",
    help="Analysis engine: [nlp] rule-based (no key), [ollama] local LLM (no key), [claude] Anthropic API",
    show_default=True,
)
_MODEL_OPTION = typer.Option(
    "llama3",
    "--model", "-m",
    help="Ollama model name (only used when --engine ollama)",
    show_default=True,
)
_API_KEY_OPTION = typer.Option(
    None,
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    help="Anthropic API key (only needed for --engine claude)",
)


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
    speaker: Optional[str] = typer.Option(None, "--speaker", "-s", help="Only analyse this speaker label"),
    engine: str = _ENGINE_OPTION,
    model: str = _MODEL_OPTION,
    api_key: Optional[str] = _API_KEY_OPTION,
):
    """Analyse a recording and update speaker personality profiles.

    Choose your engine:\n
      --engine nlp     Rule-based NLP (no API key, works offline)\n
      --engine ollama  Local LLM via Ollama (no API key, needs Ollama running)\n
      --engine claude  Claude API (best quality, needs ANTHROPIC_API_KEY)
    """
    _validate_engine(engine, api_key)

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

    analyzer = get_analyzer(engine, api_key=api_key, model=model)
    _print_engine_banner(engine, model)

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
        console.print("[green]Profile saved.[/green]")

    console.print(f"\n[dim]Run 'plaud-profiler profile <speaker-id>' for full details and tips.[/dim]")


# ── Profiles ──────────────────────────────────────────────────────────────────

@app.command(name="profiles")
def list_profiles():
    """List all saved speaker profiles."""
    reporter.print_profiles_list(profile_store.list_all())


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
def engines():
    """Show available analysis engines and their status."""
    console.print("\n[bold]Available engines:[/bold]\n")

    console.print("[cyan]nlp[/cyan]    Rule-based NLP — always available, no setup needed")
    console.print("         Grounded in Mairesse et al. (2007) LIWC linguistic markers\n")

    try:
        models = check_ollama()
        model_list = ", ".join(models) if models else "(no models pulled yet — run: ollama pull llama3)"
        console.print(f"[cyan]ollama[/cyan] Local LLM — [green]Ollama is running[/green]")
        console.print(f"         Available models: {model_list}\n")
    except RuntimeError as e:
        console.print(f"[cyan]ollama[/cyan] Local LLM — [red]not available[/red]")
        console.print(f"         {e}")
        console.print("         Install: https://ollama.ai  then: ollama pull llama3\n")

    import os
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    console.print(f"[cyan]claude[/cyan] Anthropic Claude API — {'[green]API key set[/green]' if has_key else '[yellow]needs ANTHROPIC_API_KEY[/yellow]'}")
    console.print("         Best quality analysis. Get a key: https://console.anthropic.com\n")


@app.command()
def delete(
    speaker_id: str = typer.Argument(help="Speaker ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a speaker's profile."""
    if not yes:
        if not Confirm.ask(f"Delete profile for '{speaker_id}'?"):
            raise typer.Abort()

    if profile_store.delete(speaker_id):
        console.print(f"[green]Deleted profile for '{speaker_id}'.[/green]")
    else:
        console.print(f"[red]No profile found for '{speaker_id}'.[/red]")
        raise typer.Exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_engine(engine: str, api_key: Optional[str]) -> None:
    if engine not in ("nlp", "ollama", "claude"):
        console.print(f"[red]Unknown engine '{engine}'. Choose: nlp, ollama, claude[/red]")
        raise typer.Exit(1)
    if engine == "claude" and not api_key:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[red]--engine claude requires ANTHROPIC_API_KEY to be set.[/red]")
            console.print("[dim]Try --engine nlp (no key needed) or --engine ollama (local LLM).[/dim]")
            raise typer.Exit(1)
    if engine == "ollama":
        try:
            check_ollama()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)


def _print_engine_banner(engine: str, model: str) -> None:
    labels = {
        "nlp": "[dim]Engine: rule-based NLP (Mairesse et al. LIWC markers)[/dim]",
        "ollama": f"[dim]Engine: Ollama local LLM ({model})[/dim]",
        "claude": "[dim]Engine: Claude API (claude-opus-4-7)[/dim]",
    }
    console.print(labels.get(engine, ""))
