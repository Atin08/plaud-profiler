"""Rich terminal output and markdown report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .models import BigFiveScores, CollaborationTip, SpeakerProfile, Trait, TraitAnalysis

console = Console()

_TRAIT_EMOJI = {
    Trait.openness: "O",
    Trait.conscientiousness: "C",
    Trait.extraversion: "E",
    Trait.agreeableness: "A",
    Trait.neuroticism: "N",
}

_TRAIT_COLOR = {
    Trait.openness: "magenta",
    Trait.conscientiousness: "blue",
    Trait.extraversion: "yellow",
    Trait.agreeableness: "green",
    Trait.neuroticism: "red",
}


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score:.0f}"


def _confidence_label(conf: float) -> str:
    if conf < 0.4:
        return "[dim]low[/dim]"
    if conf < 0.7:
        return "[yellow]moderate[/yellow]"
    return "[green]high[/green]"


def print_profile_summary(profile: SpeakerProfile) -> None:
    console.print()
    console.print(Panel(
        f"[bold]{profile.speaker.display_name}[/bold]\n"
        f"[dim]{len(profile.analyzed_recording_ids)} recording(s) analysed · "
        f"last updated {profile.last_updated.strftime('%Y-%m-%d')}[/dim]\n\n"
        f"{profile.communication_style_summary}",
        title="[bold cyan]Speaker Profile[/bold cyan]",
        border_style="cyan",
    ))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Trait", style="bold", width=20)
    table.add_column("Score", width=28)
    table.add_column("Confidence", width=12)
    table.add_column("Interpretation")

    for ta in profile.trait_analyses:
        color = _TRAIT_COLOR[ta.trait]
        table.add_row(
            f"[{color}]{_TRAIT_EMOJI[ta.trait]} {ta.trait.value.title()}[/{color}]",
            f"[{color}]{_score_bar(ta.score)}[/{color}]",
            _confidence_label(ta.confidence),
            ta.interpretation,
        )

    console.print(table)


def print_collaboration_tips(profile: SpeakerProfile) -> None:
    console.print()
    console.print("[bold cyan]Collaboration Tips[/bold cyan]")
    for tip in profile.collaboration_tips:
        color = _TRAIT_COLOR[tip.trait]
        console.print(Panel(
            f"[bold]{tip.tip}[/bold]\n\n"
            f"[dim italic]{tip.rationale}[/dim italic]\n\n"
            f"[green]DO:[/green] {', '.join(tip.approach_do)}\n"
            f"[red]AVOID:[/red] {', '.join(tip.approach_avoid)}",
            title=f"[{color}]{tip.trait.value.title()}[/{color}]",
            border_style=color,
        ))


def print_recordings_table(recordings: list) -> None:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", width=4)
    table.add_column("ID", style="dim", width=12)
    table.add_column("Name")
    table.add_column("Date", width=12)
    table.add_column("Duration", width=10)

    for i, rec in enumerate(recordings, 1):
        mins = int(rec.duration_seconds // 60)
        secs = int(rec.duration_seconds % 60)
        table.add_row(
            str(i),
            rec.id[:10] + "…" if len(rec.id) > 10 else rec.id,
            rec.name,
            rec.created_at.strftime("%Y-%m-%d"),
            f"{mins}m {secs:02d}s",
        )

    console.print(table)


def print_profiles_list(profiles: list[SpeakerProfile]) -> None:
    if not profiles:
        console.print("[dim]No profiles found. Run 'plaud-profiler analyze <recording-id>' to create one.[/dim]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Speaker")
    table.add_column("O", width=6)
    table.add_column("C", width=6)
    table.add_column("E", width=6)
    table.add_column("A", width=6)
    table.add_column("N", width=6)
    table.add_column("Recordings", width=11)
    table.add_column("Updated", width=12)

    for p in profiles:
        s = p.scores
        table.add_row(
            p.speaker.display_name,
            f"{s.openness:.0f}",
            f"{s.conscientiousness:.0f}",
            f"{s.extraversion:.0f}",
            f"{s.agreeableness:.0f}",
            f"{s.neuroticism:.0f}",
            str(len(p.analyzed_recording_ids)),
            p.last_updated.strftime("%Y-%m-%d"),
        )

    console.print(table)


def export_markdown(profile: SpeakerProfile, output_dir: Path = Path(".")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = profile.speaker.display_name.replace(" ", "_").lower()
    path = output_dir / f"{safe_name}_profile.md"

    s = profile.scores
    lines = [
        f"# {profile.speaker.display_name} — Big Five Profile",
        f"",
        f"_Generated {datetime.now().strftime('%Y-%m-%d')} · "
        f"{len(profile.analyzed_recording_ids)} recording(s)_",
        f"",
        f"## Communication Style",
        f"",
        profile.communication_style_summary,
        f"",
        f"## Big Five Scores",
        f"",
        f"| Trait | Score | Confidence | Interpretation |",
        f"|---|---|---|---|",
    ]

    for ta in profile.trait_analyses:
        bar = "#" * int(ta.score / 10) + "-" * (10 - int(ta.score / 10))
        conf = "low" if ta.confidence < 0.4 else ("moderate" if ta.confidence < 0.7 else "high")
        lines.append(f"| **{ta.trait.value.title()}** | `{bar}` {ta.score:.0f} | {conf} | {ta.interpretation} |")

    lines += [
        f"",
        f"## Collaboration Tips",
        f"",
    ]

    for tip in profile.collaboration_tips:
        lines += [
            f"### {tip.trait.value.title()}",
            f"",
            f"**{tip.tip}**",
            f"",
            f"_{tip.rationale}_",
            f"",
            f"**Do:** {'; '.join(tip.approach_do)}",
            f"",
            f"**Avoid:** {'; '.join(tip.approach_avoid)}",
            f"",
        ]

    lines += [
        f"## Evidence Quotes",
        f"",
    ]
    for ta in profile.trait_analyses:
        lines.append(f"### {ta.trait.value.title()}")
        for q in ta.evidence_quotes:
            lines.append(f'> "{q}"')
        lines.append("")

    path.write_text("\n".join(lines))
    return path
