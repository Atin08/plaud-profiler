# plaud-profiler

A CLI tool that reads your [Plaud.ai](https://plaud.ai) recordings and builds scientifically-grounded **Big Five (OCEAN) personality profiles** of the speakers — so you can learn to collaborate with them more effectively.

## How it works

1. Connects to the Plaud.ai MCP server to fetch your recordings and transcripts
2. Groups transcript segments by speaker
3. Sends each speaker's speech to Claude (with Big Five linguistic marker analysis) 
4. Stores profiles locally as JSON — profiles improve as more recordings are analysed
5. Generates actionable collaboration tips grounded in personality research

The Big Five model is used because it has the strongest scientific literature: ~50 years of peer-reviewed research, cross-cultural validation, and proven predictive validity for communication and collaboration behaviour (Mairesse et al., 2007; Goldberg, 1992).

## Prerequisites

- Python 3.11+
- Node.js (for the Plaud MCP server)
- An [Anthropic API key](https://console.anthropic.com)
- A Plaud.ai account with recordings

## Installation

```bash
# 1. Install the Plaud MCP server
npx -y @plaud-ai/mcp@latest install

# 2. Install plaud-profiler
cd plaud-profiler
pip install -e .
```

## Setup

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Log in to Plaud.ai (opens browser)
plaud-profiler login
```

## Usage

```bash
# List your recordings
plaud-profiler recordings

# Analyse a recording (updates all speakers' profiles)
plaud-profiler analyze <recording-id>

# Analyse only one speaker from a recording
plaud-profiler analyze <recording-id> --speaker "Speaker 1"

# View all saved profiles (summary table)
plaud-profiler profiles

# View a speaker's full profile + collaboration tips
plaud-profiler profile "Speaker 1"

# Export a profile as a markdown report
plaud-profiler report "Speaker 1" --output ./reports

# Delete a profile
plaud-profiler delete "Speaker 1"
```

## Big Five dimensions

| Dimension | What it reveals for collaboration |
|---|---|
| **O** Openness | Receptiveness to ideas, creativity, exploration |
| **C** Conscientiousness | Reliability, structure preference, follow-through |
| **E** Extraversion | Communication energy, group dynamics |
| **A** Agreeableness | Directness vs. diplomacy, conflict style |
| **N** Neuroticism | Stress response, emotional predictability |

## Project structure

```
specs/openapi.yaml          # OpenAPI spec — source of truth for all schemas
src/plaud_profiler/
  models.py                 # Pydantic models (derived from OpenAPI spec)
  plaud_client.py           # MCP client for plaud.ai
  analyzer.py               # Claude-based Big Five analysis
  profiles.py               # Local JSON profile storage
  reporter.py               # Rich terminal output + markdown export
  cli.py                    # Typer CLI
```

## References

- Goldberg, L.R. (1992). The development of markers for the Big-Five factor structure. *Psychological Assessment*.
- Mairesse, F. et al. (2007). Using linguistic cues for the automatic recognition of personality in conversation. *JAIR*.
- Pennebaker, J.W. & King, L.A. (1999). Linguistic styles: Language use as an individual difference. *JPSP*.
