# Enriched Sentence Analysis Pipeline & API

An event-sourced system for processing interview transcripts with AI-powered multi-dimensional sentence analysis.

> **Status:** M4.4 Complete (Layer 5: OKF export, front-matter capture, query endpoints) | **Tests:** 997 passing | **Coverage:** 90.8%
>
> See [ROADMAP.md](docs/ROADMAP.md) for milestones and [docs/architecture/](docs/architecture/) for detailed diagrams.

## What It Does

1. **Ingests** interview transcripts (text files — labeled or raw unlabeled prose)
2. **Segments** text into offset-grounded fragments using spaCy NLP (the map: every fragment ties back to exact source positions)
3. **Attributes** speakers (parsed from labels, or inferred with confidence when absent — fully correctable)
4. **Stitches** interrupted utterances via relationship overlay (verbatim text untouched; interruptions become queryable data)
5. **Enriches** via a registry of focused extractors — one LLM call per dimension (function, structure, purpose, topics, keywords, entities, claims), each schema-enforced with numeric confidence, behind a provider failover chain (Anthropic Haiku → Claude Code → OpenAI)
6. **Embeds** fragments and utterances into per-model Neo4j vector indexes for semantic search
7. **Applies lenses** — purpose-built views like meeting minutes (objectives, decisions, action items, follow-ups) extracted by a fully generic engine; adding a lens is one YAML profile + one prompts file, zero code
8. **Stores** everything as events in EventStoreDB (source of truth); the projection service is Neo4j's sole writer
9. **Exports** OKF bundles — markdown files with YAML frontmatter, git-versionable and agent-consumable, grounding every lens item back to the verbatim transcript
10. **Exposes** REST API for querying and user corrections (edits, speakers, stitches, lens items)

**Run it:** `python -m src.ingestion <file> --enrich` (ingest + enrich in one shot), or `make ingest FILE=<path>`.
Then apply a lens: `python -m src.lens <interview_id> meeting_minutes`.
Then export it: `python -m src.export <interview_id> meeting_minutes`.
Then ask it a question: `python -m src.ask <project_id> "What did they decide about Acme Corp?"` (or `POST /ask/{project_id}`).

## Architecture

```
Transcript Ingestion / Edit & Correction APIs
    ↓
Aggregates (Interview, Sentence)
    └──→ EventStoreDB (events only) ← Source of Truth

EventStoreDB
    ↓
Projection Service (12 lanes, sole Neo4j writer)
    ↓
Neo4j (read model: fragments, speakers, utterances, analysis)
```

**Key Patterns:** Event Sourcing, CQRS, async/await throughout

> **Details:** [docs/architecture/](docs/architecture/) — system overview, data flow, event sourcing, database schema

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10 |
| API | FastAPI + Uvicorn | 0.117.0+ |
| Event Store | EventStoreDB | 23.10.1 |
| Graph DB | Neo4j | 5.22.0 |
| Task Queue | Celery + Redis | 5.5.3 / 7 |
| NLP | spaCy | 3.8.11 |
| LLM APIs | OpenAI, Anthropic, Gemini | Various |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### Setup

```bash
# Clone and enter directory
git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining
cd interview_analyzer_chaining

# Create .env with your API keys
cat > .env << EOF
OPENAI_API_KEY=your_key_here
NEO4J_PASSWORD=your_password_here
EOF

# Build and start all services
docker compose up -d
```

### Access Points

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |
| EventStoreDB UI | http://localhost:2113 |

### Run the Pipeline

```bash
# Process files in data/input/
docker compose run --rm app python src/main.py --run-pipeline

# Or use Makefile
make run-pipeline
```

> **Detailed Setup:** [docs/onboarding/](docs/onboarding/) — prerequisites, configuration, troubleshooting

## Project Structure

```
src/
├── agents/          # LLM interaction (OpenAI, Anthropic, Gemini)
├── api/routers/     # FastAPI endpoints (files, analysis, edits)
├── commands/        # CQRS command handlers
├── events/          # Event sourcing (aggregates, store, repository)
├── projections/     # Event-to-Neo4j projection service
├── pipeline.py      # Core processing pipeline
└── main.py          # FastAPI entry point

docs/
├── architecture/    # Mermaid diagrams (system, data flow, schema)
├── onboarding/      # Getting started guides
└── ROADMAP.md       # Project status and milestones

tests/               # 691 tests (unit, integration, e2e)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/files/` | GET | List analysis files |
| `/files/{filename}` | GET | Get analysis content |
| `/analysis/` | POST | Trigger background analysis |
| `/edits/sentences/{id}/{index}/edit` | POST | Edit sentence text |
| `/edits/sentences/{id}/{index}/analysis/override` | POST | Override AI analysis |
| `/lenses/{interview_id}/items/{item_id}/override` | POST | Correct a lens item (locks it) |
| `/exports/{interview_id}/{lens_name}` | GET | Download an OKF bundle (zip) |
| `/interviews/{interview_id}/lenses/{lens}/items` | GET | List a lens's items for an interview |
| `/review/worklist` | GET | Low-confidence + unresolved-reference review queue |
| `/speakers/rollup` | GET | Speaker rollup by display name, across interviews |
| `/ask/{project_id}` | POST | Ask-the-corpus: hybrid retrieval + cited synthesis (GraphRAG) |

> Full API documentation at http://localhost:8000/docs

## Common Commands

```bash
make build          # Build Docker images
make run            # Start all services
make test           # Run test suite
make run-pipeline   # Process input files
make clean          # Stop and remove containers
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Project status, milestones, upgrade schedule |
| [docs/architecture/](docs/architecture/) | System diagrams, data flow, database schema |
| [docs/onboarding/](docs/onboarding/) | Setup guides, troubleshooting, dev workflow |

## Contributing

Contributions welcome. Please follow standard practices (issues, feature branches, tests, PRs). Code style enforced via `black` and `flake8`.

## License

MIT License. See `LICENSE` file.
