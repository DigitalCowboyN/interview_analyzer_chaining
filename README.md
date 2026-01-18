# Enriched Sentence Analysis Pipeline & API

An event-sourced system for processing interview transcripts with AI-powered multi-dimensional sentence analysis.

> **Status:** M2.8 Complete (Production Ready) | **Tests:** 691 passing | **Coverage:** 72.2%
>
> See [ROADMAP.md](docs/ROADMAP.md) for milestones and [docs/architecture/](docs/architecture/) for detailed diagrams.

## What It Does

1. **Ingests** interview transcripts (text files)
2. **Segments** text into sentences using spaCy NLP
3. **Analyzes** each sentence across 7 dimensions via LLM (function, structure, purpose, topics, keywords)
4. **Stores** results in EventStoreDB (source of truth) and Neo4j (graph queries)
5. **Exposes** REST API for querying and user corrections

## Architecture

```
User Upload / Edit API
    ↓
Pipeline / Command Handlers
    ├──→ EventStoreDB (events) ← Source of Truth
    └──→ Neo4j (direct write)  ← Temporary (M3.0 removes)

EventStoreDB
    ↓
Projection Service (12 lanes)
    ↓
Neo4j (read model)
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
