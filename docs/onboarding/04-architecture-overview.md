# Architecture Overview

This guide explains how the Interview Analyzer system works, what each component does, and how data flows through the application.

**Prerequisites:** Complete [03-running-the-system.md](./03-running-the-system.md) first.

---

## System Architecture

The Interview Analyzer is an **event-sourced microservices application** that processes interview transcripts using AI and stores results in a graph database.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client / User                             │
└───────────────┬─────────────────────────┬───────────────────────┘
                │                         │
                │ HTTP API Requests       │ Pipeline CLI
                │                         │
┌───────────────▼─────────────────────────▼───────────────────────┐
│                      FastAPI Application                         │
│                         (app service)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  API Routes  │  │   Pipeline   │  │   Commands   │          │
│  │  /files/     │  │   Processor  │  │   Handlers   │          │
│  │  /analysis/  │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───┬────────────────────┬────────────────────┬──────────────────┘
    │                    │                    │
    │ Queue Tasks        │ Write Events       │ Query Data
    │                    │                    │
┌───▼─────────┐    ┌────▼──────────────┐    ┌▼─────────────────┐
│   Redis     │    │  EventStoreDB     │    │     Neo4j        │
│  (Broker)   │    │  (Event Stream)   │    │  (Read Model)    │
└───┬─────────┘    └────┬──────────────┘    └▲─────────────────┘
    │                   │                     │
    │ Pull Tasks        │ Subscribe           │ Write Projections
    │                   │                     │
┌───▼─────────┐    ┌────▼──────────────────┐ │
│   Celery    │    │  Projection Service    │─┘
│   Worker    │    │   (Event Consumer)     │
└─────────────┘    └────────────────────────┘
```

---

## Service Descriptions

### 1. FastAPI Application (`app`)

**Purpose:** Main API server and application entry point

**Key Responsibilities:**
- Exposes REST API endpoints
- Runs pipeline processing
- Handles command requests (edits, overrides)
- Emits domain events to EventStoreDB

**Technology:** Python 3.10, FastAPI, Uvicorn

**Endpoints:**
- `GET /` - Health check
- `GET /files/` - List analysis files
- `GET /files/{filename}` - Get analysis file content
- `GET /files/{filename}/sentences/{sentence_id}` - Get specific sentence
- `POST /analysis/` - Trigger analysis via background task

**Port:** 8000

### 2. Celery Worker (`worker`)

**Purpose:** Background task processor

**Key Responsibilities:**
- Executes long-running pipeline tasks
- Processes files asynchronously
- Prevents API timeouts for heavy operations

**Technology:** Python 3.10, Celery

**Communication:** Receives tasks from Redis broker, sends results to Redis backend

**Configuration:**
- Reads from Redis at `redis:6379`
- Same codebase as `app` service
- Different entry point: `celery -A src.celery_app worker`

### 3. Redis (`redis`)

**Purpose:** Message broker and result backend for Celery

**Key Responsibilities:**
- Queues background tasks
- Stores task results
- Provides pub/sub for worker communication

**Technology:** Redis 7 Alpine

**Port:** 6379

**Data Persistence:** Volume-backed (survives container restarts)

### 4. Neo4j Main Database (`neo4j`)

**Purpose:** Primary graph database for read models (projections)

**Key Responsibilities:**
- Stores sentence analysis results as graph nodes
- Maintains relationships between sentences, topics, keywords
- Enables complex queries and graph traversals
- Acts as the "read model" in CQRS pattern

**Technology:** Neo4j 5.22.0

**Ports:**
- 7474 - HTTP (Browser UI)
- 7687 - Bolt (Driver protocol)

**Authentication:**
- Username: `neo4j`
- Password: [from your `.env` file's `NEO4J_PASSWORD`]

**Schema:** See "Neo4j Graph Schema" section below

### 5. Neo4j Test Database (`neo4j-test`)

**Purpose:** Isolated database for running tests

**Key Responsibilities:**
- Provides clean database for each test run
- Prevents test data from polluting main database
- Same configuration as main Neo4j but different ports

**Technology:** Neo4j 5.22.0

**Ports:**
- 7475 - HTTP (Browser UI)
- 7688 - Bolt (Driver protocol)

**Authentication:**
- Username: `neo4j`
- Password: `testpassword`

**Usage:** Only accessed during `pytest` test runs

### 6. EventStoreDB (`eventstore`)

**Purpose:** Event sourcing database - single source of truth

**Key Responsibilities:**
- Stores all domain events (InterviewCreated, SentenceCreated, AnalysisGenerated, etc.)
- Provides event stream for projections
- Enables event replay and audit trail
- Supports CQRS "write model"

**Technology:** EventStoreDB 23.10.1

**Ports:**
- 2113 - HTTP API and UI
- 1113 - TCP (optional)

**Configuration:**
- Single-node cluster
- In-memory mode for faster local dev
- All projections enabled
- Insecure mode (no auth for local dev)

**Event Streams:** See "Event Sourcing Model" section below

### 7. Projection Service (`projection-service`)

**Purpose:** Reads events from EventStoreDB and updates Neo4j

**Key Responsibilities:**
- Subscribes to EventStoreDB `$all` stream
- Processes events in order
- Updates Neo4j graph (creates nodes, relationships)
- Implements eventual consistency between write and read models
- Handles idempotency (safe to replay events)

**Technology:** Python 3.10, esdbclient

**Configuration:**
- 12 parallel processing lanes
- Checkpoint management for resume capability
- Retry logic for transient failures

**Entry Point:** `python -m src.run_projection_service`

---

## Data Flow

### Pipeline Processing Flow

```
1. User uploads file or runs CLI
   ↓
2. App reads .txt file from data/input/
   ↓
3. spaCy segments text into sentences
   ↓
4. ContextBuilder creates context windows
   ↓
5. SentenceAnalyzer calls OpenAI API (parallel, 10 workers)
   ↓
6. Results queued internally
   ↓
7. Dual-write happens:
   ├─→ Write to JSONL file (data/output/)
   ├─→ Emit events to EventStoreDB
   └─→ Write to Neo4j directly
   ↓
8. Projection service picks up events
   ↓
9. Updates Neo4j graph (idempotent)
   ↓
10. User queries API or Neo4j Browser
```

### Event Sourcing Flow

```
Command → Aggregate → Event → EventStore → Projection → Neo4j
```

**Example: User edits a sentence**

1. **Command:** `EditSentenceCommand` via API
2. **Aggregate:** `Sentence` aggregate loads from events
3. **Event:** `SentenceEdited` event emitted
4. **EventStore:** Event appended to `Sentence-{id}` stream
5. **Projection:** `SentenceEditedHandler` receives event
6. **Neo4j:** Sentence node updated with new text

---

## Neo4j Graph Schema

### Node Types

**`:SourceFile`**
- Properties: `filename` (string)
- Represents input transcript files

**`:Sentence`**
- Properties:
  - `sentence_id` (integer)
  - `filename` (string)
  - `text` (string)
  - `sequence_order` (integer)
  - `event_version` (integer) - for idempotency
- Represents individual sentences from transcripts

**`:FunctionType`**
- Properties: `name` (string)
- Examples: "question", "statement", "command"

**`:StructureType`**
- Properties: `name` (string)
- Examples: "simple", "compound", "complex"

**`:Purpose`**
- Properties: `name` (string)
- Examples: "inform", "persuade", "request"

**`:Topic`**
- Properties: `name` (string)
- Extracted topics from sentence analysis

**`:Keyword`**
- Properties: `text` (string)
- Keywords mentioned in sentences

### Relationship Types

- `(:Sentence)-[:PART_OF_FILE]→(:SourceFile)`
- `(:Sentence)-[:HAS_FUNCTION_TYPE]→(:FunctionType)`
- `(:Sentence)-[:HAS_STRUCTURE_TYPE]→(:StructureType)`
- `(:Sentence)-[:HAS_PURPOSE]→(:Purpose)`
- `(:Sentence)-[:HAS_TOPIC]→(:Topic)`
- `(:Sentence)-[:MENTIONS_OVERALL_KEYWORD]→(:Keyword)`
- `(:Sentence)-[:MENTIONS_DOMAIN_KEYWORD]→(:Keyword)`
- `(:Sentence)-[:FOLLOWS]→(:Sentence)` - Links to previous sentence

### Example Query

```cypher
// Find all questions about a specific topic
MATCH (s:Sentence)-[:HAS_FUNCTION_TYPE]->(f:FunctionType {name: "question"})
MATCH (s)-[:HAS_TOPIC]->(t:Topic {name: "product development"})
RETURN s.text, s.sequence_order
ORDER BY s.sequence_order
```

---

## Event Sourcing Model

### Aggregate Types

**`Interview`**
- Aggregate Root for interview transcripts
- Stream: `Interview-{uuid}`
- Events:
  - `InterviewCreated` - New interview started
  - `InterviewMetadataUpdated` - Metadata changed

**`Sentence`**
- Aggregate for individual sentences
- Stream: `Sentence-{uuid}`
- Events:
  - `SentenceCreated` - New sentence added
  - `SentenceEdited` - Text changed by user
  - `AnalysisGenerated` - AI analysis completed
  - `AnalysisOverridden` - User overrode AI analysis

### Event Envelope

All events are wrapped in `EventEnvelope`:
```json
{
  "event_id": "uuid",
  "aggregate_id": "uuid",
  "aggregate_type": "Sentence",
  "event_type": "SentenceCreated",
  "event_version": 1,
  "timestamp": "2025-11-09T12:00:00Z",
  "correlation_id": "uuid",
  "causation_id": "uuid",
  "actor": {
    "actor_type": "system",
    "user_id": "pipeline"
  },
  "payload": { /* event-specific data */ }
}
```

### Projection Handlers

Each event type has a handler in `src/projections/handlers/`:

- `InterviewCreatedHandler` → Creates `:Interview` node
- `SentenceCreatedHandler` → Creates `:Sentence` node
- `SentenceEditedHandler` → Updates sentence text
- `AnalysisGeneratedHandler` → Creates analysis relationships
- `AnalysisOverriddenHandler` → Updates with user corrections

---

## Directory Structure

```
interview_analyzer_chaining/
├── config.yaml              # Main app configuration
├── docker-compose.yml       # Service definitions
├── Makefile                 # Development commands
├── requirements.txt         # Python dependencies
├── pytest.ini               # Test configuration
├── .env                     # Secrets (not in Git)
├── .gitignore
│
├── data/                    # Data files (gitignored)
│   ├── input/              # Input .txt transcripts
│   ├── output/             # Analysis .jsonl files
│   └── maps/               # Intermediate sentence maps
│
├── docker/
│   └── Dockerfile          # Application image definition
│
├── docs/                   # Documentation
│   └── onboarding/        # This guide!
│
├── logs/                   # Application logs (gitignored)
│
├── prompts/                # LLM prompt templates
│   ├── domain_prompts.yaml
│   └── task_prompts.yaml
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # FastAPI app entry point
│   ├── pipeline.py        # Core pipeline logic
│   ├── config.py          # Configuration loading
│   ├── celery_app.py      # Celery configuration
│   ├── tasks.py           # Celery tasks
│   ├── run_projection_service.py  # Projection service entry
│   ├── pipeline_event_emitter.py  # Event emission
│   │
│   ├── agents/            # LLM interaction
│   │   ├── agent.py              # OpenAI agent wrapper
│   │   ├── context_builder.py    # Context window builder
│   │   └── sentence_analyzer.py  # Sentence analysis logic
│   │
│   ├── api/               # FastAPI routes
│   │   ├── routers/
│   │   │   ├── files.py          # File endpoints
│   │   │   ├── analysis.py       # Analysis endpoints
│   │   │   └── edits.py          # Edit endpoints
│   │   └── schemas.py            # Pydantic request/response models
│   │
│   ├── commands/          # Command handlers (CQRS)
│   │   ├── handlers.py           # Command execution logic
│   │   ├── interview_commands.py # Interview commands
│   │   └── sentence_commands.py  # Sentence commands
│   │
│   ├── events/            # Event sourcing
│   │   ├── aggregates.py         # Aggregate roots (Interview, Sentence)
│   │   ├── envelope.py           # Event envelope wrapper
│   │   ├── store.py              # EventStoreDB client wrapper
│   │   ├── repository.py         # Aggregate repository
│   │   ├── interview_events.py   # Interview event definitions
│   │   └── sentence_events.py    # Sentence event definitions
│   │
│   ├── io/                # Input/Output abstractions
│   │   ├── protocols.py          # IO protocol definitions
│   │   ├── local_storage.py      # Local file storage
│   │   ├── neo4j_map_storage.py  # Neo4j map writer
│   │   └── neo4j_analysis_writer.py  # Neo4j analysis writer
│   │
│   ├── models/            # Pydantic data models
│   │   ├── llm_responses.py      # LLM response schemas
│   │   └── analysis_result.py    # Analysis result schema
│   │
│   ├── persistence/       # Database persistence
│   │   └── graph_persistence.py  # Neo4j write operations
│   │
│   ├── projections/       # Event projection system
│   │   ├── config.py             # Projection service config
│   │   ├── lane_manager.py       # Parallel lane processing
│   │   └── handlers/             # Event handlers
│   │       ├── base_handler.py
│   │       ├── interview_handlers.py
│   │       └── sentence_handlers.py
│   │
│   ├── services/          # Business logic services
│   │   └── analysis_service.py   # Orchestrates analysis
│   │
│   └── utils/             # Utilities
│       ├── logger.py             # Logging configuration
│       ├── metrics.py            # Metrics tracking
│       ├── neo4j_driver.py       # Neo4j connection manager
│       ├── environment.py        # Environment detection
│       ├── helpers.py            # Helper functions
│       └── text_processing.py    # Text utilities
│
└── tests/                 # Test suite (691 passing, 84 skipped)
    ├── conftest.py        # Pytest fixtures
    ├── api/              # API tests
    ├── commands/         # Command handler tests
    ├── events/           # Event sourcing tests
    ├── integration/      # Integration tests
    ├── projections/      # Projection tests
    ├── services/         # Service tests
    └── utils/            # Utility tests
```

---

## Configuration System

### config.yaml

**Purpose:** Application-level configuration

**Key Sections:**
- `openai`: OpenAI API settings (model, tokens, temperature)
- `google`: Gemini API settings
- `paths`: Directory paths (input, output, maps, logs)
- `pipeline`: Worker counts, concurrency settings
- `neo4j`: Database connection settings
- `event_sourcing`: EventStoreDB connection
- `classification`: Prompt files and thresholds

**Environment Variables:** Syntax `${ENV_VAR_NAME}` auto-expands from `.env`

### .env

**Purpose:** Secrets and environment-specific config

**Not in Git:** This file is gitignored for security

**Required Variables:**
```bash
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
GEMINI_API_KEY=[YOUR_GEMINI_API_KEY]
NEO4J_PASSWORD=[CHECK_DOCKER_COMPOSE_YML]
```

### docker-compose.yml

**Purpose:** Service orchestration

**Defines:**
- 7 services with their images, ports, volumes
- Network configuration
- Health checks
- Dependencies between services
- Volume mounts

### Environment Detection

The system auto-detects its environment using `src/utils/environment.py`:

**Detection Methods:**
- **Docker:** Checks for `/.dockerenv` file or `DOCKER_CONTAINER` environment variable
- **CI:** Checks for `CI` environment variable (GitHub Actions, GitLab CI, etc.)
- **Host:** Default if neither Docker nor CI detected

**Impact on Configuration:**
- **Connection Strings:** Docker uses service names (`neo4j:7687`), Host uses `localhost:7687`
- **EventStoreDB:** Docker: `esdb://eventstore:2113?tls=false`, Host: `esdb://localhost:2113?tls=false`
- **Neo4j:** Docker: `bolt://neo4j:7687`, Host: `bolt://localhost:7687`

**Function:** `detect_environment()` returns `"docker"`, `"ci"`, or `"host"`

See implementation in: `src/utils/environment.py`

---

## Key Concepts

### CQRS (Command Query Responsibility Segregation)

**Write Side (Commands):**
- Commands handled by aggregates
- Events written to EventStoreDB
- Single source of truth

**Read Side (Queries):**
- Projections build Neo4j graph
- Optimized for queries
- Eventually consistent

### Event Sourcing

**Core Principles:**
- Events are immutable
- Current state derived from event history
- Full audit trail
- Time travel possible (replay events)

### Idempotency

**Projections are idempotent:**
- Safe to process same event multiple times
- Uses `event_version` to detect duplicates
- Enables retry logic and replay

### Dual-Write Pattern

During pipeline processing, data is written to **both**:
1. **EventStoreDB** - Events for audit and replay
2. **Neo4j** - Direct write for immediate querying

Projection service provides eventual consistency and handles conflicts.

---

## Technology Stack Summary

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10.14 |
| API Framework | FastAPI | 0.117.0+ |
| Task Queue | Celery | 5.5.3 |
| Message Broker | Redis | 7 Alpine |
| Graph Database | Neo4j | 5.22.0 |
| Event Store | EventStoreDB | 23.10.1 Jammy |
| LLM API | OpenAI | `gpt-4o-mini-2024-07-18` |
| LLM API | Google Gemini | `gemini-2.0-flash` |
| NLP | spaCy + Model | 3.8.11 + `en_core_web_sm` 3.7.0 |
| Data Validation | Pydantic | 2.12.5 |
| Testing | pytest | 8.3.3 |
| Test Coverage | pytest-cov | 6.0.0 |
| Linting | flake8 | 7.3.0 |
| Formatting | black | 24.3.0 |
| Containerization | Docker Desktop | 24.0.0+ |
| Orchestration | Docker Compose | v2.0.0+ |

---

## What's Next?

You now understand how the system works!

Next: [Development Workflow →](./05-development-workflow.md)

Learn the daily development tasks, commands, and best practices for working with this codebase.

