# Enriched Sentence Analysis Pipeline & API

## Overview

This project provides an asynchronous pipeline and a FastAPI interface for processing text files (e.g., interview transcripts) and performing detailed, multi-dimensional analysis on each sentence. It leverages OpenAI's language models (with Anthropic and Google Gemini support), the spaCy library for NLP tasks, and an **event-sourced architecture** to produce structured, auditable output.

The pipeline segments input text, builds contextual information around each sentence, and uses an `AnalysisService` to orchestrate concurrent API calls to LLM providers for classifying sentences based on function, structure, purpose, topic, and keywords according to configurable prompts.

**Architecture:** The system uses CQRS (Command Query Responsibility Segregation) with EventStoreDB as the source of truth and Neo4j as the read model. All state changes are captured as immutable events, enabling full audit trails and event replay.

The accompanying FastAPI application allows interaction with the generated analysis files and provides edit endpoints for user corrections.

The project is containerized using Docker and Docker Compose for a consistent development and runtime environment.

> **Current Status:** M2.8 Complete (Event-Sourced Architecture - Production Ready)
> **Tests:** 691 passing | **Coverage:** 72.2%
> See [ROADMAP.md](docs/ROADMAP.md) for milestone details.

## System Architecture

```
User Upload / Edit API
    ↓
Pipeline / Command Handlers
    ├──→ EventStoreDB (events) ← Source of Truth
    └──→ Neo4j (direct write)  ← Temporary (dual-write phase)

EventStoreDB
    ↓
Projection Service (12 lanes)
    ↓
Neo4j (materialized view)
```

### Processing Flow

1. **API Request:** User sends a request to FastAPI (e.g., `POST /analysis/`) to process an input file.
2. **Task Queuing:** The API queues a background task using Celery with Redis as the message broker.
3. **Pipeline Execution:** A Celery worker executes the analysis pipeline:
   - Reads input text file and segments into sentences (spaCy)
   - Builds context windows around each sentence
   - Analyzes sentences using LLM (7 parallel classification calls)
4. **Event-First Persistence:**
   - Events emitted to EventStoreDB (`InterviewCreated`, `SentenceCreated`, `AnalysisGenerated`)
   - Results written to JSONL files
   - Direct write to Neo4j (temporary during dual-write phase)
5. **Projection Service:** Subscribes to EventStoreDB, projects events to Neo4j graph

> **Architecture Docs:** See [docs/architecture/](docs/architecture/) for detailed Mermaid diagrams.

## Features

- **Asynchronous Pipeline:** Uses `asyncio` for efficient processing, particularly for I/O-bound LLM API calls and result writing.
- **Decoupled IO:** Defines `TextDataSource`, `ConversationMapStorage`, and `SentenceAnalysisWriter` protocols (`src/io/protocols.py`) for flexible data handling, enabling easy extension to different storage systems (e.g., cloud, databases).
- **Local File Storage:** Provides concrete implementations (`src/io/local_storage.py`) using `aiofiles` for reading text files and writing map/analysis data as JSON Lines (`*.jsonl`).
- **Sentence Segmentation:** Utilizes `spaCy` via `src/utils/text_processing.py` for accurate text segmentation.
- **Configurable Context Building:** `ContextBuilder` generates textual context windows (e.g., immediate, broader, observer) around each sentence based on settings in `config.yaml`.
- **Multi-Dimensional LLM Analysis:** `SentenceAnalyzer` interacts with OpenAI API (via `OpenAIAgent`) to classify sentences across multiple dimensions (function, structure, purpose, topic, keywords).
- **Service Layer Orchestration:** `AnalysisService` coordinates context building and sentence analysis.
- **Prompt-Driven Analysis:** Classification logic is driven by prompts defined in external YAML files.
- **Pydantic Validation:** Uses Pydantic V2 models (`src/models/llm_responses.py`) to validate the structure of LLM responses.
- **Configuration Management:** Centralized configuration via `config.yaml` and environment variables (`.env`) using `python-dotenv`.
- **Centralized Logging:** Uses Python's standard `logging` module configured for file and console output (`src/utils/logger.py`).
- **Metrics Tracking:** Tracks API calls, token usage, processing time, successes, and errors (`src/utils/metrics.py`).
- **FastAPI Interface:** Provides RESTful endpoints (`src/api/`) for listing analysis files and triggering background analysis via Celery.
- **Background Processing:** Uses **Celery** with a **Redis** broker/backend to run the potentially long-running analysis pipeline asynchronously, initiated via the API (`worker` service executes tasks defined likely around `src/pipeline.py`).
- **Modular Architecture:** Code organized into logical components (`pipeline`, `agents`, `services`, `api`, `models`, `utils`, `io`).
- **Robust Testing:** Comprehensive unit and integration tests using `pytest`, executable within the Docker environment.
- **Event Sourcing:** All state changes captured as immutable events in EventStoreDB, enabling audit trails and replay.
- **CQRS Pattern:** Command handlers emit events; projection service updates Neo4j read model.
- **Neo4j Graph Database:** Neo4j 5.22 serves as the read model with rich relationship queries.
- **Containerized Environment:** Defined via `Dockerfile` and `docker-compose.yml` for reproducible setup.
- **Dev Container Support:** Configured for use with VS Code Remote - Containers (`.devcontainer/devcontainer.json`).

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10 |
| API Framework | FastAPI + Uvicorn | 0.117.0+ |
| Event Store | EventStoreDB | 23.10.1 |
| Graph Database | Neo4j | 5.22.0 |
| Task Queue | Celery | 5.5.3 |
| Message Broker | Redis | 7 Alpine |
| NLP | spaCy + en_core_web_sm | 3.8.11 |
| LLM APIs | OpenAI, Anthropic, Gemini | Various |
| Validation | Pydantic | 2.12.5 |
| Testing | pytest, pytest-asyncio | 8.3.3 |
| Containerization | Docker, Docker Compose | Latest |

**Core Libraries:** `asyncio`, `openai`, `anthropic`, `spacy`, `pydantic`, `esdbclient`, `neo4j`, `celery`, `aiofiles`

## Databases

### EventStoreDB (Source of Truth)

EventStoreDB stores all domain events as immutable records:

- **Event Streams:** `Interview-{uuid}`, `Sentence-{uuid}`
- **Event Types:** `InterviewCreated`, `SentenceCreated`, `SentenceEdited`, `AnalysisGenerated`, `AnalysisOverridden`
- **Features:** Optimistic concurrency, persistent subscriptions, event replay

### Neo4j (Read Model)

Neo4j 5.22 serves as the **read model** in the CQRS architecture, providing rich graph queries for analysis exploration.

**Write Paths:**
1. **Direct writes** (temporary during dual-write phase) - Pipeline writes directly
2. **Projection service** - Subscribes to EventStoreDB and projects events to Neo4j

**Purpose:**
The graph database enables complex querying and exploration of relationships between sentences, topics, keywords, and analysis dimensions.

**Schema:**
_(Based on `src/persistence/graph_persistence.py`)_

- **Nodes:**
  - `:SourceFile {filename: string}`
  - `:Sentence {sentence_id: integer, filename: string, text: string, sequence_order: integer}`
  - `:FunctionType {name: string}`
  - `:StructureType {name: string}`
  - `:Purpose {name: string}`
  - `:Topic {name: string}`
  - `:Keyword {text: string}`
- **Relationships:**
  - `(:Sentence)-[:PART_OF_FILE]->(:SourceFile)`
  - `(:Sentence)-[:HAS_FUNCTION_TYPE]->(:FunctionType)`
  - `(:Sentence)-[:HAS_STRUCTURE_TYPE]->(:StructureType)`
  - `(:Sentence)-[:HAS_PURPOSE]->(:Purpose)`
  - `(:Sentence)-[:HAS_TOPIC]->(:Topic)` (Connects to Level 1 and Level 3 topics)
  - `(:Sentence)-[:MENTIONS_OVERALL_KEYWORD]->(:Keyword)`
  - `(:Sentence)-[:MENTIONS_DOMAIN_KEYWORD]->(:Keyword)`
  - `(:Sentence)-[:FOLLOWS]->(:Sentence)` (Links to previous sentence in the same file)

**Utilities:**

- The `Neo4jConnectionManager` provides asynchronous connection pooling and methods for executing Cypher queries.

## Project Structure

```
├── data/
│   ├── input/              # Input .txt files
│   ├── output/             # Output analysis .jsonl files
│   └── maps/               # Intermediate map .jsonl files
├── docs/
│   ├── architecture/       # Mermaid architecture diagrams
│   ├── onboarding/         # Getting started guides
│   └── ROADMAP.md          # Project roadmap (canonical)
├── docker/                 # Dockerfile for application services
├── .devcontainer/          # VS Code Dev Container configuration
├── logs/                   # Log files
├── prompts/                # LLM prompt YAML files
├── src/
│   ├── agents/             # LLM interaction, context building, analysis
│   ├── api/                # FastAPI routers and schemas
│   │   └── routers/        # files.py, analysis.py, edits.py
│   ├── commands/           # Command handlers (CQRS write side)
│   ├── events/             # Event sourcing: aggregates, store, repository
│   ├── io/                 # IO protocols and implementations
│   ├── models/             # Pydantic models
│   ├── persistence/        # Neo4j graph persistence
│   ├── projections/        # Event projection service
│   │   └── handlers/       # Event-to-Neo4j projection handlers
│   ├── services/           # Business logic services
│   ├── utils/              # Helpers, config, logger, metrics
│   ├── main.py             # FastAPI entry point
│   ├── pipeline.py         # Core pipeline processing
│   └── run_projection_service.py  # Projection service entry
├── tests/                  # Test suite (691 tests)
│   ├── agents/
│   ├── api/
│   ├── commands/
│   ├── events/
│   ├── integration/
│   ├── projections/
│   └── ...
├── config.yaml             # Main application configuration
├── docker-compose.yml      # Docker Compose services
├── Makefile                # Build, run, test commands
└── requirements.txt        # Python dependencies
```

## Prerequisites

- **Docker:** Install Docker Desktop or Docker Engine. See [Docker documentation](https://docs.docker.com/get-docker/).
- **Docker Compose:** Usually included with Docker Desktop. If not, see [Docker Compose documentation](https://docs.docker.com/compose/install/).
- **Git:** For cloning the repository.
- **VS Code + Remote - Containers Extension (Optional):** For using the integrated development environment.

## Setup and Configuration

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining # Replace with your repo URL if different
    cd interview_analyzer_chaining
    ```

2.  **Configure Environment Variables:**

    - Create a file named `.env` in the project root directory (you can copy `.env.example` if it exists and rename it).
    - This file defines **runtime secrets and configuration** needed by the services defined in `docker-compose.yml`.
    - **Important:** Add your required secrets to this `.env` file. Ensure it includes at least:
      ```dotenv
      # .env (Example - Use your actual values!)
      OPENAI_API_KEY='your_openai_api_key_here'
      GEMINI_API_KEY='your_gemini_api_key_here'
      NEO4J_PASSWORD='your_chosen_strong_neo4j_password'
      # Add any other necessary runtime variables
      ```
    - This `.env` file is listed in `.gitignore` and **should not be committed to version control.**

3.  **Build Docker Images:**
    Build the images for the application services (`app`, `worker`):
    ```bash
    docker compose build
    # Or use the Makefile target:
    # make build
    ```
    (Note: `docker compose up` will also build images if they don't exist).

## Usage

The application and its services are run using Docker Compose.

### Running the Application Services

To start all services (API, Celery worker, Redis, Neo4j databases) in the background:

```bash
docker compose up -d
```

- The API will be accessible at `http://localhost:8000`.
- Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.
- The main Neo4j browser should be accessible at `http://localhost:7474`. Login with user `neo4j` and the password set in your `.env` file.

To stop the services:

```bash
docker compose down
# To also remove persistent volumes (Redis data, Neo4j data):
# docker compose down -v
```

### Running the Pipeline

To run the analysis pipeline on files within the container:

```bash
docker compose run --rm app python src/main.py --run-pipeline
# Or use the Makefile:
# make run-pipeline [ARGS="--input_dir ./custom_in"] # Pass args via ARGS
```

This starts a temporary container based on the `app` service image, runs the command, and then removes the container. Input/output paths are relative to the container's filesystem (`/workspaces/interview_analyzer_chaining`, which maps to your project root via the volume mount).

### Development Environment (VS Code Dev Container)

This project is configured for use with the VS Code Remote - Containers extension.

1.  Ensure Docker Desktop is running.
2.  Make sure prerequisites are met (Docker, VS Code, Remote-Containers extension).
3.  Open the project folder in VS Code.
4.  VS Code should detect the `.devcontainer/devcontainer.json` file and prompt you to "Reopen in Container". Click it.
5.  Alternatively, open the Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`) and select "Remote-Containers: Reopen in Container".

This will use the Docker Compose integration defined in `.devcontainer/devcontainer.json` to start all the necessary services (if not already running) and attach VS Code to the `app` container, providing a fully configured development environment.

### Makefile Targets

Common tasks can be run using the Makefile within the Docker environment:

- `make build`: Builds the `app` and `worker` Docker images.
- `make run`: Starts all services in detached mode (`docker compose up -d`).
- `make run-pipeline`: Runs the analysis pipeline within a temporary `app` container.
- `make test`: Runs pytest tests within a temporary `app` container.
- `make lint`: Runs flake8 linter within a temporary `app` container.
- `make format`: Runs black code formatter within a temporary `app` container.
- `make clean`: Stops and removes containers and volumes (`docker compose down -v`).
- `make db-up`: Starts only the database services (`neo4j`, `redis`).
- `make db-down`: Stops only the database services.

## API Endpoints

### Files & Analysis
- **`GET /`** - Health check
- **`GET /files/`** - List analysis files
- **`GET /files/{filename}`** - Get analysis file content
- **`GET /files/{filename}/sentences/{sentence_id}`** - Get specific sentence analysis
- **`POST /analysis/`** - Trigger background analysis (returns `202 Accepted`)

### Edit Endpoints (M2.9)
- **`POST /edits/sentences/{interview_id}/{sentence_index}/edit`** - Edit sentence text
- **`POST /edits/sentences/{interview_id}/{sentence_index}/analysis/override`** - Override AI analysis
- **`GET /edits/sentences/{interview_id}/{sentence_index}/history`** - Get edit history

> Interactive documentation available at `http://localhost:8000/docs`

## Input and Output Files

- **Input:** Plain text files (`.txt`) in the input directory.
- **Output:**
  1.  **Map Files (`*_map.jsonl`):** In the map directory. Contains `sentence_id`, `sequence_order`, `sentence` per line.
  2.  **Analysis Files (`*_analysis.jsonl`):** In the output directory. Contains detailed analysis results per sentence (JSON object per line).
  3.  **Log File (`pipeline.log`):** In the logs directory.

## Testing

Run tests using the Makefile target, which executes pytest within the container environment:

```bash
make test
# Or directly using Docker Compose:
# docker compose run --rm app pytest tests/
```

Tests utilize `pytest`, `pytest-asyncio`, `unittest.mock`, FastAPI's `TestClient`, and the separate `neo4j-test` database service.

## Neo4j Setup and Testing

### Environment Detection and Configuration

The project includes utilities to detect the runtime environment (Docker, CI, or host) and configure Neo4j connections accordingly. This ensures reliable connections across different deployment contexts.

- **Environment Detection**: The `detect_environment()` function identifies the current environment and adjusts configurations.
- **Configuration**: The `get_available_neo4j_config()` function provides environment-specific Neo4j URIs, usernames, and passwords.

### Running Neo4j Tests

To ensure Neo4j connections are reliable and the database is correctly set up, integration tests are provided.

1. **Start Neo4j Test Service**:

   - Use Docker Compose to start the Neo4j test service:
     ```bash
     make db-test-up
     ```
   - Ensure the service is ready before running tests.

2. **Run Integration Tests**:

   - Execute the integration tests to verify Neo4j operations:
     ```bash
     make test
     ```
   - This will run tests that check connection reliability, data integrity, and fault tolerance.

3. **Stop Neo4j Test Service**:
   - After testing, stop the Neo4j test service:
     ```bash
     make db-test-down
     ```

### Troubleshooting

- **Connection Issues**: Ensure the Neo4j service is running and accessible. Check environment configurations if connections fail.
- **Data Integrity**: Run the data integrity tests to verify that all nodes and relationships are correctly persisted.

This setup ensures that the Neo4j database is consistently configured and tested across different environments, providing a robust foundation for graph-based data persistence.

## Contributing

Contributions are welcome. Please follow standard practices (issues, feature branches, tests, PRs). Adherence to PEP 8 and code style (`black`, `flake8`) is encouraged.

## License

MIT License. See `LICENSE` file.
