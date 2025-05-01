# Enriched Sentence Analysis Pipeline & API

## Overview

This project provides an asynchronous pipeline and a FastAPI interface for processing text files (e.g., interview transcripts) and performing detailed, multi-dimensional analysis on each sentence. It leverages OpenAI's language models, the spaCy library for NLP tasks, and a robust, configurable architecture to produce structured JSON output.

The pipeline segments input text, builds contextual information around each sentence, and uses an `AnalysisService` to orchestrate concurrent API calls to an OpenAI model (via `OpenAIAgent`) for classifying sentences based on function, structure, purpose, topic, and keywords according to configurable prompts. Results are written asynchronously using a **decoupled Input/Output (IO) layer defined by protocols**, allowing for different storage backends (e.g., local files, cloud storage, databases). Utilities for interacting with Neo4j are also included.

The accompanying FastAPI application allows interaction with the generated analysis files.

The project is containerized using Docker and Docker Compose for a consistent development and runtime environment.

## System Architecture

1.  **API Request:** A user sends a request to the FastAPI application (e.g., `POST /analysis/`) to process an input file.
2.  **Task Queuing:** The API endpoint queues a background task using Celery, with Redis acting as the message broker.
3.  **Worker Processing:** A Celery worker (`worker` service) picks up the task.
4.  **Pipeline Execution:** The worker executes the main analysis pipeline (`src/pipeline.py`):
    - Reads the input text file (`TextDataSource`).
    - Segments text into sentences (`spaCy`).
    - Creates an intermediate map file (`ConversationMapStorage`).
    - Builds context for each sentence (`ContextBuilder` via `AnalysisService`).
    - Analyzes sentences using the LLM (`SentenceAnalyzer` via `AnalysisService`).
    - Analysis results are queued internally.
5.  **Results Persistence (`_result_writer` in `pipeline.py`):**
    - Results are dequeued.
    - Each result is written to a JSONL file using `SentenceAnalysisWriter` (e.g., `*_analysis.jsonl`).
    - **In parallel**, each result is also persisted to the Neo4j database via `src.persistence.graph_persistence.save_analysis_to_graph`, using the `Neo4jConnectionManager`.

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
- **Neo4j Utilities:** Includes `Neo4jConnectionManager` for managing asynchronous connections to a Neo4j 4.4 database.
- **Containerized Environment:** Defined via `Dockerfile` and `docker-compose.yml` for reproducible setup.
- **Dev Container Support:** Configured for use with VS Code Remote - Containers (`.devcontainer/devcontainer.json`).

## Technology Stack

- **Programming Language:** Python 3.10 (specified in Dockerfile)
- **Core Libraries:** `asyncio`, `openai`, `spacy`, `pydantic`, `pyyaml`, `aiofiles`, `celery`
- **Database Driver (Optional):** `neo4j` (Async driver)
- **API Framework:** `fastapi`, `uvicorn`
- **HTTP Client (Testing):** `httpx`
- **NLP Model (Segmentation):** `en_core_web_sm` (spaCy)
- **LLM:** Configurable OpenAI model (e.g., `gpt-4o`)
- **Testing:** `pytest`, `pytest-asyncio`, `unittest.mock`
- **Linting/Formatting:** `flake8`, `black` (Recommended)
- **Database:** Neo4j 4.4
- **Containerization:** Docker, Docker Compose
- **Development Environment:** VS Code Dev Containers (optional)
- **Task Queue:** Celery
- **Message Broker:** Redis

## Database (Neo4j)

This project uses a Neo4j graph database (currently v4.4 via Docker) as a **secondary persistence layer** alongside the primary JSONL output files.

**Purpose:**
The graph database stores the structured analysis results, enabling complex querying and exploration of relationships between sentences, topics, keywords, and analysis dimensions that might be difficult with flat files.

**Interaction:**
- During pipeline execution, the `_result_writer` function (`src/pipeline.py`) iterates through analysis results.
- For each result, it calls `save_analysis_to_graph` (`src/persistence/graph_persistence.py`).
- This function uses the `Neo4jConnectionManager` (`src/utils/neo4j_driver.py`) to acquire an asynchronous Neo4j session.
- Within the session, it executes a series of Cypher queries primarily using `MERGE` to idempotently create or update nodes (`SourceFile`, `Sentence`, `Topic`, `Keyword`, etc.) and their relationships based on the analysis data.
- The operations for a single sentence analysis are grouped within an async session context.

**Schema:**
*(Based on `src/persistence/graph_persistence.py`)*
-   **Nodes:**
    -   `:SourceFile {filename: string}`
    -   `:Sentence {sentence_id: integer, filename: string, text: string, sequence_order: integer}`
    -   `:FunctionType {name: string}`
    -   `:StructureType {name: string}`
    -   `:Purpose {name: string}`
    -   `:Topic {name: string}`
    -   `:Keyword {text: string}`
-   **Relationships:**
    -   `(:Sentence)-[:PART_OF_FILE]->(:SourceFile)`
    -   `(:Sentence)-[:HAS_FUNCTION_TYPE]->(:FunctionType)`
    -   `(:Sentence)-[:HAS_STRUCTURE_TYPE]->(:StructureType)`
    -   `(:Sentence)-[:HAS_PURPOSE]->(:Purpose)`
    -   `(:Sentence)-[:HAS_TOPIC]->(:Topic)` (Connects to Level 1 and Level 3 topics)
    -   `(:Sentence)-[:MENTIONS_OVERALL_KEYWORD]->(:Keyword)`
    -   `(:Sentence)-[:MENTIONS_DOMAIN_KEYWORD]->(:Keyword)`
    -   `(:Sentence)-[:FOLLOWS]->(:Sentence)` (Links to previous sentence in the same file)

**Utilities:**
-   The `Neo4jConnectionManager` provides asynchronous connection pooling and methods for executing Cypher queries.

## Project Structure

```
├── data/
│   ├── input/         # Default directory for input .txt files
│   ├── output/        # Default directory for output analysis .jsonl files
│   └── maps/          # Default directory for intermediate map .jsonl files
├── docker/            # Dockerfile for the application services
├── .devcontainer/     # VS Code Dev Container configuration
│   ├── devcontainer.json
│   └── devcontainer.env # Example/Placeholder env vars for dev container context (not primary source)
├── logs/              # Default directory for log files
├── prompts/           # Directory for LLM prompt YAML files
├── src/
│   ├── agents/        # LLM interaction, context building, sentence analysis logic
│   ├── api/           # FastAPI application: main app, routers, schemas
│   ├── io/            # Input/Output protocols and implementations
│   │   ├── __init__.py
│   │   ├── protocols.py
│   │   └── local_storage.py
│   ├── models/        # Pydantic models for LLM responses
│   ├── persistence/   # Modules for saving data to persistent stores (e.g., graph)
│   │   ├── __init__.py
│   │   └── graph_persistence.py # Logic for saving analysis to Neo4j
│   ├── services/      # Service layer coordinating agents
│   ├── utils/         # Helper functions, config, logger, metrics, text processing
│   ├── __init__.py
│   ├── config.py      # Configuration loading logic
│   ├── main.py        # FastAPI application entry point & CLI pipeline runner
│   └── pipeline.py    # Core pipeline processing functions
├── tests/
│   ├── agents/
│   ├── api/
│   ├── integration/
│   ├── io/            # Tests for IO implementations
│   ├── services/
│   └── utils/
├── .env               # Runtime environment variables (API Keys, DB Passwords) - **Gitignored**
├── .env.example       # Example environment variable file
├── .gitignore
├── config.yaml        # Main application configuration
├── docker-compose.yml # Docker Compose configuration defining services
├── Makefile           # Commands for build, run, test, etc. within Docker
├── pytest.ini         # Pytest configuration
├── README.md          # This file
└── requirements.txt   # Python package dependencies (installed in Docker image)
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

-   The API will be accessible at `http://localhost:8000`.
-   Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.
-   The main Neo4j browser should be accessible at `http://localhost:7474`. Login with user `neo4j` and the password set in your `.env` file.

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

*   **`GET /`**: Health check endpoint.
*   **`GET /files/`**: Lists the filenames of generated analysis (`_analysis.jsonl`) files found in the configured output directory.
*   **`GET /files/{filename}`**: Retrieves the full content of a specific analysis file.
*   **`GET /files/{filename}/sentences/{sentence_id}`**: Retrieves the analysis result for a specific sentence ID within a given analysis file.
*   **`POST /analysis/`**: Accepts an `input_filename` and schedules the analysis pipeline for that file using **FastAPI BackgroundTasks** (returns `202 Accepted`).
*   _(More endpoints could be added for detailed task status, specific analysis requests without running the full pipeline, etc.)_

## Input and Output Files

-   **Input:** Plain text files (`.txt`) in the input directory.
-   **Output:**
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

## Contributing

Contributions are welcome. Please follow standard practices (issues, feature branches, tests, PRs). Adherence to PEP 8 and code style (`black`, `flake8`) is encouraged.

## License

MIT License. See `LICENSE` file.
