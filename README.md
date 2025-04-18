# Enriched Sentence Analysis Pipeline & API

## Overview

This project provides an asynchronous pipeline and a FastAPI interface for processing text files (e.g., interview transcripts) and performing detailed, multi-dimensional analysis on each sentence. It leverages OpenAI's language models, the spaCy library for NLP tasks, and a robust, configurable architecture to produce structured JSON output.

The pipeline segments input text, builds contextual information around each sentence, and uses an `AnalysisService` to orchestrate concurrent API calls to an OpenAI model (via `OpenAIAgent`) for classifying sentences based on function, structure, purpose, topic, and keywords according to configurable prompts. Results are written asynchronously.

The accompanying FastAPI application allows interaction with the generated analysis files.

## Features

- **Asynchronous Pipeline:** Uses `asyncio` for efficient processing, particularly for I/O-bound LLM API calls and result writing.
- **Sentence Segmentation:** Utilizes `spaCy` via `src/utils/text_processing.py` for accurate text segmentation.
- **Configurable Context Building:** `ContextBuilder` generates textual context windows (e.g., immediate, broader, observer) around each sentence based on settings in `config.yaml`.
- **Multi-Dimensional LLM Analysis:** `SentenceAnalyzer` interacts with OpenAI API (via `OpenAIAgent`) to classify sentences across multiple dimensions (function, structure, purpose, topic, keywords).
- **Service Layer Orchestration:** `AnalysisService` coordinates context building and sentence analysis.
- **Prompt-Driven Analysis:** Classification logic is driven by prompts defined in external YAML files.
- **Pydantic Validation:** Uses Pydantic V2 models (`src/models/llm_responses.py`) to validate the structure of LLM responses.
- **Configuration Management:** Centralized configuration via `config.yaml` and environment variables (`.env`) using `python-dotenv`.
- **Centralized Logging:** Uses Python's standard `logging` module configured for file and console output (`src/utils/logger.py`).
- **Metrics Tracking:** Tracks API calls, token usage, processing time, successes, and errors (`src/utils/metrics.py`).
- **FastAPI Interface:** Provides RESTful endpoints (`src/api/`) for listing analysis files and potentially triggering/viewing analysis.
- **Modular Architecture:** Code organized into logical components (`pipeline`, `agents`, `services`, `api`, `models`, `utils`).
- **Robust Testing:** Comprehensive unit and integration tests using `pytest`, `TestClient`, and `unittest.mock`.

## Technology Stack

- **Programming Language:** Python 3.11+
- **Core Libraries:** `asyncio`, `openai`, `spacy`, `pydantic`, `python-dotenv`, `pyyaml`
- **API Framework:** `fastapi`, `uvicorn`
- **HTTP Client (Testing):** `httpx`
- **NLP Model (Segmentation):** `en_core_web_sm` (spaCy)
- **LLM:** Configurable OpenAI model (e.g., `gpt-4o`)
- **Testing:** `pytest`, `pytest-asyncio`, `unittest.mock`
- **Linting/Formatting:** `flake8`, `black` (Recommended)

## Project Structure

```
├── data/
│   ├── input/         # Default directory for input .txt files
│   ├── output/        # Default directory for output analysis .jsonl files
│   └── maps/          # Default directory for intermediate map .jsonl files
├── docker/            # Dockerfile for the application
├── logs/              # Default directory for log files
├── prompts/           # Directory for LLM prompt YAML files
├── src/
│   ├── agents/        # LLM interaction, context building, sentence analysis logic
│   ├── api/           # FastAPI application: main app, routers, schemas
│   ├── models/        # Pydantic models for LLM responses
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
│   ├── services/
│   └── utils/
├── .env               # Environment variables (OpenAI Key) - Gitignored
├── .gitignore
├── config.yaml        # Main application configuration
├── docker-compose.yml # Docker Compose configuration
├── Makefile           # Commands for build, run, test, etc.
├── pytest.ini         # Pytest configuration
├── README.md          # This file
└── requirements.txt   # Python package dependencies
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory (copy from `.env.example` if provided) and add your OpenAI API key:
    ```dotenv
    # .env
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

## Configuration

Key application behaviors are configured in `config.yaml`:

-   **`openai`:** API key (loaded from `.env`), model name, request parameters.
-   **`openai_api.retry`:** Settings for API call retry logic.
-   **`preprocessing.context_windows`:** Sizes for different context types.
-   **`classification.local.prompt_files`:** Path to the YAML file containing classification prompts.
-   **`paths`:** Default input/output/map/log directories and file suffixes.
-   **`domain_keywords`:** List of keywords for domain-specific extraction.

Modify `config.yaml` to tune performance and analysis focus.

## Usage

### Running the Pipeline (CLI)

The main analysis pipeline can be run via the command line:

```bash
python src/main.py --run-pipeline [--input_dir <path>] [--output_dir <path>] [--map_dir <path>]
```

-   `--run-pipeline`: Flag to execute the pipeline.
-   Optional arguments override paths defined in `config.yaml`.
-   The pipeline processes each `.txt` file in the input directory.

Alternatively, use the Makefile:

```bash
make run-pipeline
# Or with custom directories:
# INPUT_DIR=./custom_in OUTPUT_DIR=./custom_out MAP_DIR=./custom_maps make run-pipeline
```

### Running the API

Start the FastAPI server using Uvicorn (or the Makefile):

```bash
make run-api
# Or manually:
# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

-   The API will be accessible at `http://localhost:8000`.
-   Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.
-   Alternative documentation (ReDoc) is available at `http://localhost:8000/redoc`.

## API Endpoints

-   **`GET /health`**: Health check endpoint.
-   **`GET /files/`**: Lists the filenames of generated analysis (`_analysis.jsonl`) files found in the configured output directory.
-   _(More endpoints to be added for file content retrieval, specific sentence analysis, and triggering pipeline runs)._

## Input and Output Files

-   **Input:** Plain text files (`.txt`) in the input directory.
-   **Output:**
    1.  **Map Files (`*_map.jsonl`):** In the map directory. Contains `sentence_id`, `sequence_order`, `sentence` per line.
    2.  **Analysis Files (`*_analysis.jsonl`):** In the output directory. Contains detailed analysis results per sentence (JSON object per line).
    3.  **Log File (`pipeline.log`):** In the logs directory.

## Testing

Run tests using `pytest` from the project root:

```bash
pytest
# Or via Makefile:
# make test
```

Tests utilize `pytest`, `pytest-asyncio`, `unittest.mock`, and FastAPI's `TestClient`.

## Contributing

Contributions are welcome. Please follow standard practices (issues, feature branches, tests, PRs). Adherence to PEP 8 and code style (`black`, `flake8`) is encouraged.

## License

MIT License. See `LICENSE` file.
