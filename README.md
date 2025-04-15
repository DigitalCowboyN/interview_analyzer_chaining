# Enriched Sentence Analysis Pipeline

## Overview

This project provides an asynchronous pipeline designed to process text files (e.g., interview transcripts) and perform detailed, multi-dimensional analysis on each sentence. It leverages OpenAI's language models, the spaCy library for NLP tasks, and a robust, configurable architecture to produce structured JSON output suitable for downstream analysis.

The pipeline segments input text, builds contextual information around each sentence, and uses concurrent API calls to an OpenAI model to classify sentences based on function, structure, purpose, topic, and keywords according to configurable prompts.

## Features

- **Asynchronous Pipeline:** Uses `asyncio` for efficient, concurrent processing of sentences within files, especially beneficial for I/O-bound LLM API calls.
- **Sentence Segmentation:** Utilizes `spaCy` for accurate splitting of input text into sentences.
- **Configurable Context Building:** Generates textual context windows (e.g., immediate, broader, observer) around each sentence based on settings in `config.yaml`.
- **Multi-Dimensional LLM Analysis:** Interacts with OpenAI API (via `src/agents/agent.py`) to classify sentences across multiple dimensions:
    - Function Type (e.g., declarative, interrogative)
    - Structure Type (e.g., simple, complex)
    - Purpose (e.g., informational, questioning)
    - Topic Level 1 (Immediate context)
    - Topic Level 3 (Broader context)
    - Overall Keywords
    - Domain Keywords
- **Prompt-Driven Analysis:** Classification logic is driven by prompts defined in external YAML files (specified in `config.yaml`).
- **Pydantic Validation:** Uses Pydantic V2 models (`src/models/llm_responses.py`) to validate the structure of JSON responses from the LLM.
- **Configuration Management:** Centralized configuration via `config.yaml` and environment variables (`.env` file) using `python-dotenv`.
- **Centralized Logging:** Uses Python's standard `logging` module configured for clear output.
- **Metrics Tracking:** Tracks key performance indicators like API calls, token usage, processing time, successes, and errors (`src/utils/metrics.py`).
- **Modular Architecture:** Code is organized into logical components (`pipeline`, `agents`, `models`, `utils`).
- **Robust Testing:** Comprehensive unit tests using `pytest` and `unittest.mock` cover core logic, concurrency handling, and error scenarios.

## Technology Stack

- **Programming Language:** Python 3.11+
- **Core Libraries:** `asyncio`, `openai`, `spacy`, `pydantic`, `python-dotenv`, `pyyaml`
- **NLP Model (Segmentation):** `en_core_web_sm` (spaCy)
- **LLM:** Configurable OpenAI model (e.g., `gpt-4o`)
- **Testing:** `pytest`, `pytest-asyncio`, `unittest.mock`
- **Linting/Formatting:** `flake8`, `black` (Recommended)

*(Note: Sentence embedding generation using `sentence-transformers` and related functionalities like clustering are currently commented out but present in the codebase for potential future use.)*

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
    Create a `.env` file in the project root directory and add your OpenAI API key:
    ```dotenv
    # .env
    OPENAI_API_KEY='your_openai_api_key_here'
    ```
    The application uses `python-dotenv` to load this variable.

## Configuration

Key pipeline behaviors are configured in `config.yaml`:

-   **`openai`:** API key (loaded from `.env`), model name, request parameters (max tokens, temperature).
-   **`openai_api.retry`:** Settings for API call retry logic (attempts, backoff factor).
-   **`preprocessing.context_windows`:** Sizes (number of sentences before/after) for different context types used in analysis prompts.
-   **`classification.local.prompt_files`:** Path to the YAML file containing the prompts used for sentence classification.
-   **`paths`:** Default input/output directories, map file suffix, analysis file suffix.
-   **`pipeline.num_analysis_workers`:** Number of concurrent workers for sentence analysis per file.
-   **`domain_keywords`:** List of keywords used for the domain-specific keyword extraction prompt.

Modify `config.yaml` to tune the pipeline's performance and analysis focus.

## Usage

The main entry point is `src/main.py`. Run the pipeline from the project root directory:

```bash
python src/main.py [--input_dir <path/to/input>] [--output_dir <path/to/output>] [--map_dir <path/to/maps>]
```

-   `--input_dir`: (Optional) Path to the directory containing input `.txt` files. Defaults to the value in `config.yaml`.
-   `--output_dir`: (Optional) Path to the directory where output analysis `.jsonl` files will be saved. Defaults to the value in `config.yaml`.
-   `--map_dir`: (Optional) Path to the directory where intermediate sentence map `.jsonl` files will be saved. Defaults to the value in `config.yaml`.

The pipeline will process each `.txt` file found in the input directory.

## Input and Output

-   **Input:** Plain text files (`.txt`) located in the specified input directory. Each file is treated as a separate conversation or document.
-   **Output:**
    1.  **Map Files (`*_map.jsonl`):** Located in the specified map directory. Each line corresponds to a sentence from the input file, containing its `sentence_id`, `sequence_order`, and the raw `sentence` text.
    2.  **Analysis Files (`*_analysis.jsonl`):** Located in the specified output directory. Each line is a JSON object representing the detailed analysis results for a single sentence, including:
        -   `sentence_id`, `sequence_order`, `sentence` (original text)
        -   `function_type`, `structure_type`, `purpose`
        -   `topic_level_1`, `topic_level_3`
        -   `overall_keywords` (list), `domain_keywords` (list)

## Testing

The project uses `pytest` for unit testing. Ensure you have installed the development dependencies if necessary (`pip install -r requirements-dev.txt` if such a file exists, otherwise dependencies are in `requirements.txt`).

Run tests from the project root directory:

```bash
pytest
```

## Contributing

Contributions are welcome. Please follow standard practices like opening issues for discussion, creating feature branches, ensuring tests pass, and submitting pull requests. Adherence to PEP 8 and code style (e.g., using `black` and `flake8`) is encouraged.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
