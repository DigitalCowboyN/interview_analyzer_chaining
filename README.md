# Interview Analyzer Chaining

## Overview

Interview Analyzer Chaining is a comprehensive pipeline designed to enhance and contextualize textual data extracted from transcripts or other conversational inputs. Using OpenAI's GPT models, advanced natural language processing techniques, and structured context building, the pipeline generates richly annotated JSON outputs. These outputs can be used for downstream analysis such as persona mining, topic extraction, sentiment analysis, and thematic clustering.

## Features

- **Sentence Segmentation**  
  Splits input text into individual sentences using spaCy.

- **Context Enrichment**  
  Constructs multi-level context windows for each sentence (e.g., immediate, observer, broader) and generates both textual and embedding-based contexts using Sentence Transformers.

- **Detailed Classification**  
  Performs sentence-level classification including:
  - Sentence function type (e.g., declarative, interrogative)
  - Sentence structure type (e.g., simple, compound)
  - Sentence purpose identification
  - Topic classification at multiple levels (e.g., immediate, broader)
  - Domain-specific keyword extraction

- **Embedding & Clustering**  
  Utilizes embeddings (via Sentence Transformers) and clustering (e.g., HDBSCAN) to group sentences thematically.

- **Visualization**  
  Provides tools for visualizing embeddings to assess clustering performance.

- **Robust Testing & Documentation**  
  Comprehensive tests ensure each module behaves as expected, with detailed inline documentation to facilitate modifications and extensions.

## Technology Stack

- **Programming Language:** Python 3.10+
- **AI Models:** OpenAI GPT-4 Turbo, Sentence Transformers
- **NLP Libraries:** spaCy, HDBSCAN
- **Utilities:** Loguru (logging), PyYAML (configuration), pandas (data handling), matplotlib (visualization)
- **Containerization:** Docker, VS Code Dev Containers
- **Testing:** pytest with asyncio support and extensive module coverage

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Configure Environmental Variables**
   Set your OpenAI AIP Key
   ```bash
   export OPENAI_API_KEY='your_api_key_here'

4. **(Optional) Using Docker**
   Set your OpenAI AIP Key
   ```bash
   docker build -t interview-analyzer .
   docker run -p 8000:8000 interview-analyzer


## Usage

The project is designed to be both a standalone application and a modular library. The main entry point is src/main.py, which orchestrates the entire pipeline. You can also import individual components into your own application.

### Example Usage

```python
from src.agents.sentence_analyzer import sentence_analyzer

# List of sentences extracted from a transcript.
sentences = ["This is a sample sentence.", "How are you doing today?"]

# Asynchronously analyze the sentences.
results = await sentence_analyzer.analyze_sentences(sentences)

for result in results:
    print(result)
```

## Configuration

Configuration is managed via the config.yaml file. This file includes settings for:
- OpenAI API: API key, model name, max tokens, temperature, and retry policies.
- Context Windows: Sizes for various context types (immediate, observer, broader, etc.).
- Embedding Model: Model name for generating sentence embeddings.
- Classification Prompts: YAML files defining classification instructions.
Adjust these settings to modify the pipeline's behavior. If you update the configuration, be sure to also update any related tests or documentation.

## Testing & Developing

The project comes with a comprehensive test suite covering:
- Agent API Calls & Retry Logic: Ensuring robust error handling and correct JSON parsing.
- Pipeline Functions: Testing sentence segmentation, file processing, and the overall pipeline run.
- Sentence Analysis: Verifying classification across multiple dimensions.
- Context Building: Confirming textual and embedding-based context generation, including edge cases.

To run the tests, execute:
```bash
   pytest
```

## Contributing

Contributions are welcome! Whether you're fixing a bug, adding a new feature, or improving documentation, please:
- Open an issue or submit a pull request.
- Follow the project's code style and ensure tests pass before submitting changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
