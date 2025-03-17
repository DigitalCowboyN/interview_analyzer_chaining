# Interview Analyzer Chaining

## Overview

Interview Analyzer Chaining is a comprehensive pipeline that enhances and contextualizes textual data extracted from transcripts or other conversational inputs. Leveraging OpenAI's GPT models, advanced NLP techniques, and structured context building, it generates richly annotated JSON outputs suitable for downstream analysis, such as persona mining, topic extraction, and sentiment analysis.

## Features

- **Sentence Segmentation**: Accurately splits input text into sentence chunks.
  
- **Context Enrichment**: Builds multi-level context windows around sentences for robust analysis.
  
- **Classification**: Performs detailed sentence-level classification, including:
  - Sentence function types (e.g., declarative, interrogative)
  - Sentence structure types (e.g., simple, compound)
  - Sentence purpose identification
  - Topic classification at multiple context depths
  - Domain-specific keyword extraction
  
- **Embedding and Clustering**: Uses embeddings and clustering (HDBSCAN) to group sentences thematically.
  
- **Visualization**: Offers embedding visualization for easy inspection of clustering effectiveness.

## Technology Stack

- **Language & Frameworks**: Python 3.10
  
- **AI Models**: OpenAI GPT-4 Turbo, Sentence Transformers
  
- **NLP Libraries**: spaCy, HDBSCAN
  
- **Utilities**: Loguru (logging), PyYAML (configuration), pandas (data handling), matplotlib (visualization)
  
- **Containerization**: Docker, VS Code Dev Container

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your_api_key_here'
   ```

4. (Optional) Build and run using Docker:
   ```bash
   docker build -t interview-analyzer .
   docker run -p 8000:8000 interview-analyzer
   ```

## Usage

To analyze a transcript or conversational input, you can use the main script or integrate the provided classes into your application. The `src/main.py` file serves as the entry point for the application. 

### Example Usage

```python
from src.agents.sentence_analyzer import sentence_analyzer

sentences = ["This is a sample sentence.", "How are you doing today?"]
results = sentence_analyzer.analyze_sentences(sentences)

for result in results:
    print(result)
```

## Configuration

The configuration for the project is managed through the `config.yaml` file. You can customize various parameters, including OpenAI API settings, context window sizes, and classification options.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
