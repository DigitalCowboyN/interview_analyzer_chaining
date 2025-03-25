# src/pipeline.py
from pathlib import Path
from src.utils.helpers import save_json
import asyncio
from src.agents.sentence_analyzer import SentenceAnalyzer
from src.utils.logger import get_logger
from src.models.analysis_result import AnalysisResult  # <- Add this import
import spacy

logger = get_logger()
nlp = spacy.load("en_core_web_sm")

def segment_text(text: str) -> list:
    # This function segments text and may reference domain-specific keywords for context.
    """
    Segment input text into sentences using spaCy.

    Parameters:
        text (str): The input text to be segmented.

    Returns:
        list: A list of segmented sentences.

    Raises:
        ValueError: If the input text is empty.
    """
    """Segment input text into sentences using spaCy."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.info(f"Segmented text into {len(sentences)} sentences.")
    return sentences

async def process_file(input_file: Path, output_dir: Path):
    """
    Process a single text file by segmenting its content into sentences,
    analyzing those sentences, and saving the results to a JSON file.

    Parameters:
        input_file (Path): The path to the input text file.
        output_dir (Path): The directory where the output JSON file will be saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    logger.info(f"Processing file: {input_file}")

    text = input_file.read_text(encoding="utf-8")
    sentences = segment_text(text)

    analyzer = SentenceAnalyzer()
    results = await analyzer.analyze_sentences(sentences)

    output_data = [result.__dict__ if isinstance(result, AnalysisResult) else result for result in results]
    output_file = output_dir / f"{input_file.stem}_analysis.json"
    save_json(output_data, output_file)

    logger.info(f"Results saved to {output_file}")

async def run_pipeline(input_dir: Path, output_dir: Path):
    """
    Run the pipeline across all text files in the specified input directory.

    This function will process each .txt file found in the input directory,
    segmenting the text into sentences and analyzing them. The results will
    be saved in the specified output directory.

    Parameters:
        input_dir (Path): The directory containing input text files.
        output_dir (Path): The directory where output JSON files will be saved.

    Returns:
        None

    Raises:
        OSError: If the output directory cannot be created.
    """
    """Run pipeline across all text files in input directory."""
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        await process_file(input_file, output_dir)

    logger.info("Pipeline run complete.")
