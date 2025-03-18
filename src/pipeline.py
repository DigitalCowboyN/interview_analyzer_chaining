# src/pipeline.py
from pathlib import Path
from src.utils.helpers import save_json
from src.agents.sentence_analyzer import sentence_analyzer
from src.utils.logger import get_logger
import spacy

logger = get_logger()
nlp = spacy.load("en_core_web_sm")


def segment_text(text: str) -> list:
    """Segment input text into sentences using spaCy."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.info(f"Segmented text into {len(sentences)} sentences.")
    return sentences


async def process_file(input_file: Path, output_dir: Path):
    """Process a single transcript file through the analysis pipeline."""
    logger.info(f"Processing file: {input_file}")

    text = input_file.read_text(encoding="utf-8")
    sentences = segment_text(text)

    results = await asyncio.gather(*[sentence_analyzer.analyze_sentence(sentence) for sentence in sentences])
    # results.append(analysis_result)  # Remove this line as it is not needed

    output_file = output_dir / f"{input_file.stem}_analysis.json"
    save_json(results, output_file)

    logger.info(f"Finished processing {input_file}, results saved to {output_file}")


async def run_pipeline(input_dir: Path, output_dir: Path):
    """Run pipeline across all text files in input directory."""
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        await process_file(input_file, output_dir)  # Add await here

    logger.info("Pipeline run complete.")
