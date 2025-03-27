"""
pipeline.py

This module defines the core functions for processing text files in the interview
analyzer pipeline. The pipeline consists of the following steps:

    1. Segmenting the input text into individual sentences using spaCy.
    2. Analyzing each sentence via the SentenceAnalyzer (which interacts with OpenAI's Responses API).
    3. Saving the analysis results to a JSON file for further processing.

Key functions:
    - segment_text: Splits input text into a list of sentences.
    - process_file: Processes a single text file by segmenting its content, analyzing the sentences,
      and saving the results.
    - run_pipeline: Iterates over all text files in an input directory and processes each file.

Usage:
    To run the pipeline for a set of text files, call run_pipeline with the input and output directories:
    
        await run_pipeline(Path("input_directory"), Path("output_directory"))
    
Modifications:
    - If the segmentation logic changes (e.g., different criteria for sentence boundaries), update segment_text.
    - If the analysis output or structure changes (e.g., additional fields), update process_file accordingly.
    - When altering file I/O behavior, ensure that both process_file and run_pipeline continue to handle errors
      (e.g., missing files or directories) appropriately.
      
Dependencies:
    - spaCy: For natural language processing and sentence segmentation.
    - SentenceAnalyzer: A class that performs sentence-level analysis using the OpenAI API.
    - save_json: A helper function to write JSON data to a file.
    - Logging: Uses a centralized logger for traceability.
"""

from pathlib import Path
from src.utils.helpers import save_json  # Helper function for saving JSON files.
import asyncio
from src.agents.sentence_analyzer import SentenceAnalyzer  # Class that analyzes sentences using OpenAI API.
from src.utils.logger import get_logger  # Centralized logger.
from src.models.analysis_result import AnalysisResult  # Pydantic model for structured analysis results.
import spacy

# Initialize the logger and load the spaCy model.
logger = get_logger()
nlp = spacy.load("en_core_web_sm")

def segment_text(text: str) -> list:
    """
    Segment input text into sentences using spaCy.
    
    This function uses the spaCy NLP library to split the provided text into sentences.
    It filters out any empty sentences after stripping whitespace.
    
    Parameters:
        text (str): The input text to be segmented.
        
    Returns:
        list: A list of sentences as strings. For example, if the text contains three
              sentences, the function returns a list with three elements.
              
    Raises:
        ValueError: If desired behavior changes (e.g., you want to raise an error for empty input).
                   Currently, if the input is empty, an empty list is returned.
                   
    Usage:
        sentences = segment_text("Hello world. How are you?")
        # sentences => ["Hello world.", "How are you?"]
    """
    # Process the text using the spaCy model.
    doc = nlp(text)
    # Extract sentences and remove any that are empty after stripping whitespace.
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.info(f"Segmented text into {len(sentences)} sentences.")
    return sentences

async def process_file(input_file: Path, output_dir: Path):
    """
    Process a single text file by segmenting its content into sentences,
    analyzing those sentences, and saving the results to a JSON file.
    
    The function performs the following steps:
      1. Reads the content of the provided input file.
      2. Segments the text into individual sentences using segment_text.
      3. Analyzes each sentence using an instance of SentenceAnalyzer, which sends
         prompts to the OpenAI API and returns structured analysis.
      4. Converts the analysis results to dictionaries (if they are instances of AnalysisResult)
         and saves them to a JSON file in the specified output directory.
         
    Parameters:
        input_file (Path): The path to the input text file.
        output_dir (Path): The directory where the output JSON file will be saved.
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If the input file does not exist.
        
    Modifications:
        - If the analysis result format changes, update the conversion logic in output_data.
        - Changing the file naming convention (e.g., suffix for the output file) should be updated here.
    """
    logger.info(f"Processing file: {input_file}")

    # Read the file's content with UTF-8 encoding.
    text = input_file.read_text(encoding="utf-8")
    # Segment the text into sentences.
    sentences = segment_text(text)

    # Create an instance of SentenceAnalyzer to analyze the segmented sentences.
    analyzer = SentenceAnalyzer()
    results = await analyzer.analyze_sentences(sentences)

    # Convert analysis results to dictionaries if they are instances of AnalysisResult.
    output_data = [result.__dict__ if isinstance(result, AnalysisResult) else result for result in results]
    # Construct the output file path based on the input file's stem.
    output_file = output_dir / f"{input_file.stem}_analysis.json"
    # Save the output data as a JSON file.
    save_json(output_data, output_file)

    logger.info(f"Results saved to {output_file}")

async def run_pipeline(input_dir: Path, output_dir: Path):
    """
    Run the pipeline across all text files in the specified input directory.
    
    This function searches for all .txt files within the input directory and processes
    each file by:
      - Segmenting the text into sentences.
      - Analyzing each sentence using the SentenceAnalyzer.
      - Saving the analysis results to the output directory.
      
    Parameters:
        input_dir (Path): The directory containing input text files.
        output_dir (Path): The directory where output JSON files will be saved.
        
    Returns:
        None
        
    Raises:
        OSError: If the output directory cannot be created.
        
    Modifications:
        - If you add support for other file types (e.g., .md files), update the file glob pattern.
        - Logging messages can be adjusted for more detailed traceability.
    """
    # Find all .txt files in the input directory.
    input_files = list(input_dir.glob("*.txt"))

    if not input_files:
        logger.warning(f"No input files found in {input_dir}")
        return

    # Ensure the output directory exists; create it if necessary.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each input file asynchronously.
    for input_file in input_files:
        await process_file(input_file, output_dir)

    logger.info("Pipeline run complete.")
