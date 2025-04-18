# src/utils/text_processing.py
import spacy
from typing import List
from src.utils.logger import get_logger

logger = get_logger()

# Initialize spaCy model globally within this module
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully for text processing.")
except OSError:
    logger.error("Could not load spaCy model 'en_core_web_sm'. "
                 "Please download it using: python -m spacy download en_core_web_sm")
    # Depending on the application's needs, you might raise an error here
    # or allow operation without spaCy if segmentation is optional/alternative exists.
    nlp = None # Indicates model loading failed

def segment_text(text: str) -> List[str]:
    """
    Segment input text into sentences using spaCy.
    
    Uses the spaCy NLP library ('en_core_web_sm') loaded in this module.
    Filters out any empty strings after stripping whitespace.
    
    Args:
        text (str): The input text to be segmented.
        
    Returns:
        List[str]: A list of sentences extracted from the text.
                   Returns an empty list if the input text is empty or contains no sentences,
                   or if the spaCy model failed to load.
    """
    if nlp is None:
        logger.error("spaCy model not loaded. Cannot segment text.")
        # Consider raising an exception if segmentation is critical
        # raise RuntimeError("spaCy model 'en_core_web_sm' is required but not loaded.")
        return []

    if not text: # Handle empty input text explicitly
        logger.debug("Input text is empty, returning empty list for segmentation.")
        return []
        
    # Process the text using the spaCy model.
    doc = nlp(text)
    # Extract sentences and remove any that are empty after stripping whitespace.
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    logger.debug(f"Segmented text into {len(sentences)} sentences.")
    return sentences 