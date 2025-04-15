"""
context_builder.py

This module defines the ContextBuilder class, which is responsible for creating
textual and embedding-based contexts for sentences. It leverages the SentenceTransformer
model to generate sentence embeddings and uses configuration settings to determine how many
sentences to include in each context window. The textual context includes the target
sentence marked within its surrounding window.

Usage Example:
    from src.agents.context_builder import context_builder
    contexts = context_builder.build_all_contexts(sentences)

Modifications:
    - To adjust the size of context windows, update the "preprocessing.context_windows"
      settings in the configuration file.
    - If a different embedding model is required, modify the "embedding.model_name" in config.
    - Changes in how context is constructed (e.g., different marking logic) should be reflected
      in the build_context method.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import config  # Project configuration settings.
from src.utils.logger import get_logger  # Centralized logger for the project.

# Initialize the logger.
logger = get_logger()


class ContextBuilder:
    """
    A class to build both textual and embedding-based contexts for sentences.

    Textual context includes the target sentence marked within its surrounding window,
    joined by newlines. Embedding-based context represents the average embedding of
    the surrounding sentences *excluding* the target.

    Attributes:
        context_windows (dict): Dictionary containing the context window sizes for different analysis types.
        embedder (SentenceTransformer): The model used for generating sentence embeddings.
    """
    def __init__(self):
        # Load context window sizes from configuration.
        self.context_windows = config["preprocessing"]["context_windows"]
        # Initialize the SentenceTransformer model using the configured model name.
        # self.embedder = SentenceTransformer(config["embedding"]["model_name"]) # Commented out unused embedder
        # logger.info(f"ContextBuilder initialized with embedding model: {config['embedding']['model_name']}") # Commented out related log
        logger.info("ContextBuilder initialized.") # Adjusted log message

    def build_context(self, sentences: List[str], idx: int, window_size: int) -> str:
        """
        Build textual context around a given sentence, marking the target sentence.

        Extracts a window of sentences, includes the target sentence marked with
        '>>> TARGET: ... <<<', and joins all sentences in the window with newlines.

        Parameters:
            sentences (List[str]): The list of sentences to build context from.
            idx (int): The index of the target sentence within the list.
            window_size (int): Number of sentences before/after target to potentially include.

        Returns:
            str: The constructed context string with newlines and marked target.
                 Returns an empty string if the input sentences list is empty or idx is invalid,
                 though primary validation should occur upstream.
        """
        if not sentences or idx < 0 or idx >= len(sentences):
            logger.warning(f"Attempted to build context with invalid input: len(sentences)={len(sentences)}, idx={idx}")
            return ""
            
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)
        
        context_parts = []
        for i in range(start, end):
            sentence_text = sentences[i]
            if i == idx:
                # Mark the target sentence
                context_parts.append(f">>> TARGET: {sentence_text} <<<")
            else:
                context_parts.append(sentence_text)
                
        # Join the parts with actual newlines
        context = "\n".join(context_parts)

        logger.debug(f"Built context for sentence {idx} with marked target.")
        return context

    # Commented out unused embedding context method
    # def build_embedding_context(self, sentences: List[str], idx: int, window_size: int) -> np.array:
    #     """
    #     Generate an embedding-based context vector (excluding the target sentence).
    #
    #     Computes the average embedding vector of the context sentences surrounding the target.
    #     Returns a zero vector if no context sentences are available.
    #
    #     Parameters:
    #         sentences (List[str]): The list of sentences to build context from.
    #         idx (int): The index of the target sentence (used to exclude it).
    #         window_size (int): The number of sentences to include in the context window.
    #
    #     Returns:
    #         np.array: The average embedding vector of the surrounding context sentences.
    #     """
    #     if not sentences or idx < 0 or idx >= len(sentences):
    #         logger.warning(f"Attempted to build embedding context with invalid input: len(sentences)={len(sentences)}, idx={idx}. Returning zero vector.")
    #         return np.zeros(self.embedder.get_sentence_embedding_dimension())
    #
    #     start = max(0, idx - window_size)
    #     end = min(len(sentences), idx + window_size + 1)
    #
    #     # Explicitly exclude the target sentence for embedding calculation
    #     context_sentences_for_embedding = [sent for i, sent in enumerate(sentences[start:end]) if i + start != idx]
    #     
    #     if not context_sentences_for_embedding:
    #         logger.debug(f"No surrounding context sentences available for embedding (sentence {idx}), returning zero vector.")
    #         # Note: Previously warning, changed to debug as zero window size is valid use case
    #         return np.zeros(self.embedder.get_sentence_embedding_dimension())
    #
    #     try:
    #         embeddings = self.embedder.encode(context_sentences_for_embedding, convert_to_numpy=True)
    #         context_embedding = np.mean(embeddings, axis=0)
    #         logger.debug(f"Generated embedding context (excluding target) for sentence {idx}.")
    #         return context_embedding
    #     except Exception as e:
    #         logger.error(f"Error generating embedding context for sentence {idx}: {e}", exc_info=True)
    #         # Return zero vector on encoding/mean calculation error
    #         return np.zeros(self.embedder.get_sentence_embedding_dimension())

    def build_all_contexts(self, sentences: List[str]) -> Dict[int, Dict[str, str]]:
        """
        Build all required textual contexts for each sentence.

        Iterates over all sentences, building a set of contexts for each based on
        different analysis types defined in the config. Each context string now
        includes the target sentence marked within its window and uses newlines.

        Parameters:
            sentences (List[str]): The list of sentences to build contexts for.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary where each key is a sentence index and each value
                                       is a dictionary containing context strings for keys like:
                                       "structure", "immediate", "observer", "broader", "overall".
                                       Returns empty dict if input sentences list is empty.
        """
        if not sentences:
            logger.warning("build_all_contexts called with empty sentences list.")
            return {}
            
        contexts = {}
        # Pre-calculate keys to avoid dictionary lookups inside the loop
        window_keys = list(self.context_windows.keys())
        
        for idx, sentence in enumerate(sentences):
            sentence_contexts = {}
            for key in window_keys:
                window_size = self.context_windows.get(key, 0) # Default to 0 if key missing
                sentence_contexts[key] = self.build_context(sentences, idx, window_size)
            contexts[idx] = sentence_contexts
            
        logger.info(f"Built all textual contexts for {len(sentences)} sentences.")
        return contexts


# Singleton instance for pipeline-wide use.
context_builder = ContextBuilder()
