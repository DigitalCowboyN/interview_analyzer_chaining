"""
context_builder.py

This module defines the ContextBuilder class, which is responsible for creating
textual and embedding-based contexts for sentences. It leverages the SentenceTransformer
model to generate sentence embeddings and uses configuration settings to determine how many
sentences to include in each context window.

Usage Example:
    from src.agents.context_builder import context_builder
    contexts = context_builder.build_all_contexts(sentences)

Modifications:
    - To adjust the size of context windows, update the "preprocessing.context_windows"
      settings in the configuration file.
    - If a different embedding model is required, modify the "embedding.model_name" in config.
    - Changes in how context is constructed (e.g., different joining logic) should be reflected
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

    This class uses window sizes specified in the configuration to determine how many
    surrounding sentences to include as context for a target sentence. For embedding-based
    contexts, it uses the SentenceTransformer to generate embeddings for the context sentences
    and then computes their average.

    Attributes:
        context_windows (dict): Dictionary containing the context window sizes for different analysis types.
        embedder (SentenceTransformer): The model used for generating sentence embeddings.
    """
    def __init__(self):
        # Load context window sizes from configuration.
        self.context_windows = config["preprocessing"]["context_windows"]
        # Initialize the SentenceTransformer model using the configured model name.
        self.embedder = SentenceTransformer(config["embedding"]["model_name"])

    def build_context(self, sentences: List[str], idx: int, window_size: int) -> str:
        """
        Build textual context around a given sentence.

        This method extracts a window of sentences surrounding the target sentence,
        excluding the target sentence itself, and joins them into a single string.

        Parameters:
            sentences (List[str]): The list of sentences to build context from.
            idx (int): The index of the target sentence within the list.
            window_size (int): The number of sentences to include before and after the target sentence.

        Returns:
            str: The constructed context as a single string.

        Raises:
            IndexError: (Implicitly) if the provided index is out of bounds.
                       (Note: The method itself does not explicitly raise IndexError,
                        but careful attention is needed if idx is not valid.)
        """
        # Determine the start and end indices for the context window.
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)

        # Exclude the target sentence from the context.
        context_sentences = [sent for i, sent in enumerate(sentences[start:end]) if i + start != idx]
        # Join the context sentences into one string.
        context = " ".join(context_sentences)

        logger.debug(f"Built context for sentence {idx}: {context}")
        return context

    def build_embedding_context(self, sentences: List[str], idx: int, window_size: int) -> np.array:
        """
        Generate an embedding-based context vector.

        This method builds a context using the same window logic as build_context, then
        computes the average embedding vector of the context sentences. If no context sentences
        are available, it returns a zero vector of the appropriate dimension.

        Parameters:
            sentences (List[str]): The list of sentences to build context from.
            idx (int): The index of the target sentence.
            window_size (int): The number of sentences to include in the context window.

        Returns:
            np.array: The average embedding vector of the context sentences.

        Raises:
            ValueError: If no context sentences are available (handled by returning a zero vector with a warning).
        """
        # Determine the window range for context.
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)

        # Exclude the target sentence from the context.
        context_sentences = [sent for i, sent in enumerate(sentences[start:end]) if i + start != idx]
        if not context_sentences:
            logger.warning("No context sentences available, returning zero vector.")
            return np.zeros(self.embedder.get_sentence_embedding_dimension())

        # Generate embeddings for the context sentences.
        embeddings = self.embedder.encode(context_sentences, convert_to_numpy=True)
        # Compute the mean embedding vector as the context representation.
        context_embedding = np.mean(embeddings, axis=0)

        logger.debug(f"Generated embedding context for sentence {idx}.")
        return context_embedding

    def build_all_contexts(self, sentences: List[str]) -> Dict[int, Dict[str, str]]:
        """
        Build all required textual contexts for each sentence.

        This method iterates over all sentences, building a set of contexts for each sentence
        based on different analysis types. It returns a dictionary mapping each sentence index to
        its corresponding contexts.

        Parameters:
            sentences (List[str]): The list of sentences to build contexts for.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary where each key is a sentence index and each value
                                       is a dictionary containing context strings for keys:
                                       "structure", "immediate", "observer", "broader", "overall".

        Raises:
            ValueError: If the sentences list is empty. (Note: Currently, the method does not explicitly
                        check for an empty list, so it will return an empty dictionary.)
        """
        contexts = {}
        # Iterate over each sentence and build various context types.
        for idx, sentence in enumerate(sentences):
            contexts[idx] = {
                "structure": self.build_context(sentences, idx, self.context_windows["structure_analysis"]),
                "immediate": self.build_context(sentences, idx, self.context_windows["immediate_context"]),
                "observer": self.build_context(sentences, idx, self.context_windows["observer_context"]),
                "broader": self.build_context(sentences, idx, self.context_windows["broader_context"]),
                "overall": self.build_context(sentences, idx, self.context_windows["overall_context"]),
            }
        logger.info("Built all contexts for sentences.")
        return contexts


# Singleton instance for pipeline-wide use.
context_builder = ContextBuilder()
