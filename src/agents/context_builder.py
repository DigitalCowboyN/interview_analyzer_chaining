"""
context_builder.py

Defines the ContextBuilder class, responsible for creating textual contexts
for sentences based on surrounding sentences within a configurable window size.

Contexts are generated for different analysis purposes (e.g., immediate, broader)
as defined in the project configuration. The target sentence within each context
string is marked.

Usage:
    from src.agents.context_builder import context_builder
    
    sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
    # Get contexts for all sentences based on config settings
    all_contexts = context_builder.build_all_contexts(sentences)
    # Example: Access context for sentence at index 1 for 'immediate' analysis
    immediate_context_for_sentence_1 = all_contexts[1]["immediate"]

"""

from typing import List, Dict
from src.config import config  # Project configuration settings.
from src.utils.logger import get_logger  # Centralized logger for the project.

# Initialize the logger.
logger = get_logger()


class ContextBuilder:
    """
    Builds textual context strings for sentences based on configuration.

    Generates different context strings for each sentence by extracting a window
    of surrounding sentences. The size of the window is configurable per context
    type (e.g., 'immediate', 'observer') in `config.yaml`.
    The target sentence is marked within the resulting context string.

    Attributes:
        context_windows (dict): Dictionary mapping context type keys (str) to
                                window sizes (int), loaded from configuration.
    """
    def __init__(self):
        """Initializes ContextBuilder by loading context window sizes from config."""
        # Load context window sizes from configuration.
        self.context_windows = config["preprocessing"]["context_windows"]
        # Initialize the SentenceTransformer model using the configured model name.
        # self.embedder = SentenceTransformer(config["embedding"]["model_name"]) # Commented out unused embedder
        # logger.info(f"ContextBuilder initialized with embedding model: {config['embedding']['model_name']}") # Commented out related log
        logger.info("ContextBuilder initialized.") # Adjusted log message

    def build_context(self, sentences: List[str], idx: int, window_size: int) -> str:
        """
        Builds a textual context string for a sentence at a specific index.

        Extracts sentences within the defined window (`idx - window_size` to
        `idx + window_size`), marks the target sentence at `idx`, and joins the
        extracted sentences with newline characters.

        Args:
            sentences (List[str]): The full list of sentences.
            idx (int): The index of the target sentence in the list.
            window_size (int): The number of sentences to include before and after
                               the target sentence.

        Returns:
            str: The generated textual context string. Returns an empty string if
                 the input `sentences` list is empty or `idx` is out of bounds.
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
        Builds all configured textual contexts for every sentence in a list.

        Iterates through each sentence and, for each context type defined in
        `self.context_windows` (loaded from config), calls `build_context`
        to generate the appropriate context string.

        Args:
            sentences (List[str]): The list of sentences for which to build contexts.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary where keys are sentence indices (int)
                and values are dictionaries. Each inner dictionary maps context type
                keys (str, e.g., "immediate", "observer") to their corresponding
                generated context strings (str).
                Returns an empty dictionary if the input `sentences` list is empty.
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


# Create a singleton instance for application-wide use.
context_builder = ContextBuilder()
