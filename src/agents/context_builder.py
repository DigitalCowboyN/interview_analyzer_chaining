"""                                                                                                                                                                                                                                    
context_builder.py                                                                                                                                                                                                                     
                                                                                                                                                                                                                                       
This module defines the ContextBuilder class, which is responsible for creating                                                                                                                                                        
textual and embedding-based contexts for sentences. It utilizes the SentenceTransformer                                                                                                                                                
model for generating embeddings and manages context windows based on configuration settings.                                                                                                                                           
                                                                                                                                                                                                                                       
Usage Example:                                                                                                                                                                                                                         
                                                                                                                                                                                                                                       
1. Import the context builder instance:                                                                                                                                                                                                
   from src.agents.context_builder import context_builder                                                                                                                                                                              
                                                                                                                                                                                                                                       
2. Build contexts for a list of sentences:                                                                                                                                                                                             
   contexts = context_builder.build_all_contexts(sentences)                                                                                                                                                                            
"""  
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


class ContextBuilder:
    """                                                                                                                                                                                                                                
    A class to build contexts for sentences.                                                                                                                                                                                           
                                                                                                                                                                                                                                       
    This class generates both textual and embedding-based contexts for sentences                                                                                                                                                       
    based on specified window sizes from the configuration.                                                                                                                                                                            
                                                                                                                                                                                                                                       
    Attributes:                                                                                                                                                                                                                        
        context_windows (dict): The context window sizes for different types of analysis.                                                                                                                                              
        embedder (SentenceTransformer): The model used for generating sentence embeddings.                                                                                                                                             
    """
    def __init__(self):
        self.context_windows = config["preprocessing"]["context_windows"]
        self.embedder = SentenceTransformer(config["embedding"]["model_name"])

    def build_context(self, sentences: List[str], idx: int, window_size: int) -> str:
        """                                                                                                                                                                                                                            
        Build textual context around a given sentence.                                                                                                                                                                                 
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            sentences (List[str]): The list of sentences to build context from.                                                                                                                                                        
            idx (int): The index of the target sentence.                                                                                                                                                                               
            window_size (int): The number of sentences to include in the context window.                                                                                                                                               
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            str: The constructed context as a single string.                                                                                                                                                                           
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            IndexError: If the index is out of bounds for the sentences list.                                                                                                                                                          
        """
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)

        context_sentences = [sent for i, sent in enumerate(sentences[start:end]) if i + start != idx]
        context = " ".join(context_sentences)

        logger.debug(f"Built context for sentence {idx}: {context}")

        return context

    def build_embedding_context(self, sentences: List[str], idx: int, window_size: int) -> np.array:
        """                                                                                                                                                                                                                            
        Generate embedding-based context vector.                                                                                                                                                                                       
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            sentences (List[str]): The list of sentences to build context from.                                                                                                                                                        
            idx (int): The index of the target sentence.                                                                                                                                                                               
            window_size (int): The number of sentences to include in the context window.                                                                                                                                               
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            np.array: The average embedding vector of the context sentences.                                                                                                                                                           
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            ValueError: If no context sentences are available.                                                                                                                                                                         
        """
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)

        context_sentences = [sent for i, sent in enumerate(sentences[start:end]) if i + start != idx]
        if not context_sentences:
            logger.warning("No context sentences available, returning zero vector.")
            return np.zeros(self.embedder.get_sentence_embedding_dimension())

        embeddings = self.embedder.encode(context_sentences, convert_to_numpy=True)
        context_embedding = np.mean(embeddings, axis=0)

        logger.debug(f"Generated embedding context for sentence {idx}.")

        return context_embedding

    def build_all_contexts(self, sentences: List[str]) -> Dict[int, Dict[str, str]]:
        """                                                                                                                                                                                                                            
        Build all required contexts for each sentence.                                                                                                                                                                                 
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            sentences (List[str]): The list of sentences to build contexts for.                                                                                                                                                        
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            Dict[int, Dict[str, str]]: A dictionary mapping sentence indices to their contexts.                                                                                                                                        
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            ValueError: If the sentences list is empty.                                                                                                                                                                                
        """
        contexts = {}
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


# Singleton instance for pipeline-wide use
context_builder = ContextBuilder()
