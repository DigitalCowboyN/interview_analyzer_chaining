"""                                                                                                                                                                                                                               
test_context_builder.py                                                                                                                                                                                                           
                                                                                                                                                                                                                                  
This module contains unit tests for the ContextBuilder class, which is responsible for                                                                                                                                            
building textual and embedding-based contexts for sentences. The tests cover various                                                                                                                                              
aspects of context building, including immediate context, embedding dimensions, and                                                                                                                                               
the construction of all contexts.                                                                                                                                                                                                 
                                                                                                                                                                                                                                  
Usage Example:                                                                                                                                                                                                                    
                                                                                                                                                                                                                                  
1. Run the tests using pytest:                                                                                                                                                                                                    
   pytest tests/test_context_builder.py                                                                                                                                                                                           
"""
import pytest
import pytest
import asyncio
from src.agents.context_builder import ContextBuilder


@pytest.fixture
def sentences():
    return [
        "First sentence.",
        "Second sentence about testing.",
        "Third sentence provides context.",
        "Fourth sentence enhances detail.",
        "Fifth and final sentence."
    ]


def test_build_context(sentences):
    """                                                                                                                                                                                                                           
    Test the construction of immediate context around a given sentence.                                                                                                                                                           
                                                                                                                                                                                                                                  
    This test verifies that the build_context method correctly constructs                                                                                                                                                         
    a context string that includes the specified number of sentences                                                                                                                                                              
    surrounding the target sentence.                                                                                                                                                                                              
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        sentences: A fixture providing a list of sentences.                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The constructed context matches the expected string.                                                                                                                                                                    
    """
    builder = ContextBuilder()

    context = builder.build_context(sentences, idx=2, window_size=1)

    assert context == "Second sentence about testing. Fourth sentence enhances detail."


def test_build_embedding_context(sentences):
    """                                                                                                                                                                                                                           
    Test the generation of embedding-based context for a given sentence.                                                                                                                                                          
                                                                                                                                                                                                                                  
    This test verifies that the build_embedding_context method returns                                                                                                                                                            
    an embedding vector of the correct dimension and is not a zero vector.                                                                                                                                                        
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        sentences: A fixture providing a list of sentences.                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The shape of the embedding context matches the expected dimension.                                                                                                                                                      
        - The embedding context is not a zero vector.                                                                                                                                                                             
    """
    builder = ContextBuilder()

    embedding_context = builder.build_embedding_context(sentences, idx=2, window_size=1)

    assert embedding_context.shape[0] == builder.embedder.get_sentence_embedding_dimension()
    assert embedding_context.any()  # not a zero vector


def test_build_all_contexts(sentences):
    """                                                                                                                                                                                                                           
    Test the construction of contexts for all sentences.                                                                                                                                                                          
                                                                                                                                                                                                                                  
    This test verifies that the build_all_contexts method returns a                                                                                                                                                               
    dictionary mapping each sentence index to its corresponding context,                                                                                                                                                          
    and that each context contains the expected keys.                                                                                                                                                                             
                                                                                                                                                                                                                                  
    Parameters:                                                                                                                                                                                                                   
        sentences: A fixture providing a list of sentences.                                                                                                                                                                       
                                                                                                                                                                                                                                  
    Asserts:                                                                                                                                                                                                                      
        - The number of contexts matches the number of sentences.                                                                                                                                                                 
        - Each context contains the keys "immediate" and "observer".                                                                                                                                                              
        - The immediate context is a string.                                                                                                                                                                                      
    """
    builder = ContextBuilder()

    contexts = builder.build_all_contexts(sentences)

    assert len(contexts) == len(sentences)
    for idx, ctx in contexts.items():
        assert "immediate" in ctx
        assert "observer" in ctx
        assert isinstance(ctx["immediate"], str)
