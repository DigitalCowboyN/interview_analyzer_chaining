"""
visualize.py

Provides utility functions for visualizing data, primarily focusing on
high-dimensional embeddings using dimensionality reduction techniques like t-SNE.
"""

# src/utils/visualize.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np
from typing import Optional, List, Union


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[Union[np.ndarray, List]] = None,
    output_file: Union[str, Path] = "embedding_plot.png",
    title: str = "Embeddings Visualization",
    figsize: tuple = (12, 8),
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
):
    """
    Generate and save a 2D scatter plot of high-dimensional embeddings using t-SNE.

    Reduces the dimensionality of the input embeddings to 2 components using t-SNE
    and plots the result as a scatter plot. If labels are provided, points are colored
    according to their labels, and a legend is included.

    Args:
        embeddings (np.ndarray): A 2D numpy array where each row is an embedding vector.
        labels (Optional[Union[np.ndarray, List]]): An optional array or list of labels 
            corresponding to each embedding. If provided, points in the plot will be 
            colored by label. Defaults to None.
        output_file (Union[str, Path]): The path (including filename) where the plot 
            image will be saved. Defaults to "embedding_plot.png".
        title (str): The title for the plot. Defaults to "Embeddings Visualization".
        figsize (tuple): The figure size (width, height) in inches for the plot. 
                       Defaults to (12, 8).
        perplexity (int): The perplexity parameter for t-SNE, related to the number 
                          of nearest neighbors. Defaults to 30.
        n_iter (int): The number of iterations for the t-SNE optimization. 
                      Defaults to 1000.
        random_state (int): The seed for the random number generator used by t-SNE 
                            for reproducibility. Defaults to 42.

    Raises:
        ImportError: If matplotlib or sklearn are not installed.
        ValueError: If input dimensions or parameters for t-SNE are invalid.
        Exception: Other exceptions related to file I/O or plotting.
    """

    # Reduce embeddings dimensionality with t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    if labels is None:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6)
    else:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(
                reduced_embeddings[idx, 0],
                reduced_embeddings[idx, 1],
                alpha=0.6,
                label=label,
            )
        plt.legend()

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Embedding visualization saved at {output_path}")
