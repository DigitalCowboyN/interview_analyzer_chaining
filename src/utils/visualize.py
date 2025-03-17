# src/utils/visualize.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np


def plot_embeddings(
    embeddings,
    labels=None,
    output_file="embedding_plot.png",
    title="Embeddings Visualization",
    figsize=(12, 8),
    perplexity=30,
    n_iter=1000,
    random_state=42,
):
    """Generate a 2D scatter plot of embeddings using t-SNE."""

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
