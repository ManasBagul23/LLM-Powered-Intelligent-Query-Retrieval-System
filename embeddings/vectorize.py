from typing import List

import numpy as np

from embeddings.embedder import Embedder


def vectorize_texts(embedder: Embedder, texts: List[str]) -> np.ndarray:
    return embedder.embed(texts)
