import numpy as np


def cosine_similarity(first, second):
    """Return a float in [-1, 1] for two real, finite, nonzero 1-D embeddings.

    Inputs may be numeric sequences or NumPy arrays of any equal, nonempty
    length. Raise ValueError for invalid inputs, including zero vectors.
    """
    vectors = []
    for embedding in (first, second):
        try:
            vector = np.asarray(embedding)
        except (TypeError, ValueError) as exc:
            raise ValueError("Embeddings must be real numeric vectors.") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Embeddings must be nonempty one-dimensional vectors.")
        if vector.dtype.kind not in "iuf":
            raise ValueError("Embeddings must contain real numbers.")
        vector = vector.astype(np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("Embeddings must contain only finite values.")
        vectors.append(vector)

    if vectors[0].shape != vectors[1].shape:
        raise ValueError("Embeddings must have the same dimensions.")

    normalized = []
    for vector in vectors:
        scale = np.max(np.abs(vector))
        if scale == 0:
            raise ValueError("Cosine similarity is undefined for zero vectors.")
        # Scaling before the norm avoids overflow/underflow for extreme magnitudes.
        vector = vector / scale
        normalized.append(vector / np.linalg.norm(vector))
    return float(np.clip(np.dot(*normalized), -1.0, 1.0))
