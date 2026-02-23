from collections.abc import Mapping
from numbers import Real

import numpy as np

from core.similarity import cosine_similarity


def match_embedding(embedding, known_embeddings, *, threshold):
    """Return (student_id, score), or ('Unknown', best_score) below threshold.

    known_embeddings maps nonempty string student IDs to one embedding each.
    An empty mapping returns ('Unknown', None). The threshold is inclusive and
    must be finite in [-1, 1]. Exact ties choose the lexicographically smallest
    ID, independently of mapping order. 'Unknown' is a reserved ID.

    Raise ValueError for any invalid input or incompatible embedding dimensions;
    invalid known entries are never silently skipped. Like cosine_similarity,
    this helper accepts any equal nonempty vector lengths, including FaceNet's 128.
    """
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not np.isfinite(threshold) or not -1 <= threshold <= 1):
        raise ValueError("Threshold must be a finite number in [-1, 1].")
    if not isinstance(known_embeddings, Mapping):
        raise ValueError("Known embeddings must be a mapping of student IDs to embeddings.")
    # Validate the query even when there are no known students.
    cosine_similarity(embedding, embedding)
    for student_id in known_embeddings:
        if (not isinstance(student_id, str) or not student_id.strip()
                or student_id == "Unknown"):
            raise ValueError("Student IDs must be nonempty strings other than 'Unknown'.")

    best_id, best_score = "Unknown", None
    for student_id in sorted(known_embeddings):
        try:
            score = cosine_similarity(embedding, known_embeddings[student_id])
        except ValueError as exc:
            raise ValueError(f"Invalid embedding for student {student_id!r}: {exc}") from exc
        if best_score is None or score > best_score:
            best_id, best_score = student_id, score
    if best_score is None or best_score < threshold:
        return "Unknown", best_score
    return best_id, best_score
