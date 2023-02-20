import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from .embedding import get_embedding


def vector_similarity(x: List[float], y: List[float]) -> float:
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(
        query: str, contexts: pd.DataFrame
):
    query_embedding = get_embedding(query)
    document_similarities = sorted([
            (vector_similarity(query_embedding, doc.embedding), idx)
            for idx, doc in contexts.iterrows()
        ], reverse=True)
    return document_similarities
