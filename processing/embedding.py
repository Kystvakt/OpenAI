import openai
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_random_exponential

EMBEDDING_MODEL = "text-embedding-ada-002"


@retry(wait=wait_random_exponential(min=1, max=50), stop=stop_after_attempt(5))
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]


def get_doc_embedding(text: str):
    return get_embedding(text, EMBEDDING_MODEL)


def get_query_embedding(text: str):
    return get_embedding(text, EMBEDDING_MODEL)


def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    return {idx: get_embedding(row.content) for idx, row in df.iterrows()}


def batched_embeddings(df: pd.DataFrame, batch_size: int = 128) -> Dict[Tuple[str, str], List[float]]:
    integrated_embeddings = pd.DataFrame()
    for _, group in df.groupby(np.arange(len(df)) // batch_size):
        contexts = list(group.content.replace("\n", " "))

        embeddings = get_batched_embedding(text=contexts, model=EMBEDDING_MODEL)
        embeddings.set_index(group.index, inplace=True)
        integrated_embeddings = pd.concat(
            [integrated_embeddings, embeddings], axis=0, copy=False
        )
    return integrated_embeddings


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(5))
def get_batched_embedding(text, model: str = EMBEDDING_MODEL) -> pd.DataFrame:
    # Warning: the response object may not return completions in the order of
    # the prompts, so always remember to match responses back to prompts using
    # the index field.
    result = openai.Embedding.create(model=model, input=text)
    embeddings = pd.DataFrame.from_dict(result["data"])
    embeddings.drop("object", axis=1, inplace=True)
    embeddings.set_index("index", inplace=True)
    return embeddings


def load_embeddings(filename: str) -> Dict[Tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    filename is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    df = pd.read_csv(filename, header=0)
    max_dim = max(
        [int(col) for col in df.columns if col != "title" and col != "heading"]
    )
    return {
        (row.title, row.heading): [row[str(i)] for i in range(max_dim + 1)]
        for _, row in df.iterrows()
    }
