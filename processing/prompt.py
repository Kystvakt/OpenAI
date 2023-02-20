import pandas as pd
import tiktoken

from .similarity import order_document_sections_by_query_similarity


MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def construct_prompt(
    question: str, context_embeddings: dict, df: pd.DataFrame, show_info: bool = False
) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, context_embeddings
    )

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indices = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indices.append(str(section_index))

    # Useful diagnostic information
    if show_info:
        print(f"Selected {len(chosen_sections)} document sections.")
        print("\n".join(chosen_sections_indices))

    header = """Answer the question as truthfully as possible using the provided context, \
    and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\nQ: " + question + "\nA:"
