from transformers import GPT2TokenizerFast

from processing import *


def answer_query_with_context(
        query, context, document_embeddings, show_prompt=False
):
    prompt = construct_prompt(
        query,
        document_embeddings,
        context
    )

    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    return response["choices"][0]["text"].strip(" \n")


if __name__ == "__main__":
    COMPLETIONS_MODEL = "text-davinci-003"
    EMBEDDING_MODEL = "text-embedding-ada-002"

    df = pd.read_csv('https://cdn.openai.com/API/examples/data/olympics_sections_text.csv')
    df = df.set_index(["title", "heading"])

    document_embeddings = batched_embeddings(df)

    MAX_SELECTION_LEN = 500
    SEPARATOR = "\n*"
    ENCODING = "gpt2"  # encoding for text-davinci-003

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    COMPLETIONS_API_PARAMS = {
        # Use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }

    question = "How many medals did the United States win at the 2020 Olympics?"
    print("Q: " + question)
    print("A:", answer_query_with_context(question, df, document_embeddings))
