from typing import Union
import pandas as pd
import numpy as np
import openai
import os
from transformers import GPT2TokenizerFast

def get_embedding(text: str) -> list[float]:
    openai.api_key = os.environ["OPENAI_TOKEN"] 
    result = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text)
    return result["data"][0]["embedding"]

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    We'll see about **THAT** OpenAI...
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    print("Embedding query...")
    query_embedding = get_embedding(query) 
    print("Fetching similar articles...")   
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    return document_similarities

def load_contexts(path:str)->pd.DataFrame:
    return pd.read_csv(path)


def fetch_data_and_prompt_GPT(prompt:str, results: list[(float, (str, str))], context_df:pd.DataFrame) -> str:
    """
    The idea here is to take the most relevant articles from the similarity search,
    fetch their text, and then feed as much of that text as context into a prompt
    that gets fed to GPT.

    This code is straight copy/pasted from the OpenAI cookbook. Thanks guys!

    Parameters:
     results(list[(float, (str, str))]): The results from the similarity search. Basically the output oforder_document_sections_by_query_similarity
     df(pd.DataFrame): The dataframe with the actual content in it, **not** the embeddings 
    """

    MAX_SECTION_LEN = 2048
    #TODO What's the actual max token count for GPT3 completion?
    SEPARATOR = "\n* "

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    print("Loading context dataset...")
    df = context_df

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, (title, header) in results:
        # Add contexts until we run out of space.        
        document_section = df.loc[(df.title == title) & (df.header == header)]
        chosen_sections_len += document_section.tokens.sum() + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        print(document_section.text.values[0])
        chosen_sections.append(SEPARATOR + document_section.text.values[0].replace("\n", " "))
        chosen_sections_indexes.append(str([title, header]))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + prompt + "\n A:"
    print("Prompting with... \n" + prompt)

    response = openai.Completion.create(
                prompt=prompt,
                model='text-ada-001',
                max_tokens=20,
                temperature=0.5,
                n=1
            )

    return response["choices"][0]["text"].strip(" \n")