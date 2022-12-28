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
    return np.dot(np.array(x), np.array(y))[0]

def prompt_GPT(prompt:str, top_articles: pd.DataFrame, articles_df:pd.DataFrame) -> str:
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
    SEPARATOR = "\n "

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    chosen_sections = []
    chosen_sections_len = 0
     
    for _, row in top_articles.iterrows():
        # Add contexts until we run out of space. 
        articleRow = articles_df.loc[(articles_df['title'] == row.title) & (articles_df['heading'] == row.heading)]      
        document_section = f"{articleRow.title.values[0]} - {articleRow.heading.values[0]}:\n{articleRow.text.values[0]}"
        print(document_section)
        chosen_sections_len += len(tokenizer.encode(document_section))
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        chosen_sections.append(SEPARATOR + document_section)
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + prompt + "\n A:"
    print("Prompting with... \n" + prompt)

    response = openai.Completion.create(
                prompt=prompt,
                model='text-ada-001',
                max_tokens=100,
                temperature=0.7,
                n=1
            )

    return response["choices"][0]["text"].strip(" \n")