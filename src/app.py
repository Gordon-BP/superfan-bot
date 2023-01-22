from typing import Union
import pandas as pd
import pinecone
import numpy as np
import cohere
import os
from transformers import GPT2TokenizerFast

def get_embedding(text:Union[str , pd.Series]) -> list[float]:
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    result = co.embed(
        texts=text,
        model='small',
        truncate='LEFT'
        ).embeddings
    #print(type(result))
    #print(result)
    print("Embeddings done!")
    return result

def query_index(query: str, index: str, top_k:int=3) -> dict[str]:
    """
    Uses similarity search to find relevent data chunks from the Pinecone index
    """
    print(f"Searching for a match to query {query}...")
    query_vector = get_embedding(query)
    print(f"Embeddings got:\n{query_vector[0:9]}")
    idx = pinecone.Index(index)
    print("Index got")
    print(idx.describe_index_stats())
    results = idx.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
        )
    print("results finished")
    print(results)
    return results

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

    MAX_SECTION_LEN = 1024
    #TODO What's the actual max token count for GPT3 completion?
    #TODO make this configurable in the settings
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
    
    header = """
    Use the included context to answer the question as truthfully as possible.\n\nContext:\n
    """
    #TODO: Prompt screen should also be configurable
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + prompt + "\n A:"
    print("Prompting with... \n" + prompt)
    #TODO: Add model features like max tokens & temp as parameters in the yaml or docker files
    response = openai.Completion.create(
                prompt=prompt,
                model='text-ada-001',
                max_tokens=100,
                temperature=0.7,
                n=1
            )
    return response["choices"][0]["text"].strip(" \n")

def prompt_completion(query:str, results:dict[list[dict]]) -> str:
    co = cohere.Client(os.environ['COHERE_API_KEY'])
    myprompt = f"""
        Using the above context, provide only the answer to the question. If you do not know the answer, say 'I don't know'
        Context:{results}
        Question:{query}
        Answer:"""
        #        
       # Context: Geralt of Rivia is a witcher
       # Question: Who is Geralt?
       # Answer: Geralt is a witcher.
       # 
    response = co.generate(
        model='medium',
        num_generations=1,
        max_tokens=40,
        temperature=0.2,
   #     stop_sequences=["--"],
        prompt=myprompt
    )
    print(response)
    print("response zero:")
    print(response[0])
    return response[0]