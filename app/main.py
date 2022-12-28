from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .app import get_embedding, vector_similarity, prompt_GPT
from dotenv import load_dotenv
import os
import pathlib
import pandas as pd
import numpy as np
import pickle
import logging
from .createData import create_or_load_dataset, connect_unix_socket
app = FastAPI()
load_dotenv() # For local dev only
api_token = os.environ['API_TOKEN']
global log
articles_df = pd.DataFrame()
embeddings_df = pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    log = logging.getLogger("uvicorn.info")
   # handler = logging.StreamHandler()
   # log.addHandler(handler)

@app.post("/api/v1/createData")
def create_or_load_data(tableName:str, url:str = '', overrideTables:bool=False):
    tableName = str.lower(tableName)
    if(pathlib.Path(f"./data/{tableName}_articles.gz").is_file()):
        print("Tables exist in memory already")
        global embeddings_df
        global articles_df
        embeddings_df = pd.read_pickle(
            f"./data/{tableName}_embeddings.gz", 
            compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        articles_df = pd.read_pickle(
            f"./data/{tableName}_articles.gz", 
            compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        return f"Loaded tables"
    tables = create_or_load_dataset(url, tableName, overrideTables)
    articles_df = tables[f"{tableName}_articles"]['dataFrame']
    embeddings_df = tables[f"{tableName}_embeddings"]['dataFrame']
    tables[f"{tableName}_articles"]['dataFrame'].to_pickle(
        path=f"./data/{tableName}_articles.gz",
        compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}, 
        protocol=-1)
    tables[f"{tableName}_embeddings"]['dataFrame'].to_pickle(
        path=f"./data/{tableName}_embeddings.gz",
        compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}, 
        protocol=-1)
    return f"Loaded tables"

@app.get("/")
def root():
    return {"message": "Hello World"} #if authToken == api_token else None

@app.get("/api/v1/{tableName}/search/")
def get_Similar_Articles(tableName:str, query:str, n:int = 5) -> pd.DataFrame:
    """
    Endpoint for similarity search using the user-provided query

    Parameters:
        tableName(str): the prefix for the tables to search
        query(str): the question to perform similarity search on
        n(int): how many results to return. Default is 5
    Returns:
        dict: the top n most similar text blurbs
    """
    queryEmbedding = get_embedding(query)
    global embeddings_df
    global articles_df
    if((embeddings_df.empty) or (articles_df.empty)):
        embeddings_df = pd.read_pickle(
            f"./data/{tableName}_embeddings.gz", 
            compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        articles_df = pd.read_pickle(
            f"./data/{tableName}_articles.gz", 
            compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    results= []
    for _, row in embeddings_df.iterrows():
        title = row.title
        heading = row.heading
        sim = vector_similarity([queryEmbedding], row.vec)
        #print(f"Title: {title}\nHeading: {heading}\nSimalarity: {sim}")
        results.append({"title":title, "heading":heading, "sim":sim})
    return pd.DataFrame.from_records(results).sort_values(by='sim', ascending=False).iloc[0:n]

@app.post('/api/v1/{tableName}/query')
def get_response(query:str, tableName:str):
    global articles_df
    top_articles = get_Similar_Articles(tableName, query)
    return prompt_GPT(query, top_articles, articles_df)
