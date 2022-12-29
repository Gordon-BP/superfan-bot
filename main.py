from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.app import get_embedding, vector_similarity, prompt_GPT
from dotenv import load_dotenv
import os
import pathlib
import pandas as pd
import numpy as np
import pickle
import logging
from src.createData import create_or_load_dataset, connect_unix_socket
app = FastAPI()
load_dotenv() # For local dev only
global log

@app.on_event("startup")
async def startup_event():
    global log
    log = logging.getLogger("uvicorn.info")
   # handler = logging.StreamHandler()
   # log.addHandler(handler)

@app.post("/api/v1/createData")
def create_or_load_data(tableName:str, url:str = '', overrideTables:bool=False):
    tableName = str.lower(tableName)
    create_or_load_dataset(url, tableName, overrideTables)
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
    #TODO: make some kind of cache to speed this up?
    pool = connect_unix_socket()
    with pool.connect() as conn:
        embeddings_df = pd.read_sql_table(f"{tableName}_embeddings", conn)
    results= []
    for _, row in embeddings_df.iterrows():
        title = row.title
        heading = row.heading
        sim = vector_similarity([queryEmbedding], row.vec)
        #print(f"Title: {title}\nHedockading: {heading}\nSimalarity: {sim}")
        results.append({"title":title, "heading":heading, "sim":sim})
    return pd.DataFrame.from_records(results).sort_values(by='sim', ascending=False).iloc[0:n]

@app.post('/api/v1/{tableName}/query')
def get_response(query:str, tableName:str):
    pool = connect_unix_socket()
    top_articles = get_Similar_Articles(tableName, query)
    with pool as conn:
        articles_df = pd.read_sql_table(f"{tableName}_articles", conn)
    return prompt_GPT(query, top_articles, articles_df)
