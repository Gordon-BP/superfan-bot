from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .app import get_embedding
from dotenv import load_dotenv
import os
import logging
from .createData import createDataset, connect_unix_socket
app = FastAPI()
load_dotenv() # For local dev only
api_token = os.environ['API_TOKEN']

@app.on_event("startup")
async def startup_event():
    log = logging.getLogger("uvicorn.info")
    handler = logging.StreamHandler()
    log.addHandler(handler)

@app.get("/")
def root(authToken:str):
    return {"message": "Hello World"} #if authToken == api_token else None

@app.get("/api/v1/rawArticle/{id}")
def getRawArticle(id:int):
    """
    Fetches an article chunk from the articles table.

    Parameters:
        id(int): The id of the article to pull
    Returns:
        I'm not entirely sure what data type but it should be something!
    """
    pool = connect_unix_socket()
    with pool.connect() as db_conn:
        print("Oh shit bois we connected!!")
        results = db_conn.execute(f"SELECT * FROM articles WHERE articles.id = {id}")     
    return results.all()

@app.post("/api/v1/createData")
def createData(url:str, tableName:str, overrideTables:bool=False):
    returnMsg = createDataset(url, tableName, overrideTables)
    return returnMsg