from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .app import get_embedding
from dotenv import load_dotenv
from .createData import createDataset, connect_unix_socket
app = FastAPI()
load_dotenv()

@app.get("/")
def root():
    return {"message": "Hello World"}

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
        result = "Oh shit bois we connected!!"
        #db_conn.execute(f"SELECT * FROM civ6_articles WHERE id = {id}")        
    return result

@app.post("/testAnswer/{prompt}")
def get_top3_docs(prompt: str) -> dict:
    return {
        0:{ 
            "Confidence":0.0835,
            "title":"Cheese",
            "heading": "Summary",
            "text":"Cheese has been around since the early ages of humanity"},
        1:{
            "Confidence":0.0665,
            "title":"Milk",
            "heading": "Production",
            "text":"Milk comes out of cows, though sometimes goats as well"},
        2:{
            "Confidence":0.0465,
            "title":"Soymilk",
            "heading": "History",
            "text":"Soymilk is arguably older than dairy milk, as plants ruled the Earth first"}
        }
@app.post("/api/v1/createData")
def createData(url:str, tableName:str):
    return createDataset(url, tableName)