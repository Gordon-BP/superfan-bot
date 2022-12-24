from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app import *
#from createData import createDataset
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/testAnswer/{prompt}")
def get_top3_docs(prompt: str) -> dict:
    return {[
        { 
            "Confidence":0.0835,
            "title":"Cheese",
            "heading": "Summary",
            "text":"Cheese has been around since the early ages of humanity"},
        {
            "Confidence":0.0665,
            "title":"Milk",
            "heading": "Production",
            "text":"Milk comes out of cows, though sometimes goats as well"},
        {
            "Confidence":0.0465,
            "title":"Soymilk",
            "heading": "History",
            "text":"Soymilk is arguably older than dairy milk, as plants ruled the Earth first"}
        ]}
@app.post("/api/v1/createData")
def createData(url:str, tableName:str):
   # createDataset(url, tableName)
    return "nope not doing that yet LOL"