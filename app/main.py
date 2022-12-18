from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from os.path import exists
from app import order_document_sections_by_query_similarity, load_embeddings, load_contexts, fetch_data_and_prompt_GPT
from apiKeys import OPENAI_TOKEN, XML_FILEPATH, EMBEDDINGS_FILEPATH, URI
from createData import createDataset
app = FastAPI()
embeddings = load_embeddings("/Users/gordy/Documents/superfan-bot/data/Civ6-Articles-Embedded2.csv")
articles_df = load_contexts("/Users/gordy/Documents/superfan-bot/data/Civ6-Atricles-Cleaned.csv")

class QueryObj(BaseModel):
    prompt:str
    token:str
    modelName:str = "text-embedding-ada-002"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/Civ6/{prompt}")
async def get_top5_docs(prompt: str) -> dict:
    return {"reply":fetch_data_and_prompt_GPT(
        prompt, 
        order_document_sections_by_query_similarity(
            prompt,
            embeddings
            )[:5],
            articles_df
            )
        }

@app.get("/data/embeddingsStatus")
def getEmbeddingsStatus() -> bool:
    return exists(EMBEDDINGS_FILEPATH)

@app.post("/data/createData")
async def createData() -> str:
    if(not(exists(XML_FILEPATH))):
        raise HTTPException(status_code=412, 
        detail="No XML File found. Please export the XML from your favorite wiki and add the XML file in data. Additionally, you will need to specify the path in apiKeys.py "
        )
    elif(OPENAI_TOKEN==''):
        raise HTTPException(status_code=412,
        detail="Please specify your OpenAI token in apiKeys.py"
        )
    elif(URI==''):
        raise HTTPException(status_code=412,
        detail="Please specify the XML URI in apiKeys.py")
    else:
        await createDataset()