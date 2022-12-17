from fastapi import FastAPI
import dotenv
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/{model}/")
async def get_answer(prompt: str, token:str) -> dict:
    return {"message":"This endpoint is not available right now"}