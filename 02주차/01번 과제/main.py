from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/posts")
def create_post():
    return {"message": "Post created successfully"}


class Post(BaseModel):
    id: int
    title: str
    content: str
