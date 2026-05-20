from database import Base, engine
from fastapi import FastAPI
from routers.comment_router import router as comment_router
from routers.post_router import router as post_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Post API")

app.include_router(post_router)
app.include_router(comment_router)


@app.get("/")
def read_root():
    return {"Hello": "Welcome to the Post API!"}
