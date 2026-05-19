from fastapi import FastAPI
from routers.post_router import router as post_router

# Spring의 Application 클래스 역할
app = FastAPI(title="Post API")

# Spring의 @ComponentScan이나 설정 파일에서 빈 등록하듯 라우터를 등록합니다.
app.include_router(post_router)

@app.get("/")
def read_root():
    return {"Hello": "Welcome to the Post API!"}
