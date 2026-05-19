from pydantic import BaseModel, Field


class PostBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=50, description="게시글 제목")
    content: str = Field(..., min_length=1, max_length=5000, description="게시글 내용")
    writer: str = Field(..., min_length=1, max_length=50, description="작성자 이름")


class PostCreate(PostBase):
    pass


class PostRead(PostBase):
    id: int = Field(..., description="게시글 고유 ID")
    view_count: int = Field(default=0, description="조회수")


class PostUpdate(PostBase):
    id: int = Field(..., description="게시글 고유 ID")
