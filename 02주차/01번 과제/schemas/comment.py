from pydantic import BaseModel, Field


class CommentBase(BaseModel):
    content: str = Field(..., min_length=1, max_length=500, description="댓글 내용")
    writer: str = Field(..., min_length=1, max_length=50, description="댓글 작성자")


class CommentCreate(CommentBase):
    pass


class Comment(CommentBase):
    id: int = Field(..., description="댓글 고유 ID")
