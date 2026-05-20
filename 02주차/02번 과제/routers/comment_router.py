from database import get_db
from fastapi import APIRouter, Depends
from schemas import Comment, CommentCreate
from services.comment_service import comment_service
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/posts", tags=["Comments"])


@router.post("/{post_id}/comments", response_model=Comment)
def create_comment(post_id: int, comment: CommentCreate, db: Session = Depends(get_db)):
    return comment_service.add_comment(db, post_id, comment)


@router.get("/{post_id}/comments", response_model=list[Comment])
def get_comments(post_id: int, db: Session = Depends(get_db)):
    return comment_service.get_comments(db, post_id)


@router.delete("/{post_id}/comments/{comment_id}")
def delete_comment(post_id: int, comment_id: int, db: Session = Depends(get_db)):
    comment_service.delete_comment(db, post_id, comment_id)
    return {"message": "Comment deleted successfully"}
