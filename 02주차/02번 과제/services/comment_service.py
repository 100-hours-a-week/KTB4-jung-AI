from fastapi import HTTPException
from models.comment import Comment
from schemas.comment import CommentCreate
from sqlalchemy.orm import Session

from services.post_service import post_service


class CommentService:
    def add_comment(
        self, db: Session, post_id: int, comment_create: CommentCreate
    ) -> Comment:
        post_service.get_post_detail(db, post_id)

        new_comment = Comment(post_id=post_id, **comment_create.model_dump())
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)
        return new_comment

    def delete_comment(self, db: Session, post_id: int, comment_id: int):
        post_service.get_post_detail(db, post_id)

        comment = (
            db.query(Comment)
            .filter(Comment.id == comment_id, Comment.post_id == post_id)
            .first()
        )
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")

        db.delete(comment)
        db.commit()

    def get_comments(self, db: Session, post_id: int) -> list[Comment]:
        post_service.get_post_detail(db, post_id)

        return db.query(Comment).filter(Comment.post_id == post_id).all()


comment_service = CommentService()
