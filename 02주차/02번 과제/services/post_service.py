from fastapi import HTTPException
from models.post import Post
from schemas.post import PostBase
from sqlalchemy.orm import Session


class PostService:
    def add_new_post(self, db: Session, post_base: PostBase) -> Post:
        new_post = Post(**post_base.model_dump())
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        return new_post

    def update_post(self, db: Session, post_id: int, post_update: PostBase) -> Post:
        post = self.get_post_detail(db, post_id)
        for key, value in post_update.model_dump().items():
            setattr(post, key, value)
        db.commit()
        db.refresh(post)
        return post

    def delete_post(self, db: Session, post_id: int):
        post = self.get_post_detail(db, post_id)
        db.delete(post)
        db.commit()

    def get_post_detail(self, db: Session, post_id: int) -> Post:
        post = db.query(Post).filter(Post.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return post

    def get_all_post(self, db: Session) -> list[Post]:
        return db.query(Post).all()

    def update_post_view_count(self, db: Session, post_id: int, view_count: int):
        post = self.get_post_detail(db, post_id)
        post.view_count = view_count
        db.commit()


post_service = PostService()
