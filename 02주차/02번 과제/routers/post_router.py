from database import get_db
from fastapi import APIRouter, Depends
from schemas import PostBase, PostRead
from services.post_service import post_service
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/posts", tags=["Posts"])


@router.post("", response_model=PostRead)
def create_post(post: PostBase, db: Session = Depends(get_db)):
    return post_service.add_new_post(db, post)


@router.get("", response_model=list[PostRead])
def get_posts(db: Session = Depends(get_db)):
    return post_service.get_all_post(db)


@router.get("/{post_id}", response_model=PostRead)
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = post_service.get_post_detail(db, post_id)
    post_service.update_post_view_count(db, post_id, post.view_count + 1)
    db.refresh(post)
    return post


@router.put("/{post_id}", response_model=PostRead)
def update_post(post_id: int, post_update: PostBase, db: Session = Depends(get_db)):
    return post_service.update_post(db, post_id, post_update)


@router.delete("/{post_id}")
def delete_post(post_id: int, db: Session = Depends(get_db)):
    post_service.delete_post(db, post_id)
    return {"message": "Post deleted successfully"}
