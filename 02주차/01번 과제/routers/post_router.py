from fastapi import APIRouter
from schemas import PostBase, PostRead
from services.post_service import post_service

router = APIRouter(prefix="/api/posts", tags=["Posts"])


@router.post("", response_model=PostRead)
def create_post(post: PostBase):
    return post_service.add_new_post(post)


@router.get("", response_model=list[PostRead])
def get_posts():
    return post_service.get_all_post()


@router.get("/{post_id}", response_model=PostRead)
def get_post(post_id: int):
    return post_service.get_post_detail(post_id)


@router.put("/{post_id}", response_model=PostRead)
def update_post(post_id: int, post_update: PostBase):
    return post_service.update_post(post_id, post_update)


@router.delete("/{post_id}")
def delete_post(post_id: int):
    post_service.delete_post(post_id)
    return {"message": "Post deleted successfully"}
