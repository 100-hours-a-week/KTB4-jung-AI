import json
from pathlib import Path
from fastapi import HTTPException
from schemas import PostBase, PostRead

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "posts.json"

_initial_posts = {}
if DATA_PATH.exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        _data = json.load(f)
        _initial_posts = {item["id"]: PostRead(**item) for item in _data}


class PostService:
    _post_dict = _initial_posts
    _max_post_id = max(_initial_posts.keys()) if _initial_posts else 0

    def add_new_post(self, post_base: PostBase):
        PostService._max_post_id += 1
        new_id = PostService._max_post_id

        post = PostRead(id=new_id, view_count=0, **post_base.model_dump())
        self._post_dict[new_id] = post

        return post

    def update_post(self, post_id: int, post_update: PostBase):
        old_post = self.get_post_detail(post_id)

        updated_post = PostRead(
            id=post_id,
            view_count=old_post.view_count,
            **post_update.model_dump(),
        )

        self._post_dict[post_id] = updated_post
        return updated_post

    def delete_post(self, post_id: int):
        if post_id not in self._post_dict:
            raise HTTPException(status_code=404, detail="Post not found")

        del self._post_dict[post_id]

    def get_post_detail(self, post_id: int):
        post = self._post_dict.get(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return post

    def get_all_post(self):
        return list(self._post_dict.values())

    def update_post_view_count(self, post_id: int, view_count: int):
        post = self.get_post_detail(post_id)
        post.view_count = view_count


post_service = PostService()
