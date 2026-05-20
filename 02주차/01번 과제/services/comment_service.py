import json
from pathlib import Path

from schemas import Comment, CommentCreate

from services.post_service import post_service

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "comments.json"

_initial_comments = {}
if DATA_PATH.exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        _data = json.load(f)
        for post_id, comment_list in _data.items():
            parsed_comments = []
            for comment in comment_list:
                parsed_comments.append(Comment(**comment))
            _initial_comments[int(post_id)] = parsed_comments


class CommentService:
    _comment_dict = _initial_comments

    def add_comment(self, post_id: int, comment_create: CommentCreate):
        post_service.get_post_detail(post_id)

        comments = self._comment_dict.setdefault(post_id, [])
        new_comment_id = len(comments) + 1

        comment = Comment(id=new_comment_id, **comment_create.model_dump())
        comments.append(comment)

        return comment

    def delete_comment(self, post_id: int, comment_id: int):
        post_service.get_post_detail(post_id)

        comments = self._comment_dict.setdefault(post_id, [])
        self._comment_dict[post_id] = [c for c in comments if c.id != comment_id]

    def get_comments(self, post_id: int):
        post_service.get_post_detail(post_id)
        return self._comment_dict.setdefault(post_id, [])


comment_service = CommentService()
