import json
from pathlib import Path
from fastapi import HTTPException
from services.post_service import post_service
from schemas import Comment, CommentCreate

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "comments.json"

_initial_comments = {}
if DATA_PATH.exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        _data = json.load(f)
        _initial_comments = {int(k): [Comment(**c) for c in v] for k, v in _data.items()}


class CommentService:
    _comment_dict = _initial_comments

    def add_comment(self, post_id: int, comment_create: CommentCreate):
        post_service.get_post_detail(post_id)
        
        comments = self._comment_dict.setdefault(post_id, [])
        new_comment_id = len(comments) + 1
        
        comment = Comment(
            id=new_comment_id,
            **comment_create.model_dump()
        )
        comments.append(comment)
        
        return comment

    def delete_comment(self, post_id: int, comment_id: int):
        post_service.get_post_detail(post_id)
        
        comments = self._comment_dict.setdefault(post_id, [])
        comment_exists = any(c.id == comment_id for c in comments)
        if not comment_exists:
            raise HTTPException(status_code=404, detail="Comment not found")
            
        self._comment_dict[post_id] = [c for c in comments if c.id != comment_id]

    def get_comments(self, post_id: int):
        post_service.get_post_detail(post_id)
        return self._comment_dict.setdefault(post_id, [])


comment_service = CommentService()
