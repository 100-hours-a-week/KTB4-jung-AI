from fastapi import HTTPException
from schemas import PostBase, PostRead


class PostService:
    post_dict = {
        1: PostRead(
            id=1, title="제목 1", content="내용 1", writer="작성자 1", view_count=0
        ),
        2: PostRead(
            id=2, title="제목 2", content="내용 2", writer="작성자 2", view_count=0
        ),
        3: PostRead(
            id=3, title="제목 3", content="내용 3", writer="작성자 3", view_count=0
        ),
    }
    post_list = list(post_dict.values())

    def add_new_post(self, post_base: PostBase):
        new_id = self.post_list[-1].id + 1 if self.post_list else 1

        post = PostRead(id=new_id, view_count=0, **post_base.model_dump())

        self.post_dict[new_id] = post
        self.post_list.append(post)

        return post

    def update_post(self, post_id: int, post_update: PostBase):
        old_post = self.get_post_detail(post_id)

        updated_post = PostRead(
            id=post_id,
            view_count=old_post.view_count,
            **post_update.model_dump(exclude={"id"}),
        )

        self.post_dict[post_id] = updated_post

        for idx, post in enumerate(self.post_list):
            if post.id == post_id:
                self.post_list[idx] = updated_post
                break

        return updated_post

    def delete_post(self, post_id: int):
        if post_id not in self.post_dict:
            raise HTTPException(status_code=404, detail="Post not found")

        del self.post_dict[post_id]
        self.post_list = [post for post in self.post_list if post.id != post_id]

    def get_post_detail(self, post_id: int):
        post = self.post_dict.get(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return post

    def get_all_post(self):
        return self.post_list

    def update_post_view_count(self, post_id: int, view_count: int):
        post = self.post_dict.get(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        post.view_count = view_count


post_service = PostService()
