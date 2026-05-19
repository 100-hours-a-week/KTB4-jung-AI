from fastapi import HTTPException
from schemas import PostBase, PostRead, PostUpdate


class PostStorage:
    post_dict = dict()
    post_list = []

    def add_new_post(self, post_base: PostBase):
        new_id = self.post_list[-1].id + 1 if self.post_list else 1

        post = PostRead(id=new_id, view_count=0, **post_base.model_dump())

        self.post_dict[new_id] = post
        self.post_list.append(post)

        return post

    def update_post(self, post_update: PostUpdate):
        old_post = self.post_dict.get(post_update.id)

        if not old_post:
            raise HTTPException(status_code=404, detail="Post not found")

        updated_post = PostRead(
            id=post_update.id,
            view_count=old_post.view_count,
            **post_update.model_dump(exclude={"id"}),
        )

        self.post_dict[post_update.id] = updated_post

        for idx, post in enumerate(self.post_list):
            if post.id == post_update.id:
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


post_service = PostStorage()
