from database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(50), nullable=False)
    content = Column(String(5000), nullable=False)
    writer = Column(String(50), nullable=False)
    view_count = Column(Integer, default=0)

    comments = relationship(
        "Comment", back_populates="post", cascade="all, delete-orphan"
    )
