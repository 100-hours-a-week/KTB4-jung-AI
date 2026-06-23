import os
import json
from langchain_core.documents import Document
from app.core.config import settings
from app.core.vectorstore import VectorStoreManager
from app.core.llm import LLMManager


class ChunkWrapper:
    def __init__(self, text: str):
        self.text = text


class RagService:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.llm_manager = LLMManager()

    def index_documents(self, chunks_file_path: str = None) -> int:
        if chunks_file_path is None:
            local_path = os.path.join(settings.BASE_DIR, "chunks.json")
            chunks_file_path = (
                local_path
                if os.path.exists(local_path)
                else os.path.join(os.path.dirname(settings.BASE_DIR), "chunks.json")
            )

        if not os.path.exists(chunks_file_path):
            raise FileNotFoundError(f"Chunks file not found at: {chunks_file_path}")

        with open(chunks_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {chunks_file_path}")

        documents_to_add = []
        ids = []

        for chunk in chunks:
            headers_str = json.dumps(chunk.get("headers", {}), ensure_ascii=False)
            metadata = {
                "id": str(chunk.get("id")),
                "header_path": chunk.get("header_path", ""),
                "headers": headers_str,
                "char_count": chunk.get("char_count", 0),
                "content": chunk.get("content", ""),
            }

            doc = Document(
                page_content=chunk.get("full_text", ""),
                metadata=metadata,
                id=str(chunk.get("id")),
            )
            documents_to_add.append(doc)
            ids.append(str(chunk.get("id")))

        self.vector_store_manager.delete_existing_documents(ids)
        self.vector_store_manager.add_documents_batch(documents_to_add, ids)

        print(f"Successfully indexed {len(ids)} documents in Chroma DB.")
        return len(ids)

    def query(self, question: str, top_k: int = 5) -> dict:
        docs = self.vector_store_manager.vector_store.similarity_search(
            question, k=top_k
        )
        contexts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        context_str = "\n\n".join(
            [f"=== Document {idx+1} ===\n{doc}" for idx, doc in enumerate(contexts)]
        )

        chain = self.llm_manager.get_chain()
        answer = chain.invoke({"context": context_str, "question": question})

        return {"answer": answer, "contexts": contexts, "metadatas": metadatas}

    def query_stream(self, question: str, top_k: int = 5):
        docs = self.vector_store_manager.vector_store.similarity_search(
            question, k=top_k
        )
        contexts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        context_str = "\n\n".join(
            [f"=== Document {idx+1} ===\n{doc}" for idx, doc in enumerate(contexts)]
        )

        chain = self.llm_manager.get_chain()

        def response_generator():
            for chunk in chain.stream({"context": context_str, "question": question}):
                yield ChunkWrapper(chunk)

        return response_generator(), contexts, metadatas
