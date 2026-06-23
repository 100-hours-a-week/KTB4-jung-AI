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
            chunks_file_path = local_path if os.path.exists(local_path) else os.path.join(os.path.dirname(settings.BASE_DIR), "chunks.json")
            
        if not os.path.exists(chunks_file_path):
            raise FileNotFoundError(f"Chunks file not found at: {chunks_file_path}")
            
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
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
                "content": chunk.get("content", "")
            }
            
            doc = Document(
                page_content=chunk.get("full_text", ""),
                metadata=metadata,
                id=str(chunk.get("id"))
            )
            documents_to_add.append(doc)
            ids.append(str(chunk.get("id")))
            
        self.vector_store_manager.delete_existing_documents(ids)
        self.vector_store_manager.add_documents_batch(documents_to_add, ids)
            
        print(f"Successfully indexed {len(ids)} documents in Chroma DB.")
        return len(ids)
        
    def query(self, question: str, top_k: int = 5) -> dict:
        # 1. Retriever 생성
        retriever = self.vector_store_manager.vector_store.as_retriever(search_kwargs={"k": top_k})
        
        # 2. Retrieval RAG Chain 구성 및 호출
        rag_chain = self.llm_manager.create_rag_chain(retriever)
        response = rag_chain.invoke({"input": question})
        
        # 3. 결과 파싱 (LangChain 표준 래퍼에서 컨텍스트 및 답변 반환)
        retrieved_docs = response.get("context", [])
        contexts = [doc.page_content for doc in retrieved_docs]
        metadatas = [doc.metadata for doc in retrieved_docs]
        
        return {
            "answer": response.get("answer", ""),
            "contexts": contexts,
            "metadatas": metadatas
        }
        
    def query_stream(self, question: str, top_k: int = 5):
        # 1. 동기 문서 검색 (FastAPI SSE 메타데이터 조기 응답 규격 만족용)
        retriever = self.vector_store_manager.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        # 2. Stuff Documents Chain 구성 및 스트리밍 호출
        stuff_chain = self.llm_manager.create_stuff_chain()
        
        def response_generator():
            for chunk in stuff_chain.stream({
                "context": docs,
                "input": question
            }):
                yield ChunkWrapper(chunk)
                
        return response_generator(), contexts, metadatas
