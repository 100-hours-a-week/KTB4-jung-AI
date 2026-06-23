import os
import json
from kiwipiepy import Kiwi
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from app.core.config import settings
from app.core.vectorstore import VectorStoreManager
from app.core.llm import LLMManager

# Initialize Kiwi morpheme analyzer once at module level
kiwi = Kiwi()

class ChunkWrapper:
    def __init__(self, text: str):
        self.text = text

class RagService:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.init_bm25_retriever()

    @staticmethod
    def kiwi_tokenize(text: str) -> list[str]:
        return [token.form for token in kiwi.tokenize(text) if token.tag.startswith("N") or token.tag.startswith("V")]

    def init_bm25_retriever(self) -> None:
        try:
            all_data = self.vector_store_manager.vector_store.get()
            documents = []
            if all_data and "documents" in all_data and all_data["documents"]:
                for doc_text, metadata, doc_id in zip(all_data["documents"], all_data["metadatas"], all_data["ids"]):
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata,
                        id=doc_id
                    ))
            
            if documents:
                print(f"Initializing BM25Retriever with {len(documents)} documents...")
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    preprocess_func=self.kiwi_tokenize
                )
            else:
                print("No documents found in Chroma DB. BM25Retriever initialization skipped.")
                self.bm25_retriever = None
        except Exception as e:
            print(f"Warning during BM25Retriever initialization: {e}")
            self.bm25_retriever = None

    def get_ensemble_retriever(self, top_k: int):
        vector_retriever = self.vector_store_manager.vector_store.as_retriever(search_kwargs={"k": top_k})
        
        if hasattr(self, 'bm25_retriever') and self.bm25_retriever is not None:
            self.bm25_retriever.k = top_k
            return EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]
            )
        return vector_retriever

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
        
        # Re-build BM25 retriever with newly indexed documents
        self.init_bm25_retriever()
            
        print(f"Successfully indexed {len(ids)} documents in Chroma DB.")
        return len(ids)
        
    def query(self, question: str, top_k: int = 5) -> dict:
        retriever = self.get_ensemble_retriever(top_k)
        
        rag_chain = self.llm_manager.create_rag_chain(retriever)
        response = rag_chain.invoke({"input": question})
        
        retrieved_docs = response.get("context", [])
        contexts = [doc.page_content for doc in retrieved_docs]
        metadatas = [doc.metadata for doc in retrieved_docs]
        
        return {
            "answer": response.get("answer", ""),
            "contexts": contexts,
            "metadatas": metadatas
        }
        
    def query_stream(self, question: str, top_k: int = 5):
        retriever = self.get_ensemble_retriever(top_k)
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        stuff_chain = self.llm_manager.create_stuff_chain()
        
        def response_generator():
            for chunk in stuff_chain.stream({
                "context": docs,
                "input": question
            }):
                yield ChunkWrapper(chunk)
                
        return response_generator(), contexts, metadatas
