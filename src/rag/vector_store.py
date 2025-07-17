import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore  # if used generically
import chromadb
from chromadb import HttpClient
from utils.logger import setup_logger

logger = setup_logger("VectorStore")

class VectorStoreManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store"""
        try:
            persist_directory = self.config.get("persist_directory", "./data/vector_store")
            os.makedirs(persist_directory, exist_ok=True)
            
            client = chromadb.PersistentClient(path=persist_directory)

            self.vector_store = Chroma(
                client=client,
                collection_name=self.config.get("collection_name", "research_documents"),
                embedding_function=self.embedding_model
            )

            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store"""
        try:
            if not documents:
                return []
            
            # Filter out empty documents
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_docs:
                logger.warning("No valid documents to add")
                return []
            
            # Add documents to vector store
            ids = self.vector_store.add_documents(valid_docs)
            
            # Persist the changes
            self.vector_store.persist()
            
            logger.info(f"Added {len(valid_docs)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Document]:
        """Search for similar documents"""
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            # Filter by threshold
            filtered_docs = [
                doc for doc, score in docs 
                if score >= threshold
            ]
            
            logger.info(f"Found {len(filtered_docs)} relevant documents for query")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, max_tokens: int = 4000) -> str:
        """Get relevant context for a query with token limit"""
        try:
            docs = self.similarity_search(query)
            
            if not docs:
                return ""
            
            # Build context within token limit
            context_parts = []
            current_tokens = 0
            
            for doc in docs:
                doc_tokens = doc.metadata.get("token_count", len(doc.page_content.split()))
                
                if current_tokens + doc_tokens <= max_tokens:
                    context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    context_parts.append(doc.page_content)
                    context_parts.append("---")
                    current_tokens += doc_tokens
                else:
                    break
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {str(e)}")
            return ""
    
    def delete_documents(self, source: str) -> bool:
        """Delete documents from a specific source"""
        try:
            self.vector_store.delete(
                where={"source": source}
            )
            logger.info(f"Deleted documents from source: {source}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            collection = self.vector_store.get()
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.config.get("collection_name"),
                "embedding_model": self.config.get("embedding_model")
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}