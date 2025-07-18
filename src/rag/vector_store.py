import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from utils.logger import setup_logger

logger = setup_logger("VectorStore")

class VectorStoreManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store"""
        try:
            persist_directory = self.config.get("persist_directory", "./data/vector_store")
            os.makedirs(persist_directory, exist_ok=True)
            
            # Create persistent client
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                client=client,
                collection_name=self.config.get("collection_name", "research_documents"),
                embedding_function=self.embedding_model,
                persist_directory=persist_directory
            )
            
            logger.info(f"Vector store initialized successfully at {persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return []
            
            # Filter out empty documents
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_docs:
                logger.warning("No valid documents to add (all empty)")
                return []
            
            # Add unique IDs to documents if not present
            for i, doc in enumerate(valid_docs):
                if "id" not in doc.metadata:
                    doc.metadata["id"] = f"doc_{i}_{hash(doc.page_content)}"
            
            # Add documents to vector store
            ids = self.vector_store.add_documents(valid_docs)
            
            logger.info(f"Added {len(valid_docs)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Document]:
        """Search for similar documents with lower threshold"""
        try:
            if not query.strip():
                logger.warning("Empty query provided")
                return []
            
            # Perform similarity search with relevance scores
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
            
            # Filter by threshold (lowered from 0.7 to 0.3)
            filtered_docs = [
                doc for doc, score in docs_with_scores 
                if score >= threshold
            ]
            
            logger.info(f"Found {len(filtered_docs)} relevant documents for query (threshold: {threshold})")
            
            # Log scores for debugging
            if docs_with_scores:
                scores = [score for _, score in docs_with_scores]
                logger.info(f"Similarity scores: {scores}")
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, max_tokens: int = 4000) -> str:
        """Get relevant context for a query with token limit"""
        try:
            # Use lower threshold for better recall
            docs = self.similarity_search(query, k=10, threshold=0.1)
            
            if not docs:
                logger.warning("No relevant documents found")
                return ""
            
            # Build context within token limit
            context_parts = []
            current_tokens = 0
            
            for i, doc in enumerate(docs):
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                doc_tokens = len(doc.page_content) // 4
                
                if current_tokens + doc_tokens <= max_tokens:
                    source = doc.metadata.get('source', 'Unknown')
                    context_parts.append(f"Document {i+1} (Source: {source}):")
                    context_parts.append(doc.page_content)
                    context_parts.append("---")
                    current_tokens += doc_tokens
                else:
                    break
            
            context = "\n".join(context_parts)
            logger.info(f"Generated context with {current_tokens} estimated tokens from {len(docs)} documents")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {str(e)}")
            return ""
    
    def delete_documents(self, source: str) -> bool:
        """Delete documents from a specific source"""
        try:
            # Get collection and delete by metadata
            collection = self.vector_store._collection
            collection.delete(where={"source": source})
            
            logger.info(f"Deleted documents from source: {source}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all documents from vector store"""
        try:
            collection = self.vector_store._collection
            collection.delete()
            logger.info("Cleared all documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get sample of documents to check sources
            sample_docs = self.vector_store.similarity_search("", k=5) if count > 0 else []
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in sample_docs]))
            
            return {
                "total_documents": count,
                "collection_name": self.config.get("collection_name"),
                "embedding_model": self.config.get("embedding_model"),
                "persist_directory": self.config.get("persist_directory"),
                "sample_sources": sources[:5]  # Show first 5 sources
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                "total_documents": 0,
                "error": str(e)
            }
    
    def test_retrieval(self, query: str = "test") -> Dict[str, Any]:
        """Test retrieval functionality"""
        try:
            docs = self.similarity_search(query, k=3, threshold=0.0)  # No threshold for testing
            
            return {
                "query": query,
                "documents_found": len(docs),
                "documents": [
                    {
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get('source', 'Unknown'),
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
            }
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "documents_found": 0
            }