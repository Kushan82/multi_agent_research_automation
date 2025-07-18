from typing import Dict, List, Any
from rag.vector_store import VectorStoreManager
from rag.document_processor import DocumentProcessor
from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt
from utils.logger import setup_logger
from config.config_loader import config
import time

logger = setup_logger("RAGAgent")

class RAGAgent:
    def __init__(self):
        self.vector_store = VectorStoreManager(config["vector_store"])
        self.document_processor = DocumentProcessor(
            chunk_size=config["vector_store"]["chunk_size"],
            chunk_overlap=config["vector_store"]["chunk_overlap"]
        )
        self.rag_prompt_template = load_prompt("rag_prompt.txt")
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest documents into the vector store"""
        try:
            all_documents = []
            processed_files = []
            failed_files = []
            
            for file_path in file_paths:
                logger.info(f"Processing file: {file_path}")
                documents = self.document_processor.process_file(file_path)
                
                if documents:
                    all_documents.extend(documents)
                    processed_files.append(file_path)
                    logger.info(f"Processed {len(documents)} chunks from {file_path}")
                else:
                    failed_files.append(file_path)
                    logger.warning(f"Failed to process {file_path}")
            
            # Add to vector store
            if all_documents:
                document_ids = self.vector_store.add_documents(all_documents)
                logger.info(f"Successfully ingested {len(all_documents)} chunks from {len(processed_files)} files")
                
                # Verify ingestion
                stats = self.vector_store.get_stats()
                logger.info(f"Vector store now contains {stats.get('total_documents', 0)} documents")
            
            return {
                "success": True,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "total_chunks": len(all_documents),
                "document_ids": document_ids if all_documents else []
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_files": [],
                "failed_files": file_paths
            }
    
    def ingest_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Ingest web content into the vector store"""
        try:
            all_documents = []
            processed_urls = []
            failed_urls = []
            
            for url in urls:
                logger.info(f"Processing URL: {url}")
                documents = self.document_processor.process_url(url)
                
                if documents:
                    all_documents.extend(documents)
                    processed_urls.append(url)
                    logger.info(f"Processed {len(documents)} chunks from {url}")
                else:
                    failed_urls.append(url)
                    logger.warning(f"Failed to process {url}")
            
            # Add to vector store
            if all_documents:
                document_ids = self.vector_store.add_documents(all_documents)
                logger.info(f"Successfully ingested {len(all_documents)} chunks from {len(processed_urls)} URLs")
            
            return {
                "success": True,
                "processed_urls": processed_urls,
                "failed_urls": failed_urls,
                "total_chunks": len(all_documents),
                "document_ids": document_ids if all_documents else []
            }
            
        except Exception as e:
            logger.error(f"URL ingestion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_urls": [],
                "failed_urls": urls
            }
    
    def query_with_rag(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Query using RAG (Retrieval-Augmented Generation)"""
        start = time.time()
        
        try:
            # First, check if we have any documents
            stats = self.vector_store.get_stats()
            total_docs = stats.get("total_documents", 0)
            
            if total_docs == 0:
                logger.warning("No documents in vector store")
                return self._fallback_response(query, debug, error="No documents in vector store")
            
            logger.info(f"Querying vector store with {total_docs} documents")
            
            # Get relevant context from vector store
            context = self.vector_store.get_relevant_context(
                query, 
                max_tokens=config["vector_store"]["retrieval"].get("max_context_tokens", 4000)
            )
            
            if not context:
                logger.warning("No relevant context found in vector store")
                # Try a test retrieval to see what's available
                test_result = self.vector_store.test_retrieval(query)
                logger.info(f"Test retrieval result: {test_result}")
                return self._fallback_response(query, debug, error="No relevant context found")
            
            # Generate response with context
            prompt = self.rag_prompt_template.replace("{{context}}", context).replace("{{query}}", query)
            
            response = run_llm_prompt(prompt)
            
            elapsed = time.time() - start
            logger.info(f"RAG query completed in {elapsed:.2f} seconds")
            
            return {
                "output": response,
                "context_used": True,
                "context_length": len(context),
                "debug": {
                    "agent": "RAGAgent",
                    "query": query,
                    "context": context[:1000] + "..." if len(context) > 1000 else context,
                    "prompt": prompt,
                    "output": response,
                    "elapsed_time": elapsed,
                    "total_docs_in_store": total_docs,
                    "vector_store_stats": stats
                } if debug else {}
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            return self._fallback_response(query, debug, error=str(e))
    
    def _fallback_response(self, query: str, debug: bool = False, error: str = None) -> Dict[str, Any]:
        """Fallback response when RAG fails"""
        fallback_prompt = f"Answer this question based on your general knowledge: {query}"
        
        try:
            response = run_llm_prompt(fallback_prompt)
            
            fallback_message = "[Fallback Response - No relevant documents found]"
            if error:
                fallback_message = f"[Fallback Response - Error: {error}]"
            
            return {
                "output": f"{fallback_message}\n\n{response}",
                "context_used": False,
                "context_length": 0,
                "debug": {
                    "agent": "RAGAgent",
                    "query": query,
                    "fallback": True,
                    "error": error,
                    "prompt": fallback_prompt,
                    "output": response,
                    "vector_store_stats": self.vector_store.get_stats()
                } if debug else {}
            }
            
        except Exception as e:
            logger.error(f"Fallback response failed: {str(e)}")
            return {
                "output": "I apologize, but I'm unable to provide a response at this time.",
                "context_used": False,
                "context_length": 0,
                "debug": {
                    "agent": "RAGAgent",
                    "query": query,
                    "fallback": True,
                    "error": str(e),
                    "vector_store_stats": self.vector_store.get_stats()
                } if debug else {}
            }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return self.vector_store.get_stats()
    
    def test_retrieval(self, query: str = "test") -> Dict[str, Any]:
        """Test retrieval functionality"""
        return self.vector_store.test_retrieval(query)
    
    def clear_vector_store(self) -> bool:
        """Clear all documents from vector store"""
        return self.vector_store.clear_all()