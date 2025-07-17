import pytest
from agent.rag_agent import RAGAgent
from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStoreManager
import tempfile
import os

class TestRAGAgent:
    def setup_method(self):
        self.rag_agent = RAGAgent()
    
    def test_document_ingestion(self):
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about artificial intelligence and machine learning.")
            temp_file = f.name
        
        try:
            result = self.rag_agent.ingest_documents([temp_file])
            assert result["success"] == True
            assert len(result["processed_files"]) == 1
            assert result["total_chunks"] > 0
        finally:
            os.unlink(temp_file)
    
    def test_rag_query(self):
        # First ingest some content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis and personalized treatment plans.")
            temp_file = f.name
        
        try:
            self.rag_agent.ingest_documents([temp_file])
            
            # Query the RAG system
            result = self.rag_agent.query_with_rag("How is AI used in healthcare?")
            
            assert "output" in result
            assert len(result["output"]) > 0
            assert result["context_used"] == True
            
        finally:
            os.unlink(temp_file)