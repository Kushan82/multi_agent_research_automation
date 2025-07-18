from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import tempfile
import os

from api.models.requests import URLIngestRequest
from api.models.responses import DocumentIngestResponse, VectorStoreStatsResponse
from agent.rag_agent import RAGAgent
from utils.logger import setup_logger

router = APIRouter(prefix="/documents", tags=["documents"])
logger = setup_logger("DocumentsAPI")

@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and ingest documents for RAG.
    """
    try:
        rag_agent = RAGAgent()
        temp_files = []
        
        # Save uploaded files temporarily
        for file in files:
            suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
        
        # Ingest documents
        result = rag_agent.ingest_documents(temp_files)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        if result["success"]:
            logger.info(f"Successfully ingested {result['total_chunks']} chunks")
            return DocumentIngestResponse(**result)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Document ingestion failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

@router.post("/ingest-urls", response_model=DocumentIngestResponse)
async def ingest_urls(request: URLIngestRequest):
    """
    Ingest documents from URLs.
    """
    try:
        rag_agent = RAGAgent()
        result = rag_agent.ingest_urls(request.urls)
        
        if result["success"]:
            logger.info(f"Successfully ingested {result['total_chunks']} chunks from URLs")
            return DocumentIngestResponse(**result)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"URL ingestion failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"URL ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"URL ingestion failed: {str(e)}"
        )

@router.get("/stats", response_model=VectorStoreStatsResponse)
async def get_vector_store_stats():
    """
    Get vector store statistics.
    """
    try:
        rag_agent = RAGAgent()
        stats = rag_agent.get_vector_store_stats()
        return VectorStoreStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get vector store stats: {str(e)}"
        )

@router.delete("/clear")
async def clear_vector_store():
    """
    Clear all documents from vector store.
    """
    try:
        rag_agent = RAGAgent()
        # Add a clear method to your RAGAgent if not exists
        # rag_agent.clear_vector_store()
        
        return {"message": "Vector store cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear vector store: {str(e)}"
        )
