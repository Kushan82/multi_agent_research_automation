from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ResearchMode(str, Enum):
    FULL = "full"
    RAG_ONLY = "rag_only"

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Research query")
    mode: ResearchMode = Field(default=ResearchMode.FULL, description="Research mode")
    debug: bool = Field(default=False, description="Enable debug mode")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens for RAG context")

class DocumentUploadRequest(BaseModel):
    files: List[str] = Field(..., description="List of file paths to ingest")

class URLIngestRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to ingest")
    
class MemoryQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query to search in memory")
