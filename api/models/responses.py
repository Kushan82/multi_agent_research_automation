from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ResearchResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    query: str = Field(..., description="Original query")
    final_report: str = Field(..., description="Generated research report")
    mode: str = Field(..., description="Research mode used")
    execution_time: float = Field(..., description="Execution time in seconds")
    memory_id: Optional[str] = Field(None, description="Memory entry ID")
    
    # Optional detailed outputs
    search_output: Optional[str] = Field(None, description="Search agent output")
    memory_output: Optional[str] = Field(None, description="Memory agent output")
    rag_output: Optional[str] = Field(None, description="RAG agent output")
    tool_output: Optional[str] = Field(None, description="Tool agent output")
    analysis_output: Optional[str] = Field(None, description="Analysis agent output")
    
    # Debug information
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information")

class DocumentIngestResponse(BaseModel):
    success: bool = Field(..., description="Whether ingestion was successful")
    total_chunks: int = Field(..., description="Total chunks processed")
    processed_files: List[str] = Field(..., description="Successfully processed files")
    failed_files: List[str] = Field(default=[], description="Failed files")
    error: Optional[str] = Field(None, description="Error message if any")

class VectorStoreStatsResponse(BaseModel):
    total_documents: int = Field(..., description="Total documents in vector store")
    collection_name: str = Field(..., description="Collection name")
    embedding_model: str = Field(..., description="Embedding model used")
    persist_directory: str = Field(..., description="Persistence directory")
    sample_sources: List[str] = Field(..., description="Sample sources")

class MemoryEntryResponse(BaseModel):
    id: str = Field(..., description="Memory entry ID")
    query: str = Field(..., description="Original query")
    final_report: str = Field(..., description="Research report")
    timestamp: datetime = Field(..., description="Creation timestamp")

class MemoryListResponse(BaseModel):
    entries: List[MemoryEntryResponse] = Field(..., description="List of memory entries")
    total: int = Field(..., description="Total number of entries")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    agents_status: Dict[str, str] = Field(..., description="Status of each agent")
