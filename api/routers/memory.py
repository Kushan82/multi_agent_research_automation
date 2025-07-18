from fastapi import APIRouter, HTTPException
from typing import List

from api.models.requests import MemoryQueryRequest
from api.models.responses import MemoryEntryResponse, MemoryListResponse
from agent.memory_agent import MemoryAgent
from utils.logger import setup_logger

router = APIRouter(prefix="/memory", tags=["memory"])
logger = setup_logger("MemoryAPI")

@router.get("/entries", response_model=MemoryListResponse)
async def get_memory_entries(limit: int = 10, offset: int = 0):
    """
    Get memory entries with pagination.
    """
    try:
        memory_agent = MemoryAgent()
        entries = memory_agent.get_all()
        
        # Apply pagination
        total = len(entries)
        paginated_entries = entries[offset:offset + limit]
        
        response_entries = [
            MemoryEntryResponse(
                id=entry["id"],
                query=entry["query"],
                final_report=entry["final_report"],
                timestamp=entry["timestamp"]
            )
            for entry in paginated_entries
        ]
        
        return MemoryListResponse(
            entries=response_entries,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Failed to get memory entries: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory entries: {str(e)}"
        )

@router.get("/entries/{entry_id}", response_model=MemoryEntryResponse)
async def get_memory_entry(entry_id: str):
    """
    Get a specific memory entry by ID.
    """
    try:
        memory_agent = MemoryAgent()
        entry = memory_agent.get_by_id(entry_id)
        
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Memory entry {entry_id} not found"
            )
        
        return MemoryEntryResponse(
            id=entry_id,
            query=entry["query"],
            final_report=entry["final_report"],
            timestamp=entry["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory entry {entry_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory entry: {str(e)}"
        )

@router.post("/analyze", response_model=dict)
async def analyze_memory_context(request: MemoryQueryRequest):
    """
    Analyze memory context for a query.
    """
    try:
        memory_agent = MemoryAgent()
        result = memory_agent.analyze_context(request.query, debug=True)
        
        return {
            "query": request.query,
            "analysis": result["output"],
            "debug": result.get("debug", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze memory context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze memory context: {str(e)}"
        )

@router.delete("/clear")
async def clear_memory():
    """
    Clear all memory entries.
    """
    try:
        memory_agent = MemoryAgent()
        memory_agent.memory.clear()
        
        return {"message": "Memory cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear memory: {str(e)}"
        )
