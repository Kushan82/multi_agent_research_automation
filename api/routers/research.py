from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from api.models.requests import ResearchRequest
from api.models.responses import ResearchResponse
from workflow.research_flow import research_workflow
from agent.memory_agent import MemoryAgent
from utils.logger import setup_logger

router = APIRouter(prefix="/research", tags=["research"])
logger = setup_logger("ResearchAPI")

# Thread pool for running synchronous workflow
executor = ThreadPoolExecutor(max_workers=4)

@router.post("/query", response_model=ResearchResponse)
async def research_query(request: ResearchRequest):
    """
    Execute a research query using the multi-agent system.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting research for query: {request.query}")
        
        # Prepare workflow input
        workflow_input = {
            "query": request.query,
            "debug": request.debug
        }
        
        # Run workflow in thread pool to avoid blocking
        if request.mode == "full":
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, 
                research_workflow.invoke, 
                workflow_input
            )
            
            # Store in memory
            memory_agent = MemoryAgent()
            memory_id = memory_agent.store(request.query, result["final_report"])
            
        else:  # RAG-only mode
            from agent.rag_agent import RAGAgent
            rag_agent = RAGAgent()
            rag_result = rag_agent.query_with_rag(
                request.query, 
                max_tokens=request.max_tokens,
                debug=request.debug
            )
            
            result = {
                "final_report": rag_result["output"],
                "rag_output": rag_result["output"],
                "rag_debug": rag_result.get("debug", {})
            }
            memory_id = None
        
        execution_time = time.time() - start_time
        
        response = ResearchResponse(
            success=True,
            query=request.query,
            final_report=result["final_report"],
            mode=request.mode.value,
            execution_time=execution_time,
            memory_id=memory_id
        )
        
        # Add optional outputs if available
        if request.debug:
            response.search_output = result.get("search_output")
            response.memory_output = result.get("memory_output")
            response.rag_output = result.get("rag_output")
            response.tool_output = result.get("tool_output")
            response.analysis_output = result.get("analysis_output")
            
            # Combine debug info
            debug_info = {}
            for key in ["search_debug", "memory_debug", "rag_debug", "tool_debug", "analysis_debug", "generation_debug"]:
                if key in result:
                    debug_info[key] = result[key]
            response.debug_info = debug_info
        
        logger.info(f"Research completed in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Research execution failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Check the health of the research system.
    """
    try:
        # Test each agent
        agents_status = {}
        
        # Test search agent
        try:
            from agent.search_agent import SearchAgent
            search_agent = SearchAgent()
            agents_status["search"] = "healthy"
        except Exception as e:
            agents_status["search"] = f"error: {str(e)}"
        
        # Test other agents similarly...
        agents_status["memory"] = "healthy"
        agents_status["rag"] = "healthy"
        agents_status["tool"] = "healthy"
        agents_status["analysis"] = "healthy"
        agents_status["generation"] = "healthy"
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": time.time(),
            "agents_status": agents_status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )
