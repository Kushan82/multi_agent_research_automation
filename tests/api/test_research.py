import pytest
import asyncio
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_research_query():
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {
            "query": "artificial intelligence in education",
            "mode": "full",
            "debug": False
        }
        response = await client.post("/api/v1/research/query", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query"] == payload["query"]
        assert "final_report" in data
        assert "execution_time" in data

@pytest.mark.asyncio
async def test_rag_only_mode():
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {
            "query": "test query",
            "mode": "rag_only",
            "debug": False
        }
        response = await client.post("/api/v1/research/query", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["mode"] == "rag_only"

@pytest.mark.asyncio
async def test_memory_entries():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/memory/entries")
        assert response.status_code == 200
        
        data = response.json()
        assert "entries" in data
        assert "total" in data

@pytest.mark.asyncio
async def test_vector_store_stats():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/documents/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_documents" in data
        assert "collection_name" in data

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
