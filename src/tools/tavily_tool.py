import os
from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

def search_tavily(query: str) -> str:
    if not tavily_client:
        return "Tavily not configured."
    try:
        results = tavily_client.search(query=query, search_depth="advanced", max_results=3)
        return "🌐 Tavily Results:\n" + "\n\n".join([f"{r['title']}: {r['url']}" for r in results["results"]])
    except Exception as e:
        return f"❌ Tavily search failed: {str(e)}"