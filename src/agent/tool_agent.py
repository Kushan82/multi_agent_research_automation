import wikipedia
from tavily import TavilyClient
import os

class ToolAgent:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=self.tavily_api_key) if self.tavily_api_key else None

    def wiki_search(self, query: str) -> str:
        try:
            summary = wikipedia.summary(query, sentences=3)
            return f"Wikipedia:\n{summary}"
        except Exception as e:
            return f"Wikipedia search failed: {str(e)}"

    def tavily_search(self, query: str) -> str:
        if not self.tavily:
            return "Tavily not configured."
        try:
            results = self.tavily.search(query=query, search_depth="advanced", max_results=3)
            return "\n\n".join([f"{r['title']}: {r['url']}" for r in results["results"]])
        except Exception as e:
            return f"Tavily search failed: {str(e)}"

    def run(self, query: str) -> str:
        sources = []
        sources.append(self.wiki_search(query))
        sources.append(self.tavily_search(query))
        return "\n\n".join(sources)
