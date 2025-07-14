import wikipedia
from tavily import TavilyClient
import os
from langchain_community.utilities.arxiv import ArxivAPIWrapper
class ToolAgent:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=self.tavily_api_key) if self.tavily_api_key else None
        self.arxiv = ArxivAPIWrapper(load_max_docs=3)

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
        
    def arxiv_search(self, query: str) -> str:
        try:
            docs = self.arxiv.run(query)
            return f"📄 Arxiv Results:\n{docs}"
        except Exception as e:
            return f"❌ Arxiv failed: {str(e)}"

    def run(self, query: str) -> str:
        wiki = self.wiki_search(query)
        tavily = self.tavily_search(query)
        arxiv = self.arxiv_search(query)
        return "\n\n".join([wiki, tavily, arxiv])