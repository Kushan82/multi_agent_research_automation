from tools.arxiv_tool import search_arxiv
from tools.wikipedia_tool import search_wikipedia
from tools.tavily_tool import search_tavily
from config.config_loader import config

class ToolAgent:
    def run(self, query: str) -> str:
        results = []
        if config["tools"].get("enable_wikipedia", True):
            results.append(search_wikipedia(query))
        if config["tools"].get("enable_tavily", True):
            results.append(search_tavily(query))
        if config["tools"].get("enable_arxiv", True):
            results.append(search_arxiv(query))
        return "\n\n".join(results)
