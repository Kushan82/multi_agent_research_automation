from tools.arxiv_tool import search_arxiv
from tools.wikipedia_tool import search_wikipedia
from tools.tavily_tool import search_tavily
from config.config_loader import config
from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt
from utils.logger import setup_logger
import time

logger = setup_logger("ToolAgent")

class ToolAgent:
    def __init__(self):
        self.prompt_template = load_prompt("tool_agent_prompt.txt")

    def run(self, query: str,debug:bool=False) -> dict:
        start = time.time()
        logger.info("running tool agent: %s",query)
        #raw results
        raw_results = self._gather_raw_data(query)
        # Process with LLM using the prompt template
        prompt_input = self.prompt_template.replace("{{input}}", query)
        prompt_input = prompt_input.replace("{{raw_data}}", raw_results)
        try:
            result = run_llm_prompt(prompt_input)
            logger.info("Tool agent result: %d characters", len(result))
        except Exception as e:
            logger.error("Tool Agent failed: %s", str(e))
            result = "Tool Agent failed"
        
        elapsed = time.time() - start
        logger.warning("⏱️ ToolAgent completed in %.2f seconds", elapsed)
        return {
            "output": result,
            "debug": {
                "agent": "ToolAgent",
                "input": query,
                "prompt": prompt_input,
                "raw_data": raw_results,
                "output": result
            } if debug else {}
        }
        
    def _gather_raw_data(self, query: str) -> str:
        results = []
        
        if config["tools"].get("enable_wikipedia", True):
            wiki_results = search_wikipedia(query)
            results.append(f"WIKIPEDIA RESULTS:\n{wiki_results}")
        
        if config["tools"].get("enable_tavily", True):
            tavily_results = search_tavily(query)
            results.append(f"TAVILY RESULTS:\n{tavily_results}")
        
        if config["tools"].get("enable_arxiv", True):
            arxiv_results = search_arxiv(query)
            results.append(f"ARXIV RESULTS:\n{arxiv_results}")
        
        return "\n\n".join(results)