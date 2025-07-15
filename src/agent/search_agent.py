from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt
from utils.logger import setup_logger
import time
logger = setup_logger("SearchAgent")

class SearchAgent:
    def __init__(self):
        self.prompt_template = load_prompt("search_prompt.txt")
    
    def run(self,query:str, debug:bool= False)-> dict:
        start = time.time()
        logger.info("Running search for query: %s", query)
        prompt= self.prompt_template.replace("{{input}}", query.strip())
        try:
            result= run_llm_prompt(prompt)
            logger.info("Search result received, %d characters", len(result))
        except Exception as e:
            logger.error("SearchAgent failed: %s", str(e))
            result = "SearchAgent failed."
        elapsed = time.time() - start
        logger.warning("⏱️ SearchAgent completed in %.2f seconds", elapsed)
        return {
            "output":result,
            "debug":{
                "agent":"SearchAgent",
                "input":query,
                "prompt":prompt,
                "output":result
            }if debug else{}
        }