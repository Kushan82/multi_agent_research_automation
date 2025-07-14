from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt
from utils.logger import setup_logger

logger = setup_logger("AnalysisAgent")

class AnalysisAgent:
    def __init__(self):
        self.prompt_template=  load_prompt("analysis_prompt.txt")

    def run(self,search_output:str, debug:bool=False)->dict:
        logger.info("Analysing search output: %s",search_output)
        prompt = self.prompt_template.replace("{{input}}",search_output.strip())
        try:
            result = run_llm_prompt(prompt)
            logger.info("Analysis result,%d characters",len(result))
        except Exception as e:
            logger.error("Analysis Agent failed:%s",str(e))
            result = "AnalysisAgent failed"
        return {
            "output":result,
            "debug":{
                "agent":"AnalysisAgent",
                "input":search_output.strip(),
                "prompt":prompt,
                "output":result
            }if debug else{}
        }
    