from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt
from utils.logger import setup_logger

logger = setup_logger("GenerationAgent")
class GenerationAgent:
    def __init__(self):
        self.prompt_template = load_prompt("generation_prompt.txt")

    def run(self, analysis_output:str, debug:bool=False)-> dict:
        logger.info("Generation Agent output %s",analysis_output)
        prompt = self.prompt_template.replace("{{input}}", analysis_output.strip())
        try:
            result = run_llm_prompt(prompt)
            logger.info("Generation Agent result, %d characters",len(result))
        except Exception as e:
            logger.error("GenerationAgent failed %s",str(e))
            result=("GenerationAgent failed")
        return {
            "output":result,
            "debug":{
                "agent":"GenerationAgent",
                "input":analysis_output,
                "prompt":prompt,
                "output":result
            }if debug else{}
        }