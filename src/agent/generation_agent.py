from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt

class GenerationAgent:
    def __init__(self):
        self.prompt_template = load_prompt("generation_prompt.txt")

    def run(self, analysis_output:str, debug:bool=False)-> dict:
        prompt = self.prompt_template.replace("{{input}}", analysis_output.strip())
        result = run_llm_prompt(prompt)
        return {
            "output":result,
            "debug":{
                "agent":"GenerationAgent",
                "input":analysis_output,
                "prompt":prompt,
                "output":result
            }if debug else{}
        }