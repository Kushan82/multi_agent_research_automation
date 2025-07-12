from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt

class AnalysisAgent:
    def __init__(self):
        self.prompt_template=  load_prompt("analysis_prompt.txt")

    def run(self,search_output:str)->str:
        prompt = self.prompt_template.replace("{{input}}",search_output.strip())
        result = run_llm_prompt(prompt)
        return result
    