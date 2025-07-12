from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt

class GenerationAgent:
    def __init__(self):
        self.prompt_template = load_prompt("report_prompt.txt")

    def run(self, analysis_output:str)-> str:
        prompt = self.prompt_template.replace("{{analysis}}", analysis_output.strip())
        result = run_llm_prompt(prompt)
        return result