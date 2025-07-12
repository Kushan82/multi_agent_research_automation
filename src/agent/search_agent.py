from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt

class SearchAgent:
    def __init__(self):
        self.prompt_template = load_prompt("search_prompt.txt")
    
    def run(self,query:str)-> str:
        prompt= self.prompt_template.replace("{{query}}", query)
        result= run_llm_prompt(prompt)
        return result