from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt

class SearchAgent:
    def __init__(self):
        self.prompt_template = load_prompt("search_prompt.txt")
    
    def run(self,query:str, debug:bool= False)-> dict:
        prompt= self.prompt_template.replace("{{input}}", query.strip())
        result= run_llm_prompt(prompt)
        return {
            "output":result,
            "debug":{
                "agent":"SearchAgent",
                "input":query,
                "prompt":prompt,
                "output":result
            }if debug else{}
        }