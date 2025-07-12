import os
PROMPT_DIR = os.path.join(os.path.dirname(__file__),"../prompt_library")

def load_prompt(filename:str)->str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path,"r",encoding="utf-8")as f:
        return f.read()
    

