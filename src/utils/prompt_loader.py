import os
PROMPT_DIR = os.path.join(os.path.dirname(__file__),"../prompt_library")

def load_prompt(filename: str) -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))  # full absolute path
    prompt_path = os.path.join(base_dir, "../prompt_library", filename)
    prompt_path = os.path.abspath(prompt_path)  # normalize ../

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"❌ Prompt not found: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

    

