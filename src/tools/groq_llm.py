from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

def get_groq_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192"
    )
def run_llm_prompt(prompt:str)->str:
    llm = get_groq_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if  content.startswith("<think>"):
        content=content.split("<think>")[-1].strip()
    return content