from typing import Dict,List
import uuid
import datetime
from tools.groq_llm import run_llm_prompt
from utils.prompt_loader import load_prompt
from utils.logger import setup_logger
import time

logger = setup_logger("MemoryAgent")

class MemoryAgent:
    def __init__(self):
        self.memory: Dict[str, Dict]= {}
        self.prompt_template = load_prompt("memory_agent_prompt.txt")

    def store(self,query:str,final_report:str)->str:
        entry_id = str(uuid.uuid4())
        self.memory[entry_id]={
            "query":query,
            "final_report":final_report,
            "timestamp":datetime.datetime.now().isoformat()
        }
        return entry_id
    
    def get_all(self)->List[Dict]:
        return[
            {"id": k, **v} for k, v in sorted(self.memory.items(), key=lambda item: item[1]['timestamp'], reverse=True)
        ]
    def by_get_id(self,entry_id:str)->Dict:
        return self.memory.get(entry_id,None)
    
    def analyze_context(self, current_query: str, debug :bool= False) -> dict:
        """Analyze current query against historical context using LLM"""
        start= time.time()
        logger.info("Analyzing context for query: %s", current_query)
        # Get recent history (last 5 entries)
        recent_history = self.get_all()[:5]
        
        # Format history for prompt
        history_text = self._format_history(recent_history)
        
        # Apply prompt template
        prompt_input = self.prompt_template.replace("{{query}}", current_query)
        prompt_input = prompt_input.replace("{{history}}", history_text)
        
        try:
            result = run_llm_prompt(prompt_input)
            logger.info("Memory analysis result: %d characters", len(result))
        except Exception as e:
            logger.error("Memory Agent failed: %s", str(e))
            result = "No previous research context available."
        
        elapsed = time.time() - start
        logger.warning("⏱️ MemoryAgent completed in %.2f seconds", elapsed)
        return {
            "output": result,
            "debug": {
                "agent": "MemoryAgent",
                "input": current_query,
                "prompt": prompt_input,
                "history_entries": len(recent_history),
                "output": result
            } if debug else {}
        }

    def _format_history(self, history: List[Dict]) -> str:
        """Format history for prompt input"""
        if not history:
            return "No previous research history available."
        
        formatted_entries = []
        for entry in history:
            # Truncate final_report to avoid token limits
            truncated_report = entry['final_report'][:300] + "..." if len(entry['final_report']) > 300 else entry['final_report']
            formatted_entries.append(
                f"Previous Query: {entry['query']}\n"
                f"Date: {entry['timestamp']}\n"
                f"Key Findings: {truncated_report}\n"
            )
        return "\n---\n".join(formatted_entries)