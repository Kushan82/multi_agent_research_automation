from typing import Dict,List
import uuid
import datetime

class MemoryAgent:
    def __init__(self):
        self.memory: Dict[str, Dict]= {}
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
        