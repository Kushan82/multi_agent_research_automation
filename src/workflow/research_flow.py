from langgraph.graph import StateGraph,END
from agent.search_agent import SearchAgent
from agent.analysis_agent import AnalysisAgent
from agent.generation_agent import GenerationAgent
from typing import TypedDict

class ResearchState(TypedDict):
    query:str
    debug:bool
    search_output:str
    analysis_output:str
    final_report:str
    search_debug: dict
    analysis_debug: dict
    generation_debug: dict

search_agent= SearchAgent()
analysis_agent= AnalysisAgent()
generation_agent= GenerationAgent()

def run_search_agent(state: ResearchState) -> dict:
    result = search_agent.run(state["query"],debug=state.get("debug",False))
    return{
        "search_output":result["output"],
        "search_debug": result["debug"]
        }

def run_analysis_agent(state: ResearchState) -> dict:
    result = analysis_agent.run(state["search_output"],debug=state.get("debug",False))
    return{"analysis_output":result["output"],
           "analysis_debug":result["output"]
           }

def run_generation_agent(state: ResearchState) -> dict:
    result = generation_agent.run(state["analysis_output"],debug=state.get("debug",False))
    return{"final_report":result["output"],
           "analysis_debug":result["output"]
           }

def build_graph():
    graph_builder = StateGraph(ResearchState)
    graph_builder.add_node("search",run_search_agent)
    graph_builder.add_node("analyse",run_analysis_agent)
    graph_builder.add_node("generate",run_generation_agent)

    graph_builder.set_entry_point("search")
    graph_builder.add_edge("search", "analyse")
    graph_builder.add_edge("analyse", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()
research_workflow = build_graph()