from langgraph.graph import StateGraph, END
from agent.search_agent import SearchAgent
from agent.analysis_agent import AnalysisAgent
from agent.generation_agent import GenerationAgent
from agent.tool_agent import ToolAgent
from agent.rag_agent import RAGAgent
from agent.memory_agent import MemoryAgent
from config.config_loader import config
from typing import TypedDict

class ResearchState(TypedDict):
    query: str
    debug: bool
    search_output: str
    analysis_output: str
    final_report: str
    tool_output: str
    rag_output: str
    memory_output:str
    search_debug: dict
    analysis_debug: dict
    generation_debug: dict
    rag_debug: dict
    memory_debug: dict
    

search_agent = SearchAgent()
analysis_agent = AnalysisAgent()
generation_agent = GenerationAgent()
tool_agent = ToolAgent()
rag_agent = RAGAgent()
memory_agent = MemoryAgent()

def run_search_agent(state: ResearchState) -> dict:
    result = search_agent.run(state["query"], debug=state.get("debug", False))
    return {
        "search_output": result["output"],
        "search_debug": result["debug"]
    }

def run_tool_agent(state: ResearchState) -> dict:
    result = tool_agent.run(state["query"], debug=state.get("debug", False))
    return {
        "tool_output": result["output"],
        "tool_debug": result["debug"]}

def run_rag_agent(state: ResearchState) -> dict:
    """Run RAG agent to get context-aware responses"""
    if not config["tools"].get("enable_rag", False):
        return {"rag_output": ""}
    
    result = rag_agent.query_with_rag(state["query"], debug=state.get("debug", False))
    return {
        "rag_output": result["output"],
        "rag_debug": result["debug"]
    }
def run_memory_agent(state: ResearchState) -> dict:
    """Run memory agent to analyze context from previous research"""
    result = memory_agent.analyze_context(state["query"])
    return {"memory_output": result["output"] if isinstance(result, dict)
            else result,
            "memory_debug":result.get("debug",{})if isinstance(result,dict)
            else{}}

def run_analysis_agent(state: ResearchState) -> dict:
    combined_input = (
        "🔎 Search Summary:\n" + state["search_output"] +
        "\n\n🧠 Memory Context:\n" + state.get("memory_output", "") +
        "\n\n📚 RAG Context:\n" + state.get("rag_output", "") +
        "\n\n🌐 External Sources:\n" + state.get("tool_output", "")
    )
    result = analysis_agent.run(combined_input, debug=state.get("debug", False))
    return {
        "analysis_output": result["output"],
        "analysis_debug": result["debug"]
    }

def run_generation_agent(state: ResearchState) -> dict:
    combined_input = (
        "Insights:\n" + state["analysis_output"] +
        "\n\nReferenced Sources:\n" + state.get("tool_output", "")
    )
    result = generation_agent.run(combined_input, debug=state.get("debug", False))
    return {
        "final_report": result["output"],
        "generation_debug": result["debug"]
    }

def build_graph():
    graph_builder = StateGraph(ResearchState)
    graph_builder.add_node("search", run_search_agent)
    graph_builder.add_node("memory", run_memory_agent)
    graph_builder.add_node("rag", run_rag_agent)
    graph_builder.add_node("tool_agent", run_tool_agent)
    graph_builder.add_node("analyse", run_analysis_agent)
    graph_builder.add_node("generate", run_generation_agent)

    graph_builder.set_entry_point("search")
    graph_builder.add_edge("search","memory")
    graph_builder.add_edge("memory", "rag")  # Search -> RAG
    graph_builder.add_edge("rag", "tool_agent")  # RAG -> Tools
    graph_builder.add_edge("tool_agent", "analyse")
    graph_builder.add_edge("analyse", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()

research_workflow = build_graph()