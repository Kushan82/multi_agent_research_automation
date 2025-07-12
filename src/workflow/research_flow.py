from langgraph.graph import StateGraph,END
from agent.search_agent import SearchAgent
from agent.analysis_agent import AnalysisAgent
from agent.generation_agent import GenerationAgent

search_agent= SearchAgent()
analysis_agent= AnalysisAgent()
generation_agent= GenerationAgent()

def run_search_agent(state):
    query = state["query"]
    result = search_agent.run(query)
    return{"search_output":result}
def run_analysis_agent(state):
    search_output = state["search_output"]
    result = analysis_agent.run(search_output)
    return{"analysis_output":result}
def run_generation_agent(state):
    analysis_output = state["analysis_output"]
    result = generation_agent.run(analysis_output)
    return{"final_output":result}

def build_graph():
    graph_builder = StateGraph(input_schema={"query":str})

    graph_builder.add_node("search",run_search_agent)
    graph_builder.add_node("analyse",run_analysis_agent)
    graph_builder.add_node("generate",run_generation_agent)

    graph_builder.set_entry_point("search")
    graph_builder.add_edge("search", "analyze")
    graph_builder.add_edge("analyze", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()
research_workflow = build_graph()