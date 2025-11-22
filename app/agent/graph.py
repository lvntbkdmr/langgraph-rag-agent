from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import mode_detector, retrieval, response_generation

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("mode_detector", mode_detector)
    workflow.add_node("retrieval", retrieval)
    workflow.add_node("response_generation", response_generation)
    
    workflow.set_entry_point("mode_detector")
    
    # Conditional logic could go here, but for now we'll do a linear flow
    # mode_detector -> retrieval -> response_generation
    # In the future we can skip retrieval if not needed
    
    workflow.add_edge("mode_detector", "retrieval")
    workflow.add_edge("retrieval", "response_generation")
    workflow.add_edge("response_generation", END)
    
    return workflow.compile()

agent_runner = create_graph()

def run_agent(messages: list):
    # Initial state
    initial_state = {
        "messages": messages,
        "mode": "general",
        "retrieved_docs": [],
        "final_answer": ""
    }
    
    result = agent_runner.invoke(initial_state)
    return result['final_answer']
