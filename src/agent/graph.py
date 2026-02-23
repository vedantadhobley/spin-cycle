"""LangGraph verification graph.

Defines the state machine for claim verification:
  decompose → research → evaluate → (loop or judge) → synthesize
"""

from langgraph.graph import StateGraph, END

from src.agent.state import VerificationState
from src.agent.nodes import decompose, research, evaluate_evidence, judge, synthesize


def should_continue_research(state: VerificationState) -> str:
    """Conditional edge: loop back to research or proceed to judge."""
    if state.get("needs_more_research") and state["research_iterations"] < state.get("max_research_iterations", 3):
        return "research"
    return "judge"


def build_verification_graph() -> StateGraph:
    """Build the LangGraph verification state machine.

    Graph structure:
        decompose → research → evaluate → [research (loop) | judge] → synthesize → END
    """
    graph = StateGraph(VerificationState)

    # Add nodes
    graph.add_node("decompose", decompose)
    graph.add_node("research", research)
    graph.add_node("evaluate", evaluate_evidence)
    graph.add_node("judge", judge)
    graph.add_node("synthesize", synthesize)

    # Add edges
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "research")
    graph.add_edge("research", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue_research)
    graph.add_edge("judge", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Singleton compiled graph
verification_graph = build_verification_graph()
