"""Tests for the LangGraph verification graph structure."""

from src.agent.graph import build_verification_graph


def test_graph_builds():
    """Graph compiles without error."""
    graph = build_verification_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    """Graph contains all verification nodes."""
    graph = build_verification_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"decompose", "research", "evaluate", "judge", "synthesize"}
    assert expected.issubset(node_names)
