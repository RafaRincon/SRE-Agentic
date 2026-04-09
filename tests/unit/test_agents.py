import app.agents.graph as graph_module


def test_sre_graph_compiles():
    """The LangGraph pipeline should compile successfully."""
    graph_module._graph = None

    graph = graph_module.get_graph()

    assert graph is not None


def test_get_graph_returns_singleton():
    """Graph compilation should be cached after the first call."""
    graph_module._graph = None

    first = graph_module.get_graph()
    second = graph_module.get_graph()

    assert first is second


def test_merge_lists_concatenates_items():
    assert graph_module.merge_lists(["a"], ["b", "c"]) == ["a", "b", "c"]
