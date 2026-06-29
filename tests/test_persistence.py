from pathlib import Path

from langgraph.graph import END, START, StateGraph

from langgraph_agent_lab.persistence import build_checkpointer


def test_sqlite_checkpointer_records_state_history(tmp_path: Path) -> None:
    database_path = tmp_path / "checkpoints.sqlite"

    builder = StateGraph(dict)
    builder.add_node("write", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "write")
    builder.add_edge("write", END)
    graph = builder.compile(checkpointer=build_checkpointer("sqlite", str(database_path)))
    config = {"configurable": {"thread_id": "sqlite-test"}}

    result = graph.invoke({"value": 1}, config=config)
    history = list(graph.get_state_history(config))

    assert database_path.exists()
    assert result["value"] == 2
    assert history
