"""验证 Agent 消息日志只记录一次且保持原有文本格式。"""

from pathlib import Path
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sql_agent_cli_stream_plus import (
    AgentConfig,
    SQLCheckpointAgentStreamingPlus,
    _sanitize_for_logging,
)


def test_chat_once_stream_logs_each_message_update_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证多模式流仅记录一次用户、AI 和工具消息。

    Args:
        tmp_path (Path): pytest 提供的临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 提供的环境变量替换工具。

    Returns:
        None: 测试无返回值。

    Raises:
        AssertionError: 当日志数量、顺序或文本格式发生变化时抛出。
    """
    monkeypatch.setenv("AGENT_MESSAGE_LOG_DIR", str(tmp_path))
    ai_call = AIMessage(
        content="",
        id="ai-call",
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "x"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    tool_result = ToolMessage(
        content="ok",
        name="lookup",
        tool_call_id="call-1",
    )
    ai_final = AIMessage(content="done", id="ai-final")
    user_update = [{"role": "user", "content": "hi"}]

    graph = mock.Mock()
    graph.stream.return_value = iter(
        [
            ("values", {"messages": [HumanMessage(content="hi")]}),
            ("updates", {"chatbot": {"messages": [ai_call]}}),
            ("values", {"messages": [HumanMessage(content="hi"), ai_call]}),
            ("updates", {"tools": {"messages": [tool_result]}}),
            (
                "values",
                {"messages": [HumanMessage(content="hi"), ai_call, tool_result]},
            ),
            ("updates", {"chatbot": {"messages": [ai_final]}}),
            (
                "values",
                {
                    "messages": [
                        HumanMessage(content="hi"),
                        ai_call,
                        tool_result,
                        ai_final,
                    ]
                },
            ),
        ]
    )
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._stateless_graph = graph
    agent._config = AgentConfig(model_name="fake:echo", use_memory_ckpt=True)

    with mock.patch("builtins.print"):
        result = agent.chat_once_stream("hi", thread_id=None)

    assert result == "done"
    graph.stream.assert_called_once_with(
        {"messages": user_update},
        {"configurable": {}},
        stream_mode=["values", "updates"],
    )
    log_paths = list(tmp_path.glob("*.log"))
    assert len(log_paths) == 1
    log_payloads = [
        line.split(" | ", 1)[1]
        for line in log_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert log_payloads == [
        _sanitize_for_logging(user_update),
        _sanitize_for_logging([ai_call]),
        _sanitize_for_logging([tool_result]),
        _sanitize_for_logging([ai_final]),
    ]
