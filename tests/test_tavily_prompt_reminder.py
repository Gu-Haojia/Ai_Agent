"""Tavily 首次超限工具返回提醒的集成测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.tavily_search_tool import (
    TAVILY_SEARCH_LIMIT_NOTICE,
    TavilySearchReminder,
)
from sql_agent_cli_stream_plus import AgentConfig, SQLCheckpointAgentStreamingPlus


def _build_ai_message(tool_names: list[str], prefix: str) -> AIMessage:
    """
    构造包含指定工具调用的 AI 消息。

    Args:
        tool_names (list[str]): 工具名称列表。
        prefix (str): 工具调用 ID 前缀。

    Returns:
        AIMessage: 包含工具调用元数据的 AI 消息。

    Raises:
        AssertionError: 当工具名称列表为空或前缀为空时抛出。
    """
    assert tool_names, "tool_names 不能为空"
    assert prefix.strip(), "prefix 不能为空"
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": f"{prefix}-{index}",
                "name": tool_name,
                "args": {"query": f"{prefix}-{index}"},
                "type": "tool_call",
            }
            for index, tool_name in enumerate(tool_names)
        ],
    )


def _completed_searches(count: int, prefix: str) -> list[BaseMessage]:
    """构造当前轮已完成的搜索调用和对应结果。

    Args:
        count (int): 已完成的搜索次数。
        prefix (str): 工具调用标识的前缀。

    Returns:
        list[BaseMessage]: 交替排列的 AI 工具调用和工具结果。

    Raises:
        None: 测试辅助函数不主动抛出异常。
    """
    messages: list[BaseMessage] = []
    for index in range(count):
        ai_message = _build_ai_message(["tavily_search"], f"{prefix}-{index}")
        messages.extend(
            [
                ai_message,
                ToolMessage(
                    content="已有搜索结果",
                    name="tavily_search",
                    tool_call_id=ai_message.tool_calls[0]["id"],
                ),
            ]
        )
    return messages


def _build_tool_graph(
    backend: mock.Mock, search_limit_getter: Callable[[], int]
) -> CompiledStateGraph:
    """构造通过真实 ToolNode 同步和异步执行工具的测试图。

    Args:
        backend (mock.Mock): 记录实际执行的搜索请求。
        search_limit_getter (Callable[[], int]): 当前搜索阈值的读取函数。

    Returns:
        CompiledStateGraph: 包含搜索工具和无关工具的已编译图。

    Raises:
        None: 测试辅助函数不主动抛出异常。
    """

    @tool("tavily_search")
    def search(query: str) -> str:
        """调用模拟搜索后端。

        Args:
            query (str): 搜索关键词。

        Returns:
            str: 模拟搜索结果。

        Raises:
            None: 模拟搜索不主动抛出异常。
        """
        return backend(query)

    @tool
    def other_tool(query: str) -> str:
        """返回无关工具的结果。

        Args:
            query (str): 查询内容。

        Returns:
            str: 无关工具的查询结果。

        Raises:
            None: 模拟工具不主动抛出异常。
        """
        return f"其他结果：{query}"

    reminder = TavilySearchReminder(search_limit_getter)
    node = ToolNode(
        [search, other_tool],
        wrap_tool_call=reminder.wrap_tool_call,
        awrap_tool_call=reminder.awrap_tool_call,
        handle_tool_errors=False,
    )
    builder = StateGraph(MessagesState)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


@pytest.mark.parametrize("call_number", [1, 5, 6, 7, 12])
@pytest.mark.parametrize("use_async", [False, True])
def test_only_first_excess_call_returns_tool_notice(
    call_number: int, use_async: bool
) -> None:
    """验证前五次及第七次起正常搜索，仅第六次返回提醒。

    Args:
        call_number (int): 当前搜索调用在本轮的序号。
        use_async (bool): 是否使用异步图执行。

    Returns:
        None: 测试不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    backend = mock.Mock(return_value="新搜索结果")
    graph = _build_tool_graph(backend, lambda: 5)
    state = {
        "messages": [
            HumanMessage(content="当前问题"),
            *_completed_searches(call_number - 1, "previous"),
            _build_ai_message(["tavily_search"], "current"),
        ]
    }
    result = asyncio.run(graph.ainvoke(state)) if use_async else graph.invoke(state)
    tool_result = result["messages"][-1]

    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "current-0"
    assert tool_result.name == "tavily_search"
    assert tool_result.status == "success"
    if call_number == 6:
        assert tool_result.content == TAVILY_SEARCH_LIMIT_NOTICE
        backend.assert_not_called()
    else:
        assert tool_result.content == "新搜索结果"
        backend.assert_called_once_with("current-0")


@pytest.mark.parametrize("use_async", [False, True])
def test_parallel_batch_reminds_once_and_keeps_other_tools(
    use_async: bool,
) -> None:
    """验证同批第六次提醒，第五和第七次及无关工具仍正常执行。

    Args:
        use_async (bool): 是否使用异步图执行。

    Returns:
        None: 测试不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    backend = mock.Mock(return_value="搜索结果")
    graph = _build_tool_graph(backend, lambda: 5)
    state = {
        "messages": [
            HumanMessage(content="当前问题"),
            *_completed_searches(4, "previous"),
            _build_ai_message(
                ["tavily_search", "other_tool", "tavily_search", "tavily_search"],
                "batch",
            ),
        ]
    }
    result = asyncio.run(graph.ainvoke(state)) if use_async else graph.invoke(state)
    results = result["messages"][-4:]

    assert [message.tool_call_id for message in results] == [
        "batch-0",
        "batch-1",
        "batch-2",
        "batch-3",
    ]
    assert [message.content for message in results] == [
        "搜索结果",
        "其他结果：batch-1",
        TAVILY_SEARCH_LIMIT_NOTICE,
        "搜索结果",
    ]
    assert backend.call_count == 2
    backend.assert_has_calls([mock.call("batch-0"), mock.call("batch-3")], any_order=True)


def test_new_user_turn_and_separate_session_reset_count() -> None:
    """验证历史轮次和其他会话的搜索次数不会影响当前轮。

    Returns:
        None: 测试不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    backend = mock.Mock(return_value="搜索结果")
    graph = _build_tool_graph(backend, lambda: 5)
    old_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="旧问题"),
                *_completed_searches(5, "old"),
                _build_ai_message(["tavily_search"], "excess"),
            ]
        }
    )
    assert old_result["messages"][-1].content == TAVILY_SEARCH_LIMIT_NOTICE
    for history in [old_result["messages"], []]:
        new_result = graph.invoke(
            {
                "messages": [
                    *history,
                    HumanMessage(content="新问题"),
                    _build_ai_message(["tavily_search"], "new"),
                ]
            }
        )
        assert new_result["messages"][-1].content == "搜索结果"
    assert backend.call_count == 2


def test_runtime_search_limit_changes_without_rebuilding_graph() -> None:
    """验证运行时修改阈值后，同一工具节点立即读取新设置。

    Returns:
        None: 测试不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    backend = mock.Mock(return_value="搜索结果")
    limit_getter = mock.Mock(return_value=5)
    graph = _build_tool_graph(backend, limit_getter)
    state = {
        "messages": [
            HumanMessage(content="当前问题"),
            *_completed_searches(5, "previous"),
            _build_ai_message(["tavily_search"], "current"),
        ]
    }
    assert graph.invoke(state)["messages"][-1].content == TAVILY_SEARCH_LIMIT_NOTICE

    limit_getter.return_value = 7
    assert graph.invoke(state)["messages"][-1].content == "搜索结果"
    state["messages"] = [
        HumanMessage(content="当前问题"),
        *_completed_searches(7, "previous"),
        _build_ai_message(["tavily_search"], "new-limit"),
    ]
    assert graph.invoke(state)["messages"][-1].content == TAVILY_SEARCH_LIMIT_NOTICE
    backend.assert_called_once_with("current-0")


def test_agent_returns_notice_as_tool_result_without_system_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """通过 Agent 完整工具循环验证第六次提醒且第七次仍可搜索。

    Args:
        tmp_path (Path): 临时测试文件目录。
        monkeypatch (pytest.MonkeyPatch): 环境变量和模型替换工具。

    Returns:
        None: 测试不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    prompt_file = tmp_path / "persona.txt"
    prompt_file.write_text("你是群聊机器人。", encoding="utf-8")
    monkeypatch.setenv("SYS_MSG_FILE", str(prompt_file))
    monkeypatch.setenv("SKIP_VENV_CHECK", "1")
    monkeypatch.setenv("CONTEXT_SUMMARY_MODEL", "fake:summary")
    monkeypatch.setenv("ENABLE_TOOLS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    model = mock.Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        *[_build_ai_message(["tavily_search"], f"call-{i}") for i in range(1, 8)],
        AIMessage(content="完成回答"),
    ]
    monkeypatch.setattr(
        "sql_agent_cli_stream_plus.init_chat_model", mock.Mock(return_value=model)
    )
    backend = mock.Mock(return_value="搜索结果")
    monkeypatch.setattr(TavilySearch, "invoke", backend)
    agent = SQLCheckpointAgentStreamingPlus(
        AgentConfig(model_name="openai:test-reminder", use_memory_ckpt=True)
    )

    try:
        result = agent._stateless_graph.invoke(
            {"messages": [HumanMessage(content="请搜索并回答")]}
        )
        tool_results = [
            message for message in result["messages"] if isinstance(message, ToolMessage)
        ]
        assert len(tool_results) == 7
        assert tool_results[5].content == TAVILY_SEARCH_LIMIT_NOTICE
        assert tool_results[5].tool_call_id == "call-6-0"
        assert tool_results[6].content == "搜索结果"
        assert backend.call_count == 6
        assert result["messages"][-1].content == "完成回答"
        for model_call in model.invoke.call_args_list:
            assert all(
                "本轮 Tavily 调用次数" not in str(message.content)
                for message in model_call.args[0]
                if isinstance(message, SystemMessage)
            )
    finally:
        agent.shutdown()
