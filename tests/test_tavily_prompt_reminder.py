"""Tavily 动态提示的单元测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sql_agent_cli_stream_plus import (
    _build_tavily_prompt_notice,
    _count_current_turn_tavily_calls,
)


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
                "args": {"query": f"query-{index}"},
                "type": "tool_call",
            }
            for index, tool_name in enumerate(tool_names)
        ],
    )


def test_count_current_turn_tavily_calls_ignores_previous_turn() -> None:
    """
    验证调用次数只统计最近一条用户消息之后的当前轮次。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    messages = [
        HumanMessage(content="旧问题"),
        _build_ai_message(["tavily_search"], "old"),
        ToolMessage(content="旧结果", tool_call_id="old-0"),
        HumanMessage(content="新问题"),
        _build_ai_message(
            ["tavily_search", "other_tool", "tavily_search"],
            "current",
        ),
        ToolMessage(content="新结果", tool_call_id="current-0"),
    ]

    assert _count_current_turn_tavily_calls(messages) == 2


def test_build_tavily_prompt_notice_starts_at_fifth_call() -> None:
    """
    验证达到第五次调用后生成动态提醒，低于阈值时不生成。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    assert _build_tavily_prompt_notice(4) == ""

    notice = _build_tavily_prompt_notice(5)

    assert "Tavily 5 次" in notice
    assert "尽量不要再次调用 Tavily" in notice


def test_count_current_turn_tavily_calls_returns_zero_without_user_message() -> None:
    """
    验证没有用户消息时不会错误统计历史或孤立工具调用。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    messages = [_build_ai_message(["tavily_search"], "orphan")]

    assert _count_current_turn_tavily_calls(messages) == 0
