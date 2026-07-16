"""
群聊上下文压缩测试。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Any
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

import qq_group_bot
from qq_group_bot import QQBotHandler
from sql_agent_cli_stream_plus import (
    AgentConfig,
    ContextCompressionConfig,
    ContextCompressor,
    SQLCheckpointAgentStreamingPlus,
)


class RecordingSummaryModel:
    """
    记录摘要 prompt 的测试模型。

    Args:
        None: 无初始化参数。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        None: 初始化不主动抛出异常。
    """

    def __init__(self) -> None:
        """
        初始化 prompt 记录列表。

        Returns:
            None: 无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.prompts: list[str] = []

    def invoke(self, messages: Iterable[Any]) -> AIMessage:
        """
        记录摘要 prompt 并返回群聊语境摘要。

        Args:
            messages (Iterable[Any]): 摘要模型输入消息。

        Returns:
            AIMessage: 固定结构的群聊摘要。

        Raises:
            AssertionError: 当 prompt 为空时抛出。
        """
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt = str(message.content)
        assert prompt, "摘要 prompt 不应为空"
        self.prompts.append(prompt)
        return AIMessage(
            content=(
                "当前话题：群友在聊演唱会抽选和行程安排。\n"
                "当前氛围：轻松吐槽，夹杂抽选焦虑。\n"
                "参与者与发言倾向：User_id 2001 关心票价，User_id 2002 爱吐槽。\n"
                "群内梗、称呼、关系：大家用又寄了调侃抽选失败。\n"
                "明确偏好或雷点：回答不要太官方。\n"
                "最近图片/视频语境：旧图只保留文件名和视觉语境，不保留 base64。\n"
                "工具调用要点：google_flights_search 查过东京到大阪，最低价约 12000 JPY。\n"
                "机器人最近说过什么：已经提醒过抽选截止时间。\n"
                "需要延续或避免重复的点：不要重复完整航班列表。"
            )
        )


def _group_human(index: int, body: str, with_image: bool = False) -> HumanMessage:
    """
    构造群聊用户消息。

    Args:
        index (int): 消息序号。
        body (str): 消息正文。
        with_image (bool): 是否附带图片 data URL。

    Returns:
        HumanMessage: 群聊格式用户消息。

    Raises:
        AssertionError: 当序号非法或正文为空时抛出。
    """
    assert index >= 0, "index 不能为负数"
    assert body.strip(), "body 不能为空"
    user_id = 2000 + index % 3
    user_name = f"群友{index % 3}"
    text = f"Group_id: [10001]; User_id: [{user_id}]; User_name: {user_name}; Msg:\n[{body}]"
    if not with_image:
        return HumanMessage(content=text)
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "text",
                "text": (
                    "第 1 张图像已经以内嵌 data URL 形式提供，"
                    f"本地文件名为 live-{index}.jpg。"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "A" * 8000},
            },
        ]
    )


def _business_messages() -> list:
    """
    构造包含图片和工具调用的真实群聊近似历史。

    Returns:
        list: 已分配消息 ID 的 LangChain 消息列表。

    Raises:
        None: 本函数不主动抛出异常。
    """
    raw_messages: list = []
    for index in range(12):
        raw_messages.append(
            _group_human(
                index,
                f"第 {index} 轮：抽选是不是又寄了，顺便看看周末怎么去会场。",
                with_image=index == 4,
            )
        )
        if index == 5:
            raw_messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "google_flights_search",
                            "args": {
                                "departure_id": "HND",
                                "arrival_id": "ITM",
                                "outbound_date": "2026-06-20",
                            },
                            "id": "call-flight-1",
                            "type": "tool_call",
                        }
                    ],
                )
            )
            raw_messages.append(
                ToolMessage(
                    name="google_flights_search",
                    tool_call_id="call-flight-1",
                    content=(
                        '{"flights":[{"flight_no":"NH001","price":"12000 JPY",'
                        '"departure":"09:00","arrival":"10:10"}]}'
                    ),
                )
            )
            raw_messages.append(AIMessage(content="查到了，最低大概 12000 日元。"))
        else:
            raw_messages.append(AIMessage(content=f"第 {index} 轮回复：先别急，看看抽选时间。"))
    return add_messages([], raw_messages)


def _heavy_tool_turn(prefix: str, tool_count: int = 8) -> list:
    """
    构造一轮包含多条 ToolMessage 的重工具调用轮次。

    Args:
        prefix (str): 用于区分消息和 tool_call_id 的前缀。
        tool_count (int): 本轮工具结果数量。

    Returns:
        list: 未合并的 LangChain 消息列表。

    Raises:
        AssertionError: 当前缀为空或工具数量非法时抛出。
    """
    assert prefix.strip(), "prefix 不能为空"
    assert tool_count > 0, "tool_count 必须为正整数"
    tool_calls = [
        {
            "name": "tavily_search",
            "args": {"query": f"{prefix} query {index}"},
            "id": f"{prefix}-call-{index}",
            "type": "tool_call",
        }
        for index in range(tool_count)
    ]
    tool_messages = [
        ToolMessage(
            name="tavily_search",
            tool_call_id=f"{prefix}-call-{index}",
            content=(f"{prefix} 工具结果 {index} " + "长文本 ") * 200,
        )
        for index in range(tool_count)
    ]
    return [
        HumanMessage(
            content=f"Group_id: [10001]; User_id: [9001]; Msg:\n[{prefix} 重工具轮]"
        ),
        AIMessage(content="", tool_calls=tool_calls),
        *tool_messages,
        AIMessage(content=f"{prefix} 工具结果整理完毕。"),
    ]


def test_context_compression_config_uses_conservative_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证未设置环境变量时使用保守的线上默认压缩配置。

    Args:
        monkeypatch (pytest.MonkeyPatch): 环境变量清理工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    monkeypatch.delenv("CONTEXT_COMPRESS_TRIGGER_TEXT_TOKENS", raising=False)
    monkeypatch.delenv("CONTEXT_KEEP_RECENT_TEXT_TOKENS", raising=False)
    monkeypatch.delenv("CONTEXT_MIN_KEEP_RECENT_MESSAGES", raising=False)
    monkeypatch.delenv("CONTEXT_MAX_SUMMARY_TEXT_TOKENS", raising=False)
    monkeypatch.delenv("CONTEXT_MIN_COMPRESSIBLE_TEXT_TOKENS", raising=False)

    config = ContextCompressionConfig.from_env()

    assert config.trigger_text_tokens == 35000
    assert config.keep_recent_text_tokens == 20000
    assert config.min_keep_recent_messages == 10
    assert config.max_summary_text_tokens == 3500
    assert config.min_compressible_text_tokens == 8000
    assert config.print_summary is False


def test_context_compressor_keeps_group_chat_context_and_tool_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    验证群聊压缩会保留完整轮次、工具要点并去掉图片 base64。

    Args:
        capsys (pytest.CaptureFixture[str]): 控制台输出捕获工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    summary_model = RecordingSummaryModel()
    compressor = ContextCompressor(
        ContextCompressionConfig(
            trigger_text_tokens=260,
            keep_recent_text_tokens=180,
            min_keep_recent_messages=8,
            max_summary_text_tokens=400,
            min_compressible_text_tokens=1,
        ),
        summary_model,
    )
    messages = _business_messages()

    update = compressor.compress(
        {
            "messages": messages,
            "group_context_summary": "当前话题：上一轮大家刚开始聊演唱会抽选。",
            "compression_round": 0,
        }
    )

    output = capsys.readouterr().out
    assert "[ContextCompression] 已压缩群聊上下文" in output
    assert "total_tokens_before=" in output
    assert "compressible_tokens=" in output
    assert "removed_messages=" in output
    assert "kept_messages=" in output
    assert "kept_turns=" in output
    assert "summary_tokens=" in output
    assert "image_tokens_before=" not in output
    assert "video_tokens_before=" not in output
    assert "摘要正文" not in output
    assert "群友在聊演唱会抽选" not in output
    assert update["compression_round"] == 1
    assert "工具调用要点" in update["group_context_summary"]
    assert "google_flights_search" in update["group_context_summary"]
    assert summary_model.prompts
    prompt = summary_model.prompts[0]
    assert "User_id" in prompt
    assert "google_flights_search" in prompt
    assert "data:image/jpeg;base64," not in prompt
    assert "A" * 200 not in prompt
    assert "[图片已省略]" in prompt

    remaining = add_messages(messages, update["messages"])
    remaining_turns = compressor.split_turns(remaining)
    assert len(remaining_turns) >= 4
    assert isinstance(remaining[0], HumanMessage)
    for message in remaining:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id == "call-flight-1"


def test_context_compressor_prints_summary_when_enabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    验证打开内嵌开关时会在控制台输出摘要正文。

    Args:
        capsys (pytest.CaptureFixture[str]): 控制台输出捕获工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    compressor = ContextCompressor(
        ContextCompressionConfig(
            trigger_text_tokens=260,
            keep_recent_text_tokens=180,
            min_keep_recent_messages=8,
            max_summary_text_tokens=400,
            min_compressible_text_tokens=1,
            print_summary=True,
        ),
        RecordingSummaryModel(),
    )

    update = compressor.compress(
        {
            "messages": _business_messages(),
            "group_context_summary": "当前话题：上一轮大家刚开始聊演唱会抽选。",
            "compression_round": 0,
        }
    )

    output = capsys.readouterr().out
    assert update["compression_round"] == 1
    assert "[ContextCompression] 已压缩群聊上下文" in output
    assert "[ContextCompression] 摘要正文:" in output
    assert "群友在聊演唱会抽选" in output


def test_context_compressor_skips_when_compressible_area_is_too_small(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    验证可压缩旧上下文过小时不会频繁调用摘要模型。

    Args:
        capsys (pytest.CaptureFixture[str]): 控制台输出捕获工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    summary_model = RecordingSummaryModel()
    compressor = ContextCompressor(
        ContextCompressionConfig(
            trigger_text_tokens=260,
            keep_recent_text_tokens=180,
            min_keep_recent_messages=8,
            max_summary_text_tokens=400,
            min_compressible_text_tokens=10000,
        ),
        summary_model,
    )

    update = compressor.compress(
        {
            "messages": _business_messages(),
            "group_context_summary": "当前话题：上一轮大家刚开始聊演唱会抽选。",
            "compression_round": 0,
        }
    )

    assert update == {}
    assert summary_model.prompts == []
    assert "[ContextCompression]" not in capsys.readouterr().out


def test_context_compressor_keeps_recent_messages_with_complete_turns() -> None:
    """
    验证按最近消息数保留时会向上取整到完整轮次。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    compressor = ContextCompressor(
        ContextCompressionConfig(
            trigger_text_tokens=1000,
            keep_recent_text_tokens=100,
            min_keep_recent_messages=10,
            max_summary_text_tokens=400,
            min_compressible_text_tokens=1,
        ),
        RecordingSummaryModel(),
    )
    messages = add_messages(
        [],
        [
            *_heavy_tool_turn("old", tool_count=8),
            *_heavy_tool_turn("new", tool_count=8),
            HumanMessage(
                content="Group_id: [10001]; User_id: [9002]; Msg:\n[下一轮消息]"
            ),
        ],
    )

    update = compressor.compress({"messages": messages, "compression_round": 0})

    assert update["compression_round"] == 1
    remaining = add_messages(messages, update["messages"])
    remaining_text = "\n".join(str(message.content) for message in remaining)
    assert "old 重工具轮" not in remaining_text
    assert "new 重工具轮" in remaining_text
    assert "下一轮消息" in remaining_text
    assert any(
        isinstance(message, ToolMessage) and str(message.tool_call_id).startswith("new-")
        for message in remaining
    )
    assert not any(
        isinstance(message, ToolMessage) and str(message.tool_call_id).startswith("old-")
        for message in remaining
    )


def test_context_compressor_skips_when_below_threshold() -> None:
    """
    验证低于阈值时不会压缩。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    compressor = ContextCompressor(
        ContextCompressionConfig(
            trigger_text_tokens=10000,
            keep_recent_text_tokens=8000,
            min_keep_recent_messages=8,
            max_summary_text_tokens=400,
        ),
        RecordingSummaryModel(),
    )

    update = compressor.compress({"messages": _business_messages()[:4]})

    assert update == {}


def test_agent_persists_group_context_summary_in_thread_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    通过真实 Agent 图模拟多轮群聊，验证摘要会写入 checkpoint state。

    Args:
        tmp_path: pytest 提供的临时目录。
        monkeypatch (pytest.MonkeyPatch): 环境变量注入工具。

    Returns:
        None: 无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    prompt_file = tmp_path / "persona.txt"
    prompt_file.write_text("你是群聊机器人。", encoding="utf-8")
    monkeypatch.setenv("SYS_MSG_FILE", str(prompt_file))
    monkeypatch.setenv("SKIP_VENV_CHECK", "1")
    monkeypatch.setenv("CONTEXT_SUMMARY_MODEL", "fake:summary")
    monkeypatch.setenv("CONTEXT_COMPRESS_TRIGGER_TEXT_TOKENS", "220")
    monkeypatch.setenv("CONTEXT_KEEP_RECENT_TEXT_TOKENS", "120")
    monkeypatch.setenv("CONTEXT_MIN_KEEP_RECENT_MESSAGES", "4")
    monkeypatch.setenv("CONTEXT_MAX_SUMMARY_TEXT_TOKENS", "500")
    monkeypatch.setenv("CONTEXT_MIN_COMPRESSIBLE_TEXT_TOKENS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REMINDER_STORE_FILE", str(tmp_path / "reminders.json"))

    agent = SQLCheckpointAgentStreamingPlus(
        AgentConfig(
            model_name="fake:echo",
            thread_id="test-context-compress",
            use_memory_ckpt=True,
        )
    )
    agent.set_token_printer(lambda _: None)

    for index in range(10):
        text = (
            f"Group_id: [10001]; User_id: [{3000 + index % 2}]; "
            f"User_name: 测试群友{index % 2}; Msg:\n"
            f"[第 {index} 轮，大家继续聊演唱会抽选、票价和怎么去现场。]"
        )
        agent.chat_once_stream(text, thread_id="test-context-compress")

    values = agent.get_latest_state_values("test-context-compress")
    summary = agent.get_group_context_summary("test-context-compress")
    messages = agent.get_latest_messages("test-context-compress")

    assert summary
    assert "当前话题" in summary
    assert values["compression_round"] >= 1
    assert len(messages) > 0
    assert len(messages) < 20


def test_get_latest_state_values_does_not_read_checkpoint_history() -> None:
    """
    验证读取最新状态时不会遍历线程的全部 checkpoint 历史。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    graph = mock.Mock()
    graph.get_state.return_value = SimpleNamespace(
        values={"group_context_summary": "当前摘要"}
    )
    graph.get_state_history.side_effect = AssertionError("不应读取全部 checkpoint 历史")
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._graph = graph
    agent._config = SimpleNamespace(thread_id="default-thread")

    values = agent.get_latest_state_values("summary-thread")

    assert values == {"group_context_summary": "当前摘要"}
    graph.get_state.assert_called_once_with(
        {"configurable": {"thread_id": "summary-thread"}}
    )
    graph.get_state_history.assert_not_called()


@pytest.mark.parametrize(
    ("summary", "expected_message"),
    [
        ("当前话题：群友在讨论演唱会。", "当前线程上下文摘要：\n当前话题：群友在讨论演唱会。"),
        ("", "当前线程还没有上下文摘要。"),
    ],
)
def test_summary_command_returns_current_thread_summary(
    summary: str,
    expected_message: str,
) -> None:
    """
    验证小写 /summary 会返回当前群线程的摘要或空摘要提示。

    Args:
        summary (str): Agent 查询接口返回的上下文摘要。
        expected_message (str): 预期发送到群聊的文本。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    handler._group_threads = {"10001/default": "thread-summary-test"}
    summary_reader = mock.Mock(return_value=summary)
    handler.agent = SimpleNamespace(get_group_context_summary=summary_reader)

    with mock.patch.dict(
        qq_group_bot.os.environ,
        {"SYS_MSG_FILE": "/tmp/default.txt"},
    ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/summary")

    assert handled is True
    summary_reader.assert_called_once_with(thread_id="thread-summary-test")
    send_mock.assert_called_once_with(
        "http://onebot",
        10001,
        expected_message,
        "token",
    )


def test_summary_command_does_not_accept_uppercase_name() -> None:
    """
    验证大写 /Summary 不会进入上下文摘要查询逻辑。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    summary_reader = mock.Mock(return_value="不应读取")
    handler.agent = SimpleNamespace(get_group_context_summary=summary_reader)

    with mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/Summary")

    assert handled is True
    summary_reader.assert_not_called()
    assert send_mock.call_args.args[2] == "无此命令"
