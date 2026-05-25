"""
群聊上下文压缩测试。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

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
    monkeypatch.delenv("CONTEXT_MIN_KEEP_RECENT_TURNS", raising=False)
    monkeypatch.delenv("CONTEXT_MAX_SUMMARY_TEXT_TOKENS", raising=False)
    monkeypatch.delenv("CONTEXT_MIN_COMPRESSIBLE_TEXT_TOKENS", raising=False)

    config = ContextCompressionConfig.from_env()

    assert config.trigger_text_tokens == 35000
    assert config.keep_recent_text_tokens == 20000
    assert config.min_keep_recent_turns == 5
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
            min_keep_recent_turns=4,
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
    assert "kept_turns=" in output
    assert "summary_tokens=" in output
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
            min_keep_recent_turns=4,
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
            min_keep_recent_turns=4,
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
            min_keep_recent_turns=4,
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
    monkeypatch.setenv("CONTEXT_MIN_KEEP_RECENT_TURNS", "2")
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
