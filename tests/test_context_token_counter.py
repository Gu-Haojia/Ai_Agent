"""群聊上下文 token 统计器测试。"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from sql_agent_cli_stream_plus import ContextCompressionConfig, ContextCompressor
from src.context_token_counter import ContextTokenCounter


class _SummaryModel:
    """
    提供上下文压缩器构造所需的最小摘要模型。

    Returns:
        None: 初始化不返回额外值。

    Raises:
        None: 初始化不主动抛出异常。
    """

    def invoke(self, messages: object) -> AIMessage:
        """
        返回固定摘要消息。

        Args:
            messages (object): 摘要输入消息，本测试不读取内容。

        Returns:
            AIMessage: 固定摘要响应。

        Raises:
            None: 本方法不主动抛出异常。
        """
        del messages
        return AIMessage(content="测试摘要")


def _identity_text(text: str) -> str:
    """
    返回未经修改的测试文本。

    Args:
        text (str): 原始文本。

    Returns:
        str: 原始文本。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return text


def _omit_test_base64(text: str) -> str:
    """
    将测试文本中的 data URL Base64 替换为占位符。

    Args:
        text (str): 原始文本。

    Returns:
        str: 已替换 Base64 的文本。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return re.sub(
        r"data:[^;\s]+;base64,[A-Za-z0-9+/=]+",
        "[BASE64_OMITTED]",
        text,
    )


def test_context_token_counter_counts_images_videos_and_safe_tool_calls() -> None:
    """
    验证共享统计器计入媒体额定 token，并忽略 Gemini 内部元数据。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    counter = ContextTokenCounter(text_sanitizer=_identity_text)
    human = HumanMessage(
        content=[
            {"type": "text", "text": "请分析图片和视频。"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "A" * 8000},
            },
            {
                "type": "media",
                "mime_type": "video/mp4",
                "data": b"video-bytes",
                "duration_seconds": 12.4,
            },
        ]
    )
    tool_call = {
        "name": "search",
        "args": {"query": "演唱会"},
        "id": "call-1",
        "type": "tool_call",
    }
    ai_with_metadata = AIMessage(
        content="",
        tool_calls=[tool_call],
        additional_kwargs={"thought_signature": "Z" * 12000},
    )
    ai_without_metadata = AIMessage(content="", tool_calls=[tool_call])

    estimate = counter.count_state("当前摘要", [human, ai_with_metadata])
    comparison = counter.count_state("当前摘要", [human, ai_without_metadata])

    assert estimate.image_count == 1
    assert estimate.image_tokens == 1120
    assert estimate.video_count == 1
    assert estimate.video_seconds == 13
    assert estimate.video_tokens == 13 * 102
    assert estimate.message_text_tokens == comparison.message_text_tokens
    assert estimate.total_tokens == (
        estimate.text_tokens + estimate.image_tokens + estimate.video_tokens
    )


def test_context_token_counter_rejects_video_without_duration() -> None:
    """
    验证新视频消息缺少时长时会显式报错。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 预期断言由 pytest 捕获。
    """
    counter = ContextTokenCounter(text_sanitizer=_identity_text)
    message = HumanMessage(
        content=[
            {
                "type": "media",
                "mime_type": "video/mp4",
                "data": b"video-bytes",
            }
        ]
    )

    with pytest.raises(AssertionError, match="视频消息缺少 duration_seconds"):
        counter.count_messages([message])


def test_context_token_counter_sanitizes_base64_text() -> None:
    """
    验证文本中的 Base64 会在编码前交给清洗函数处理。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    counter = ContextTokenCounter(text_sanitizer=_omit_test_base64)
    message = HumanMessage(
        content="图片数据 data:image/png;base64," + "A" * 12000
    )

    estimate = counter.count_messages([message])

    assert estimate.message_text_tokens < 100


def test_context_token_counter_truncates_text_with_same_encoder() -> None:
    """
    验证文本截断和 token 统计使用同一个编码器。

    Returns:
        None: 测试用例不主动抛出异常。

    Raises:
        None: 预期行为由断言验证。
    """
    counter = ContextTokenCounter(text_sanitizer=_identity_text)

    truncated = counter.truncate_text_tokens("long " * 100, 20)

    assert counter.count_text_tokens(truncated) <= 20
    assert truncated


def test_context_token_counter_ignores_display_format_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证 FORMAT 显示配置不会改变原始消息 token 统计。

    Args:
        monkeypatch (pytest.MonkeyPatch): 环境变量修改工具。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    counter = ContextTokenCounter(text_sanitizer=_identity_text)
    message = HumanMessage(content=r"**结果**是 $\alpha + \beta$")
    monkeypatch.setenv("FORMAT", "0")
    without_format = counter.count_messages([message]).message_text_tokens
    monkeypatch.setenv("FORMAT", "1")
    with_format = counter.count_messages([message]).message_text_tokens

    assert with_format == without_format


def test_context_compressor_uses_shared_context_token_counter() -> None:
    """
    验证压缩器与共享统计器对相同消息采用完全相同的 token 总数。

    Returns:
        None: 测试用例无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    counter = ContextTokenCounter(text_sanitizer=_identity_text)
    compressor = ContextCompressor(
        ContextCompressionConfig(),
        _SummaryModel(),
        token_counter=counter,
    )
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "带一张图。"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ]
        )
    ]

    assert compressor.count_messages_tokens(messages) == counter.count_messages(
        messages
    ).total_tokens
