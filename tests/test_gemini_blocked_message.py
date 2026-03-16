"""
Gemini 受限内容空回复的归一化测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_agent_cli_stream_plus import (
    _extract_text_content,
    _normalize_blocked_ai_message,
)


def test_normalize_blocked_ai_message_converts_empty_content_to_text_block() -> None:
    """
    验证受限内容空回复会被转换为可显示的文本块。

    Returns:
        None: 无返回值。
    """

    message = AIMessage(
        content=[],
        response_metadata={"finish_reason": "PROHIBITED_CONTENT"},
    )

    normalized = _normalize_blocked_ai_message(message)

    assert normalized.content == [
        {
            "type": "text",
            "text": "（该请求触发安全策略，未返回内容）",
        }
    ]
    assert _extract_text_content(normalized) == "（该请求触发安全策略，未返回内容）"


def test_normalize_blocked_ai_message_keeps_non_blocked_message() -> None:
    """
    验证非受限内容消息保持原样返回。

    Returns:
        None: 无返回值。
    """

    message = AIMessage(
        content=[{"type": "text", "text": "正常回复"}],
        response_metadata={"finish_reason": "STOP"},
    )

    normalized = _normalize_blocked_ai_message(message)

    assert normalized is message
