"""
Pseudo tool-call marker compatibility tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_agent_cli_stream_plus import _convert_pseudo_tool_call_message


def test_convert_pseudo_tool_call_alias_to_real_tool_call() -> None:
    """
    文本里的 daytime_now 兼容别名会被转换成真实 datetime_now tool_call。
    """

    message = AIMessage(content="我调用一下看看。\n[TOOL_CALL]daytime_now")

    converted = _convert_pseudo_tool_call_message(message, {"datetime_now"})

    assert converted.content == ""
    assert converted.tool_calls == [
        {
            "name": "datetime_now",
            "args": {},
            "id": converted.tool_calls[0]["id"],
            "type": "tool_call",
        }
    ]
    assert converted.tool_calls[0]["id"].startswith("pseudo_")


def test_convert_pseudo_tool_call_with_json_args() -> None:
    """
    支持在标记后追加 JSON 对象参数。
    """

    message = AIMessage(content='[TOOL_CALL]datetime_now {"tz":"Asia/Tokyo"}')

    converted = _convert_pseudo_tool_call_message(message, {"datetime_now"})

    assert converted.content == ""
    assert converted.tool_calls[0]["name"] == "datetime_now"
    assert converted.tool_calls[0]["args"] == {"tz": "Asia/Tokyo"}


def test_convert_pseudo_tool_call_ignores_unknown_tool() -> None:
    """
    未注册工具名不会被转换。
    """

    message = AIMessage(content="[TOOL_CALL]missing_tool")

    converted = _convert_pseudo_tool_call_message(message, {"datetime_now"})

    assert converted is message
    assert converted.tool_calls == []
