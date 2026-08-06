"""QQ 群 Gemini 限流提示测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

from google.genai.errors import ClientError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import qq_group_bot
from qq_group_bot import BotConfig, QQBotHandler


def test_gemini_rate_limit_uses_fixed_group_message() -> None:
    """
    验证 Gemini 最终返回 HTTP 429 时不向群聊暴露原始异常。

    Returns:
        None: 无返回值。

    Raises:
        None: 预期行为由断言验证。
    """
    rate_limit_error = ChatGoogleGenerativeAIError("raw Gemini HTTP 429 detail")
    rate_limit_error.__cause__ = ClientError(
        429,
        {
            "message": "quota exceeded",
            "status": "RESOURCE_EXHAUSTED",
        },
    )
    agent = SimpleNamespace(
        _config=SimpleNamespace(model_name="google_genai:gemini-test"),
        set_token_printer=mock.Mock(),
        set_memory_namespace=mock.Mock(),
        chat_once_stream=mock.Mock(side_effect=rate_limit_error),
    )
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = BotConfig(
        api_base="http://onebot",
        access_token="token",
    )
    handler.agent = agent
    handler.headers = {}
    handler._read_body = mock.Mock(
        return_value=(
            json.dumps(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "group_id": 10001,
                    "user_id": 20002,
                    "sender": {"card": "测试用户", "nickname": "测试用户"},
                }
            ).encode("utf-8"),
            None,
        )
    )
    handler._handle_commands = mock.Mock(return_value=False)
    handler._namespace_for = mock.Mock(return_value="group-10001")
    handler._thread_id_for = mock.Mock(return_value="thread-10001")
    handler._send_no_content = mock.Mock()
    parsed = SimpleNamespace(
        text="你好",
        images=[],
        videos=[],
        reply_message_ids=[],
        at_me=True,
    )

    with mock.patch.object(
        qq_group_bot,
        "_parse_message_and_at",
        return_value=parsed,
    ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_group_msg, mock.patch.dict(
        qq_group_bot.os.environ,
        {"ENABLE_DATETIME_SYSTEM_REMINDER": "0"},
    ):
        handler._handle_post_locked()

    send_group_msg.assert_called_once_with(
        "http://onebot",
        10001,
        "（模型服务繁忙，请稍后再试）",
        "token",
    )
