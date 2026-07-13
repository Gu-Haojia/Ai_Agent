"""QQ Bot 引用消息上下文测试。"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from qq_group_bot import MessageContent, _fetch_message_content, _format_reply_context


class QQBotReplyContextTests(unittest.TestCase):
    """验证引用消息发送者信息会进入 Agent 上下文。"""

    def test_fetch_message_content_keeps_sender(self) -> None:
        """
        get_msg 返回的发送者 ID 和名称应保留在消息内容中。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        payload = {
            "status": "ok",
            "retcode": 0,
            "data": {
                "time": 1704067200,
                "message": [{"type": "text", "data": {"text": "原消息"}}],
                "raw_message": "原消息",
                "sender": {
                    "user_id": 123456,
                    "card": "群名片",
                    "nickname": "昵称",
                },
            },
        }
        response = mock.MagicMock(status=200)
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__.return_value = response

        with mock.patch("qq_group_bot.urlopen", return_value=response):
            content = _fetch_message_content("http://127.0.0.1:3000", "789")

        self.assertEqual(content.user_id, "123456")
        self.assertEqual(content.user_name, "群名片")
        self.assertEqual(content.text, "原消息")
        self.assertEqual(content.sent_at, 1704067200)

    def test_format_reply_context_includes_sender(self) -> None:
        """
        引用上下文应使用与当前消息一致的发送者字段名。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        content = MessageContent(
            text="原消息",
            images=(),
            videos=(),
            user_id="123456",
            user_name="群名片",
        )

        result = _format_reply_context(1, content)

        self.assertEqual(
            result,
            "引用消息1: User_id: [123456]; User_name: 群名片; Msg:\n[原消息]",
        )

    def test_format_reply_context_includes_timestamp_when_enabled(self) -> None:
        """
        开启时间戳格式化时，引用上下文应包含东京时间。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        content = MessageContent(
            text="原消息",
            images=(),
            videos=(),
            user_id="123456",
            user_name="群名片",
            sent_at=1704067200,
        )

        result = _format_reply_context(1, content, include_timestamp=True)

        self.assertEqual(
            result,
            "引用消息1: User_id: [123456]; User_name: 群名片; "
            "Sent_at: [2024-01-01 09:00:00 (JST)]; Msg:\n[原消息]",
        )

    def test_format_reply_context_requires_timestamp_when_enabled(self) -> None:
        """
        开启时间戳格式化但引用消息缺少时间时应显式失败。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 被测断言由测试捕获。
        """
        content = MessageContent(
            text="原消息",
            images=(),
            videos=(),
            user_id="123456",
            user_name="群名片",
        )

        with self.assertRaisesRegex(AssertionError, "引用消息时间戳不能为空"):
            _format_reply_context(1, content, include_timestamp=True)


if __name__ == "__main__":
    unittest.main()
