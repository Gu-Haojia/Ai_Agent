"""Token 消费日志命令测试。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

import qq_group_bot
from qq_group_bot import QQBotHandler
from src.token_usage_logger import TokenUsageSummary


def _handler() -> QQBotHandler:
    """构造不启动 HTTP 服务的命令测试 Handler。

    Returns:
        QQBotHandler: 注入最小命令配置的 Handler。

    Raises:
        None: 本函数不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    return handler


def test_log_command_displays_usage_summary() -> None:
    """验证 /log 输出完整消费汇总。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    summary = TokenUsageSummary(
        start_time=datetime(2026, 8, 10, 12, 30, tzinfo=ZoneInfo("Asia/Tokyo")),
        total_tokens=13220,
        input_tokens=12800,
        cache_read=9600,
        output_tokens=420,
    )
    with mock.patch.object(
        qq_group_bot.TOKEN_USAGE_LOGGER,
        "summarize",
        return_value=summary,
    ) as summarize, mock.patch.object(
        qq_group_bot,
        "_send_group_msg",
    ) as send_group_msg:
        handled = _handler()._handle_commands(10001, 20002, "/log")

    assert handled is True
    summarize.assert_called_once_with(None)
    assert send_group_msg.call_args.args[2] == (
        "Token 消费记录\n"
        "记录开始时间：2026-08-10 12:30\n"
        "总消费：13,220\n"
        "输入消费：12,800\n"
        "输入命中：9,600\n"
        "输出消费：420"
    )


def test_log_command_filters_from_input_time() -> None:
    """验证 /log 时间 将东京时间传给日志汇总。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    summary = TokenUsageSummary(None, 0, 0, 0, 0)
    with mock.patch.object(
        qq_group_bot.TOKEN_USAGE_LOGGER,
        "summarize",
        return_value=summary,
    ) as summarize, mock.patch.object(
        qq_group_bot,
        "_send_group_msg",
    ) as send_group_msg:
        handled = _handler()._handle_commands(
            10001,
            20002,
            "/log 2026-08-10 12:30",
        )

    assert handled is True
    summarize.assert_called_once_with(
        datetime(2026, 8, 10, 12, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    )
    assert send_group_msg.call_args.args[2] == (
        "2026-08-10 12:30 之后暂无 Token 消费记录。"
    )


def test_log_clear_command_clears_usage_file() -> None:
    """验证 /log clear 调用日志清空操作。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    with mock.patch.object(
        qq_group_bot.TOKEN_USAGE_LOGGER,
        "clear",
    ) as clear, mock.patch.object(
        qq_group_bot,
        "_send_group_msg",
    ) as send_group_msg:
        handled = _handler()._handle_commands(10001, 20002, "/log clear")

    assert handled is True
    clear.assert_called_once_with()
    assert send_group_msg.call_args.args[2] == "Token 消费记录已清空。"


def test_log_command_is_not_added_to_command_list() -> None:
    """验证 /cmd 展示文本中不包含 /log。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    with mock.patch.object(qq_group_bot, "_send_group_msg") as send_group_msg:
        handled = _handler()._handle_commands(10001, 20002, "/cmd")

    assert handled is True
    assert "/log" not in send_group_msg.call_args.args[2]
