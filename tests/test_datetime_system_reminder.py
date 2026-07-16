"""动态时间系统提醒测试。"""

from __future__ import annotations

from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from qq_group_bot import QQBotHandler
from sql_agent_cli_stream_plus import (
    AgentConfig,
    SQLCheckpointAgentStreamingPlus,
    _FakeStreamingEcho,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_datetime_system_reminder_switches_legacy_system_date(
    enabled: bool,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证动态提醒开关反向控制旧系统日期。

    Args:
        enabled (bool): 是否开启动态时间提醒模式。
        tmp_path: pytest 提供的临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 环境变量与属性替换工具。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    prompt_file = tmp_path / "persona.txt"
    prompt_file.write_text("你是群聊机器人。", encoding="utf-8")
    monkeypatch.setenv("SYS_MSG_FILE", str(prompt_file))
    monkeypatch.setenv("SKIP_VENV_CHECK", "1")
    monkeypatch.setenv("CONTEXT_SUMMARY_MODEL", "fake:summary")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REMINDER_STORE_FILE", str(tmp_path / "reminders.json"))
    monkeypatch.setenv(
        "ENABLE_DATETIME_SYSTEM_REMINDER",
        "1" if enabled else "0",
    )
    invoke_mock = mock.Mock(return_value=AIMessage(content="ok"))
    monkeypatch.setattr(_FakeStreamingEcho, "invoke", invoke_mock)

    agent = SQLCheckpointAgentStreamingPlus(
        AgentConfig(
            model_name="fake:echo",
            thread_id=f"test-datetime-reminder-{enabled}",
            use_memory_ckpt=True,
        )
    )
    agent.set_token_printer(lambda _: None)

    try:
        user_input = (
            QQBotHandler._build_multimodal_content(
                "你好",
                [],
                include_datetime_system_reminder=True,
            )
            if enabled
            else "你好"
        )
        result = agent.chat_once_stream(
            user_input,
            thread_id=f"test-datetime-reminder-{enabled}",
        )
        model_messages = list(invoke_mock.call_args.args[0])
        system_message = model_messages[0]
        human_messages = [
            message for message in model_messages if isinstance(message, HumanMessage)
        ]

        assert result == "ok"
        assert isinstance(system_message, SystemMessage)
        assert len(human_messages) == 1
        if enabled:
            assert "当前时间是东京时间" not in str(system_message.content)
            assert isinstance(human_messages[0].content, list)
            assert human_messages[0].content[0] == {
                "type": "text",
                "text": "你好",
            }
            assert str(human_messages[0].content[1]["text"]).startswith(
                "<system_reminder>Current datetime:"
            )
        else:
            assert "当前时间是东京时间" in str(system_message.content)
            assert human_messages[0].content == "你好"
        persisted_messages = agent.get_latest_messages(
            f"test-datetime-reminder-{enabled}"
        )
        persisted_human_messages = [
            message
            for message in persisted_messages
            if isinstance(message, HumanMessage)
        ]
        assert len(persisted_human_messages) == 1
        assert persisted_human_messages[0].content == human_messages[0].content
    finally:
        agent.shutdown()
