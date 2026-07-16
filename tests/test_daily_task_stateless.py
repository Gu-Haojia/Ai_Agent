"""Daily 定时任务无状态调用测试。"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from daily_task import DailyTicketTask, DailyWeatherTask
from sql_agent_cli_stream_plus import AgentConfig, SQLCheckpointAgentStreamingPlus


def test_agent_builds_stateless_graph_with_shared_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    验证无状态 Graph 不带 checkpointer 且与持久化 Graph 共用 Store。

    Args:
        tmp_path (Path): pytest 提供的临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 环境变量替换工具。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    prompt_file = tmp_path / "persona.txt"
    prompt_file.write_text("你是测试助手。", encoding="utf-8")
    monkeypatch.setenv("SYS_MSG_FILE", str(prompt_file))
    monkeypatch.setenv("SKIP_VENV_CHECK", "1")
    monkeypatch.setenv("CONTEXT_SUMMARY_MODEL", "fake:summary")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_TOOLS", "0")
    monkeypatch.setenv("REMINDER_STORE_FILE", str(tmp_path / "reminders.json"))
    agent = SQLCheckpointAgentStreamingPlus(
        AgentConfig(
            model_name="fake:echo",
            thread_id="persistent-thread",
            use_memory_ckpt=True,
        )
    )

    try:
        assert agent._graph.checkpointer is agent._saver
        assert agent._stateless_graph.checkpointer is None
        assert agent._graph.store is agent._store
        assert agent._stateless_graph.store is agent._store
    finally:
        agent.shutdown()


def test_daily_weather_task_calls_agent_without_thread_id() -> None:
    """
    验证每日简报不为单次执行创建 checkpoint 线程。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    send_func = mock.Mock()
    agent_provider = mock.Mock(return_value=agent)
    task = DailyWeatherTask(
        send_func,
        [10001],
        question="今日简报",
        agent_provider=agent_provider,
    )

    with mock.patch.object(
        SQLCheckpointAgentStreamingPlus,
        "chat_once_stream",
        return_value="简报内容",
    ) as chat_once_stream:
        task._execute_once()

    chat_once_stream.assert_called_once_with("今日简报")
    send_func.assert_called_once_with(10001, "📅 简报内容")


def test_daily_ticket_task_calls_agent_without_thread_id() -> None:
    """
    验证抽選更新任务在发现更新后执行无状态 Agent 调用。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    send_func = mock.Mock()
    agent_provider = mock.Mock(return_value=agent)
    query = mock.Mock()
    query.run.return_value = '{"has_update": true}'
    task = DailyTicketTask(
        send_func,
        [10001],
        prompt="整理抽選更新",
        query=query,
        agent_provider=agent_provider,
    )

    with mock.patch.object(
        SQLCheckpointAgentStreamingPlus,
        "chat_once_stream",
        return_value="抽選内容",
    ) as chat_once_stream:
        task._execute_once()

    query.run.assert_called_once_with("check")
    chat_once_stream.assert_called_once_with("整理抽選更新")
    send_func.assert_called_once_with(10001, "🎟️ 抽選内容")
