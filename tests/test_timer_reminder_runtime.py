"""QQ Bot 共享计时器生命周期测试。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

import qq_group_bot
from image_storage import ImageStorageManager
from qq_group_bot import BotConfig, QQBotHandler
from sql_agent_cli_stream_plus import SQLCheckpointAgentStreamingPlus
from src.timer_reminder import (
    JsonReminderStore,
    ReminderPruneResult,
    TimerReminderManager,
)


class _FakeReminderStore:
    """提供内存记录的提醒存储测试替身。"""

    def __init__(self, records: tuple[dict[str, object], ...]) -> None:
        """初始化测试存储。

        Args:
            records (tuple[dict[str, object], ...]): 待恢复的提醒记录。
        """
        self.records = records
        self.added: list[dict[str, object]] = []
        self.removed: list[tuple[object, ...]] = []

    def add(self, record: dict[str, object]) -> None:
        """记录新增提醒。

        Args:
            record (dict[str, object]): 新增提醒记录。
        """
        self.added.append(record)

    def remove_one(
        self,
        ts: int,
        group_id: int,
        user_id: int,
        description: str,
        answer: str,
    ) -> None:
        """记录删除提醒参数。

        Args:
            ts (int): 触发时间戳。
            group_id (int): 群号。
            user_id (int): 用户号。
            description (str): 提醒描述。
            answer (str): 提醒文本。

        Returns:
            None: 无返回值。
        """
        self.removed.append((ts, group_id, user_id, description, answer))

    def prune_and_get_active(self, now_ts: int) -> ReminderPruneResult:
        """返回固定的提醒清理结果。

        Args:
            now_ts (int): 当前 Unix 时间戳。

        Returns:
            ReminderPruneResult: 固定的有效提醒与统计。
        """
        assert now_ts >= 0
        return ReminderPruneResult(
            active_records=self.records,
            expired_count=2,
            invalid_count=1,
        )


def _future_record() -> dict[str, object]:
    """构造一分钟后触发的提醒记录。

    Returns:
        dict[str, object]: 有效提醒记录。
    """
    return {
        "ts": int(time.time()) + 60,
        "group_id": 10001,
        "user_id": 20002,
        "description": "测试提醒",
        "answer": "该提醒了",
    }


def test_restore_pending_returns_summary_without_per_item_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证历史提醒恢复只返回统计，不逐条写控制台。

    Args:
        capsys (pytest.CaptureFixture[str]): pytest 标准输出捕获工具。
    """
    store = _FakeReminderStore((_future_record(),))
    manager = TimerReminderManager(store, send_reminder=mock.Mock())

    try:
        result = manager.restore_pending()
    finally:
        manager.stop()

    assert result.restored_count == 1
    assert result.failed_count == 0
    assert result.expired_count == 2
    assert result.invalid_count == 1
    assert capsys.readouterr().out == ""


def test_timer_trigger_uses_injected_sender_and_removes_record() -> None:
    """验证提醒触发的文本和删除规则保持不变。"""
    store = _FakeReminderStore(())
    sender = mock.Mock()
    manager = TimerReminderManager(store, send_reminder=sender)
    record = _future_record()

    try:
        manager.create_timer(
            "after:1m",
            int(record["group_id"]),
            int(record["user_id"]),
            str(record["description"]),
            str(record["answer"]),
        )
        scheduled_job = manager._scheduler.jobs[0]
        scheduled_job.job_func()
    finally:
        manager.stop()

    sender.assert_called_once_with(10001, 20002, "📣[提醒]：该提醒了")
    assert len(store.removed) == 1


def test_timer_trigger_removes_record_when_sender_fails() -> None:
    """验证 QQ 发送失败后仍沿用既有的一次性提醒删除规则。"""
    store = _FakeReminderStore(())
    sender = mock.Mock(side_effect=RuntimeError("OneBot unavailable"))
    manager = TimerReminderManager(store, send_reminder=sender)

    try:
        manager.create_timer("after:1m", 10001, 20002, "测试提醒", "该提醒了")
        manager._scheduler.jobs[0].job_func()
    finally:
        manager.stop()

    assert len(store.removed) == 1


def test_json_store_preserves_schema_and_reports_pruned_records(tmp_path: Path) -> None:
    """验证存储迁移不改变文件字段，并汇总过期与非法记录。

    Args:
        tmp_path (Path): pytest 临时目录。
    """
    store_path = tmp_path / "reminders.json"
    future = _future_record()
    store_path.write_text(
        json.dumps(
            [
                future,
                {**future, "ts": 1},
                {"ts": "invalid"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = JsonReminderStore(str(store_path))

    result = store.prune_and_get_active(int(time.time()))

    assert result.active_records == (future,)
    assert result.expired_count == 1
    assert result.invalid_count == 1
    assert json.loads(store_path.read_text(encoding="utf-8")) == [future]


def test_json_store_rejects_malformed_file(tmp_path: Path) -> None:
    """验证损坏的 JSON 会中止启动恢复，不伪装为 READY。

    Args:
        tmp_path (Path): pytest 临时目录。
    """
    store_path = tmp_path / "reminders.json"
    store_path.write_text("{broken", encoding="utf-8")
    store = JsonReminderStore(str(store_path))

    with pytest.raises(json.JSONDecodeError):
        store.prune_and_get_active(int(time.time()))


def test_qq_timer_sender_reads_current_bot_config_each_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证每次触发都重新读取当前 OneBot 配置。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 属性替换工具。
    """
    configs = iter(
        (
            BotConfig(api_base="http://onebot-1", access_token="token-1"),
            BotConfig(api_base="http://onebot-2", access_token="token-2"),
        )
    )
    monkeypatch.setattr(BotConfig, "from_env", staticmethod(lambda: next(configs)))
    send_mock = mock.Mock()
    monkeypatch.setattr(qq_group_bot, "_send_group_at_message", send_mock)

    qq_group_bot._send_timer_reminder(10001, 20002, "提醒一")
    qq_group_bot._send_timer_reminder(10001, 20002, "提醒二")

    assert send_mock.call_args_list == [
        mock.call("http://onebot-1", 10001, 20002, "提醒一", "token-1"),
        mock.call("http://onebot-2", 10001, 20002, "提醒二", "token-2"),
    ]


def test_rebuild_agent_reuses_timer_and_keeps_old_agent_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证重建复用计时器，并在新 Agent 成功后才关闭旧实例。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 属性替换工具。
    """
    old_agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    new_agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    image_manager = object.__new__(ImageStorageManager)
    reminder_manager = mock.Mock(spec=TimerReminderManager)
    events: list[str] = []

    monkeypatch.setattr(QQBotHandler, "agent", old_agent, raising=False)
    monkeypatch.setattr(QQBotHandler, "image_storage", image_manager, raising=False)
    monkeypatch.setattr(
        QQBotHandler,
        "reminder_manager",
        reminder_manager,
        raising=False,
    )
    monkeypatch.setattr(
        qq_group_bot,
        "_build_agent_from_env",
        lambda manager: events.append("build") or new_agent,
    )
    monkeypatch.setattr(
        SQLCheckpointAgentStreamingPlus,
        "shutdown",
        lambda self: events.append("shutdown"),
    )

    result = QQBotHandler.rebuild_agent()

    assert result is new_agent
    assert QQBotHandler.agent is new_agent
    assert events == ["build", "shutdown"]
    reminder_manager.restore_pending.assert_not_called()


def test_rebuild_agent_keeps_old_agent_when_new_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证新 Agent 构建失败时旧实例和共享计时器保持运行。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 属性替换工具。
    """
    old_agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    image_manager = object.__new__(ImageStorageManager)
    reminder_manager = mock.Mock(spec=TimerReminderManager)
    shutdown_mock = mock.Mock()

    monkeypatch.setattr(QQBotHandler, "agent", old_agent, raising=False)
    monkeypatch.setattr(QQBotHandler, "image_storage", image_manager, raising=False)
    monkeypatch.setattr(
        QQBotHandler, "reminder_manager", reminder_manager, raising=False
    )
    monkeypatch.setattr(
        qq_group_bot,
        "_build_agent_from_env",
        mock.Mock(side_effect=RuntimeError("build failed")),
    )
    monkeypatch.setattr(
        SQLCheckpointAgentStreamingPlus,
        "shutdown",
        shutdown_mock,
    )

    with pytest.raises(RuntimeError, match="build failed"):
        QQBotHandler.rebuild_agent()

    assert QQBotHandler.agent is old_agent
    shutdown_mock.assert_not_called()
    reminder_manager.stop.assert_not_called()
