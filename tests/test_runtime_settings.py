"""运行时设置持久化与搜索上限命令测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import qq_group_bot
from qq_group_bot import BotConfig, QQBotHandler, _send_pending_restart_notification
from src.runtime_settings import RuntimeSettings, RuntimeSettingsStore


def test_runtime_settings_store_creates_and_loads_default_file(
    tmp_path: Path,
) -> None:
    """验证配置缺失时创建默认文件并可再次加载。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    path = tmp_path / ".runtime_settings.json"
    store = RuntimeSettingsStore(path)

    settings = store.load()

    assert settings == RuntimeSettings()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 3,
        "tavily_search_limit": 5,
        "prompt_file": "",
        "daily_city": "",
        "restart_notification_group_id": None,
    }
    assert store.load() == settings


def test_runtime_settings_store_migrates_version_one(tmp_path: Path) -> None:
    """验证现有版本 1 配置会保留设置并增加重启通知字段。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    path = tmp_path / ".runtime_settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tavily_search_limit": 9,
                "prompt_file": "takina.txt",
            }
        ),
        encoding="utf-8",
    )

    settings = RuntimeSettingsStore(path).load()

    assert settings == RuntimeSettings(
        tavily_search_limit=9,
        prompt_file="takina.txt",
    )
    assert json.loads(path.read_text(encoding="utf-8"))[
        "restart_notification_group_id"
    ] is None
    assert json.loads(path.read_text(encoding="utf-8"))["daily_city"] == ""


def test_runtime_settings_store_migrates_version_two(tmp_path: Path) -> None:
    """验证版本 2 配置迁移后增加简报地点字段。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    path = tmp_path / ".runtime_settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "tavily_search_limit": 9,
                "prompt_file": "takina.txt",
                "restart_notification_group_id": 10001,
            }
        ),
        encoding="utf-8",
    )

    settings = RuntimeSettingsStore(path).load()

    assert settings == RuntimeSettings(
        tavily_search_limit=9,
        prompt_file="takina.txt",
        restart_notification_group_id=10001,
    )
    assert json.loads(path.read_text(encoding="utf-8"))["daily_city"] == ""


def test_pending_restart_notification_sends_and_clears_group(
    tmp_path: Path,
) -> None:
    """验证启动通知发送成功后清空持久化群号。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    store = RuntimeSettingsStore(tmp_path / ".runtime_settings.json")
    settings = RuntimeSettings(restart_notification_group_id=10001)
    store.save(settings)
    bot_config = BotConfig(api_base="http://onebot", access_token="token")

    with mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        updated_settings = _send_pending_restart_notification(
            bot_config,
            store,
            settings,
        )

    send_mock.assert_called_once_with(
        "http://onebot",
        10001,
        "✅ app 重启成功。",
        "token",
    )
    assert updated_settings.restart_notification_group_id is None
    assert store.load().restart_notification_group_id is None


def test_pending_restart_notification_keeps_group_when_send_fails(
    tmp_path: Path,
) -> None:
    """验证启动通知发送失败时保留群号供下次启动重试。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 预期的 OSError 由测试捕获。
    """
    store = RuntimeSettingsStore(tmp_path / ".runtime_settings.json")
    settings = RuntimeSettings(restart_notification_group_id=10001)
    store.save(settings)

    with mock.patch.object(
        qq_group_bot,
        "_send_group_msg",
        side_effect=OSError("onebot unavailable"),
    ), pytest.raises(OSError, match="onebot unavailable"):
        _send_pending_restart_notification(BotConfig(), store, settings)

    assert store.load().restart_notification_group_id == 10001


def test_searchlimit_command_saves_and_updates_current_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证搜索上限命令同时更新文件和当前 Agent。

    Args:
        tmp_path (Path): pytest 临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 属性替换工具。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    agent = mock.Mock()
    store = RuntimeSettingsStore(tmp_path / ".runtime_settings.json")
    monkeypatch.setattr(QQBotHandler, "agent", agent, raising=False)
    monkeypatch.setattr(
        QQBotHandler,
        "runtime_settings",
        RuntimeSettings(),
        raising=False,
    )
    monkeypatch.setattr(
        QQBotHandler,
        "runtime_settings_store",
        store,
        raising=False,
    )

    with mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/searchlimit 7")

    assert handled is True
    assert store.load().tavily_search_limit == 7
    assert QQBotHandler.runtime_settings.tavily_search_limit == 7
    agent.set_tavily_search_limit.assert_called_once_with(7)
    assert "已设置为：7" in send_mock.call_args.args[2]


def test_location_command_saves_and_updates_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证地点命令完整保存含空格的地点并同步环境变量。

    Args:
        tmp_path (Path): pytest 临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 环境变量替换工具。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    store = RuntimeSettingsStore(tmp_path / ".runtime_settings.json")
    monkeypatch.setattr(
        QQBotHandler, "runtime_settings", RuntimeSettings(), raising=False
    )
    monkeypatch.setattr(
        QQBotHandler, "runtime_settings_store", store, raising=False
    )

    with mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/location New York,US")

    assert handled is True
    assert store.load().daily_city == "New York,US"
    assert QQBotHandler.runtime_settings.daily_city == "New York,US"
    assert os.environ["DAILY_TASK_CITY"] == "New York,US"
    assert "已设置为：New York,US" in send_mock.call_args.args[2]


@pytest.mark.parametrize("search_limit", [5, 999])
def test_runtime_settings_accepts_search_limit_boundaries(search_limit: int) -> None:
    """验证搜索上限允许 5 到 999 的边界值。

    Args:
        search_limit (int): 待验证的合法边界值。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    settings = RuntimeSettings(tavily_search_limit=search_limit)

    assert settings.tavily_search_limit == search_limit


@pytest.mark.parametrize("search_limit", [4, 1000])
def test_runtime_settings_rejects_search_limit_outside_range(
    search_limit: int,
) -> None:
    """验证搜索上限拒绝小于 5 或大于 999 的数值。

    Args:
        search_limit (int): 待验证的非法边界值。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 预期的 AssertionError 由测试捕获。
    """
    with pytest.raises(AssertionError, match="必须在 5 到 999 之间"):
        RuntimeSettings(tavily_search_limit=search_limit)
