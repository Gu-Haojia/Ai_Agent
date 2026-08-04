"""运行时设置持久化与搜索上限命令测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import qq_group_bot
from qq_group_bot import QQBotHandler
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
        "schema_version": 1,
        "tavily_search_limit": 5,
    }
    assert store.load() == settings


def test_searchlimit_command_saves_and_updates_current_agent(
    tmp_path: Path,
) -> None:
    """验证搜索上限命令同时更新文件和当前 Agent。

    Args:
        tmp_path (Path): pytest 临时目录。

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
    QQBotHandler.agent = agent
    QQBotHandler.runtime_settings = RuntimeSettings()
    QQBotHandler.runtime_settings_store = store

    with mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/searchlimit 7")

    assert handled is True
    assert store.load().tavily_search_limit == 7
    assert QQBotHandler.runtime_settings.tavily_search_limit == 7
    agent.set_tavily_search_limit.assert_called_once_with(7)
    assert "已设置为：7" in send_mock.call_args.args[2]


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
