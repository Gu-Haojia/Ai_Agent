"""QQ 群 Prompt 与线程映射测试。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import qq_group_bot
from qq_group_bot import QQBotHandler
from src.napcat_account_profile import PromptAccountProfileManager


@pytest.fixture(autouse=True)
def reset_thread_mapping_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """隔离每个测试使用的 Handler 类级映射与环境变量。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 仅设置测试隔离状态。

    Raises:
        None: 本 fixture 不主动抛出异常。
    """
    monkeypatch.setattr(QQBotHandler, "_group_threads", {})
    monkeypatch.setattr(QQBotHandler, "_thread_store_file", str(tmp_path / "threads.json"))
    monkeypatch.setattr(QQBotHandler, "_env_consistency_checked", False)
    monkeypatch.setattr(QQBotHandler, "_group_namespaces", {})
    monkeypatch.setattr(QQBotHandler, "_ns_store_file", str(tmp_path / "namespaces.json"))
    profile_config = tmp_path / "account_profiles.json"
    profile_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        QQBotHandler,
        "account_profile_manager",
        PromptAccountProfileManager(profile_config, tmp_path / "avatars"),
    )
    monkeypatch.setenv("SYS_MSG_FILE", "/app/prompts/default.txt")


def _handler() -> QQBotHandler:
    """构造不启动 HTTP 服务的测试 Handler。

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


def _agent() -> SimpleNamespace:
    """构造包含母线程与长期记忆配置的测试 Agent。

    Returns:
        SimpleNamespace: 测试使用的最小 Agent。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return SimpleNamespace(
        _config=SimpleNamespace(thread_id="mother", store_id="memory"),
    )


def test_prompt_switch_restores_each_prompt_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证同一群在不同 Prompt 间切换时恢复各自线程。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = _handler()
    QQBotHandler.agent = _agent()
    timestamps = iter((101, 102))
    monkeypatch.setattr(qq_group_bot.time, "time_ns", lambda: next(timestamps))

    default_thread = handler._thread_id_for(10001)
    monkeypatch.setenv("SYS_MSG_FILE", "/app/prompts/kotone.txt")
    kotone_thread = handler._thread_id_for(10001)
    monkeypatch.setenv("SYS_MSG_FILE", "/app/prompts/default.txt")

    assert handler._thread_id_for(10001) == default_thread
    assert default_thread == "thread-10001-mother-101"
    assert kotone_thread == "thread-10001-mother-102"
    assert QQBotHandler._group_threads == {
        "10001/default": default_thread,
        "10001/kotone": kotone_thread,
    }


def test_switch_command_selects_target_prompt_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `/switch` 重建 Agent 后立即选择目标 Prompt 线程。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "kotone.txt").write_text("prompt", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qq_group_bot.time, "time_ns", lambda: 202)
    handler = _handler()
    agent = _agent()
    QQBotHandler.agent = agent
    QQBotHandler._group_threads = {
        "10001/default": "thread-10001-mother-101",
    }

    with mock.patch.object(
        QQBotHandler,
        "rebuild_agent",
        return_value=agent,
    ), mock.patch.object(
        qq_group_bot, "_set_qq_nickname"
    ) as nickname_mock, mock.patch.object(
        qq_group_bot, "_set_qq_avatar"
    ) as avatar_mock, mock.patch.object(
        qq_group_bot, "_send_group_msg"
    ) as send_mock:
        handled = handler._handle_commands(10001, 20002, "/switch kotone")

    assert handled is True
    assert QQBotHandler._group_threads == {
        "10001/default": "thread-10001-mother-101",
        "10001/kotone": "thread-10001-mother-202",
    }
    assert send_mock.call_args.args[2].startswith(
        "已切换到 kotone 并恢复对应线程：thread-10001-mother-202。"
    )
    nickname_mock.assert_not_called()
    avatar_mock.assert_not_called()


def test_switch_command_applies_tracked_account_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证已登记 Prompt 会同步昵称和 Base64 头像。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    prompts_dir = tmp_path / "prompts"
    avatar_dir = prompts_dir / "avatars"
    avatar_dir.mkdir(parents=True)
    (prompts_dir / "藤田ことね.txt").write_text("prompt", encoding="utf-8")
    (avatar_dir / "藤田ことね.png").write_bytes(b"avatar")
    profile_config = prompts_dir / "account_profiles.json"
    profile_config.write_text(
        json.dumps(
            {
                "藤田ことね": {
                    "nickname": "藤田ことね",
                    "avatar_file": "藤田ことね.png",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qq_group_bot.time, "time_ns", lambda: 404)
    handler = _handler()
    agent = _agent()
    QQBotHandler.agent = agent
    QQBotHandler.account_profile_manager = PromptAccountProfileManager(
        profile_config,
        avatar_dir,
    )

    with mock.patch.object(
        QQBotHandler,
        "rebuild_agent",
        return_value=agent,
    ), mock.patch.object(
        qq_group_bot, "_set_qq_nickname"
    ) as nickname_mock, mock.patch.object(
        qq_group_bot, "_set_qq_avatar"
    ) as avatar_mock, mock.patch.object(
        qq_group_bot, "_send_group_msg"
    ) as send_mock:
        handled = handler._handle_commands(10001, 20002, "/switch 藤田ことね")

    assert handled is True
    nickname_mock.assert_called_once_with("http://onebot", "藤田ことね", "token")
    expected_avatar = base64.b64encode(b"avatar").decode("ascii")
    avatar_mock.assert_called_once_with("http://onebot", expected_avatar, "token")
    assert "账号昵称和头像已切换为 藤田ことね" in send_mock.call_args.args[2]


def test_switch_command_validates_tracked_profile_before_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证已登记资料非法时不会修改 Prompt 或重建 Agent。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "tracked.txt").write_text("prompt", encoding="utf-8")
    profile_config = prompts_dir / "account_profiles.json"
    profile_config.write_text(
        json.dumps(
            {
                "tracked": {
                    "nickname": "已登记昵称",
                    "avatar_file": "missing.png",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    handler = _handler()
    agent = _agent()
    QQBotHandler.agent = agent
    QQBotHandler.account_profile_manager = PromptAccountProfileManager(
        profile_config,
        prompts_dir / "avatars",
    )

    with mock.patch.object(
        QQBotHandler,
        "rebuild_agent",
        return_value=agent,
    ) as rebuild_mock, mock.patch.object(
        qq_group_bot, "_send_group_msg"
    ) as send_mock:
        handled = handler._handle_commands(10001, 20002, "/switch tracked")

    assert handled is True
    rebuild_mock.assert_not_called()
    assert qq_group_bot.os.environ["SYS_MSG_FILE"] == "/app/prompts/default.txt"
    assert "头像文件不存在" in send_mock.call_args.args[2]


def test_switch_command_reports_avatar_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证头像 API 失败时明确保留 Prompt 与昵称已成功的状态。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    prompts_dir = tmp_path / "prompts"
    avatar_dir = prompts_dir / "avatars"
    avatar_dir.mkdir(parents=True)
    (prompts_dir / "tracked.txt").write_text("prompt", encoding="utf-8")
    (avatar_dir / "tracked.png").write_bytes(b"avatar")
    profile_config = prompts_dir / "account_profiles.json"
    profile_config.write_text(
        json.dumps(
            {
                "tracked": {
                    "nickname": "已登记昵称",
                    "avatar_file": "tracked.png",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qq_group_bot.time, "time_ns", lambda: 505)
    handler = _handler()
    agent = _agent()
    QQBotHandler.agent = agent
    QQBotHandler.account_profile_manager = PromptAccountProfileManager(
        profile_config,
        avatar_dir,
    )

    with mock.patch.object(
        QQBotHandler,
        "rebuild_agent",
        return_value=agent,
    ), mock.patch.object(
        qq_group_bot, "_set_qq_nickname"
    ) as nickname_mock, mock.patch.object(
        qq_group_bot,
        "_set_qq_avatar",
        side_effect=RuntimeError("avatar rejected"),
    ), mock.patch.object(
        qq_group_bot, "_send_group_msg"
    ) as send_mock:
        handled = handler._handle_commands(10001, 20002, "/switch tracked")

    assert handled is True
    nickname_mock.assert_called_once()
    assert qq_group_bot.os.environ["SYS_MSG_FILE"].endswith("prompts/tracked.txt")
    assert "昵称已切换为 已登记昵称，但头像同步失败" in send_mock.call_args.args[2]


def test_clear_replaces_only_current_prompt_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `/clear` 只替换当前 Prompt 的群聊线程。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 环境与属性替换工具。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    monkeypatch.setattr(qq_group_bot.time, "time_ns", lambda: 303)
    handler = _handler()
    QQBotHandler.agent = _agent()
    QQBotHandler._group_threads = {
        "10001/default": "thread-10001-mother-101",
        "10001/kotone": "thread-10001-mother-202",
    }

    with mock.patch.object(qq_group_bot, "_send_group_msg"):
        handled = handler._handle_commands(10001, 20002, "/clear")

    assert handled is True
    assert QQBotHandler._group_threads["10001/default"] == "thread-10001-mother-303"
    assert QQBotHandler._group_threads["10001/kotone"] == "thread-10001-mother-202"


def test_forget_targets_only_current_prompt_thread() -> None:
    """验证 `/forget` 删除当前 Prompt 对应的 checkpoint 线程。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = _handler()
    clear_thread = mock.Mock()
    agent = _agent()
    agent.clear_thread_history_fast = clear_thread
    QQBotHandler.agent = agent
    QQBotHandler._group_threads = {
        "10001/default": "thread-10001-mother-101",
        "10001/kotone": "thread-10001-mother-202",
    }

    with mock.patch.object(qq_group_bot, "_send_group_msg"):
        handled = handler._handle_commands(10001, 20002, "/forget")

    assert handled is True
    clear_thread.assert_called_once_with(thread_id="thread-10001-mother-101")


def test_whoami_and_token_use_current_prompt_thread() -> None:
    """验证身份询问与 token 统计读取当前 Prompt 线程。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = _handler()
    agent = _agent()
    agent.set_token_printer = mock.Mock()
    agent.chat_once_stream = mock.Mock(return_value="我是当前 Prompt。")
    agent.count_tokens = mock.Mock(return_value=(123, 4))
    QQBotHandler.agent = agent
    QQBotHandler._group_threads = {
        "10001/default": "thread-10001-mother-101",
        "10001/kotone": "thread-10001-mother-202",
    }

    with mock.patch.object(qq_group_bot, "_send_group_msg"):
        whoami_handled = handler._handle_commands(10001, 20002, "/whoami")
        token_handled = handler._handle_commands(10001, 20002, "/token")

    assert whoami_handled is True
    assert token_handled is True
    agent.chat_once_stream.assert_called_once_with(
        "你是谁",
        thread_id="thread-10001-mother-101",
    )
    agent.count_tokens.assert_called_once_with(
        thread_id="thread-10001-mother-101"
    )


def test_setup_thread_store_migrates_legacy_group_key(tmp_path: Path) -> None:
    """验证旧版纯群号映射迁移到启动 Prompt 且立即落盘。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    store_path = tmp_path / "legacy-threads.json"
    store_path.write_text(
        json.dumps({"10001": "thread-10001-mother-101"}),
        encoding="utf-8",
    )

    QQBotHandler.setup_thread_store(str(store_path), "mother")

    expected = {"10001/default": "thread-10001-mother-101"}
    assert QQBotHandler._group_threads == expected
    assert json.loads(store_path.read_text(encoding="utf-8")) == expected


def test_setup_thread_store_restores_composite_prompt_keys(tmp_path: Path) -> None:
    """验证重启加载时保留同一母线程下的全部 Prompt 映射。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    store_path = tmp_path / "prompt-threads.json"
    expected = {
        "10001/default": "thread-10001-mother-101",
        "10001/kotone": "thread-10001-mother-202",
    }
    store_path.write_text(json.dumps(expected), encoding="utf-8")

    QQBotHandler.setup_thread_store(str(store_path), "mother")

    assert QQBotHandler._group_threads == expected


def test_setup_thread_store_keeps_mother_thread_reset_behavior(tmp_path: Path) -> None:
    """验证母线程环境变量变化后仍清空全部群 Prompt 映射。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    store_path = tmp_path / "mismatched-threads.json"
    store_path.write_text(
        json.dumps(
            {
                "10001/default": "thread-10001-old-mother-101",
                "10001/kotone": "thread-10001-old-mother-202",
            }
        ),
        encoding="utf-8",
    )

    QQBotHandler.setup_thread_store(str(store_path), "new-mother")

    assert QQBotHandler._group_threads == {}
    assert json.loads(store_path.read_text(encoding="utf-8")) == {}
