"""共享 Git 仓库更新与 QQ `/update` 命令测试。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import qq_group_bot
from qq_group_bot import QQBotHandler
from src.repository_updater import GitRepositoryUpdater, RepositoryUpdateResult


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    """构造成功的 Git 命令结果。

    Args:
        stdout (str): 模拟的标准输出。

    Returns:
        subprocess.CompletedProcess[str]: 返回码为零的命令结果。

    Raises:
        None: 不主动抛出异常。
    """
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout)


def _build_updater(repository_path: Path) -> GitRepositoryUpdater:
    """构造指向测试目录的仓库更新器。

    Args:
        repository_path (Path): 模拟共享仓库的临时目录。

    Returns:
        GitRepositoryUpdater: 使用固定测试仓库地址的更新器。

    Raises:
        AssertionError: 当更新器初始化参数无效时抛出。
    """
    return GitRepositoryUpdater(
        repository_path=repository_path,
        repository_url="https://github.com/example/project.git",
    )


def test_repository_updater_skips_merge_when_already_up_to_date(
    tmp_path: Path,
) -> None:
    """验证本地与远端提交相同时不执行合并。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    updater = _build_updater(tmp_path)
    commit = "a" * 40
    responses = [
        _completed("main\n"),
        _completed(),
        _completed(),
        _completed(f"{commit}\n"),
        _completed(f"{commit}\n"),
    ]

    with mock.patch(
        "src.repository_updater.subprocess.run",
        side_effect=responses,
    ) as run_mock:
        result = updater.update()

    assert result == RepositoryUpdateResult(False, commit, commit)
    assert run_mock.call_count == 5
    assert "merge" not in [call.args[0][3] for call in run_mock.call_args_list]


def test_repository_updater_fast_forwards_to_remote_commit(tmp_path: Path) -> None:
    """验证远端领先时检查祖先关系并完成快进更新。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    updater = _build_updater(tmp_path)
    old_commit = "a" * 40
    new_commit = "b" * 40
    responses = [
        _completed("main\n"),
        _completed(),
        _completed(),
        _completed(f"{old_commit}\n"),
        _completed(f"{new_commit}\n"),
        _completed(),
        _completed(),
        _completed(f"{new_commit}\n"),
    ]

    with mock.patch(
        "src.repository_updater.subprocess.run",
        side_effect=responses,
    ) as run_mock:
        result = updater.update()

    assert result == RepositoryUpdateResult(True, old_commit, new_commit)
    commands = [call.args[0][3:] for call in run_mock.call_args_list]
    assert ["merge-base", "--is-ancestor", old_commit, new_commit] in commands
    assert ["merge", "--ff-only", new_commit] in commands


def test_repository_updater_rejects_dirty_worktree(tmp_path: Path) -> None:
    """验证共享工作区存在修改时直接终止更新。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 预期的 AssertionError 由测试捕获。
    """
    updater = _build_updater(tmp_path)

    with mock.patch(
        "src.repository_updater.subprocess.run",
        side_effect=[_completed("main\n"), _completed(" M local.py\n")],
    ) as run_mock, pytest.raises(
        AssertionError,
        match="工作区存在未提交修改",
    ):
        updater.update()

    assert run_mock.call_count == 2


def test_repository_updater_propagates_non_fast_forward_failure(
    tmp_path: Path,
) -> None:
    """验证无法快进时原样抛出 Git 错误且不执行合并。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 预期的 CalledProcessError 由测试捕获。
    """
    updater = _build_updater(tmp_path)
    old_commit = "a" * 40
    new_commit = "b" * 40
    merge_base_error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["git", "merge-base"],
    )
    responses: list[object] = [
        _completed("main\n"),
        _completed(),
        _completed(),
        _completed(f"{old_commit}\n"),
        _completed(f"{new_commit}\n"),
        merge_base_error,
    ]

    with mock.patch(
        "src.repository_updater.subprocess.run",
        side_effect=responses,
    ) as run_mock, pytest.raises(subprocess.CalledProcessError):
        updater.update()

    commands = [call.args[0][3:] for call in run_mock.call_args_list]
    assert not any(command[0] == "merge" for command in commands)


def test_update_command_skips_restart_when_repository_is_current() -> None:
    """验证 `/update` 无新提交时仅发送当前版本消息。

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
    commit = "a" * 40
    updater = mock.Mock()
    updater.update.return_value = RepositoryUpdateResult(False, commit, commit)
    restart_scheduler = mock.Mock()

    with mock.patch.object(qq_group_bot, "_REPOSITORY_UPDATER", updater), mock.patch.object(
        qq_group_bot,
        "_APP_RESTART_SCHEDULER",
        restart_scheduler,
    ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/update")

    assert handled is True
    restart_scheduler.schedule.assert_not_called()
    assert "已经是最新版本" in send_mock.call_args.args[2]
    assert commit[:7] in send_mock.call_args.args[2]


def test_update_command_schedules_restart_after_fast_forward() -> None:
    """验证 `/update` 拉到新提交后调度 app 重启。

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
    old_commit = "a" * 40
    new_commit = "b" * 40
    updater = mock.Mock()
    updater.update.return_value = RepositoryUpdateResult(
        True,
        old_commit,
        new_commit,
    )
    restart_scheduler = mock.Mock()

    with mock.patch.object(qq_group_bot, "_REPOSITORY_UPDATER", updater), mock.patch.object(
        qq_group_bot,
        "_APP_RESTART_SCHEDULER",
        restart_scheduler,
    ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/update")

    assert handled is True
    restart_scheduler.schedule.assert_called_once_with()
    assert f"{old_commit[:7]} → {new_commit[:7]}" in send_mock.call_args.args[2]


def test_update_command_reuses_existing_command_allowlist() -> None:
    """验证 `/update` 继续受现有命令用户白名单保护。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 测试用例不主动抛出异常。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(99999,),
    )
    updater = mock.Mock()

    with mock.patch.object(qq_group_bot, "_REPOSITORY_UPDATER", updater), mock.patch.object(
        qq_group_bot,
        "_send_group_msg",
    ) as send_mock:
        handled = handler._handle_commands(10001, 20002, "/update")

    assert handled is True
    updater.update.assert_not_called()
    assert "无权执行命令" in send_mock.call_args.args[2]


def test_update_command_still_restarts_when_success_message_fails() -> None:
    """验证拉取成功后即使群消息失败也仍然调度 app 重启。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 预期的 RuntimeError 由测试捕获。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    updater = mock.Mock()
    updater.update.return_value = RepositoryUpdateResult(
        True,
        "a" * 40,
        "b" * 40,
    )
    restart_scheduler = mock.Mock()

    with mock.patch.object(qq_group_bot, "_REPOSITORY_UPDATER", updater), mock.patch.object(
        qq_group_bot,
        "_APP_RESTART_SCHEDULER",
        restart_scheduler,
    ), mock.patch.object(
        qq_group_bot,
        "_send_group_msg",
        side_effect=RuntimeError("send failed"),
    ), pytest.raises(
        RuntimeError,
        match="send failed",
    ):
        handler._handle_commands(10001, 20002, "/update")

    restart_scheduler.schedule.assert_called_once_with()


def test_update_command_propagates_repository_failure() -> None:
    """验证未知仓库更新异常继续向上传递且不安排重启。

    Returns:
        None: 测试通过时无返回值。

    Raises:
        None: 预期的 RuntimeError 由测试捕获。
    """
    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(
        api_base="http://onebot",
        access_token="token",
        cmd_allowed_users=(),
    )
    updater = mock.Mock()
    updater.update.side_effect = RuntimeError("fetch failed")
    restart_scheduler = mock.Mock()

    with mock.patch.object(qq_group_bot, "_REPOSITORY_UPDATER", updater), mock.patch.object(
        qq_group_bot,
        "_APP_RESTART_SCHEDULER",
        restart_scheduler,
    ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock, pytest.raises(
        RuntimeError,
        match="fetch failed",
    ):
        handler._handle_commands(10001, 20002, "/update")

    restart_scheduler.schedule.assert_not_called()
    send_mock.assert_not_called()


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (AssertionError("工作区存在未提交修改"), "工作区存在未提交修改"),
        (
            subprocess.CalledProcessError(
                128,
                ["git", "fetch"],
                stderr="无法访问远端仓库",
            ),
            "无法访问远端仓库",
        ),
        (subprocess.TimeoutExpired(["git", "fetch"], 120), "Git 命令执行超时"),
        (OSError("无法启动 Git"), "无法启动 Git"),
    ],
)
def test_update_command_reports_known_repository_failure(
    error: BaseException,
    expected_message: str,
) -> None:
    """验证已知更新失败会回群说明原因且不安排重启。

    Args:
        error (BaseException): 模拟的仓库更新异常。
        expected_message (str): 群消息中应包含的原因。

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
    updater = mock.Mock()
    updater.update.side_effect = error
    restart_scheduler = mock.Mock()

    with mock.patch.object(qq_group_bot, "_REPOSITORY_UPDATER", updater), mock.patch.object(
        qq_group_bot,
        "_APP_RESTART_SCHEDULER",
        restart_scheduler,
    ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_mock:
        handled = handler._handle_commands(10001, 20002, "/update")

    assert handled is True
    restart_scheduler.schedule.assert_not_called()
    message = send_mock.call_args.args[2]
    assert expected_message in message
    assert "app 未重启" in message
