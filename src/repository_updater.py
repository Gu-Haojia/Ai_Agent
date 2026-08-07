"""提供共享部署仓库的快进更新与应用重启调度。"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Timer


@dataclass(frozen=True)
class RepositoryUpdateResult:
    """描述一次仓库更新的结果。

    Args:
        updated (bool): 是否拉取到了新提交。
        old_commit (str): 更新前的完整提交 SHA。
        new_commit (str): 更新后的完整提交 SHA。

    Raises:
        AssertionError: 当提交 SHA 为空或未更新时提交不一致时抛出。
    """

    updated: bool
    old_commit: str
    new_commit: str

    def __post_init__(self) -> None:
        """校验更新结果字段。

        Returns:
            None: 校验通过后不返回额外值。

        Raises:
            AssertionError: 当提交 SHA 为空或未更新时提交不一致时抛出。
        """
        assert self.old_commit.strip(), "old_commit 不能为空"
        assert self.new_commit.strip(), "new_commit 不能为空"
        if self.updated:
            assert self.old_commit != self.new_commit, "更新时提交 SHA 必须发生变化"
        else:
            assert self.old_commit == self.new_commit, "未更新时提交 SHA 必须一致"


class GitRepositoryUpdater:
    """通过固定 HTTPS 仓库快进更新共享工作区。

    Args:
        repository_path (Path): 容器内共享仓库路径。
        repository_url (str): 用于匿名拉取的 HTTPS 仓库地址。
        branch (str): 固定更新的远端分支。
        command_timeout_seconds (float): 单条 Git 命令的超时秒数。

    Raises:
        AssertionError: 当初始化参数不合法时抛出。
    """

    def __init__(
        self,
        repository_path: Path,
        repository_url: str,
        branch: str = "main",
        command_timeout_seconds: float = 120.0,
    ) -> None:
        """初始化仓库更新器。

        Args:
            repository_path (Path): 容器内共享仓库路径。
            repository_url (str): 用于匿名拉取的 HTTPS 仓库地址。
            branch (str): 固定更新的远端分支。
            command_timeout_seconds (float): 单条 Git 命令的超时秒数。

        Returns:
            None: 初始化完成后不返回额外值。

        Raises:
            AssertionError: 当路径、URL、分支或超时不合法时抛出。
        """
        assert isinstance(repository_path, Path), "repository_path 必须为 Path"
        assert repository_url.startswith("https://"), "repository_url 必须使用 HTTPS"
        assert branch.strip(), "branch 不能为空"
        assert command_timeout_seconds > 0, "Git 命令超时必须为正数"
        self._repository_path = repository_path
        self._repository_url = repository_url
        self._branch = branch
        self._command_timeout_seconds = command_timeout_seconds
        self._lock = Lock()

    def update(self) -> RepositoryUpdateResult:
        """检查远端 main 并在可快进时更新当前共享工作区。

        Returns:
            RepositoryUpdateResult: 是否更新以及更新前后的提交 SHA。

        Raises:
            AssertionError: 当仓库路径、分支、工作区或快进结果不符合预期时抛出。
            subprocess.CalledProcessError: 当任一 Git 命令失败时抛出。
            subprocess.TimeoutExpired: 当任一 Git 命令超时时抛出。
        """
        with self._lock:
            assert self._repository_path.is_dir(), "共享仓库目录不存在"
            current_branch = self._run_git("branch", "--show-current")
            assert current_branch == self._branch, (
                f"当前分支必须为 {self._branch}，实际为 {current_branch or 'DETACHED'}"
            )
            worktree_status = self._run_git("status", "--porcelain")
            assert not worktree_status, "工作区存在未提交修改，已终止更新"

            self._run_git(
                "fetch",
                "--no-tags",
                self._repository_url,
                self._branch,
            )
            old_commit = self._run_git("rev-parse", "HEAD")
            remote_commit = self._run_git("rev-parse", "FETCH_HEAD")
            if old_commit == remote_commit:
                return RepositoryUpdateResult(
                    updated=False,
                    old_commit=old_commit,
                    new_commit=remote_commit,
                )

            self._run_git(
                "merge-base",
                "--is-ancestor",
                old_commit,
                remote_commit,
            )
            self._run_git("merge", "--ff-only", remote_commit)
            new_commit = self._run_git("rev-parse", "HEAD")
            assert new_commit == remote_commit, "快进更新后的 HEAD 与远端提交不一致"
            return RepositoryUpdateResult(
                updated=True,
                old_commit=old_commit,
                new_commit=new_commit,
            )

    def _run_git(self, *arguments: str) -> str:
        """在共享仓库中执行固定 Git 子命令。

        Args:
            *arguments (str): 由调用方固定提供的 Git 子命令及参数。

        Returns:
            str: 去除首尾空白后的标准输出。

        Raises:
            subprocess.CalledProcessError: 当 Git 命令返回非零状态时抛出。
            subprocess.TimeoutExpired: 当 Git 命令执行超时时抛出。
        """
        assert arguments, "Git 子命令不能为空"
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={self._repository_path}",
                *arguments,
            ],
            cwd=self._repository_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=self._command_timeout_seconds,
        )
        return completed.stdout.strip()


class ApplicationRestartScheduler:
    """延迟终止当前应用进程并交由 Docker 重启策略拉起。

    Args:
        delay_seconds (float): 终止进程前等待的秒数。

    Raises:
        AssertionError: 当延迟秒数不是正数时抛出。
    """

    def __init__(self, delay_seconds: float = 2.0) -> None:
        """初始化应用重启调度器。

        Args:
            delay_seconds (float): 终止进程前等待的秒数。

        Returns:
            None: 初始化完成后不返回额外值。

        Raises:
            AssertionError: 当延迟秒数不是正数时抛出。
        """
        assert delay_seconds > 0, "重启延迟必须为正数"
        self._delay_seconds = delay_seconds

    def schedule(self) -> None:
        """调度当前进程在短暂延迟后接收 SIGTERM。

        Returns:
            None: 调度完成后不返回额外值。

        Raises:
            RuntimeError: 当系统无法创建计时线程时抛出。
        """
        timer = Timer(self._delay_seconds, self._terminate_current_process)
        timer.daemon = True
        timer.start()

    @staticmethod
    def _terminate_current_process() -> None:
        """向当前应用进程发送 SIGTERM。

        Returns:
            None: 信号发送成功后不返回额外值。

        Raises:
            OSError: 当系统无法发送进程信号时抛出。
        """
        os.kill(os.getpid(), signal.SIGTERM)
