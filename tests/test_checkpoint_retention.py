"""Postgres checkpoint 保留策略测试。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from unittest import mock

import pytest
from langchain_core.messages import AIMessage

from sql_agent_cli_stream_plus import AgentConfig, SQLCheckpointAgentStreamingPlus
from src.checkpoint_retention import (
    CheckpointPruneResult,
    CheckpointRetentionError,
    RetainingPostgresSaver,
)


class _FakeConnection:
    """为保留策略测试提供事务上下文。"""

    def __init__(self) -> None:
        """
        初始化事务调用计数。

        Returns:
            None: 初始化不返回值。

        Raises:
            None.
        """
        self.transaction_calls = 0

    def transaction(self) -> nullcontext[None]:
        """
        返回空事务上下文。

        Returns:
            nullcontext[None]: 测试使用的空事务上下文。

        Raises:
            None.
        """
        self.transaction_calls += 1
        return nullcontext()


class _FakeCursor:
    """记录清理 SQL 及其参数的测试游标。"""

    def __init__(
        self,
        checkpoint_rows: list[dict[str, object]],
        *,
        deleted_writes: int = 0,
        deleted_checkpoints: int = 0,
        deleted_blobs: int = 0,
    ) -> None:
        """
        初始化测试游标。

        Args:
            checkpoint_rows (list[dict[str, object]]): 最新优先的检查点行。
            deleted_writes (int): 模拟删除的中间写入数。
            deleted_checkpoints (int): 模拟删除的检查点数。
            deleted_blobs (int): 模拟删除的 Blob 数。

        Returns:
            None: 初始化不返回值。

        Raises:
            None.
        """
        self.connection = _FakeConnection()
        self._checkpoint_rows = checkpoint_rows
        self._deleted_writes = deleted_writes
        self._deleted_checkpoints = deleted_checkpoints
        self._deleted_blobs = deleted_blobs
        self._selected_rows: list[dict[str, object]] = []
        self.rowcount = -1
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def execute(
        self,
        query: str,
        params: tuple[object, ...],
    ) -> "_FakeCursor":
        """
        记录 SQL 并设置对应模拟结果。

        Args:
            query (str): 待执行 SQL。
            params (tuple[object, ...]): SQL 参数。

        Returns:
            _FakeCursor: 当前游标。

        Raises:
            AssertionError: 当测试收到未声明的 SQL 时抛出。
        """
        normalized = " ".join(query.split())
        self.statements.append((normalized, params))
        if normalized.startswith("SELECT checkpoint_id"):
            limit = int(params[-1])
            self._selected_rows = self._checkpoint_rows[:limit]
            self.rowcount = len(self._selected_rows)
        elif normalized.startswith("DELETE FROM checkpoint_writes"):
            self.rowcount = self._deleted_writes
        elif normalized.startswith("DELETE FROM checkpoints"):
            self.rowcount = self._deleted_checkpoints
        elif normalized.startswith("UPDATE checkpoints"):
            self.rowcount = 1
        elif normalized.startswith("DELETE FROM checkpoint_blobs"):
            self.rowcount = self._deleted_blobs
        else:
            raise AssertionError(f"测试收到未声明的 SQL：{normalized}")
        return self

    def fetchall(self) -> list[dict[str, object]]:
        """
        返回最近一次查询的模拟行。

        Returns:
            list[dict[str, object]]: 模拟检查点行。

        Raises:
            None.
        """
        return list(self._selected_rows)


class _FakeRetainingPostgresSaver(RetainingPostgresSaver):
    """使用测试游标运行真实保留策略。"""

    def __init__(self, cursor: _FakeCursor) -> None:
        """
        保存测试游标。

        Args:
            cursor (_FakeCursor): 用于记录 SQL 的测试游标。

        Returns:
            None: 初始化不返回值。

        Raises:
            None.
        """
        self._test_cursor = cursor
        self.pipe = None

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[_FakeCursor]:
        """
        返回测试游标上下文。

        Args:
            pipeline (bool): 是否请求 pipeline；测试中必须为 False。

        Yields:
            Iterator[_FakeCursor]: 测试游标。

        Raises:
            AssertionError: 当被要求使用 pipeline 时抛出。
        """
        assert pipeline is False, (
            "保留策略必须使用显式事务而不是 pipeline"
        )
        yield self._test_cursor


def _checkpoint_rows(count: int, *, version: int = 4) -> list[dict[str, object]]:
    """
    构造最新优先的模拟 checkpoint 行。

    Args:
        count (int): 需要生成的行数。
        version (int): checkpoint 格式版本。

    Returns:
        list[dict[str, object]]: 最新 checkpoint 排在前面的行列表。

    Raises:
        AssertionError: 当 count 为负数时抛出。
    """
    assert count >= 0, "count 不能为负数"
    return [
        {
            "checkpoint_id": f"checkpoint-{index:04d}",
            "checkpoint_version": version,
        }
        for index in range(count, 0, -1)
    ]


def test_prune_thread_keeps_latest_five_and_cleans_related_rows() -> None:
    """
    验证清理事务保留最近五个 checkpoint 并处理关联数据。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    cursor = _FakeCursor(
        _checkpoint_rows(8),
        deleted_writes=7,
        deleted_checkpoints=3,
        deleted_blobs=4,
    )
    saver = _FakeRetainingPostgresSaver(cursor)

    result = saver.prune_thread("thread-1", checkpoint_ns="", keep_last=5)

    assert result.kept_checkpoints == 5
    assert result.deleted_checkpoints == 3
    assert result.deleted_writes == 7
    assert result.deleted_blobs == 4
    assert [statement.split()[0] for statement, _ in cursor.statements] == [
        "SELECT",
        "DELETE",
        "DELETE",
        "UPDATE",
        "DELETE",
    ]
    assert "DELETE FROM checkpoint_writes" in cursor.statements[1][0]
    assert "DELETE FROM checkpoints" in cursor.statements[2][0]
    assert "SET parent_checkpoint_id = NULL" in cursor.statements[3][0]
    assert "jsonb_each_text" in cursor.statements[4][0]
    assert cursor.statements[1][1][-1] == "checkpoint-0004"
    assert all(
        params[:2] == ("thread-1", "")
        for _, params in cursor.statements
    )
    assert cursor.connection.transaction_calls == 1


@pytest.mark.parametrize("count", [0, 1, 4, 5])
def test_prune_thread_does_not_delete_when_history_is_within_limit(
    count: int,
) -> None:
    """
    验证 checkpoint 数量未超过上限时只执行读取。

    Args:
        count (int): 模拟 checkpoint 数量。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    cursor = _FakeCursor(_checkpoint_rows(count))
    saver = _FakeRetainingPostgresSaver(cursor)

    result = saver.prune_thread("thread-1", checkpoint_ns="", keep_last=5)

    assert result.kept_checkpoints == count
    assert result.deleted_checkpoints == 0
    assert result.deleted_writes == 0
    assert result.deleted_blobs == 0
    assert len(cursor.statements) == 1
    assert cursor.statements[0][0].startswith("SELECT checkpoint_id")


def test_prune_thread_rejects_legacy_retained_checkpoint() -> None:
    """
    验证保留范围含旧格式 checkpoint 时整次清理失败。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    rows = _checkpoint_rows(6)
    rows[2]["checkpoint_version"] = 3
    cursor = _FakeCursor(rows)
    saver = _FakeRetainingPostgresSaver(cursor)

    with pytest.raises(CheckpointRetentionError, match="格式版本"):
        saver.prune_thread("thread-legacy", checkpoint_ns="", keep_last=5)

    assert len(cursor.statements) == 1


def test_agent_prunes_only_after_graph_stream_completes() -> None:
    """
    验证 graph 正常结束后才调用当前线程清理。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    graph = mock.Mock()
    graph.stream.return_value = iter(
        [{"messages": [AIMessage(content="已完成")]}]
    )
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._graph = graph
    agent._config = AgentConfig(model_name="fake:echo", thread_id="default-thread")
    agent._prune_completed_thread = mock.Mock()

    with mock.patch("builtins.print"):
        answer = agent.chat_once_stream("你好", thread_id="target-thread")

    assert answer == "已完成"
    agent._prune_completed_thread.assert_called_once_with("target-thread")


def test_agent_uses_stateless_graph_without_thread_id() -> None:
    """
    验证未传入线程 ID 时使用无 checkpoint Graph。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    persistent_graph = mock.Mock()
    stateless_graph = mock.Mock()
    stateless_graph.stream.return_value = iter(
        [{"messages": [AIMessage(content="无状态完成")]}]
    )
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._graph = persistent_graph
    agent._stateless_graph = stateless_graph
    agent._config = AgentConfig(model_name="fake:echo", thread_id="default-thread")
    agent._prune_completed_thread = mock.Mock()

    with mock.patch("builtins.print"):
        answer = agent.chat_once_stream("你好")

    assert answer == "无状态完成"
    persistent_graph.stream.assert_not_called()
    stateless_graph.stream.assert_called_once_with(
        {"messages": [{"role": "user", "content": "你好"}]},
        {"configurable": {}},
        stream_mode="values",
    )
    agent._prune_completed_thread.assert_not_called()


def test_agent_rejects_explicit_empty_thread_id() -> None:
    """
    验证显式空线程 ID 不会被当作无状态调用。

    Returns:
        None: 测试无返回值。

    Raises:
        None: 断言失败时由 pytest 报告。
    """
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)

    with pytest.raises(AssertionError, match="thread_id 不能为空"):
        agent.chat_once_stream("你好", thread_id="   ")


def test_agent_does_not_prune_when_graph_stream_fails() -> None:
    """
    验证 graph 异常退出时保留中间 checkpoint，不执行清理。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    graph = mock.Mock()
    graph.stream.side_effect = RuntimeError("graph failed")
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._graph = graph
    agent._config = AgentConfig(model_name="fake:echo", thread_id="default-thread")
    agent._prune_completed_thread = mock.Mock()

    with mock.patch("builtins.print"), pytest.raises(
        RuntimeError, match="graph failed"
    ):
        agent.chat_once_stream("你好", thread_id="target-thread")

    agent._prune_completed_thread.assert_not_called()


def test_agent_does_not_prune_when_graph_stream_is_interrupted() -> None:
    """
    验证生成被中断时保留中间 checkpoint，不执行清理。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    graph = mock.Mock()
    graph.stream.side_effect = KeyboardInterrupt()
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._graph = graph
    agent._config = AgentConfig(model_name="fake:echo", thread_id="default-thread")
    agent._prune_completed_thread = mock.Mock()

    with mock.patch("builtins.print"):
        answer = agent.chat_once_stream("你好", thread_id="target-thread")

    assert answer == ""
    agent._prune_completed_thread.assert_not_called()


def test_agent_keeps_reply_available_when_retention_fails() -> None:
    """
    验证已完成回复不会被可识别的清理失败覆盖。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    saver = object.__new__(RetainingPostgresSaver)
    saver.prune_thread = mock.Mock(
        side_effect=CheckpointRetentionError("database unavailable")
    )
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._config = AgentConfig(
        model_name="fake:echo",
        pg_conn="postgresql://test",
        thread_id="target-thread",
    )
    agent._saver = saver

    with mock.patch("sys.stderr.write") as write_error:
        agent._prune_completed_thread("target-thread")

    write_error.assert_called_once()
    assert "database unavailable" in write_error.call_args.args[0]


def test_agent_does_not_log_successful_retention() -> None:
    """
    验证 checkpoint 清理成功时不产生逐轮标准输出。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    saver = object.__new__(RetainingPostgresSaver)
    saver.prune_thread = mock.Mock(
        return_value=CheckpointPruneResult(
            kept_checkpoints=5,
            deleted_checkpoints=3,
            deleted_writes=7,
            deleted_blobs=4,
        )
    )
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._config = AgentConfig(
        model_name="fake:echo",
        pg_conn="postgresql://test",
        thread_id="target-thread",
    )
    agent._saver = saver

    with mock.patch("builtins.print") as print_output:
        agent._prune_completed_thread("target-thread")

    print_output.assert_not_called()


def test_agent_skips_retention_for_memory_checkpointer() -> None:
    """
    验证内存 checkpointer 不进入 Postgres 清理分支。

    Returns:
        None: 测试无返回值。

    Raises:
        None.
    """
    agent = object.__new__(SQLCheckpointAgentStreamingPlus)
    agent._config = AgentConfig(
        model_name="fake:echo",
        thread_id="memory-thread",
        use_memory_ckpt=True,
    )

    agent._prune_completed_thread("memory-thread")
