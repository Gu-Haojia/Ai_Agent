"""Postgres checkpoint 保留策略。"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Cursor
from psycopg import Error as PsycopgError
from psycopg.rows import DictRow


_SELECT_RECENT_CHECKPOINTS_SQL = """
SELECT
    checkpoint_id,
    COALESCE((checkpoint ->> 'v')::integer, 0) AS checkpoint_version
FROM checkpoints
WHERE thread_id = %s
  AND checkpoint_ns = %s
ORDER BY checkpoint_id DESC
LIMIT %s
"""

_DELETE_OLD_WRITES_SQL = """
DELETE FROM checkpoint_writes
WHERE thread_id = %s
  AND checkpoint_ns = %s
  AND checkpoint_id < %s
"""

_DELETE_OLD_CHECKPOINTS_SQL = """
DELETE FROM checkpoints
WHERE thread_id = %s
  AND checkpoint_ns = %s
  AND checkpoint_id < %s
"""

_DETACH_OLDEST_RETAINED_CHECKPOINT_SQL = """
UPDATE checkpoints
SET parent_checkpoint_id = NULL
WHERE thread_id = %s
  AND checkpoint_ns = %s
  AND checkpoint_id = %s
"""

_DELETE_UNREFERENCED_BLOBS_SQL = """
DELETE FROM checkpoint_blobs AS blob
WHERE blob.thread_id = %s
  AND blob.checkpoint_ns = %s
  AND NOT EXISTS (
      SELECT 1
      FROM checkpoints AS checkpoint
      CROSS JOIN LATERAL jsonb_each_text(
          COALESCE(
              checkpoint.checkpoint -> 'channel_versions',
              '{}'::jsonb
          )
      ) AS referenced(channel, version)
      WHERE checkpoint.thread_id = blob.thread_id
        AND checkpoint.checkpoint_ns = blob.checkpoint_ns
        AND referenced.channel = blob.channel
        AND referenced.version = blob.version
  )
"""


@dataclass(frozen=True)
class CheckpointPruneResult:
    """
    记录一次 checkpoint 清理结果。

    Args:
        kept_checkpoints (int): 清理后保留的 checkpoint 数量。
        deleted_checkpoints (int): 删除的 checkpoint 数量。
        deleted_writes (int): 删除的中间写入数量。
        deleted_blobs (int): 删除的无引用 Blob 数量。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None.
    """

    kept_checkpoints: int
    deleted_checkpoints: int
    deleted_writes: int
    deleted_blobs: int


class CheckpointRetentionError(RuntimeError):
    """
    表示 checkpoint 清理在数据库边界内失败。

    Args:
        message (str): 清理失败说明。

    Returns:
        None: 异常初始化不返回额外值。

    Raises:
        None.
    """


class RetainingPostgresSaver(PostgresSaver):
    """
    在原生 PostgresSaver 上增加按线程保留 checkpoint 的能力。

    正常的 checkpoint 写入、读取与恢复完全沿用父类实现；清理仅在
    调用方明确执行 ``prune_thread`` 时发生。

    Args:
        conn: 由 PostgresSaver 接收的数据库连接或连接池。
        pipe: 可选 psycopg pipeline。
        serde: 可选 checkpoint 序列化器。

    Returns:
        None: 类初始化不返回额外值。

    Raises:
        ValueError: 当父类发现连接和 pipeline 组合非法时抛出。
    """

    def prune_thread(
        self,
        thread_id: str,
        *,
        checkpoint_ns: str = "",
        keep_last: int = 5,
    ) -> CheckpointPruneResult:
        """
        在一个事务中保留指定线程最近的 checkpoint。

        Args:
            thread_id (str): 需要清理的 LangGraph 线程 ID。
            checkpoint_ns (str): checkpoint 命名空间，
                默认清理根命名空间。
            keep_last (int): 需要保留的最新 checkpoint 数量。

        Returns:
            CheckpointPruneResult: 本次保留与删除数量。

        Raises:
            AssertionError: 当输入参数非法或 saver 使用 pipeline 时抛出。
            CheckpointRetentionError: 当数据库操作失败或保留范围包含
                不支持安全裁剪的旧 checkpoint 格式时抛出。
        """
        normalized_thread_id = thread_id.strip()
        assert normalized_thread_id, "thread_id 不能为空"
        assert isinstance(checkpoint_ns, str), "checkpoint_ns 必须是字符串"
        assert keep_last >= 1, "keep_last 必须大于等于 1"
        assert self.pipe is None, "checkpoint 清理不支持 pipeline 模式"

        try:
            with self._cursor() as cursor:
                with cursor.connection.transaction():
                    return self._prune_with_cursor(
                        cursor,
                        thread_id=normalized_thread_id,
                        checkpoint_ns=checkpoint_ns,
                        keep_last=keep_last,
                    )
        except (PsycopgError, AssertionError) as exc:
            raise CheckpointRetentionError(
                f"线程 {normalized_thread_id} 的 checkpoint 清理失败：{exc}"
            ) from exc

    @staticmethod
    def _prune_with_cursor(
        cursor: Cursor[DictRow],
        *,
        thread_id: str,
        checkpoint_ns: str,
        keep_last: int,
    ) -> CheckpointPruneResult:
        """
        使用已进入事务的游标执行 checkpoint 清理。

        Args:
            cursor (Cursor[DictRow]): 已进入事务的 psycopg 游标。
            thread_id (str): 需要清理的线程 ID。
            checkpoint_ns (str): 需要清理的 checkpoint 命名空间。
            keep_last (int): 需要保留的 checkpoint 数量。

        Returns:
            CheckpointPruneResult: 本次保留与删除数量。

        Raises:
            AssertionError: 当保留范围包含 v4 以前的 checkpoint 时抛出。
            PsycopgError: 当 SQL 执行失败时抛出。
        """
        cursor.execute(
            _SELECT_RECENT_CHECKPOINTS_SQL,
            (thread_id, checkpoint_ns, keep_last + 1),
        )
        checkpoint_rows = cursor.fetchall()
        retained_rows = checkpoint_rows[:keep_last]
        if len(checkpoint_rows) <= keep_last:
            return CheckpointPruneResult(
                kept_checkpoints=len(retained_rows),
                deleted_checkpoints=0,
                deleted_writes=0,
                deleted_blobs=0,
            )

        unsupported_versions = [
            int(row["checkpoint_version"])
            for row in retained_rows
            if int(row["checkpoint_version"]) < 4
        ]
        assert not unsupported_versions, (
            "最近保留范围包含 v4 以前的 checkpoint 格式版本："
            f"{unsupported_versions}"
        )

        cutoff_id = str(retained_rows[-1]["checkpoint_id"])
        params = (thread_id, checkpoint_ns, cutoff_id)

        cursor.execute(_DELETE_OLD_WRITES_SQL, params)
        deleted_writes = cursor.rowcount

        cursor.execute(_DELETE_OLD_CHECKPOINTS_SQL, params)
        deleted_checkpoints = cursor.rowcount

        cursor.execute(_DETACH_OLDEST_RETAINED_CHECKPOINT_SQL, params)

        cursor.execute(
            _DELETE_UNREFERENCED_BLOBS_SQL,
            (thread_id, checkpoint_ns),
        )
        deleted_blobs = cursor.rowcount

        return CheckpointPruneResult(
            kept_checkpoints=len(retained_rows),
            deleted_checkpoints=deleted_checkpoints,
            deleted_writes=deleted_writes,
            deleted_blobs=deleted_blobs,
        )
