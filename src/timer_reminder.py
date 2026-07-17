"""
计时提醒调度模块。

封装计时器后端逻辑，提供创建与恢复提醒的统一入口，降低主程序耦合度。
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol
from zoneinfo import ZoneInfo

import schedule

_RELATIVE_TIME_TOKEN_PATTERN = re.compile(
    r"^(?P<value>\d+)(?P<unit>[dhms])$", re.IGNORECASE
)

SendReminder = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ReminderPruneResult:
    """描述持久化提醒清理结果。

    Args:
        active_records (tuple[dict[str, Any], ...]): 仍需恢复的有效记录。
        expired_count (int): 已删除的过期记录数。
        invalid_count (int): 已删除的非法记录数。
    """

    active_records: tuple[dict[str, Any], ...]
    expired_count: int
    invalid_count: int


@dataclass(frozen=True)
class ReminderRestoreResult:
    """描述启动时的提醒恢复结果。

    Args:
        restored_count (int): 已成功注册的提醒数。
        failed_count (int): 注册失败的提醒数。
        expired_count (int): 已清理的过期记录数。
        invalid_count (int): 已清理的非法记录数。
    """

    restored_count: int
    failed_count: int
    expired_count: int
    invalid_count: int

    def display(self) -> str:
        """生成启动摘要文本。

        Returns:
            str: 提醒恢复与清理统计。

        Raises:
            AssertionError: 当任一统计值为负数时抛出。
        """
        values = (
            self.restored_count,
            self.failed_count,
            self.expired_count,
            self.invalid_count,
        )
        assert all(value >= 0 for value in values), "提醒统计值不能为负数"
        result = f"已恢复 {self.restored_count} 个"
        if self.failed_count:
            result += f" · 失败 {self.failed_count} 个"
        result += f" · 已清理 {self.expired_count} 个"
        if self.invalid_count:
            result += f" · 无效 {self.invalid_count} 个"
        return result


class ReminderStoreProtocol(Protocol):
    """
    持久化存储协议。

    提醒存储需要实现添加、移除以及获取有效提醒的能力，以便调度器管理。
    """

    def add(self, record: dict[str, Any]) -> None:
        """
        写入新的提醒记录。

        Args:
            record (dict[str, Any]): 单条提醒记录。
        """

    def remove_one(
        self,
        ts: int,
        group_id: int,
        user_id: int,
        description: str,
        answer: str,
    ) -> None:
        """
        移除指定提醒记录。

        Args:
            ts (int): 触发时间戳。
            group_id (int): 群号。
            user_id (int): 用户号。
            description (str): 提醒描述。
            answer (str): 提醒文本。
        """

    def prune_and_get_active(self, now_ts: int) -> ReminderPruneResult:
        """
        清理过期提醒并返回仍需触发的提醒列表。

        Args:
            now_ts (int): 当前 Unix 时间戳。

        Returns:
            ReminderPruneResult: 有效提醒及清理统计。
        """


class JsonReminderStore:
    """使用 JSON 文件持久化提醒记录。

    Args:
        path (str): 提醒文件路径。
    """

    _LOCK = threading.Lock()

    def __init__(self, path: str) -> None:
        """初始化 JSON 提醒存储。

        Args:
            path (str): 提醒文件路径。

        Raises:
            AssertionError: 当路径为空时抛出。
        """
        assert isinstance(path, str) and path.strip(), "持久化文件路径无效"
        self._path = os.path.abspath(path)

    def _read_all(self) -> list[dict[str, Any]]:
        """读取全部提醒记录。

        Returns:
            list[dict[str, Any]]: 文件中的提醒记录。

        Raises:
            AssertionError: 当文件根节点不是列表时抛出。
            json.JSONDecodeError: 当 JSON 内容损坏时抛出。
        """
        if not os.path.isfile(self._path):
            return []
        with open(self._path, "r", encoding="utf-8") as file:
            raw = file.read()
        if not raw.strip():
            return []
        data = json.loads(raw)
        assert isinstance(data, list), "提醒存储文件格式应为列表"
        return data

    def _write_all(self, items: list[dict[str, Any]]) -> None:
        """原子写入全部提醒记录。

        Args:
            items (list[dict[str, Any]]): 需要保存的提醒记录。

        Returns:
            None: 无返回值。
        """
        temporary_path = self._path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, self._path)

    @staticmethod
    def _validate(record: dict[str, Any]) -> None:
        """校验单条提醒记录。

        Args:
            record (dict[str, Any]): 待校验的提醒记录。

        Raises:
            AssertionError: 当字段缺失或类型不合法时抛出。
        """
        assert isinstance(record, dict), "提醒记录必须为对象"
        ts = record.get("ts")
        group_id = record.get("group_id")
        user_id = record.get("user_id")
        description = record.get("description")
        answer = record.get("answer")
        assert isinstance(ts, int) and ts > 0, "ts 必须为正整数时间戳"
        assert isinstance(group_id, int) and group_id > 0, "group_id 必须为正整数"
        assert isinstance(user_id, int) and user_id > 0, "user_id 必须为正整数"
        assert isinstance(description, str) and description.strip(), "description 不能为空"
        assert isinstance(answer, str) and answer.strip(), "answer 不能为空"

    @staticmethod
    def _normalize(record: dict[str, Any]) -> dict[str, Any]:
        """生成兼容既有文件格式的规范记录。

        Args:
            record (dict[str, Any]): 已通过校验的提醒记录。

        Returns:
            dict[str, Any]: 仅包含既有五个字段的记录。
        """
        return {
            "ts": int(record["ts"]),
            "group_id": int(record["group_id"]),
            "user_id": int(record["user_id"]),
            "description": str(record["description"]),
            "answer": str(record["answer"]),
        }

    def add(self, record: dict[str, Any]) -> None:
        """追加一条提醒记录。

        Args:
            record (dict[str, Any]): 待保存的提醒记录。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当提醒记录不合法时抛出。
        """
        self._validate(record)
        with self._LOCK:
            items = self._read_all()
            items.append(self._normalize(record))
            self._write_all(items)

    def prune_and_get_active(self, now_ts: int) -> ReminderPruneResult:
        """删除过期或非法记录并返回有效提醒。

        Args:
            now_ts (int): 当前 Unix 时间戳。

        Returns:
            ReminderPruneResult: 有效提醒及清理统计。

        Raises:
            AssertionError: 当当前时间戳不合法时抛出。
        """
        assert isinstance(now_ts, int) and now_ts >= 0, "now_ts 必须为非负整数"
        with self._LOCK:
            items = self._read_all()
            active: list[dict[str, Any]] = []
            expired_count = 0
            invalid_count = 0
            for record in items:
                try:
                    self._validate(record)
                except AssertionError:
                    invalid_count += 1
                    continue
                if int(record["ts"]) > now_ts:
                    active.append(self._normalize(record))
                else:
                    expired_count += 1
            self._write_all(active)
        return ReminderPruneResult(
            active_records=tuple(active),
            expired_count=expired_count,
            invalid_count=invalid_count,
        )

    def remove_one(
        self,
        ts: int,
        group_id: int,
        user_id: int,
        description: str,
        answer: str,
    ) -> None:
        """移除第一条完全匹配的提醒记录。

        Args:
            ts (int): 触发时间戳。
            group_id (int): 群号。
            user_id (int): 用户号。
            description (str): 提醒描述。
            answer (str): 提醒文本。

        Returns:
            None: 无返回值；记录不存在时不修改文件。
        """
        with self._LOCK:
            items = self._read_all()
            matched_index: int | None = None
            for index, record in enumerate(items):
                if not isinstance(record, dict):
                    continue
                if (
                    record.get("ts") == ts
                    and record.get("group_id") == group_id
                    and record.get("user_id") == user_id
                    and record.get("description") == description
                    and record.get("answer") == answer
                ):
                    matched_index = index
                    break
            if matched_index is not None:
                items.pop(matched_index)
                self._write_all(items)


class TimerReminderManager:
    """
    计时提醒管理器，负责解析时间表达式、创建调度任务以及恢复未完成提醒。
    """

    def __init__(
        self,
        reminder_store: ReminderStoreProtocol,
        send_reminder: SendReminder,
        timezone: ZoneInfo | None = None,
    ) -> None:
        """
        初始化管理器。

        Args:
            reminder_store (ReminderStoreProtocol): 提醒持久化存储实现。
            send_reminder (SendReminder): QQ 提醒发送函数。
            timezone (ZoneInfo | None): 目标时区，默认使用东京时间。

        Raises:
            AssertionError: 当 `reminder_store` 缺失时抛出。
        """
        assert reminder_store is not None, "reminder_store 不能为空"
        assert callable(send_reminder), "send_reminder 必须可调用"
        self._store = reminder_store
        self._send_reminder = send_reminder
        self._tz = timezone or ZoneInfo("Asia/Tokyo")
        self._scheduler = schedule.Scheduler()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reminder-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """
        停止后台调度线程。

        Raises:
            None.
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def create_timer(
        self,
        time_expression: str,
        group_id: int,
        user_id: int,
        description: str,
        answer: str,
        log_prefix: str = "[TimerTool]",
    ) -> str:
        """
        新建提醒任务。

        Args:
            time_expression (str): 时间表达式，支持 at/after 语法。
            group_id (int): 群号。
            user_id (int): 用户号。
            description (str): 提醒描述。
            answer (str): 提醒消息文本。
            log_prefix (str): 日志前缀，便于标识调用来源。

        Returns:
            str: 创建成功的提示信息。

        Raises:
            AssertionError: 当时间表达式无效或早于当前时间时抛出。
        """
        trigger_ts = self._parse_time_expression(time_expression)
        now_ts = int(time.time())
        remain_seconds = max(1, trigger_ts - now_ts)
        record = {
            "ts": trigger_ts,
            "group_id": group_id,
            "user_id": user_id,
            "description": description,
            "answer": answer,
            "time_expression": time_expression,
        }
        self._store.add(record)
        message = self._schedule_job(
            trigger_ts=trigger_ts,
            group_id=group_id,
            user_id=user_id,
            description=description,
            answer=answer,
            log_prefix=log_prefix,
            action="create",
            remain_seconds_hint=remain_seconds,
        )
        return message

    def restore_pending(self, log_prefix: str = "[TimerStore]") -> ReminderRestoreResult:
        """
        恢复尚未触发的提醒任务。

        Args:
            log_prefix (str): 日志前缀，用于区分初始化恢复。

        Returns:
            ReminderRestoreResult: 本次恢复及清理统计。
        """
        now_ts = int(time.time())
        prune_result = self._store.prune_and_get_active(now_ts)
        restored_count = 0
        failed_count = 0
        for record in prune_result.active_records:
            try:
                self._schedule_job(
                    trigger_ts=int(record.get("ts")),
                    group_id=int(record.get("group_id")),
                    user_id=int(record.get("user_id")),
                    description=str(record.get("description")),
                    answer=str(record.get("answer")),
                    log_prefix=log_prefix,
                    action="restore",
                )
                restored_count += 1
            except (AssertionError, TypeError, ValueError) as err:
                failed_count += 1
                sys.stderr.write(f"{log_prefix} 恢复计时器失败：{err}\n")
        return ReminderRestoreResult(
            restored_count=restored_count,
            failed_count=failed_count,
            expired_count=prune_result.expired_count,
            invalid_count=prune_result.invalid_count,
        )

    def _run_loop(self) -> None:
        """
        后台线程循环执行调度任务，并根据 idle 时间动态休眠。
        """
        while not self._stop_event.is_set():
            with self._lock:
                try:
                    self._scheduler.run_pending()
                except Exception as err:
                    sys.stderr.write(f"[TimerScheduler] 调度执行失败：{err}\n")
                idle_attr = getattr(self._scheduler, "idle_seconds", None)
                idle = idle_attr() if callable(idle_attr) else idle_attr
            if idle is None:
                idle = 1.0
            if idle < 0:
                idle = 0.0
            wait_seconds = min(idle, 60.0)
            self._stop_event.wait(wait_seconds)

    def _parse_time_expression(self, expression: str) -> int:
        """
        解析时间表达式，输出触发时间戳。

        Args:
            expression (str): at/after 格式的时间描述。

        Returns:
            int: 触发时刻的 Unix 时间戳。

        Raises:
            AssertionError: 当表达式无效或指向过去时抛出。
        """
        assert (
            isinstance(expression, str) and expression.strip()
        ), "time_expression 不能为空"
        normalized = expression.strip()
        now = datetime.now(tz=self._tz)
        if normalized.startswith("at:"):
            raw = normalized[3:].strip()
            assert raw, "at: 之后必须提供时间，如 at:2024-12-16T13:56"
            try:
                dt = datetime.fromisoformat(raw)
            except ValueError as exc:
                raise AssertionError(
                    "time 使用 at: 时必须为 ISO 日期时间，例如 at:2024-12-16T13:56"
                ) from exc
            if dt.tzinfo:
                target = dt.astimezone(self._tz)
            else:
                target = dt.replace(tzinfo=self._tz)
            assert target > now, "time 指定的绝对时间必须晚于当前时间"
            return int(target.timestamp())
        if normalized.startswith("after:"):
            raw = normalized[6:].strip()
            assert raw, "after: 之后必须提供时间片段，如 after:1h-30m"
            tokens = [token.strip() for token in raw.split("-") if token.strip()]
            assert tokens, "after: 至少提供一个时间片段"
            total_seconds = 0
            for token in tokens:
                match = _RELATIVE_TIME_TOKEN_PATTERN.match(token)
                assert match, f"time 片段格式非法：{token}"
                value = int(match.group("value"))
                unit = match.group("unit").lower()
                if unit == "d":
                    total_seconds += value * 86400
                elif unit == "h":
                    total_seconds += value * 3600
                elif unit == "m":
                    total_seconds += value * 60
                else:
                    total_seconds += value
            assert total_seconds > 0, "after: 累计秒数必须大于 0"
            target = now + timedelta(seconds=total_seconds)
            return int(target.timestamp())
        raise AssertionError("time_expression 必须以 at: 或 after: 开头")

    def _schedule_job(
        self,
        trigger_ts: int,
        group_id: int,
        user_id: int,
        description: str,
        answer: str,
        log_prefix: str,
        action: str,
        remain_seconds_hint: int | None = None,
    ) -> str:
        """
        注册调度任务。

        Args:
            trigger_ts (int): 触发时间戳。
            group_id (int): 群号。
            user_id (int): 用户号。
            description (str): 提醒描述。
            answer (str): 提醒文本。
            log_prefix (str): 日志前缀。

        Raises:
            AssertionError: 当触发时间早于当前时刻时抛出。
        """
        now_ts = int(time.time())
        wait_seconds = trigger_ts - now_ts
        assert wait_seconds >= 0, "触发时间必须晚于当前时刻"
        wait_seconds = max(1, wait_seconds)
        target_dt = datetime.fromtimestamp(trigger_ts, tz=self._tz)
        readable_time = target_dt.strftime("%Y-%m-%d %H:%M")
        offset_text = self._format_offset(target_dt)
        remain_seconds = (
            remain_seconds_hint if remain_seconds_hint is not None else wait_seconds
        )
        remain_seconds = max(1, remain_seconds)

        def _job() -> Any:
            """到达触发时间后发送提醒并删除持久化记录。"""
            try:
                text = f"📣[提醒]：{answer}"
                self._send_reminder(group_id, user_id, text)
                print(
                    f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
                    f"\033[33m{log_prefix}\033[0m 计时器触发，已在群 {group_id} 内提醒 @({user_id})：{description}",
                    flush=True,
                )
            except Exception as err:
                sys.stderr.write(f"\033[31m{log_prefix}\033[0m 发送提醒失败：{err}\n")
            finally:
                try:
                    self._store.remove_one(
                        trigger_ts, group_id, user_id, description, answer
                    )
                except Exception as cleanup_err:
                    sys.stderr.write(
                        f"\033[31m{log_prefix}\033[0m 移除记录失败：{cleanup_err}\n"
                    )
            return schedule.CancelJob

        with self._lock:
            job = self._scheduler.every(wait_seconds).seconds.do(_job)
            job.tag(f"group:{group_id}", f"user:{user_id}", f"ts:{trigger_ts}")

        if action == "restore":
            action_text = "恢复计时器"
        else:
            action_text = "已创建计时器"

        message = (
            f"{action_text}：预计 {readable_time} ({offset_text}) "
            f"将在群 {group_id} 内提醒 @({user_id})：{description}，约 {remain_seconds} 秒后触发。"
        )
        if action != "restore":
            print(
                f"\n\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
                f"\033[33m{log_prefix}\033[0m {message}",
                flush=True,
            )
        return message

    def _format_offset(self, dt: datetime) -> str:
        """
        根据目标时间生成 UTC 偏移字符串。

        Args:
            dt (datetime): 目标时间。

        Returns:
            str: 形如 `UTC+09:00` 的偏移描述。
        """
        offset = dt.utcoffset()
        if offset is None:
            return "UTC+00:00"
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"UTC{sign}{hours:02d}:{minutes:02d}"
