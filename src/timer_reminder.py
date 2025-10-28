"""
计时提醒调度模块。

封装计时器后端逻辑，提供创建与恢复提醒的统一入口，降低主程序耦合度。
"""

from __future__ import annotations

import re
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Protocol, Sequence
from zoneinfo import ZoneInfo

import schedule

_RELATIVE_TIME_TOKEN_PATTERN = re.compile(
    r"^(?P<value>\d+)(?P<unit>[dhms])$", re.IGNORECASE
)


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

    def prune_and_get_active(self, now_ts: int) -> Sequence[dict[str, Any]]:
        """
        清理过期提醒并返回仍需触发的提醒列表。

        Args:
            now_ts (int): 当前 Unix 时间戳。

        Returns:
            str: 描述当前任务状态的提示信息。
            Sequence[dict[str, Any]]: 尚未过期的提醒集合。
        """

class TimerReminderManager:
    """
    计时提醒管理器，负责解析时间表达式、创建调度任务以及恢复未完成提醒。
    """

    def __init__(
        self,
        reminder_store: ReminderStoreProtocol,
        timezone: ZoneInfo | None = None,
    ) -> None:
        """
        初始化管理器。

        Args:
            reminder_store (ReminderStoreProtocol): 提醒持久化存储实现。
            timezone (ZoneInfo | None): 目标时区，默认使用东京时间。

        Raises:
            AssertionError: 当 `reminder_store` 缺失时抛出。
        """
        assert reminder_store is not None, "reminder_store 不能为空"
        self._store = reminder_store
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

    def restore_pending(self, log_prefix: str = "[TimerStore]") -> None:
        """
        恢复尚未触发的提醒任务。

        Args:
            log_prefix (str): 日志前缀，用于区分初始化恢复。
        """
        now_ts = int(time.time())
        active = self._store.prune_and_get_active(now_ts)
        if not active:
            return
        for record in active:
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
            except AssertionError as err:
                sys.stderr.write(f"{log_prefix} 恢复计时器失败：{err}\n")

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
        assert isinstance(expression, str) and expression.strip(), "time_expression 不能为空"
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
        remain_seconds = remain_seconds_hint if remain_seconds_hint is not None else wait_seconds
        remain_seconds = max(1, remain_seconds)

        def _job() -> Any:
            """到达触发时间后发送提醒并删除持久化记录。"""
            try:
                from qq_group_bot import BotConfig, _send_group_at_message

                cfg = BotConfig.from_env()
                text = f"[提醒]：{answer}"
                _send_group_at_message(
                    cfg.api_base,
                    group_id,
                    user_id,
                    text,
                    cfg.access_token,
                )
                print(
                    f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
                    f"\033[33m{log_prefix}\033[0m 计时器触发，已在群 {group_id} 内提醒 @({user_id})：{description}",
                    flush=True,
                )
            except Exception as err:
                sys.stderr.write(f"\033[31m{log_prefix}\033[0m 发送提醒失败：{err}\n")
            finally:
                try:
                    self._store.remove_one(trigger_ts, group_id, user_id, description, answer)
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
        prefix = "\n" if action != "restore" else ""
        print(
            f"{prefix}\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
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
