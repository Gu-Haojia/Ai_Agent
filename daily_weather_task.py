"""
每日天气提问调度模块。

提供独立的调度器，负责在指定时间触发对 Agent 的提问并将文本回复广播到多个群聊。
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from threading import Event, Thread
from typing import Callable, Optional, Sequence, TypeAlias

import schedule

from sql_agent_cli_stream_plus import SQLCheckpointAgentStreamingPlus

if False:  # pragma: no cover - 类型检查使用，避免循环导入
    from qq_group_bot import BotConfig  # noqa: F401


SendGroupText: TypeAlias = Callable[[int, str], None]


def parse_daily_task_groups(raw: str) -> tuple[int, ...]:
    """
    解析 DAILY_TASK 环境变量得到目标群号。

    Args:
        raw (str): 原始环境变量字符串，使用半角逗号分隔群号。

    Returns:
        tuple[int, ...]: 去重后的有效群号元组，按输入顺序保留。
    """
    if not raw:
        return ()
    groups: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        entry = token.strip()
        if not entry:
            continue
        assert entry.isdigit(), f"DAILY_TASK 包含非法群号：{entry}"
        gid = int(entry)
        assert gid > 0, f"DAILY_TASK 中的群号必须为正整数：{entry}"
        if gid in seen:
            continue
        seen.add(gid)
        groups.append(gid)
    return tuple(groups)


class DailyWeatherTask:
    """
    每日定时向 Agent 提问天气并广播文本结果的调度器。

    该调度器独立于消息处理逻辑运行，通过 schedule 库计算下一次触发时间，
    避免线程持续进行秒级倒计时。
    """

    def __init__(
        self,
        agent: SQLCheckpointAgentStreamingPlus,
        send_func: SendGroupText,
        group_ids: Sequence[int],
        run_time: str = "09:00",
        question: str = "今天的天气",
    ) -> None:
        """
        初始化调度器。

        Args:
            agent (SQLCheckpointAgentStreamingPlus): 已初始化的 Agent 实例。
            send_func (SendGroupText): 发送文本到群聊的回调函数。
            group_ids (Sequence[int]): 准备广播的目标群号列表。
            run_time (str): 每日触发时间，必须为 HH:MM（24 小时制）。
            question (str): 提问内容，默认“今天的天气”。

        Raises:
            AssertionError: 当参数不符合预期时抛出。
        """
        assert isinstance(question, str) and question.strip(), "question 不能为空"
        assert isinstance(run_time, str) and run_time.strip(), "run_time 不能为空"
        try:
            datetime.strptime(run_time, "%H:%M")
        except ValueError as exc:
            raise AssertionError("run_time 必须为 HH:MM（24 小时制）") from exc
        assert callable(send_func), "send_func 必须可调用"
        assert isinstance(agent, SQLCheckpointAgentStreamingPlus), "agent 类型非法"
        normalized_groups = tuple(int(gid) for gid in group_ids if int(gid) > 0)

        self._agent = agent
        self._send_func = send_func
        self._group_ids: tuple[int, ...] = normalized_groups
        self._run_time = run_time
        self._question = question.strip()
        self._scheduler = schedule.Scheduler()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._started = False

    def start(self) -> None:
        """
        启动调度线程（若未配置群号则直接返回）。

        Raises:
            AssertionError: 当调度器已启动时重复调用。
        """
        if not self._group_ids:
            print(
                f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
                "[DailyTask] 未配置 DAILY_TASK，跳过自动提问。",
                flush=True,
            )
            return
        assert not self._started, "调度器已启动，请勿重复调用 start()"
        self._scheduler.every().day.at(self._run_time).do(self._execute_once)
        self._thread = Thread(target=self._run_loop, name="daily-weather-task", daemon=True)
        self._started = True
        self._thread.start()
        print(
            f"\033[94m{time.strftime('[%m-%d %H:%M:%S]', time.localtime())}\033[0m "
            f"[DailyTask] 调度已启动，将在每日 {self._run_time} 提问。目标群：{self._group_ids}",
            flush=True,
        )

    def _run_loop(self) -> None:
        """运行调度循环，通过动态等待避免频繁唤醒。"""
        while not self._stop_event.is_set():
            self._scheduler.run_pending()
            idle = self._scheduler.idle_seconds()
            if idle is None:
                idle = 60.0
            if idle < 0:
                idle = 0.0
            wait_seconds = min(idle, 3600.0)
            self._stop_event.wait(wait_seconds)

    def _execute_once(self) -> None:
        """执行一次提问并广播结果。"""
        timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
        print(
            f"\033[94m{timestamp}\033[0m [DailyTask] 准备提问：{self._question}",
            flush=True,
        )
        try:
            answer = self._agent.chat_once_stream(self._question)
            assert isinstance(answer, str) and answer.strip(), "Agent 未返回文本内容"
            reply = answer.strip()
        except Exception as err:
            sys.stderr.write(f"[DailyTask] 调用 Agent 失败: {err}\n")
            return

        # 添加一个表情
        reply = f"📅 {reply}"

        for gid in self._group_ids:
            try:
                self._send_func(gid, reply)
                print(
                    f"\033[94m{timestamp}\033[0m [DailyTask] 已发送到群 {gid}",
                    flush=True,
                )
            except Exception as err:
                sys.stderr.write(f"[DailyTask] 群 {gid} 发送失败: {err}\n")

    def stop(self) -> None:
        """停止调度线程（若未启动则忽略）。"""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
