"""
X 推文监控 Agent 工具核心模块。

复用进程内唯一的 XMonitorManager，并为 Agent 工具提供监控与链接发送能力。
"""

from __future__ import annotations

import os
from functools import partial
from threading import Lock
from typing import Sequence

from src.x_monitor import DEFAULT_LIMIT, XMonitorManager, XPostResult
from src.x_monitor_media import _send_group_msg, send_x_message_with_images

DEFAULT_ONEBOT_API_BASE = "http://127.0.0.1:3000"

_X_MONITOR_MANAGER: XMonitorManager | None = None
_X_MONITOR_MANAGER_LOCK = Lock()


def get_x_monitor_manager() -> XMonitorManager:
    """
    获取进程内共享的 X 推文监控管理器。

    Returns:
        XMonitorManager: 进程内唯一的监控管理器。

    Raises:
        AssertionError: 当持久化文件路径为空时抛出。
    """
    global _X_MONITOR_MANAGER
    with _X_MONITOR_MANAGER_LOCK:
        if _X_MONITOR_MANAGER is None:
            store_path = os.environ.get("X_MONITOR_STORE", ".x_monitor.json").strip()
            assert store_path, "X_MONITOR_STORE 不能为空"
            _X_MONITOR_MANAGER = XMonitorManager(store_path=store_path)
        return _X_MONITOR_MANAGER


def _get_cmd_allowed_user_ids() -> tuple[int, ...]:
    """
    读取允许执行受限 X 监控工具的 QQ 用户号白名单。

    Args:
        None: 本函数不接收参数。

    Returns:
        tuple[int, ...]: 允许执行工具的 QQ 用户号。空元组表示不限制。

    Raises:
        AssertionError: 当 `CMD_ALLOWED_USERS` 包含非正整数用户号时抛出。
    """
    raw = os.environ.get("CMD_ALLOWED_USERS", "").strip()
    if not raw:
        return ()
    user_ids: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        assert token.isdigit(), "CMD_ALLOWED_USERS 只能包含逗号分隔的正整数用户号"
        user_id = int(token)
        assert user_id > 0, "CMD_ALLOWED_USERS 只能包含正整数用户号"
        user_ids.append(user_id)
    return tuple(user_ids)


def is_x_monitor_tool_user_allowed(user_id: int) -> bool:
    """
    判断用户是否允许执行 xmonitor 工具。

    Args:
        user_id (int): 当前用户消息中的 User_id。

    Returns:
        bool: True 表示允许执行；False 表示拒绝执行。

    Raises:
        AssertionError: 当用户号非法或权限配置非法时抛出。
    """
    assert user_id > 0, "user_id 必须为正整数"
    allowed_user_ids = _get_cmd_allowed_user_ids()
    return not allowed_user_ids or user_id in allowed_user_ids


def build_x_monitor_permission_failure(
    action: str,
    group_id: int,
    user_id: int,
) -> dict[str, object]:
    """
    构造 xmonitor 工具权限拒绝结果。

    Args:
        action (str): xmonitor 操作类型。
        group_id (int): 当前用户消息中的 Group_id。
        user_id (int): 当前用户消息中的 User_id。

    Returns:
        dict[str, object]: 便于 Agent 继续生成的失败结果。

    Raises:
        AssertionError: 当输入参数非法时抛出。
    """
    normalized_action = action.strip().lower()
    assert normalized_action, "action 不能为空"
    assert group_id > 0, "group_id 必须为正整数"
    assert user_id > 0, "user_id 必须为正整数"
    return {
        "action": normalized_action,
        "group_id": group_id,
        "user_id": user_id,
        "status": "failed",
        "error": "permission_denied",
        "message": "无权执行 xmonitor 工具，请联系管理员加入 CMD_ALLOWED_USERS。",
    }


def start_x_monitor(
    username: str,
    interval_seconds: float,
    group_id: int,
    user_id: int,
) -> None:
    """
    启动指定群的 X 账号推文监控。

    Args:
        username (str): X 用户名，可带 `@` 前缀。
        interval_seconds (float): 轮询间隔秒数。
        group_id (int): 接收推文通知的群号。
        user_id (int): 创建任务的用户 QQ 号。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当输入参数非法时抛出。
        RuntimeError: 当 X API 请求失败或任务数量超过限制时抛出。
    """
    assert username.strip(), "username 不能为空"
    assert interval_seconds > 0, "interval_seconds 必须大于 0"
    assert group_id > 0, "group_id 必须为正整数"
    assert user_id > 0, "user_id 必须为正整数"
    get_x_monitor_manager().start_watch(
        username=username,
        interval=interval_seconds,
        limit_per_cycle=DEFAULT_LIMIT,
        group_id=group_id,
        user_id=user_id,
        notify=partial(_send_x_monitor_text, group_id),
        notify_media=partial(_send_x_monitor_media, group_id),
    )


def stop_x_monitor(username: str, group_id: int) -> int:
    """
    停止指定群中的 X 账号推文监控。

    Args:
        username (str): X 用户名，可带 `@` 前缀。
        group_id (int): 目标群号。

    Returns:
        int: 停止的任务数量。

    Raises:
        AssertionError: 当输入参数非法时抛出。
    """
    assert username.strip(), "username 不能为空"
    assert group_id > 0, "group_id 必须为正整数"
    return get_x_monitor_manager().stop_watch(username, group_id=group_id)


def list_x_monitor_tasks(group_id: int) -> list[dict[str, object]]:
    """
    列出指定群中的 X 推文监控任务。

    Args:
        group_id (int): 目标群号。

    Returns:
        list[dict[str, object]]: 当前群中的监控任务快照。

    Raises:
        AssertionError: 当群号非法时抛出。
    """
    assert group_id > 0, "group_id 必须为正整数"
    tasks = get_x_monitor_manager().list_watch_tasks()
    return [task for task in tasks if task.get("group_id") == group_id]


def send_x_link(url: str, group_id: int, user_id: int) -> XPostResult:
    """
    拉取指定 X 推文并向目标群发送截图。

    Args:
        url (str): X/Twitter 推文链接。
        group_id (int): 接收截图的群号。
        user_id (int): 发起请求的用户 QQ 号。

    Returns:
        XPostResult: 已发送的推文结果。

    Raises:
        AssertionError: 当输入参数或推文链接非法时抛出。
        RuntimeError: 当 X API 或 OneBot 请求失败时抛出。
        ValueError: 当 X API 返回字段格式非法时抛出。
    """
    assert url.strip(), "url 不能为空"
    assert group_id > 0, "group_id 必须为正整数"
    assert user_id > 0, "user_id 必须为正整数"
    manager = get_x_monitor_manager()
    item = manager.fetch_link(url)
    message = manager.format_lines([item], "LINK")
    api_base, access_token = _onebot_config()
    send_x_message_with_images(
        api_base,
        group_id,
        access_token,
        message,
        [item],
    )
    print(
        f"[XMonitor Tool] 链接拉取 url='{url}' group={group_id} user={user_id}",
        flush=True,
    )
    return item


def _onebot_config() -> tuple[str, str]:
    """
    从环境变量读取 OneBot 发送配置。

    Returns:
        tuple[str, str]: OneBot API 基地址与访问令牌。

    Raises:
        AssertionError: 当 API 基地址为空时抛出。
    """
    api_base = os.environ.get("ONEBOT_API_BASE", DEFAULT_ONEBOT_API_BASE).strip()
    access_token = os.environ.get("ONEBOT_ACCESS_TOKEN", "").strip()
    assert api_base, "ONEBOT_API_BASE 不能为空"
    return api_base.rstrip("/"), access_token


def _send_x_monitor_text(group_id: int, text: str) -> None:
    """
    向指定群发送 X 推文监控文本通知。

    Args:
        group_id (int): 目标群号。
        text (str): 文本内容。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当参数非法时抛出。
        RuntimeError: 当 OneBot 请求失败时抛出。
    """
    assert group_id > 0, "group_id 必须为正整数"
    assert text.strip(), "text 不能为空"
    api_base, access_token = _onebot_config()
    _send_group_msg(api_base, group_id, text, access_token)


def _send_x_monitor_media(
    group_id: int,
    text: str,
    items: Sequence[XPostResult],
    tag: str,
) -> None:
    """
    向指定群发送 X 推文监控截图。

    Args:
        group_id (int): 目标群号。
        text (str): 文本内容。
        items (Sequence[XPostResult]): 推文列表。
        tag (str): 通知标签。

    Returns:
        None: 无返回值。

    Raises:
        AssertionError: 当参数非法时抛出。
        RuntimeError: 当 OneBot 请求失败时抛出。
    """
    assert group_id > 0, "group_id 必须为正整数"
    assert text.strip(), "text 不能为空"
    assert items, "items 不能为空"
    assert tag.strip(), "tag 不能为空"
    api_base, access_token = _onebot_config()
    send_x_message_with_images(
        api_base,
        group_id,
        access_token,
        text,
        items,
    )
