"""
X 推文监控 Agent 工具核心与注册包装层单元测试。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sql_agent_cli_stream_plus as agent_module
import src.x_monitor_tool as tool_module
from src.x_monitor import XMonitorToolError, XPostResult


def _make_post() -> XPostResult:
    """
    构造固定的推文结果。

    Returns:
        XPostResult: 用于测试的推文结果。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return XPostResult(
        username="kana_hanaiwa",
        display_name="Kana Hanaiwa",
        post_id="1",
        text="linked post",
        created_label="05-05 10:02",
        url="https://x.com/kana_hanaiwa/status/1",
        image_urls=("https://example.com/1.jpg",),
        source_payload={"data": [{"id": "1"}]},
    )


class FakeXMonitorManager:
    """
    记录 Agent 工具核心调用参数的 X 监控管理器替身。
    """

    def __init__(self) -> None:
        """
        初始化调用记录。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.start_calls: list[dict[str, object]] = []
        self.stop_calls: list[tuple[str, int | None]] = []
        self.fetch_link_calls: list[str] = []
        self.tasks: list[dict[str, object]] = [
            {"username": "kana_hanaiwa", "group_id": 123, "user_id": 456},
            {"username": "other", "group_id": 999, "user_id": 888},
        ]

    def start_watch(self, **kwargs: object) -> None:
        """
        记录启动监控参数。

        Args:
            **kwargs (object): 监控管理器启动参数。

        Returns:
            None: 无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.start_calls.append(kwargs)

    def stop_watch(self, username: str, group_id: int | None = None) -> int:
        """
        记录停止监控参数。

        Args:
            username (str): X 用户名。
            group_id (int | None): 目标群号。

        Returns:
            int: 固定返回一个已停止任务。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.stop_calls.append((username, group_id))
        return 1

    def list_watch_tasks(self) -> list[dict[str, object]]:
        """
        返回固定的监控任务列表。

        Returns:
            list[dict[str, object]]: 固定任务列表。

        Raises:
            None: 本方法不主动抛出异常。
        """
        return list(self.tasks)

    def fetch_link(self, url: str) -> XPostResult:
        """
        返回固定的链接推文。

        Args:
            url (str): X 推文链接。

        Returns:
            XPostResult: 固定推文结果。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.fetch_link_calls.append(url)
        return _make_post()

    def format_lines(self, items: list[XPostResult], tag: str) -> str:
        """
        返回固定的推文通知文本。

        Args:
            items (list[XPostResult]): 推文列表。
            tag (str): 通知标签。

        Returns:
            str: 固定通知文本。

        Raises:
            AssertionError: 当参数为空时抛出。
        """
        assert items, "items 不能为空"
        assert tag.strip(), "tag 不能为空"
        return f"[X {tag}] | @{items[0].username}"


class XMonitorToolCoreTests(unittest.TestCase):
    """
    验证 X 推文监控 Agent 工具核心。
    """

    def setUp(self) -> None:
        """
        注入监控管理器替身。

        Returns:
            None: 无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.old_manager = tool_module._X_MONITOR_MANAGER
        self.manager = FakeXMonitorManager()
        tool_module._X_MONITOR_MANAGER = self.manager  # type: ignore[assignment]

    def tearDown(self) -> None:
        """
        恢复原始监控管理器。

        Returns:
            None: 无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        tool_module._X_MONITOR_MANAGER = self.old_manager

    def test_get_manager_builds_one_shared_instance(self) -> None:
        """
        首次获取 Manager 时应按环境变量构造，并在后续调用中复用。
        """
        tool_module._X_MONITOR_MANAGER = None
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = str(Path(tmpdir) / "x_monitor.json")
            with mock.patch.dict("os.environ", {"X_MONITOR_STORE": store_path}):
                first = tool_module.get_x_monitor_manager()
                second = tool_module.get_x_monitor_manager()

        self.assertIs(first, second)
        self.assertEqual(first._store_path, Path(store_path))

    def test_start_monitor_builds_onebot_callbacks(self) -> None:
        """
        启动监控时应注入可向指定群发送文本和截图的回调。
        """
        with mock.patch.dict(
            "os.environ",
            {
                "ONEBOT_API_BASE": "http://onebot/",
                "ONEBOT_ACCESS_TOKEN": "token",
            },
        ), mock.patch.object(tool_module, "_send_group_msg") as send_text, mock.patch.object(
            tool_module, "send_x_message_with_images"
        ) as send_media:
            tool_module.start_x_monitor("@kana_hanaiwa", 60, 123, 456)
            call = self.manager.start_calls[0]
            notify = call["notify"]
            notify_media = call["notify_media"]
            assert callable(notify), "notify 必须为可调用对象"
            assert callable(notify_media), "notify_media 必须为可调用对象"
            notify("new post")
            notify_media("new post", [_make_post()], "NEW")

        self.assertEqual(call["username"], "@kana_hanaiwa")
        self.assertEqual(call["group_id"], 123)
        self.assertEqual(call["user_id"], 456)
        send_text.assert_called_once_with("http://onebot", 123, "new post", "token")
        send_media.assert_called_once()
        self.assertEqual(send_media.call_args.args[:3], ("http://onebot", 123, "token"))

    def test_list_monitor_tasks_filters_other_groups(self) -> None:
        """
        Agent 工具只能列出模型传入群号对应的任务。
        """
        tasks = tool_module.list_x_monitor_tasks(123)

        self.assertEqual(tasks, [self.manager.tasks[0]])

    def test_xmonitor_permission_uses_cmd_allowed_users(self) -> None:
        """
        xmonitor 工具权限应复用 CMD_ALLOWED_USERS 白名单。
        """
        with mock.patch.dict("os.environ", {"CMD_ALLOWED_USERS": ""}):
            self.assertTrue(tool_module.is_x_monitor_tool_user_allowed(456))

        with mock.patch.dict("os.environ", {"CMD_ALLOWED_USERS": "456,789"}):
            self.assertTrue(tool_module.is_x_monitor_tool_user_allowed(456))
            self.assertFalse(tool_module.is_x_monitor_tool_user_allowed(123))

    def test_xmonitor_permission_failure_payload(self) -> None:
        """
        xmonitor 工具权限拒绝结果应明确返回失败状态。
        """
        payload = tool_module.build_x_monitor_permission_failure("START", 123, 456)

        self.assertEqual(payload["action"], "start")
        self.assertEqual(payload["group_id"], 123)
        self.assertEqual(payload["user_id"], 456)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "permission_denied")

    def test_send_link_uses_shared_manager_and_target_group(self) -> None:
        """
        链接工具应复用共享 Manager，并将截图发送到模型传入的群。
        """
        url = "https://x.com/kana_hanaiwa/status/1"
        with mock.patch.dict(
            "os.environ",
            {
                "ONEBOT_API_BASE": "http://onebot",
                "ONEBOT_ACCESS_TOKEN": "token",
            },
        ), mock.patch.object(tool_module, "send_x_message_with_images") as send_media:
            item = tool_module.send_x_link(url, 123, 456)

        self.assertEqual(self.manager.fetch_link_calls, [url])
        self.assertEqual(item.post_id, "1")
        send_media.assert_called_once()
        self.assertEqual(send_media.call_args.args[:3], ("http://onebot", 123, "token"))
        self.assertEqual(send_media.call_args.args[4], [item])


class XMonitorToolRegistrationTests(unittest.TestCase):
    """
    验证 Agent 文件内的 X 工具参数包装和返回值处理。
    """

    def _build_tools(self) -> dict[str, object]:
        """
        构造 Agent 文件中定义的 X 工具映射。

        Returns:
            dict[str, object]: 工具名到 LangChain 工具对象的映射。

        Raises:
            AssertionError: 当工具注册缺失时抛出。
        """
        mapped = {
            str(agent_module.xmonitor.name): agent_module.xmonitor,
            str(agent_module.xlink.name): agent_module.xlink,
        }
        assert set(mapped) == {"xmonitor", "xlink"}, "X 工具注册结果不完整"
        return mapped

    def test_xmonitor_wrapper_validates_and_returns_json(self) -> None:
        """
        xmonitor 包装层应校验参数，并返回便于模型消费的 JSON。
        """
        tools = self._build_tools()
        xmonitor = tools["xmonitor"]
        with mock.patch.dict(
            "os.environ", {"CMD_ALLOWED_USERS": ""}
        ), mock.patch.object(agent_module, "start_x_monitor") as start:
            output = xmonitor.invoke(  # type: ignore[attr-defined]
                {
                    "action": "start",
                    "group_id": 123,
                    "user_id": 456,
                    "username": "@kana_hanaiwa",
                    "interval_seconds": 60,
                }
            )

        payload = json.loads(output)
        self.assertEqual(payload["status"], "started")
        self.assertEqual(payload["username"], "kana_hanaiwa")
        start.assert_called_once_with(
            username="@kana_hanaiwa",
            interval_seconds=60,
            group_id=123,
            user_id=456,
        )

    def test_xmonitor_wrapper_rejects_unknown_action(self) -> None:
        """
        xmonitor 包装层遇到未知动作时应显式失败。
        """
        tools = self._build_tools()
        xmonitor = tools["xmonitor"]

        with self.assertRaises(AssertionError):
            xmonitor.invoke(  # type: ignore[attr-defined]
                {
                    "action": "unknown",
                    "group_id": 123,
                    "user_id": 456,
                }
            )

    def test_xmonitor_wrapper_uses_default_interval(self) -> None:
        """
        xmonitor 包装层省略轮询间隔时应默认使用 300 秒。
        """
        tools = self._build_tools()
        xmonitor = tools["xmonitor"]
        with mock.patch.dict(
            "os.environ", {"CMD_ALLOWED_USERS": ""}
        ), mock.patch.object(agent_module, "start_x_monitor") as start:
            output = xmonitor.invoke(  # type: ignore[attr-defined]
                {
                    "action": "start",
                    "group_id": 123,
                    "user_id": 456,
                    "username": "@kana_hanaiwa",
                }
            )

        payload = json.loads(output)
        self.assertEqual(payload["interval_seconds"], 300)
        start.assert_called_once_with(
            username="@kana_hanaiwa",
            interval_seconds=300,
            group_id=123,
            user_id=456,
        )

    def test_xmonitor_wrapper_returns_structured_x_api_failure(self) -> None:
        """
        xmonitor 包装层应把预期的 X API 错误结构化返回给模型。
        """
        tools = self._build_tools()
        xmonitor = tools["xmonitor"]
        with mock.patch.dict(
            "os.environ", {"CMD_ALLOWED_USERS": ""}
        ), mock.patch.object(
            agent_module,
            "start_x_monitor",
            side_effect=XMonitorToolError(
                "x_user_not_found",
                "未找到该 X 用户，请确认 username 是账号 handle。",
            ),
        ):
            output = xmonitor.invoke(  # type: ignore[attr-defined]
                {
                    "action": "start",
                    "group_id": 123,
                    "user_id": 456,
                    "username": "mizuki_yumina",
                }
            )

        payload = json.loads(output)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "x_user_not_found")
        self.assertEqual(payload["username"], "mizuki_yumina")
        self.assertIn("账号 handle", payload["message"])

    def test_xmonitor_wrapper_denies_unlisted_user_for_all_actions(self) -> None:
        """
        xmonitor 包装层应拒绝未列入白名单的用户且不执行底层操作。
        """
        tools = self._build_tools()
        xmonitor = tools["xmonitor"]
        cases: list[tuple[str, dict[str, object]]] = [
            ("start", {"username": "@kana_hanaiwa", "interval_seconds": 60}),
            ("stop", {"username": "@kana_hanaiwa"}),
            ("list", {}),
        ]

        for action, extra_payload in cases:
            with self.subTest(action=action):
                payload: dict[str, object] = {
                    "action": action,
                    "group_id": 123,
                    "user_id": 456,
                }
                payload.update(extra_payload)
                with mock.patch.dict(
                    "os.environ", {"CMD_ALLOWED_USERS": "999"}
                ), mock.patch.object(
                    agent_module, "start_x_monitor"
                ) as start, mock.patch.object(
                    agent_module, "stop_x_monitor"
                ) as stop, mock.patch.object(
                    agent_module, "list_x_monitor_tasks"
                ) as list_tasks:
                    output = xmonitor.invoke(payload)  # type: ignore[attr-defined]

                parsed = json.loads(output)
                self.assertEqual(parsed["action"], action)
                self.assertEqual(parsed["status"], "failed")
                self.assertEqual(parsed["error"], "permission_denied")
                start.assert_not_called()
                stop.assert_not_called()
                list_tasks.assert_not_called()

    def test_xlink_wrapper_returns_text_only(self) -> None:
        """
        xlink 包装层只应把推文正文返回给模型。
        """
        tools = self._build_tools()
        xlink = tools["xlink"]
        with mock.patch.object(agent_module, "send_x_link", return_value=_make_post()):
            output = xlink.invoke(  # type: ignore[attr-defined]
                {
                    "url": "https://x.com/kana_hanaiwa/status/1",
                    "group_id": 123,
                    "user_id": 456,
                }
            )

        self.assertEqual(output, "linked post")


if __name__ == "__main__":
    unittest.main()
