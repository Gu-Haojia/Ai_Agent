"""
Playwright Browser Toolkit 注册测试。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import sql_agent_cli_stream_plus as agent_module


class PlaywrightBrowserToolkitRegistrationTests(unittest.TestCase):
    """
    验证 Agent 对 Playwright Browser Toolkit 的注册与资源释放。
    """

    def test_build_playwright_browser_tools_uses_sync_browser(self) -> None:
        """
        Playwright Browser Toolkit 应使用同步浏览器并返回完整工具集合。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当工具注册行为不符合预期时由断言抛出。
        """
        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        agent._playwright_browser = None
        fake_browser = mock.Mock()
        fake_tools = [
            SimpleNamespace(name=name)
            for name in sorted(agent_module.PLAYWRIGHT_BROWSER_TOOL_NAMES)
        ]
        fake_toolkit = mock.Mock()
        fake_toolkit.get_tools.return_value = fake_tools

        with mock.patch.object(
            agent_module,
            "create_sync_playwright_browser",
            return_value=fake_browser,
        ) as create_browser, mock.patch.object(
            agent_module.PlayWrightBrowserToolkit,
            "from_browser",
            return_value=fake_toolkit,
        ) as from_browser:
            tools = agent._build_playwright_browser_tools()

        self.assertEqual(tools, fake_tools)
        self.assertEqual(agent._playwright_browser, fake_browser)
        create_browser.assert_called_once_with(headless=True)
        from_browser.assert_called_once_with(sync_browser=fake_browser)
        fake_toolkit.get_tools.assert_called_once_with()

    def test_build_playwright_browser_tools_rejects_missing_tools(self) -> None:
        """
        Toolkit 返回的工具名不完整时应显式失败。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当工具名集合不完整时由被测方法抛出。
        """
        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        agent._playwright_browser = None
        fake_toolkit = mock.Mock()
        fake_toolkit.get_tools.return_value = [SimpleNamespace(name="navigate_browser")]

        with mock.patch.object(
            agent_module,
            "create_sync_playwright_browser",
            return_value=mock.Mock(),
        ), mock.patch.object(
            agent_module.PlayWrightBrowserToolkit,
            "from_browser",
            return_value=fake_toolkit,
        ):
            with self.assertRaises(AssertionError):
                agent._build_playwright_browser_tools()

    def test_shutdown_closes_playwright_browser(self) -> None:
        """
        Agent 关闭时应释放 Playwright 浏览器实例。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当浏览器关闭行为不符合预期时由断言抛出。
        """
        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        fake_browser = mock.Mock()
        agent._playwright_browser = fake_browser
        agent._reminder_scheduler = object()

        agent.shutdown()

        fake_browser.close.assert_called_once_with()
        self.assertIsNone(agent._playwright_browser)


if __name__ == "__main__":
    unittest.main()
