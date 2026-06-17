"""
Playwright Browser Toolkit 注册测试。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import sql_agent_cli_stream_plus as agent_module


class _FakeBindableLLM:
    """
    测试用的可绑定工具 LLM。
    """

    def __init__(self) -> None:
        """
        初始化测试用 LLM。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.bound_tools: list[tuple[list[object], object]] = []

    def bind_tools(
        self, tools: list[object], tool_choice: object = None
    ) -> "_FakeBindableLLM":
        """
        记录传入的工具列表并返回自身。

        Args:
            tools (list[object]): 被绑定到 LLM 的工具列表。
            tool_choice (object): LangChain 工具选择参数。

        Returns:
            _FakeBindableLLM: 当前测试 LLM 实例。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.bound_tools.append((list(tools), tool_choice))
        return self


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

    def test_build_graph_keeps_legacy_web_browser_disabled(self) -> None:
        """
        启用工具时应跳过旧 web_browser 总结工具并保留 Playwright 工具。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当旧工具未被屏蔽时由断言抛出。
        """

        @agent_module.tool("fake_playwright_browser")
        def fake_playwright_browser() -> str:
            """
            返回测试用 Playwright 工具结果。

            Returns:
                str: 固定的测试字符串。

            Raises:
                None: 本函数不主动抛出异常。
            """
            return "ok"

        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        agent._config = agent_module.AgentConfig(
            model_name="openai:test", use_memory_ckpt=True
        )
        agent._enable_tools = True
        agent._sys_msg_content = "test"
        agent._reminder_scheduler = mock.Mock()
        agent._asobi_query = mock.Mock()
        agent._image_manager = None
        agent._generated_images = []
        agent._memory_namespace = ""
        fake_llm = _FakeBindableLLM()

        with mock.patch.dict(
            "os.environ",
            {
                "SUMMARY_MODEL": "",
                "TAVILY_API_KEY": "",
                "SERPAPI_API_KEY": "",
                "VISUAL_CROSSING_API_KEY": "",
            },
        ), mock.patch.object(
            agent_module,
            "init_chat_model",
            return_value=fake_llm,
        ) as init_model, mock.patch.object(
            agent_module.SQLCheckpointAgentStreamingPlus,
            "_build_playwright_browser_tools",
            return_value=[fake_playwright_browser],
        ), mock.patch.object(
            agent_module,
            "WebBrowserTool",
        ) as web_browser_tool, mock.patch.object(
            agent_module,
            "InMemoryStore",
            return_value=mock.Mock(),
        ):
            graph = agent._build_graph()

        self.assertIsNotNone(graph)
        init_model.assert_called_once_with("openai:test")
        web_browser_tool.assert_not_called()
        self.assertTrue(fake_llm.bound_tools)
        bound_tool_names = {str(tool.name) for tool in fake_llm.bound_tools[0][0]}
        self.assertIn("fake_playwright_browser", bound_tool_names)
        self.assertNotIn("web_browser", bound_tool_names)


if __name__ == "__main__":
    unittest.main()
