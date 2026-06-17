"""
Playwright Browser Toolkit 注册测试。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import unittest
from types import SimpleNamespace

import sql_agent_cli_stream_plus as agent_module
from pydantic import BaseModel
from src.playwright_browser_toolkit_runner import (
    PLAYWRIGHT_BROWSER_TOOL_NAMES,
    PlaywrightBrowserToolSpec,
    PlaywrightBrowserThreadRunner,
    ThreadBoundPlaywrightTool,
)


class _FakePlaywrightRunner(PlaywrightBrowserThreadRunner):
    """
    测试用 Playwright runner，避免启动真实浏览器。
    """

    def __init__(self, tools: list[object]) -> None:
        """
        初始化测试 runner。

        Args:
            tools (list[object]): 模拟返回的工具列表。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.tools = tools
        self.closed = False

    def build_tools(self) -> list[object]:
        """
        返回预置的测试工具列表。

        Returns:
            list[object]: 模拟工具列表。

        Raises:
            None: 本方法不主动抛出异常。
        """
        return self.tools

    def close(self) -> None:
        """
        标记测试 runner 已关闭。

        Returns:
            None: 函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.closed = True


class _RaisingPlaywrightRunner(PlaywrightBrowserThreadRunner):
    """
    测试用 Playwright runner，用于模拟工具执行异常。
    """

    def __init__(self) -> None:
        """
        初始化测试 runner。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        pass

    def invoke(self, tool_name: str, arguments: dict[str, object]) -> str:
        """
        模拟固定线程工具调用失败。

        Args:
            tool_name (str): 被调用的工具名。
            arguments (dict[str, object]): 工具参数。

        Returns:
            str: 本测试方法不会返回。

        Raises:
            RuntimeError: 始终抛出，用于验证代理工具错误返回。
        """
        raise RuntimeError(f"{tool_name} failed with {arguments}")


class _EmptyToolArgs(BaseModel):
    """
    测试用空参数 schema。
    """


class PlaywrightBrowserToolkitRegistrationTests(unittest.TestCase):
    """
    验证 Agent 对 Playwright Browser Toolkit 的注册与资源释放。
    """

    def test_build_playwright_browser_tools_uses_runner(self) -> None:
        """
        Agent 应通过固定线程 runner 构造 Playwright 工具。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当工具注册行为不符合预期时由断言抛出。
        """
        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        fake_tools = [
            SimpleNamespace(name=name)
            for name in sorted(PLAYWRIGHT_BROWSER_TOOL_NAMES)
        ]
        agent._playwright_runner = _FakePlaywrightRunner(fake_tools)

        tools = agent._build_playwright_browser_tools()

        self.assertEqual(tools, fake_tools)

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
        agent._playwright_runner = _FakePlaywrightRunner(
            [SimpleNamespace(name="navigate_browser")]
        )

        with self.assertRaises(AssertionError):
            agent._build_playwright_browser_tools()

    def test_shutdown_closes_playwright_runner(self) -> None:
        """
        Agent 关闭时应释放 Playwright runner。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当浏览器关闭行为不符合预期时由断言抛出。
        """
        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        runner = _FakePlaywrightRunner([])
        agent._playwright_runner = runner
        agent._reminder_scheduler = object()

        agent.shutdown()

        self.assertTrue(runner.closed)

    def test_proxy_tool_returns_error_json_on_exception(self) -> None:
        """
        代理工具内部异常应作为工具返回值暴露给 LLM。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当异常继续冒泡或返回值格式不符合预期时抛出。
        """
        spec = PlaywrightBrowserToolSpec(
            name="current_webpage",
            description="Returns the URL of the current page",
            args_schema=_EmptyToolArgs,
        )
        tool = ThreadBoundPlaywrightTool(_RaisingPlaywrightRunner(), spec)

        output = tool.invoke({})
        payload = json.loads(output)

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["tool_name"], "current_webpage")
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertIn("current_webpage failed", payload["error"])

    def test_playwright_tool_runs_from_worker_thread(self) -> None:
        """
        Playwright 工具被 LangGraph 线程池调用时不应触发 greenlet 跨线程错误。

        Returns:
            None: 测试用例无返回值。

        Raises:
            AssertionError: 当工具无法在线程池中执行时由断言抛出。
        """
        agent = agent_module.SQLCheckpointAgentStreamingPlus.__new__(
            agent_module.SQLCheckpointAgentStreamingPlus
        )
        agent._playwright_runner = PlaywrightBrowserThreadRunner()
        agent._reminder_scheduler = object()

        try:
            tools = {
                str(playwright_tool.name): playwright_tool
                for playwright_tool in agent._build_playwright_browser_tools()
            }
            current_webpage = tools["current_webpage"]

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(current_webpage.invoke, {})
                output = future.result(timeout=10)

            self.assertIn("about:blank", output)
        finally:
            agent.shutdown()


if __name__ == "__main__":
    unittest.main()
