"""
Playwright Browser Toolkit 的固定线程执行器。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from langchain_community.agent_toolkits.playwright.toolkit import (
    PlayWrightBrowserToolkit,
)
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_core.tools import BaseTool
from playwright.sync_api import Browser as SyncBrowser
from pydantic import BaseModel, PrivateAttr


PLAYWRIGHT_BROWSER_TOOL_NAMES: set[str] = {
    "click_element",
    "navigate_browser",
    "previous_webpage",
    "extract_text",
    "extract_hyperlinks",
    "get_elements",
    "current_webpage",
}


@dataclass(frozen=True)
class PlaywrightBrowserToolSpec:
    """
    保存 Playwright 工具暴露给 Agent 的公开定义。

    Args:
        name (str): 工具名称。
        description (str): 工具说明。
        args_schema (type[BaseModel]): 工具参数 schema。

    Returns:
        None: 数据类构造函数无返回值。

    Raises:
        AssertionError: 当字段为空时由调用方校验抛出。
    """

    name: str
    description: str
    args_schema: type[BaseModel]


class ThreadBoundPlaywrightTool(BaseTool):
    """
    将 LangChain 工具调用代理到固定 Playwright 线程。
    """

    _runner: PlaywrightBrowserThreadRunner = PrivateAttr()
    _tool_name: str = PrivateAttr()

    def __init__(
        self,
        runner: PlaywrightBrowserThreadRunner,
        spec: PlaywrightBrowserToolSpec,
    ) -> None:
        """
        初始化代理工具。

        Args:
            runner (PlaywrightBrowserThreadRunner): 固定线程执行器。
            spec (PlaywrightBrowserToolSpec): 原始工具公开定义。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当工具定义非法时抛出。
        """
        assert isinstance(runner, PlaywrightBrowserThreadRunner), "runner 类型无效"
        assert isinstance(spec, PlaywrightBrowserToolSpec), "spec 类型无效"
        assert spec.name.strip(), "工具名称不能为空"
        assert spec.description.strip(), "工具说明不能为空"
        super().__init__(
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_schema,
        )
        self._runner = runner
        self._tool_name = spec.name

    def _run(self, **kwargs: Any) -> str:
        """
        在固定 Playwright 线程中执行原始工具。

        Args:
            **kwargs (Any): 已通过 LangChain schema 校验的工具参数。

        Returns:
            str: 原始 Playwright 工具返回文本。

        Raises:
            AssertionError: 当执行器已关闭或返回值类型非法时抛出。
        """
        return self._runner.invoke(self._tool_name, dict(kwargs))


class PlaywrightBrowserThreadRunner:
    """
    在单一专用线程中创建并调用 Playwright 同步浏览器。
    """

    def __init__(self) -> None:
        """
        初始化固定线程执行器。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="playwright-browser",
        )
        self._browser: SyncBrowser | None = None
        self._tools: dict[str, BaseTool] | None = None
        self._tool_specs: list[PlaywrightBrowserToolSpec] | None = None
        self._closed: bool = False

    def build_tools(self) -> list[BaseTool]:
        """
        构造可注册到 Agent 的固定线程代理工具。

        Returns:
            list[BaseTool]: 与 Playwright Browser Toolkit 同名同 schema 的代理工具。

        Raises:
            AssertionError: 当执行器已关闭或工具集合不完整时抛出。
        """
        assert not self._closed, "Playwright browser runner 已关闭"
        specs = self._executor.submit(self._build_tool_specs_in_worker).result()
        return [ThreadBoundPlaywrightTool(self, spec) for spec in specs]

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        在线程绑定的 Playwright 环境中调用指定工具。

        Args:
            tool_name (str): Playwright 工具名称。
            arguments (dict[str, Any]): 工具参数。

        Returns:
            str: 工具执行结果。

        Raises:
            AssertionError: 当执行器已关闭、工具名非法或返回值非字符串时抛出。
        """
        assert not self._closed, "Playwright browser runner 已关闭"
        assert isinstance(tool_name, str) and tool_name.strip(), "tool_name 不能为空"
        assert isinstance(arguments, dict), "arguments 必须为 dict"
        output = self._executor.submit(
            self._invoke_in_worker,
            tool_name,
            dict(arguments),
        ).result()
        assert isinstance(output, str), "Playwright 工具返回值必须为字符串"
        return output

    def close(self) -> None:
        """
        关闭 Playwright 浏览器和固定线程。

        Returns:
            None: 函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        if self._closed:
            return
        self._executor.submit(self._close_in_worker).result()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def _build_tool_specs_in_worker(self) -> list[PlaywrightBrowserToolSpec]:
        """
        在固定线程中初始化原始 Toolkit 并导出工具定义。

        Returns:
            list[PlaywrightBrowserToolSpec]: 工具公开定义列表。

        Raises:
            AssertionError: 当 Toolkit 返回的工具集合不符合预期时抛出。
        """
        self._ensure_tools_in_worker()
        assert self._tool_specs is not None, "Playwright 工具定义未初始化"
        return list(self._tool_specs)

    def _invoke_in_worker(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        在固定线程中调用原始 Playwright 工具。

        Args:
            tool_name (str): Playwright 工具名称。
            arguments (dict[str, Any]): 工具参数。

        Returns:
            str: 原始工具执行结果。

        Raises:
            AssertionError: 当工具未初始化、工具名未知或返回值非法时抛出。
        """
        self._ensure_tools_in_worker()
        assert self._tools is not None, "Playwright 工具未初始化"
        assert tool_name in self._tools, f"未知 Playwright 工具: {tool_name}"
        output = self._tools[tool_name].invoke(arguments)
        assert isinstance(output, str), "Playwright 工具返回值必须为字符串"
        return output

    def _ensure_tools_in_worker(self) -> None:
        """
        确保浏览器与原始 Playwright 工具已在固定线程中初始化。

        Returns:
            None: 函数无返回值。

        Raises:
            AssertionError: 当 Toolkit 返回工具集合不完整时抛出。
        """
        if self._tools is not None:
            return

        self._browser = create_sync_playwright_browser(headless=True)
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=self._browser)
        playwright_tools = toolkit.get_tools()
        tool_names = {str(playwright_tool.name) for playwright_tool in playwright_tools}
        assert tool_names == PLAYWRIGHT_BROWSER_TOOL_NAMES, (
            "Playwright Browser Toolkit 工具集合不符合预期: "
            f"{sorted(tool_names)}"
        )
        self._tools = {
            str(playwright_tool.name): playwright_tool
            for playwright_tool in playwright_tools
        }
        self._tool_specs = [
            PlaywrightBrowserToolSpec(
                name=str(playwright_tool.name),
                description=str(playwright_tool.description),
                args_schema=playwright_tool.args_schema,
            )
            for playwright_tool in playwright_tools
        ]

    def _close_in_worker(self) -> None:
        """
        在固定线程中关闭 Playwright 浏览器。

        Returns:
            None: 函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        self._tools = None
        self._tool_specs = None
