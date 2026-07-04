"""
按图片搜索参数路由的 Tavily 搜索工具。

该模块保持原生 TavilySearch 的模型可见接口不变，仅在请求图片时
使用启用了图片描述的 TavilySearch 实例。
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from pydantic import PrivateAttr


class RoutedTavilySearch(BaseTool):
    """
    保持原生接口并按 include_images 路由的 Tavily 搜索工具。

    模型看到的名称、说明和参数 schema 均直接复用原生 TavilySearch。
    当 include_images 为 true 时，内部选择启用了图片描述的实例；
    其他请求选择普通实例。
    """

    _text_search: TavilySearch = PrivateAttr()
    _image_search: TavilySearch = PrivateAttr()

    def __init__(self, max_results: int = 3) -> None:
        """
        初始化文本搜索和图片描述搜索实例。

        Args:
            max_results (int): Tavily 返回的最大网页搜索结果数。

        Returns:
            None: 本方法仅初始化工具实例。

        Raises:
            AssertionError: 当 max_results 不是正整数时抛出。
            ValueError: 当 Tavily API Key 未配置时由原生工具抛出。
        """
        assert isinstance(max_results, int) and max_results > 0, (
            "max_results 必须为正整数"
        )
        text_search = TavilySearch(max_results=max_results)
        image_search = TavilySearch(
            max_results=max_results,
            include_image_descriptions=True,
        )
        super().__init__(
            name=text_search.name,
            description=text_search.description,
            args_schema=text_search.args_schema,
            return_direct=text_search.return_direct,
            response_format=text_search.response_format,
            handle_tool_error=text_search.handle_tool_error,
            handle_validation_error=text_search.handle_validation_error,
        )
        self._text_search = text_search
        self._image_search = image_search

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **arguments: Any,
    ) -> Any:
        """
        根据 include_images 同步调用对应的原生 Tavily 工具。

        Args:
            run_manager (CallbackManagerForToolRun | None): LangChain 回调管理器。
            **arguments (Any): 原生 TavilySearch 接收的调用参数。

        Returns:
            Any: 原生 TavilySearch 未经转换的返回值。

        Raises:
            Exception: 原生 TavilySearch 调用抛出的异常。
        """
        selected_tool = self._select_tool(arguments)
        return selected_tool.invoke(arguments)

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **arguments: Any,
    ) -> Any:
        """
        根据 include_images 异步调用对应的原生 Tavily 工具。

        Args:
            run_manager (AsyncCallbackManagerForToolRun | None):
                LangChain 回调管理器。
            **arguments (Any): 原生 TavilySearch 接收的调用参数。

        Returns:
            Any: 原生 TavilySearch 未经转换的返回值。

        Raises:
            Exception: 原生 TavilySearch 调用抛出的异常。
        """
        selected_tool = self._select_tool(arguments)
        return await selected_tool.ainvoke(arguments)

    def _select_tool(self, arguments: dict[str, Any]) -> TavilySearch:
        """
        按原生 include_images 参数选择内部工具。

        Args:
            arguments (dict[str, Any]): 已通过原生 schema 校验的调用参数。

        Returns:
            TavilySearch: 应执行当前请求的原生 TavilySearch 实例。

        Raises:
            AssertionError: 当 arguments 不是字典时抛出。
        """
        assert isinstance(arguments, dict), "arguments 必须为字典"
        if arguments.get("include_images") is True:
            return self._image_search
        return self._text_search
