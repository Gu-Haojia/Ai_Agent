"""
按图片搜索参数路由并提供单轮超限提醒的 Tavily 搜索工具。

该模块保持原生 TavilySearch 的模型可见接口不变，仅在请求图片时
使用启用了图片描述的 TavilySearch 实例。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import PrivateAttr


TAVILY_SEARCH_TOOL_NAME: str = "tavily_search"
TAVILY_SEARCH_LIMIT_NOTICE: str = (
    "本轮 Tavily 调用次数已经超过规定上限。"
    "请尽量不要再次调用 Tavily，优先基于已有搜索结果完成回答；"
    "只有在现有信息无法回答且确实需要新的独立事实时才继续搜索。"
)


class TavilySearchReminder:
    """在单轮首次超限的 Tavily 调用中返回提醒，后续调用正常执行。

    Args:
        search_limit_getter (Callable[[], int]): 获取当前搜索提醒阈值的函数。

    Returns:
        None: 初始化不返回额外值。

    Raises:
        None: 初始化不主动抛出异常。
    """

    def __init__(self, search_limit_getter: Callable[[], int]) -> None:
        """保存读取实时阈值的函数，不维护跨会话共享计数。

        Args:
            search_limit_getter (Callable[[], int]): 当前阈值的读取函数。

        Returns:
            None: 初始化不返回额外值。

        Raises:
            None: 初始化不主动抛出异常。
        """
        self._search_limit_getter = search_limit_getter

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """首次超限时同步返回提醒，其余调用交给原工具执行。

        Args:
            request (ToolCallRequest): 当前工具调用及其会话状态。
            execute (Callable): 原工具的同步执行函数。

        Returns:
            ToolMessage | Command: 提醒消息或原工具返回值。

        Raises:
            AssertionError: 当阈值或 Tavily 调用状态非法时抛出。
        """
        notice = self._build_notice(request)
        if notice is not None:
            return notice
        return execute(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """首次超限时异步返回提醒，其余调用交给原工具执行。

        Args:
            request (ToolCallRequest): 当前工具调用及其会话状态。
            execute (Callable): 原工具的异步执行函数。

        Returns:
            ToolMessage | Command: 提醒消息或原工具返回值。

        Raises:
            AssertionError: 当阈值或 Tavily 调用状态非法时抛出。
        """
        notice = self._build_notice(request)
        if notice is not None:
            return notice
        return await execute(request)

    def _build_notice(self, request: ToolCallRequest) -> ToolMessage | None:
        """根据本轮调用顺序构造首次超限的工具提醒。

        Args:
            request (ToolCallRequest): 当前工具调用及其会话状态。

        Returns:
            ToolMessage | None: 首次超限时返回提醒，其他调用返回 None。

        Raises:
            AssertionError: 当阈值、状态或工具调用标识非法时抛出。
        """
        if request.tool_call["name"] != TAVILY_SEARCH_TOOL_NAME:
            return None
        search_limit = self._search_limit_getter()
        assert type(search_limit) is int and search_limit > 0, (
            "搜索提醒阈值必须为正整数"
        )
        assert isinstance(request.state, dict), "工具调用状态必须为字典"
        call_number = self._current_call_number(
            request.state["messages"], request.tool_call["id"]
        )
        if call_number != search_limit + 1:
            return None
        return ToolMessage(
            content=TAVILY_SEARCH_LIMIT_NOTICE,
            name=TAVILY_SEARCH_TOOL_NAME,
            tool_call_id=request.tool_call["id"],
        )

    @staticmethod
    def _current_call_number(
        messages: Sequence[BaseMessage], tool_call_id: str
    ) -> int:
        """按消息中声明的顺序计算当前 Tavily 调用在本轮的序号。

        Args:
            messages (Sequence[BaseMessage]): 当前会话的消息序列。
            tool_call_id (str): 当前 Tavily 调用的标识。

        Returns:
            int: 从最近一条用户消息开始计算的调用序号。

        Raises:
            AssertionError: 当消息不是序列、缺少用户消息或调用时抛出。
        """
        assert isinstance(messages, Sequence), "messages 必须是序列"
        last_human_index = -1
        for index, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                last_human_index = index
        assert last_human_index >= 0, "Tavily 调用必须包含当前轮用户消息"
        call_number = 0
        for message in messages[last_human_index + 1 :]:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                if tool_call["name"] != TAVILY_SEARCH_TOOL_NAME:
                    continue
                call_number += 1
                if tool_call["id"] == tool_call_id:
                    return call_number
        raise AssertionError("当前 Tavily 调用未出现在本轮消息中")


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
