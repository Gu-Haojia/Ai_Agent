"""
Meru 商品搜索工具。

将现有 Mercari 搜索能力封装为 LangChain 工具，供 Agent 直接调用。
"""

from __future__ import annotations

import requests
from langchain_core.tools import BaseTool, tool

from src.meru_monitor import DEFAULT_LIMIT, MeruMonitorManager


def build_meru_search_tool(
    monitor: MeruMonitorManager | None = None,
) -> BaseTool:
    """
    创建 Meru 商品搜索工具。

    Args:
        monitor (MeruMonitorManager | None): 可选的 Meru 搜索管理器，主要用于
            复用实例或测试注入。

    Returns:
        BaseTool: 可注册到 LangGraph ToolNode 的搜索工具。

    Raises:
        AssertionError: 当 monitor 不是 MeruMonitorManager 实例时抛出。
    """
    assert monitor is None or isinstance(
        monitor, MeruMonitorManager
    ), "monitor 类型无效"
    search_monitor = monitor or MeruMonitorManager()

    @tool("meru_search")
    def meru_search(keyword: str, limit: int = DEFAULT_LIMIT) -> str:
        """
        搜索日本 Mercari 当前在售商品。

        Args:
            keyword (str): 商品搜索关键词，支持日语和英语；搜索日本 Mercari
                商品时优先使用日语关键词。
            limit (int): 最大返回数量。

        Returns:
            str: 搜索结果或错误文本。

        Raises:
            Exception: 底层发生非网络类的未预期错误时原样抛出。
        """
        if not isinstance(keyword, str) or not keyword.strip():
            return "Mercari 搜索失败：关键词不能为空。"
        if not isinstance(limit, int) or limit <= 0:
            return "Mercari 搜索失败：limit 必须为正整数。"

        normalized_keyword = keyword.strip()
        try:
            results = search_monitor.search(normalized_keyword, limit)
        except requests.RequestException as exc:
            return f"Mercari 搜索失败：{exc}"

        if not results:
            return f"未找到与「{normalized_keyword}」相关的在售商品。"
        return search_monitor.format_lines(results, "SEARCH")

    return meru_search


meru_search: BaseTool = build_meru_search_tool()

__all__ = ["build_meru_search_tool", "meru_search"]
