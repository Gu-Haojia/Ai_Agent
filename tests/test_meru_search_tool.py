"""Meru 商品搜索工具测试。"""

from unittest.mock import Mock

import requests

from src.meru_monitor import MeruMonitorManager, MeruSearchResult
from src.meru_search_tool import build_meru_search_tool


def _build_monitor() -> MeruMonitorManager:
    """
    创建可注入模拟行为的 Meru 管理器。

    Returns:
        MeruMonitorManager: 独立的 Meru 管理器实例。
    """
    return MeruMonitorManager()


class TestMeruSearchTool:
    """验证 Meru 搜索工具的参数说明、结果与错误文本。"""

    def test_description_prioritizes_japanese_keywords(self) -> None:
        """工具说明应告知模型支持日英语且优先使用日语。"""
        tool_instance = build_meru_search_tool(_build_monitor())

        assert "支持日语和英语" in tool_instance.description
        assert "优先使用日语关键词" in tool_instance.description

    def test_returns_formatted_search_results(self) -> None:
        """搜索成功时应返回现有 Meru 文本格式。"""
        monitor = _build_monitor()
        item = MeruSearchResult(
            keyword="最上静香",
            item_id="m123",
            name="アクリルスタンド",
            price=1500,
            created_label="08-14 12:00",
            url="https://jp.mercari.com/item/m123",
        )
        monitor.search = Mock(return_value=[item])  # type: ignore[method-assign]
        tool_instance = build_meru_search_tool(monitor)

        output = tool_instance.invoke({"keyword": " 最上静香 ", "limit": 3})

        monitor.search.assert_called_once_with("最上静香", 3)  # type: ignore[attr-defined]
        assert "アクリルスタンド" in output
        assert "¥1500" in output

    def test_returns_text_for_request_exception(self) -> None:
        """网络请求异常应直接转换为简短错误文本。"""
        monitor = _build_monitor()
        monitor.search = Mock(  # type: ignore[method-assign]
            side_effect=requests.ConnectionError("连接中断")
        )
        tool_instance = build_meru_search_tool(monitor)

        output = tool_instance.invoke({"keyword": "最上静香", "limit": 5})

        assert output == "Mercari 搜索失败：连接中断"

    def test_returns_text_for_invalid_keyword(self) -> None:
        """空关键词应直接返回参数错误文本且不执行搜索。"""
        monitor = _build_monitor()
        monitor.search = Mock()  # type: ignore[method-assign]
        tool_instance = build_meru_search_tool(monitor)

        output = tool_instance.invoke({"keyword": "   ", "limit": 5})

        assert output == "Mercari 搜索失败：关键词不能为空。"
        monitor.search.assert_not_called()  # type: ignore[attr-defined]
