"""
Tavily 图片描述路由工具单元测试。
"""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest import mock

from langchain_tavily import TavilySearch

from src.tavily_search_tool import RoutedTavilySearch


class RoutedTavilySearchTests(unittest.TestCase):
    """验证路由工具保持原生接口并选择正确实例。"""

    def setUp(self) -> None:
        """
        创建无需真实网络请求的路由工具和原生工具。

        Returns:
            None: 本方法仅初始化测试对象。

        Raises:
            None: 本方法不主动抛出异常。
        """
        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
            self.tool = RoutedTavilySearch(max_results=3)
            self.native_tool = TavilySearch(max_results=3)

    def test_exposes_native_tool_contract(self) -> None:
        """模型可见的工具定义应与原生 TavilySearch 一致。"""
        self.assertEqual(self.tool.name, self.native_tool.name)
        self.assertEqual(self.tool.description, self.native_tool.description)
        self.assertIs(self.tool.args_schema, self.native_tool.args_schema)
        self.assertEqual(
            self.tool.tool_call_schema.model_json_schema(),
            self.native_tool.tool_call_schema.model_json_schema(),
        )

    def test_routes_regular_search_to_text_instance(self) -> None:
        """未请求图片时应调用未启用图片描述的实例。"""
        expected = {"query": "佳村遥", "images": [], "results": []}

        with mock.patch.object(
            TavilySearch,
            "invoke",
            autospec=True,
            return_value=expected,
        ) as invoke:
            actual = self.tool.invoke({"query": "佳村遥"})

        self.assertIs(actual, expected)
        selected_tool = invoke.call_args.args[0]
        forwarded_arguments = invoke.call_args.args[1]
        self.assertIsNone(selected_tool.include_image_descriptions)
        self.assertEqual(forwarded_arguments["query"], "佳村遥")
        self.assertFalse(forwarded_arguments["include_images"])

    def test_routes_image_search_to_description_instance(self) -> None:
        """请求图片时应调用启用了图片描述的实例。"""
        expected = {
            "query": "佳村遥 写真",
            "images": [{"url": "https://example.com/image.jpg", "description": ""}],
            "results": [],
        }

        with mock.patch.object(
            TavilySearch,
            "invoke",
            autospec=True,
            return_value=expected,
        ) as invoke:
            actual = self.tool.invoke(
                {"query": "佳村遥 写真", "include_images": True}
            )

        self.assertIs(actual, expected)
        selected_tool = invoke.call_args.args[0]
        forwarded_arguments = invoke.call_args.args[1]
        self.assertTrue(selected_tool.include_image_descriptions)
        self.assertEqual(forwarded_arguments["query"], "佳村遥 写真")
        self.assertTrue(forwarded_arguments["include_images"])

    def test_routes_async_image_search_to_description_instance(self) -> None:
        """异步图片请求也应调用启用了图片描述的实例。"""
        expected = {
            "query": "佳村遥 写真",
            "images": [{"url": "https://example.com/image.jpg", "description": ""}],
            "results": [],
        }

        with mock.patch.object(
            TavilySearch,
            "ainvoke",
            autospec=True,
        ) as ainvoke:
            ainvoke.return_value = expected
            actual = asyncio.run(
                self.tool.ainvoke(
                    {"query": "佳村遥 写真", "include_images": True}
                )
            )

        self.assertIs(actual, expected)
        selected_tool = ainvoke.call_args.args[0]
        forwarded_arguments = ainvoke.call_args.args[1]
        self.assertTrue(selected_tool.include_image_descriptions)
        self.assertEqual(forwarded_arguments["query"], "佳村遥 写真")
        self.assertTrue(forwarded_arguments["include_images"])


if __name__ == "__main__":
    unittest.main()
