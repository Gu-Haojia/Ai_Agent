"""Serper 图片搜索工具单元测试。"""

from __future__ import annotations

import asyncio
import json
import unittest
from io import BytesIO
from types import TracebackType
from unittest import mock

import requests
from PIL import Image

from src.serper_image_search_tool import (
    ImageUrlValidator,
    RejectedImage,
    SerperImageCandidate,
    SerperImageSearchClient,
    SerperImageSearchError,
    SerperImageSearchRequest,
    SerperImageSearchTool,
    ValidatedImage,
)


class FakeSession:
    """
    为图片验证测试提供固定 HTTP 响应。

    Args:
        responses (list[requests.Response]): 按调用顺序返回的响应。

    Returns:
        FakeSession: 支持上下文管理器的测试会话。

    Raises:
        AssertionError: 当请求次数超过预设响应数量时抛出。
    """

    def __init__(self, responses: list[requests.Response]) -> None:
        """
        保存测试响应队列。

        Args:
            responses (list[requests.Response]): 按调用顺序返回的响应。

        Returns:
            None: 本方法仅保存测试数据。

        Raises:
            AssertionError: 当 responses 为空时抛出。
        """

        assert responses, "responses 不能为空"
        self._responses = list(responses)
        self.requested_urls: list[str] = []

    def __enter__(self) -> "FakeSession":
        """
        进入测试会话上下文。

        Returns:
            FakeSession: 当前测试会话。

        Raises:
            None: 本方法不主动抛出异常。
        """

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """
        退出测试会话上下文。

        Args:
            exc_type (type[BaseException] | None): 异常类型。
            exc_value (BaseException | None): 异常实例。
            traceback (TracebackType | None): 异常调用栈。

        Returns:
            bool: False 表示不吞掉异常。

        Raises:
            None: 本方法不主动抛出异常。
        """

        del exc_type, exc_value, traceback
        return False

    def get(self, url: str, **kwargs: object) -> requests.Response:
        """
        返回下一条测试响应。

        Args:
            url (str): 本次请求 URL。
            **kwargs (object): requests.Session.get 的其余参数。

        Returns:
            requests.Response: 预设的下一条响应。

        Raises:
            AssertionError: 当没有剩余响应时抛出。
        """

        del kwargs
        assert self._responses, "测试响应已耗尽"
        self.requested_urls.append(url)
        return self._responses.pop(0)


def build_response(
    body: bytes,
    content_type: str,
    status_code: int = 200,
    url: str = "https://example.com/image.png",
) -> requests.Response:
    """
    构造可由 requests 流式读取的测试响应。

    Args:
        body (bytes): 响应正文。
        content_type (str): Content-Type 响应头。
        status_code (int): HTTP 状态码。
        url (str): 响应 URL。

    Returns:
        requests.Response: 已消费正文的测试响应。

    Raises:
        AssertionError: 当 content_type 为空时抛出。
    """

    assert content_type.strip(), "content_type 不能为空"
    response = requests.Response()
    response.status_code = status_code
    response.url = url
    response.headers["Content-Type"] = content_type
    response._content = body
    response._content_consumed = True
    return response


def build_png(width: int, height: int) -> bytes:
    """
    创建指定尺寸的内存 PNG 图片。

    Args:
        width (int): 图片宽度。
        height (int): 图片高度。

    Returns:
        bytes: PNG 图片字节。

    Raises:
        AssertionError: 当尺寸不是正整数时抛出。
    """

    assert width > 0, "width 必须为正整数"
    assert height > 0, "height 必须为正整数"
    with BytesIO() as buffer:
        Image.new("RGB", (width, height), color="blue").save(buffer, format="PNG")
        return buffer.getvalue()


class SerperImageSearchRequestTests(unittest.TestCase):
    """验证模型可见的 Serper 图片搜索参数。"""

    def test_schema_only_exposes_query_and_size_filter(self) -> None:
        """工具 schema 应仅暴露 query 与 size_filter。"""

        schema = SerperImageSearchRequest.model_json_schema()

        self.assertEqual(set(schema["properties"]), {"query", "size_filter"})
        self.assertEqual(
            schema["properties"]["size_filter"]["enum"],
            ["none", "medium", "large"],
        )
        self.assertEqual(
            schema["properties"]["size_filter"]["default"],
            "none",
        )

    def test_query_is_stripped_and_validated(self) -> None:
        """query 应去除首尾空白且拒绝纯空白。"""

        request = SerperImageSearchRequest(query="  橘ありす  ")

        self.assertEqual(request.query, "橘ありす")
        with self.assertRaises(ValueError):
            SerperImageSearchRequest(query="   ")


class SerperImageSearchClientTests(unittest.TestCase):
    """验证 Serper 请求参数和候选解析。"""

    def setUp(self) -> None:
        """
        创建固定请求 15 条候选的客户端。

        Returns:
            None: 本方法仅初始化测试对象。

        Raises:
            None: 本方法不主动抛出异常。
        """

        self.client = SerperImageSearchClient(api_key="test-key")

    def test_maps_size_filter_and_requests_fifteen_candidates(self) -> None:
        """尺寸枚举应映射为 Serper tbs，且 num 固定为 15。"""

        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "images": [
                {
                    "title": "图片",
                    "imageUrl": "https://example.com/image.jpg",
                }
            ]
        }
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with mock.patch(
            "src.serper_image_search_tool.requests.post",
            return_value=response,
        ) as post:
            candidates = self.client.search("梅澤めぐ", "medium")

        request_json = post.call_args.kwargs["json"]
        self.assertEqual(request_json["num"], 15)
        self.assertEqual(request_json["gl"], "jp")
        self.assertEqual(request_json["hl"], "ja")
        self.assertEqual(request_json["tbs"], "isz:m")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].image_url, "https://example.com/image.jpg")

    def test_none_size_filter_omits_tbs(self) -> None:
        """none 尺寸过滤不应向 Serper 发送 tbs。"""

        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {"images": []}
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with mock.patch(
            "src.serper_image_search_tool.requests.post",
            return_value=response,
        ) as post:
            self.client.search("橘ありす", "none")

        self.assertNotIn("tbs", post.call_args.kwargs["json"])

    def test_converts_rate_limit_to_stable_error(self) -> None:
        """HTTP 429 应转换为 rate_limited 外部错误。"""

        response = mock.MagicMock()
        response.status_code = 429
        response.__enter__.return_value = response
        response.__exit__.return_value = False

        with mock.patch(
            "src.serper_image_search_tool.requests.post",
            return_value=response,
        ):
            with self.assertRaises(SerperImageSearchError) as context:
                self.client.search("橘ありす", "large")

        self.assertEqual(context.exception.error_code, "rate_limited")
        self.assertEqual(context.exception.http_status, 429)


class ImageUrlValidatorTests(unittest.TestCase):
    """验证完整下载、MIME 校验和实际尺寸读取。"""

    def setUp(self) -> None:
        """
        创建图片验证器和固定候选。

        Returns:
            None: 本方法仅初始化测试对象。

        Raises:
            None: 本方法不主动抛出异常。
        """

        self.validator = ImageUrlValidator()
        self.candidate = SerperImageCandidate(
            original_position=1,
            title="测试图片",
            image_url="https://example.com/image.png",
        )

    def test_downloads_and_decodes_actual_dimensions(self) -> None:
        """有效图片应完整解码并使用实际宽高。"""

        response = build_response(build_png(37, 23), "image/png")
        session = FakeSession([response])

        with mock.patch.object(
            ImageUrlValidator,
            "_is_public_http_url",
            return_value=True,
        ), mock.patch(
            "src.serper_image_search_tool.requests.Session",
            return_value=session,
        ):
            result = self.validator.validate(self.candidate)

        self.assertIsInstance(result, ValidatedImage)
        assert isinstance(result, ValidatedImage)
        self.assertEqual((result.width, result.height), (37, 23))

    def test_rejects_http_200_html_body_without_decoding(self) -> None:
        """HTTP 200 但 Content-Type 为 HTML 时应拒绝。"""

        response = build_response(b"<html></html>", "text/html; charset=utf-8")
        session = FakeSession([response])

        with mock.patch.object(
            ImageUrlValidator,
            "_is_public_http_url",
            return_value=True,
        ), mock.patch(
            "src.serper_image_search_tool.requests.Session",
            return_value=session,
        ):
            result = self.validator.validate(self.candidate)

        self.assertIsInstance(result, RejectedImage)
        assert isinstance(result, RejectedImage)
        self.assertEqual(result.reason, "non_image_content")
        self.assertEqual(result.http_status, 200)

    def test_rejects_undecodable_image_content(self) -> None:
        """伪造为 image MIME 的损坏正文应拒绝。"""

        response = build_response(b"not-a-real-image", "image/jpeg")
        session = FakeSession([response])

        with mock.patch.object(
            ImageUrlValidator,
            "_is_public_http_url",
            return_value=True,
        ), mock.patch(
            "src.serper_image_search_tool.requests.Session",
            return_value=session,
        ):
            result = self.validator.validate(self.candidate)

        self.assertIsInstance(result, RejectedImage)
        assert isinstance(result, RejectedImage)
        self.assertEqual(result.reason, "decode_failed")


class SerperImageSearchToolTests(unittest.TestCase):
    """验证工具编排、返回截断和 Reject 隐藏。"""

    def setUp(self) -> None:
        """
        创建使用测试 Key 的 Serper 图片工具。

        Returns:
            None: 本方法仅初始化测试对象。

        Raises:
            None: 本方法不主动抛出异常。
        """

        self.tool = SerperImageSearchTool(api_key="test-key")
        self.candidates = [
            SerperImageCandidate(
                original_position=position,
                title=f"图片 {position}",
                image_url=f"https://example.com/{position}.jpg",
            )
            for position in range(1, 16)
        ]

    def test_validates_all_candidates_and_returns_at_most_ten(self) -> None:
        """15 条候选应全部验证，最终最多返回 10 条。"""

        self.tool._client = mock.Mock(spec=SerperImageSearchClient)
        self.tool._client.search.return_value = self.candidates
        self.tool._validator = mock.Mock(spec=ImageUrlValidator)

        def validate(
            candidate: SerperImageCandidate,
        ) -> ValidatedImage | RejectedImage:
            """
            按固定位置生成有效或拒绝结果。

            Args:
                candidate (SerperImageCandidate): 当前候选。

            Returns:
                ValidatedImage | RejectedImage: 测试验证结果。

            Raises:
                None: 本函数不主动抛出异常。
            """

            if candidate.original_position in {3, 11}:
                return RejectedImage(candidate, "non_image_content", 200, "text/html")
            return ValidatedImage(candidate, 800, 1200)

        self.tool._validator.validate.side_effect = validate

        with mock.patch.object(self.tool, "_log_rejection") as log_rejection:
            output = self.tool.invoke(
                {"query": "梅澤めぐ ライブ", "size_filter": "none"}
            )

        payload = json.loads(output)
        self.assertEqual(self.tool._validator.validate.call_count, 15)
        self.assertEqual(log_rejection.call_count, 2)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["count"], 10)
        self.assertEqual(
            [item["position"] for item in payload["images"]],
            list(range(1, 11)),
        )
        self.assertNotIn("provider", payload)
        self.assertNotIn("filters", payload)
        self.assertNotIn("rejected", payload)
        self.assertNotIn("rejected_count", payload)

    def test_returns_not_found_when_every_candidate_is_rejected(self) -> None:
        """没有有效图片时应返回 not_found 且隐藏 Reject。"""

        self.tool._client = mock.Mock(spec=SerperImageSearchClient)
        self.tool._client.search.return_value = self.candidates[:2]
        self.tool._validator = mock.Mock(spec=ImageUrlValidator)
        self.tool._validator.validate.side_effect = [
            RejectedImage(self.candidates[0], "http_error", 404),
            RejectedImage(self.candidates[1], "decode_failed", 200, "image/jpeg"),
        ]

        with mock.patch.object(self.tool, "_log_rejection"):
            payload = json.loads(self.tool.invoke({"query": "不存在的图片"}))

        self.assertEqual(payload["status"], "not_found")
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["images"], [])
        self.assertNotIn("rejected", payload)

    def test_returns_structured_serper_failure(self) -> None:
        """预期的 Serper 外部错误应转换为短结构化失败。"""

        self.tool._client = mock.Mock(spec=SerperImageSearchClient)
        self.tool._client.search.side_effect = SerperImageSearchError(
            "authentication_failed",
            "Serper API 认证失败。",
            401,
        )

        payload = json.loads(self.tool.invoke({"query": "橘ありす"}))

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "authentication_failed")
        self.assertEqual(payload["http_status"], 401)
        self.assertEqual(payload["images"], [])

    def test_async_call_returns_same_shape(self) -> None:
        """异步调用应复用同步搜索并返回相同结构。"""

        self.tool._client = mock.Mock(spec=SerperImageSearchClient)
        self.tool._client.search.return_value = self.candidates[:1]
        self.tool._validator = mock.Mock(spec=ImageUrlValidator)
        self.tool._validator.validate.return_value = ValidatedImage(
            self.candidates[0],
            640,
            960,
        )

        output = asyncio.run(self.tool.ainvoke({"query": "橘ありす"}))
        payload = json.loads(output)

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["images"][0]["width"], 640)
        self.assertEqual(payload["images"][0]["height"], 960)


if __name__ == "__main__":
    unittest.main()
