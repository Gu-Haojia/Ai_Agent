"""
Serper 图片搜索 Agent 工具。

该模块固定从 Serper Images 获取候选图片，并通过五路并发完整
下载、MIME 校验和 Pillow 解码，仅向模型返回当前可用的原图
链接与实际尺寸。
下载内容只保存在验证线程的内存中，验证结束后立即释放。
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from typing import Literal
from urllib.parse import urljoin, urlparse

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


SERPER_IMAGES_ENDPOINT = "https://google.serper.dev/images"
SERPER_CANDIDATE_COUNT = 15
MAX_VALID_IMAGE_RESULTS = 10
IMAGE_VALIDATION_WORKERS = 5
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
CONNECT_TIMEOUT_SECONDS = 3
READ_TIMEOUT_SECONDS = 15
MAX_REDIRECTS = 5

SizeFilter = Literal["none", "medium", "large"]

SIZE_FILTER_TBS: dict[str, str | None] = {
    "none": None,
    "medium": "isz:m",
    "large": "isz:l",
}

REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class SerperImageSearchRequest(BaseModel):
    """
    表示模型可调用的 Serper 图片搜索参数。

    Args:
        query (str): 图片搜索关键词，最长 200 个字符。
        size_filter (SizeFilter): Google Images 尺寸分档。

    Returns:
        SerperImageSearchRequest: 完成校验和空白清理的请求参数。

    Raises:
        ValidationError: 当 query 为空、过长或 size_filter 非法时抛出。
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="图片搜索关键词。",
    )
    size_filter: SizeFilter = Field(
        "none",
        description=(
            "图片尺寸过滤，默认 none。用户未明确要求图片尺寸时，"
            "必须省略本参数或使用 none，不要主动选择 medium 或 large。"
            "仅当用户明确要求中等尺寸时使用 medium；明确要求大图、"
            "高清或高分辨率图片时使用 large。"
        ),
    )

    model_config = ConfigDict(str_strip_whitespace=True)


@dataclass(frozen=True)
class SerperImageCandidate:
    """
    保存一条 Serper 原始图片候选。

    Args:
        original_position (int): Serper 返回顺序中的位置。
        title (str): 图片结果标题。
        image_url (str): Serper 返回的原图链接。

    Returns:
        SerperImageCandidate: 不可变的候选图片数据。

    Raises:
        AssertionError: 当 original_position 非正整数时抛出。
    """

    original_position: int
    title: str
    image_url: str

    def __post_init__(self) -> None:
        """
        校验候选图片的内部排名。

        Returns:
            None: 本方法只执行数据前提校验。

        Raises:
            AssertionError: 当 original_position 非正整数时抛出。
        """

        assert self.original_position > 0, "original_position 必须为正整数"


@dataclass(frozen=True)
class ValidatedImage:
    """
    表示通过完整下载和解码验证的图片。

    Args:
        candidate (SerperImageCandidate): 对应的 Serper 候选。
        width (int): 实际解码宽度。
        height (int): 实际解码高度。

    Returns:
        ValidatedImage: 不可变的有效图片数据。

    Raises:
        AssertionError: 当图片尺寸不是正整数时抛出。
    """

    candidate: SerperImageCandidate
    width: int
    height: int

    def __post_init__(self) -> None:
        """
        校验实际图片尺寸。

        Returns:
            None: 本方法只执行数据前提校验。

        Raises:
            AssertionError: 当图片尺寸不是正整数时抛出。
        """

        assert self.width > 0, "width 必须为正整数"
        assert self.height > 0, "height 必须为正整数"

    def to_dict(self, position: int) -> dict[str, object]:
        """
        转换为模型可读取的精简图片结构。

        Args:
            position (int): 过滤后的连续排名。

        Returns:
            dict[str, object]: 包含标题、原图 URL 和实际尺寸的字典。

        Raises:
            AssertionError: 当 position 非正整数时抛出。
        """

        assert position > 0, "position 必须为正整数"
        return {
            "position": position,
            "title": self.candidate.title,
            "image_url": self.candidate.image_url,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class RejectedImage:
    """
    表示未通过本地验证的图片候选。

    Args:
        candidate (SerperImageCandidate): 被拒绝的候选。
        reason (str): 稳定的拒绝原因代码。
        http_status (int | None): 已收到的 HTTP 状态码。
        content_type (str): 已收到的 Content-Type。

    Returns:
        RejectedImage: 仅供本地日志使用的拒绝结果。

    Raises:
        AssertionError: 当 reason 为空时抛出。
    """

    candidate: SerperImageCandidate
    reason: str
    http_status: int | None = None
    content_type: str = ""

    def __post_init__(self) -> None:
        """
        校验拒绝原因不为空。

        Returns:
            None: 本方法只执行数据前提校验。

        Raises:
            AssertionError: 当 reason 为空时抛出。
        """

        assert self.reason.strip(), "reason 不能为空"


class SerperImageSearchError(RuntimeError):
    """
    表示可转换为 Agent 结构化失败结果的 Serper 外部错误。

    Args:
        error_code (str): 稳定错误码。
        message (str): 面向模型的简短错误说明。
        http_status (int | None): Serper 返回的 HTTP 状态码。

    Returns:
        SerperImageSearchError: 带错误码和状态码的异常对象。

    Raises:
        AssertionError: 当 error_code 或 message 为空时抛出。
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        http_status: int | None = None,
    ) -> None:
        """
        初始化 Serper 外部错误。

        Args:
            error_code (str): 稳定错误码。
            message (str): 面向模型的简短错误说明。
            http_status (int | None): Serper 返回的 HTTP 状态码。

        Returns:
            None: 本方法仅初始化异常。

        Raises:
            AssertionError: 当 error_code 或 message 为空时抛出。
        """

        assert error_code.strip(), "error_code 不能为空"
        assert message.strip(), "message 不能为空"
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status


class SerperImageSearchClient:
    """
    调用 Serper Images 并转换原始候选。

    Args:
        api_key (str): Serper API Key。
        candidate_count (int): 单次请求的候选数量。
        gl (str): Google 搜索国家代码。
        hl (str): Google 搜索语言代码。
        timeout_seconds (int): Serper 请求读取超时秒数。

    Returns:
        SerperImageSearchClient: 可执行图片搜索的客户端。

    Raises:
        AssertionError: 当初始化参数非法时抛出。
    """

    def __init__(
        self,
        api_key: str,
        candidate_count: int = SERPER_CANDIDATE_COUNT,
        gl: str = "jp",
        hl: str = "ja",
        timeout_seconds: int = READ_TIMEOUT_SECONDS,
    ) -> None:
        """
        初始化 Serper Images 客户端。

        Args:
            api_key (str): Serper API Key。
            candidate_count (int): 单次请求的候选数量。
            gl (str): Google 搜索国家代码。
            hl (str): Google 搜索语言代码。
            timeout_seconds (int): Serper 请求读取超时秒数。

        Returns:
            None: 本方法仅保存客户端配置。

        Raises:
            AssertionError: 当初始化参数非法时抛出。
        """

        assert api_key.strip(), "api_key 不能为空"
        assert candidate_count > 0, "candidate_count 必须为正整数"
        assert gl.strip(), "gl 不能为空"
        assert hl.strip(), "hl 不能为空"
        assert timeout_seconds > 0, "timeout_seconds 必须为正整数"
        self._api_key = api_key.strip()
        self._candidate_count = candidate_count
        self._gl = gl.strip()
        self._hl = hl.strip()
        self._timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        size_filter: SizeFilter,
    ) -> list[SerperImageCandidate]:
        """
        调用 Serper Images 并返回固定数量的原始候选。

        Args:
            query (str): 图片搜索关键词。
            size_filter (SizeFilter): none、medium 或 large。

        Returns:
            list[SerperImageCandidate]: 按 Serper 排序的图片候选。

        Raises:
            AssertionError: 当 query 或 size_filter 非法时抛出。
            SerperImageSearchError:
                当请求、认证、限流或响应解析失败时抛出。
        """

        normalized_query = query.strip()
        assert normalized_query, "query 不能为空"
        assert size_filter in SIZE_FILTER_TBS, "size_filter 非法"

        request_payload: dict[str, object] = {
            "q": normalized_query,
            "gl": self._gl,
            "hl": self._hl,
            "num": self._candidate_count,
        }
        tbs = SIZE_FILTER_TBS[size_filter]
        if tbs is not None:
            request_payload["tbs"] = tbs

        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }
        try:
            with requests.post(
                SERPER_IMAGES_ENDPOINT,
                headers=headers,
                json=request_payload,
                timeout=(CONNECT_TIMEOUT_SECONDS, self._timeout_seconds),
            ) as response:
                self._raise_for_http_error(response.status_code)
                try:
                    payload = response.json()
                except requests.exceptions.JSONDecodeError as exc:
                    raise SerperImageSearchError(
                        "invalid_response",
                        "Serper API 返回了无法解析的 JSON。",
                        response.status_code,
                    ) from exc
        except requests.Timeout as exc:
            raise SerperImageSearchError(
                "request_timeout",
                "Serper API 请求超时。",
            ) from exc
        except requests.RequestException as exc:
            raise SerperImageSearchError(
                "serper_http_error",
                "Serper API 请求失败，请稍后重试。",
            ) from exc

        return self._parse_candidates(payload)

    @staticmethod
    def _raise_for_http_error(status_code: int) -> None:
        """
        将 Serper HTTP 状态转换为稳定外部错误。

        Args:
            status_code (int): Serper 返回的 HTTP 状态码。

        Returns:
            None: 状态码正常时不返回内容。

        Raises:
            SerperImageSearchError: 当状态码不是 2xx 时抛出。
        """

        if 200 <= status_code < 300:
            return
        if status_code in {401, 403}:
            raise SerperImageSearchError(
                "authentication_failed",
                "Serper API 认证失败，请检查 SERPER_API_KEY。",
                status_code,
            )
        if status_code == 429:
            raise SerperImageSearchError(
                "rate_limited",
                "Serper API 请求频率超过限制。",
                status_code,
            )
        raise SerperImageSearchError(
            "serper_http_error",
            f"Serper API 返回异常状态：HTTP {status_code}。",
            status_code,
        )

    def _parse_candidates(
        self,
        payload: object,
    ) -> list[SerperImageCandidate]:
        """
        将 Serper JSON 转换为内部候选列表。

        Args:
            payload (object): Serper 解码后的 JSON 数据。

        Returns:
            list[SerperImageCandidate]: 最多 candidate_count 条候选。

        Raises:
            SerperImageSearchError:
                当响应顶层或 images 字段类型非法时抛出。
        """

        if not isinstance(payload, dict):
            raise SerperImageSearchError(
                "invalid_response",
                "Serper API 返回结构不是对象。",
            )
        raw_images = payload.get("images", [])
        if not isinstance(raw_images, list):
            raise SerperImageSearchError(
                "invalid_response",
                "Serper API 返回的 images 字段不是列表。",
            )

        candidates: list[SerperImageCandidate] = []
        for position, item in enumerate(
            raw_images[: self._candidate_count],
            start=1,
        ):
            if not isinstance(item, dict):
                raise SerperImageSearchError(
                    "invalid_response",
                    "Serper API 返回了非法的图片结果项。",
                )
            raw_title = item.get("title")
            if raw_title is None:
                title = ""
            elif isinstance(raw_title, str):
                title = raw_title.strip()
            else:
                raise SerperImageSearchError(
                    "invalid_response",
                    "Serper API 图片标题类型非法。",
                )
            raw_url = item.get("imageUrl")
            image_url = raw_url.strip() if isinstance(raw_url, str) else ""
            candidates.append(
                SerperImageCandidate(
                    original_position=position,
                    title=title,
                    image_url=image_url,
                )
            )
        return candidates


class ImageUrlValidator:
    """
    完整下载并解码验证 Serper 原图链接。

    Args:
        max_image_bytes (int): 单张图片允许的最大下载字节数。
        max_image_pixels (int): 单张图片允许的最大总像素数。
        connect_timeout_seconds (int): HTTP 连接超时秒数。
        read_timeout_seconds (int): HTTP 读取超时秒数。
        max_redirects (int): 最大手动重定向次数。

    Returns:
        ImageUrlValidator: 不使用磁盘缓存的图片验证器。

    Raises:
        AssertionError: 当初始化参数非法时抛出。
    """

    def __init__(
        self,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        max_image_pixels: int = MAX_IMAGE_PIXELS,
        connect_timeout_seconds: int = CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds: int = READ_TIMEOUT_SECONDS,
        max_redirects: int = MAX_REDIRECTS,
    ) -> None:
        """
        初始化图片验证限制。

        Args:
            max_image_bytes (int): 单张图片允许的最大下载字节数。
            max_image_pixels (int): 单张图片允许的最大总像素数。
            connect_timeout_seconds (int): HTTP 连接超时秒数。
            read_timeout_seconds (int): HTTP 读取超时秒数。
            max_redirects (int): 最大手动重定向次数。

        Returns:
            None: 本方法仅保存验证配置。

        Raises:
            AssertionError: 当初始化参数非法时抛出。
        """

        assert max_image_bytes > 0, "max_image_bytes 必须为正整数"
        assert max_image_pixels > 0, "max_image_pixels 必须为正整数"
        assert connect_timeout_seconds > 0, (
            "connect_timeout_seconds 必须为正整数"
        )
        assert read_timeout_seconds > 0, "read_timeout_seconds 必须为正整数"
        assert max_redirects >= 0, "max_redirects 不能为负数"
        self._max_image_bytes = max_image_bytes
        self._max_image_pixels = max_image_pixels
        self._connect_timeout_seconds = connect_timeout_seconds
        self._read_timeout_seconds = read_timeout_seconds
        self._max_redirects = max_redirects

    def validate(
        self,
        candidate: SerperImageCandidate,
    ) -> ValidatedImage | RejectedImage:
        """
        完整下载一张候选图片并验证实际可解码性。

        Args:
            candidate (SerperImageCandidate): 待验证图片候选。

        Returns:
            ValidatedImage | RejectedImage: 有效图片或本地拒绝结果。

        Raises:
            AssertionError: 当 candidate 类型非法时抛出。
        """

        assert isinstance(candidate, SerperImageCandidate), (
            "candidate 必须为 SerperImageCandidate"
        )
        if not self._is_public_http_url(candidate.image_url):
            return RejectedImage(candidate=candidate, reason="invalid_url")

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": (
                "image/avif,image/webp,image/apng,image/jpeg,image/png,"
                "image/gif,image/*,*/*;q=0.8"
            ),
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
        }
        current_url = candidate.image_url
        try:
            with requests.Session() as session:
                for redirect_count in range(self._max_redirects + 1):
                    if not self._is_public_http_url(current_url):
                        return RejectedImage(
                            candidate=candidate,
                            reason="unsafe_address",
                        )
                    with session.get(
                        current_url,
                        headers=headers,
                        stream=True,
                        allow_redirects=False,
                        timeout=(
                            self._connect_timeout_seconds,
                            self._read_timeout_seconds,
                        ),
                    ) as response:
                        if response.status_code in REDIRECT_STATUS_CODES:
                            if redirect_count >= self._max_redirects:
                                return RejectedImage(
                                    candidate=candidate,
                                    reason="too_many_redirects",
                                    http_status=response.status_code,
                                )
                            location = response.headers.get("Location", "").strip()
                            if not location:
                                return RejectedImage(
                                    candidate=candidate,
                                    reason="invalid_redirect",
                                    http_status=response.status_code,
                                )
                            current_url = urljoin(current_url, location)
                            continue
                        return self._validate_response(candidate, response)
        except requests.Timeout:
            return RejectedImage(
                candidate=candidate,
                reason="download_timeout",
            )
        except requests.RequestException:
            return RejectedImage(
                candidate=candidate,
                reason="request_failed",
            )

        raise AssertionError("图片重定向处理未返回验证结果")

    def _validate_response(
        self,
        candidate: SerperImageCandidate,
        response: requests.Response,
    ) -> ValidatedImage | RejectedImage:
        """
        验证最终 HTTP 响应并完整解码图片。

        Args:
            candidate (SerperImageCandidate): 当前候选图片。
            response (requests.Response): 已关闭自动重定向的最终响应。

        Returns:
            ValidatedImage | RejectedImage: 有效图片或拒绝结果。

        Raises:
            AssertionError: 当 response 类型非法时抛出。
        """

        assert isinstance(response, requests.Response), (
            "response 必须为 requests.Response"
        )
        content_type = response.headers.get("Content-Type", "")
        normalized_content_type = content_type.split(";", 1)[0].strip().lower()
        if not 200 <= response.status_code < 300:
            return RejectedImage(
                candidate=candidate,
                reason="http_error",
                http_status=response.status_code,
                content_type=normalized_content_type,
            )
        if not normalized_content_type.startswith("image/"):
            return RejectedImage(
                candidate=candidate,
                reason="non_image_content",
                http_status=response.status_code,
                content_type=normalized_content_type,
            )

        content = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > self._max_image_bytes:
                    return RejectedImage(
                        candidate=candidate,
                        reason="image_too_large",
                        http_status=response.status_code,
                        content_type=normalized_content_type,
                    )

            if not content:
                return RejectedImage(
                    candidate=candidate,
                    reason="empty_image",
                    http_status=response.status_code,
                    content_type=normalized_content_type,
                )

            try:
                with BytesIO(content) as buffer:
                    with Image.open(buffer) as image:
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            return RejectedImage(
                                candidate=candidate,
                                reason="invalid_dimensions",
                                http_status=response.status_code,
                                content_type=normalized_content_type,
                            )
                        if width * height > self._max_image_pixels:
                            return RejectedImage(
                                candidate=candidate,
                                reason="image_too_many_pixels",
                                http_status=response.status_code,
                                content_type=normalized_content_type,
                            )
                        image.load()
            except (
                Image.DecompressionBombError,
                UnidentifiedImageError,
                OSError,
            ):
                return RejectedImage(
                    candidate=candidate,
                    reason="decode_failed",
                    http_status=response.status_code,
                    content_type=normalized_content_type,
                )

            return ValidatedImage(
                candidate=candidate,
                width=width,
                height=height,
            )
        finally:
            content.clear()

    @staticmethod
    def _is_public_http_url(url: str) -> bool:
        """
        判断 URL 是否指向可解析的公网 HTTP 地址。

        Args:
            url (str): 待检查 URL。

        Returns:
            bool: True 表示 URL 仅解析到公网地址。

        Raises:
            None: DNS 或 URL 解析失败时返回 False。
        """

        if not isinstance(url, str) or not url.strip():
            return False
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme not in {"http", "https"}:
                return False
            if not parsed.hostname or parsed.username or parsed.password:
                return False
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addresses = socket.getaddrinfo(
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except (ValueError, socket.gaierror, UnicodeError):
            return False
        if not addresses:
            return False
        for address in addresses:
            raw_ip = address[4][0].split("%", 1)[0]
            try:
                parsed_ip = ipaddress.ip_address(raw_ip)
            except ValueError:
                return False
            if not parsed_ip.is_global:
                return False
        return True


class SerperImageSearchTool(BaseTool):
    """
    向模型提供经过完整下载验证的 Serper 图片搜索结果。

    Args:
        api_key (str): Serper API Key。
        candidate_count (int): Serper 候选数量，默认 15。
        max_results (int): 最多返回的有效图片数量，默认 10。
        validation_workers (int): 并发图片验证线程数，默认 5。

    Returns:
        SerperImageSearchTool: 可注册到 LangGraph ToolNode 的工具。

    Raises:
        AssertionError: 当初始化参数非法时抛出。
    """

    name: str = "serper_image_search"
    description: str = (
        "使用 Google Images 搜索图片，返回经过本地完整下载验证的"
        "原图 URL 和实际分辨率。仅在用户需要图片、照片或视觉"
        "素材时调用。除非用户明确提出图片尺寸、高清或大图要求，"
        "否则不要主动设置尺寸过滤。"
    )
    args_schema: type[BaseModel] = SerperImageSearchRequest

    _client: SerperImageSearchClient = PrivateAttr()
    _validator: ImageUrlValidator = PrivateAttr()
    _max_results: int = PrivateAttr()
    _validation_workers: int = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        candidate_count: int = SERPER_CANDIDATE_COUNT,
        max_results: int = MAX_VALID_IMAGE_RESULTS,
        validation_workers: int = IMAGE_VALIDATION_WORKERS,
    ) -> None:
        """
        初始化 Serper 客户端和本地图片验证器。

        Args:
            api_key (str): Serper API Key。
            candidate_count (int): Serper 候选数量。
            max_results (int): 最多返回的有效图片数量。
            validation_workers (int): 并发图片验证线程数。

        Returns:
            None: 本方法仅初始化工具依赖。

        Raises:
            AssertionError: 当初始化参数非法时抛出。
        """

        assert api_key.strip(), "api_key 不能为空"
        assert candidate_count > 0, "candidate_count 必须为正整数"
        assert max_results > 0, "max_results 必须为正整数"
        assert max_results <= candidate_count, (
            "max_results 不能大于 candidate_count"
        )
        assert validation_workers > 0, "validation_workers 必须为正整数"
        super().__init__()
        self._client = SerperImageSearchClient(
            api_key=api_key,
            candidate_count=candidate_count,
        )
        self._validator = ImageUrlValidator()
        self._max_results = max_results
        self._validation_workers = validation_workers

    def _run(
        self,
        query: str,
        size_filter: SizeFilter = "none",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """
        同步搜索并返回最多十张有效原图。

        Args:
            query (str): 图片搜索关键词。
            size_filter (SizeFilter): none、medium 或 large。
            run_manager (CallbackManagerForToolRun | None): LangChain 回调管理器。

        Returns:
            str: 包含 status、query、count 和 images 的 JSON 字符串。

        Raises:
            AssertionError: 当 query 或 size_filter 非法时抛出。
        """

        del run_manager
        normalized_query = query.strip()
        assert normalized_query, "query 不能为空"
        assert size_filter in SIZE_FILTER_TBS, "size_filter 非法"

        try:
            candidates = self._client.search(normalized_query, size_filter)
        except SerperImageSearchError as exc:
            return self._build_failure(normalized_query, exc)

        unique_candidates = self._deduplicate_candidates(candidates)
        with ThreadPoolExecutor(
            max_workers=self._validation_workers,
        ) as executor:
            validation_results = list(
                executor.map(self._validator.validate, unique_candidates)
            )

        valid_images: list[ValidatedImage] = []
        for result in validation_results:
            if isinstance(result, ValidatedImage):
                valid_images.append(result)
            else:
                self._log_rejection(result)

        selected_images = valid_images[: self._max_results]
        if not selected_images:
            result_payload: dict[str, object] = {
                "status": "not_found",
                "query": normalized_query,
                "count": 0,
                "images": [],
                "message": (
                    "没有找到可用图片，请调整关键词或尺寸条件。"
                ),
            }
        else:
            result_payload = {
                "status": "success",
                "query": normalized_query,
                "count": len(selected_images),
                "images": [
                    image.to_dict(position)
                    for position, image in enumerate(selected_images, start=1)
                ],
            }
        return json.dumps(result_payload, ensure_ascii=False)

    async def _arun(
        self,
        query: str,
        size_filter: SizeFilter = "none",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """
        在线程中执行同步图片搜索，避免阻塞异步调用链。

        Args:
            query (str): 图片搜索关键词。
            size_filter (SizeFilter): none、medium 或 large。
            run_manager (AsyncCallbackManagerForToolRun | None):
                异步回调管理器。

        Returns:
            str: 与同步调用一致的 JSON 字符串。

        Raises:
            AssertionError: 当 query 或 size_filter 非法时抛出。
        """

        del run_manager
        return await asyncio.to_thread(
            self._run,
            query=query,
            size_filter=size_filter,
            run_manager=None,
        )

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[SerperImageCandidate],
    ) -> list[SerperImageCandidate]:
        """
        按原图 URL 保序去重。

        Args:
            candidates (list[SerperImageCandidate]): Serper 原始候选列表。

        Returns:
            list[SerperImageCandidate]: 首次出现顺序不变的唯一候选。

        Raises:
            AssertionError: 当候选项类型非法时抛出。
        """

        seen_urls: set[str] = set()
        unique_candidates: list[SerperImageCandidate] = []
        for candidate in candidates:
            assert isinstance(candidate, SerperImageCandidate), (
                "candidates 必须只包含 SerperImageCandidate"
            )
            normalized_url = candidate.image_url.strip()
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _log_rejection(rejection: RejectedImage) -> None:
        """
        将拒绝原因写入本地日志，不暴露给模型。

        Args:
            rejection (RejectedImage): 图片验证拒绝结果。

        Returns:
            None: 本方法仅输出本地日志。

        Raises:
            AssertionError: 当 rejection 类型非法时抛出。
        """

        assert isinstance(rejection, RejectedImage), (
            "rejection 必须为 RejectedImage"
        )
        domain = urlparse(rejection.candidate.image_url).hostname or "unknown"
        print(
            "[SerperImageValidation] "
            f"position={rejection.candidate.original_position} "
            f"domain={domain} reason={rejection.reason} "
            f"status={rejection.http_status} "
            f"content_type={rejection.content_type or 'unknown'}",
            flush=True,
        )

    @staticmethod
    def _build_failure(
        query: str,
        error: SerperImageSearchError,
    ) -> str:
        """
        将预期的 Serper 外部错误转换为结构化失败 JSON。

        Args:
            query (str): 当前图片搜索关键词。
            error (SerperImageSearchError): 带稳定错误码的外部异常。

        Returns:
            str: 便于 Agent 继续生成回复的失败 JSON 字符串。

        Raises:
            AssertionError: 当 query 为空或 error 类型非法时抛出。
        """

        assert query.strip(), "query 不能为空"
        assert isinstance(error, SerperImageSearchError), (
            "error 必须为 SerperImageSearchError"
        )
        payload: dict[str, object] = {
            "status": "failed",
            "query": query,
            "count": 0,
            "images": [],
            "error": error.error_code,
            "message": str(error),
        }
        if error.http_status is not None:
            payload["http_status"] = error.http_status
        return json.dumps(payload, ensure_ascii=False)
