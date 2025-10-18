"""
Google Reverse Image Tool 模块，提供本地图像上传与搜索能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.google_reverse_image_client import GoogleReverseImageClient


@dataclass
class ReverseImageUploader:
    """
    将本地图片上传到临时公网存储的工具类。

    Args:
        endpoint (str): 上传服务地址，默认 ``https://0x0.st``。
        user_agent (str): 提供给上传服务的 User-Agent，默认 ``curl/8.1.2``。
        timeout_seconds (int): 上传请求超时时间（秒），默认 30。
    """

    endpoint: str = "https://0x0.st"
    user_agent: str = "curl/8.1.2"
    timeout_seconds: int = 30

    def upload(self, path: Path) -> str:
        """
        上传本地图片并返回可公开访问的 URL。

        Args:
            path (Path): 待上传的图片路径。

        Returns:
            str: 上传成功后返回的公网 URL。

        Raises:
            AssertionError: 当路径不存在或不是文件时抛出。
            ValueError: 当上传失败或返回内容异常时抛出。
        """

        assert path.is_file(), f"本地图片不存在：{path}"
        headers = {"User-Agent": self.user_agent}
        with path.open("rb") as file_handle:
            files = {"file": (path.name, file_handle.read())}
        try:
            response = requests.post(
                self.endpoint,
                files=files,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ValueError("上传图片到临时存储失败，请检查网络。") from exc

        if response.status_code >= 400:
            raise ValueError(f"上传服务返回异常状态码：HTTP {response.status_code}。")

        uploaded_url = response.text.strip()
        if not uploaded_url.startswith("http"):
            raise ValueError("上传服务返回内容异常，未提供可访问的 URL。")
        return uploaded_url


@dataclass
class GoogleReverseImageTool:
    """
    整合上传、路径解析与 SerpAPI 调用的反向搜图工具。

    Args:
        client (GoogleReverseImageClient): 已配置 API Key 的 SerpAPI 客户端。
        uploader (ReverseImageUploader): 本地图片上传器。
        locale (str): 请求使用的语言参数，默认 ``zh-CN``。
        max_results (int): ``image_results`` 保留的最大条目数，默认 7。
    """

    client: GoogleReverseImageClient
    uploader: ReverseImageUploader
    locale: str = "zh-CN"
    max_results: int = 7

    def run(self, image_url_or_name: str) -> dict[str, Any]:
        """
        根据图片 URL 或本地文件名执行反向搜图。

        Args:
            image_url_or_name (str): 图片的网络地址或本地文件名。

        Returns:
            dict[str, Any]: 清洗后的 SerpAPI 响应数据。

        Raises:
            AssertionError: 当输入参数为空时抛出。
            ValueError: 当上传或 SerpAPI 调用失败时抛出。
            FileNotFoundError: 当本地文件找不到时抛出。
        """

        assert (
            isinstance(image_url_or_name, str) and image_url_or_name.strip()
        ), "image_url_or_name 不能为空。"

        target_url = self._prepare_image_url(image_url_or_name.strip())
        payload = self.client.search(target_url, hl=self.locale)
        sanitized = self._sanitize_payload(payload)
        sanitized["source_image_url"] = target_url
        return sanitized

    def _prepare_image_url(self, image_identifier: str) -> str:
        """
        根据输入判断是否需要上传本地图片并返回网络地址。

        Args:
            image_identifier (str): 图片标识，可能是 URL 或文件名。

        Returns:
            str: 对应的网络可访问地址。
        """

        if image_identifier.lower().startswith(("http://", "https://")):
            return image_identifier

        path_candidate = Path(image_identifier).expanduser()
        if path_candidate.is_file():
            return self.uploader.upload(path_candidate)

        raise FileNotFoundError(f"本地图片不存在：{image_identifier}")

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        过滤 SerpAPI 返回字段并限制结果数量。

        Args:
            payload (dict[str, Any]): 原始的 SerpAPI 返回数据。

        Returns:
            dict[str, Any]: 已过滤的结果字典。
        """

        retained: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"search_metadata", "search_parameters"}:
                continue
            if key == "image_results" and isinstance(value, list):
                retained[key] = value[: self.max_results]
            else:
                retained[key] = value
        return retained
