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
class ReverseImageLocalResolver:
    """
    根据文件名在本地图片目录中定位真实路径的解析器。

    Args:
        image_root (Path): 图像根目录，默认 ``images``。
        categories (tuple[str, ...]): 支持的子目录类别，默认 ``(\"incoming\", \"generated\")``。
    """

    image_root: Path = Path("images")
    categories: tuple[str, ...] = ("incoming", "generated")

    def resolve(self, file_name: str) -> Path:
        """
        查找文件名对应的本地图片路径。

        Args:
            file_name (str): 图片文件名，仅包含文件名部分。

        Returns:
            Path: 匹配到的图片绝对路径。

        Raises:
            AssertionError: 当 ``file_name`` 为空字符串时抛出。
            FileNotFoundError: 当未找到对应文件时抛出。
            ValueError: 当找到多个匹配文件时抛出。
        """

        assert isinstance(file_name, str) and file_name.strip(), "file_name 不能为空。"

        normalized_name = Path(file_name).name
        candidates: list[Path] = []
        if not self.image_root.exists():
            raise FileNotFoundError(f"本地图像根目录不存在：{self.image_root}")

        run_directories = sorted(
            (
                path
                for path in self.image_root.iterdir()
                if path.is_dir() and path.name.endswith("~run")
            ),
            reverse=True,
        )

        for run_dir in run_directories:
            for category in self.categories:
                candidate = run_dir / category / normalized_name
                if candidate.is_file():
                    candidates.append(candidate)

        if not candidates:
            raise FileNotFoundError(f"未找到匹配的本地图片：{normalized_name}")
        if len(candidates) > 1:
            locations = ", ".join(str(path) for path in candidates)
            raise ValueError(f"找到多个同名图片，请确认文件名：{locations}")
        return candidates[0].resolve()


@dataclass
class GoogleReverseImageTool:
    """
    整合上传、路径解析与 SerpAPI 调用的反向搜图工具。

    Args:
        client (GoogleReverseImageClient): 已配置 API Key 的 SerpAPI 客户端。
        uploader (ReverseImageUploader): 本地图片上传器。
        resolver (ReverseImageLocalResolver): 本地路径解析器。
        locale (str): 请求使用的语言参数，默认 ``zh-CN``。
        max_results (int): ``image_results`` 保留的最大条目数，默认 7。
    """

    client: GoogleReverseImageClient
    uploader: ReverseImageUploader
    resolver: ReverseImageLocalResolver
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

        local_path = self.resolver.resolve(image_identifier)
        return self.uploader.upload(local_path)

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
