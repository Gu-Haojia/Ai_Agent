"""
Google Lens 工具模块，提供本地与在线图片的识别与匹配查询能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.google_reverse_image_tool import ReverseImageUploader


@dataclass(frozen=True)
class GoogleLensClient:
    """
    调用 SerpAPI Google Lens 接口的轻量客户端。

    Args:
        api_key (str): SerpAPI API Key。
        endpoint (str): SerpAPI 接口地址，默认 ``https://serpapi.com/search``。
        timeout_seconds (int): 请求超时时间（秒），默认 30。
    """

    api_key: str
    endpoint: str = "https://serpapi.com/search"
    timeout_seconds: int = 30

    def search(self, image_url: str) -> dict[str, Any]:
        """
        使用指定图片 URL 调用 Google Lens。

        Args:
            image_url (str): 待识别图片的公网 URL。

        Returns:
            dict[str, Any]: SerpAPI 返回的 JSON 数据。

        Raises:
            AssertionError: 当 ``image_url`` 为空字符串时抛出。
            ValueError: 当网络请求失败、鉴权失败或 SerpAPI 返回错误时抛出。
        """

        assert isinstance(image_url, str) and image_url.strip(), "image_url 必须为非空字符串。"

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": self.api_key,
        }

        try:
            response = requests.get(
                self.endpoint,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ValueError("请求 SerpAPI 失败，请检查网络连接。") from exc

        if response.status_code == 401:
            raise ValueError("SerpAPI 鉴权失败：请确认 SERPAPI_API_KEY 是否正确。")
        if response.status_code == 403:
            raise ValueError("SerpAPI 拒绝访问：请检查订阅状态或配额限制。")
        if response.status_code >= 400:
            raise ValueError(f"SerpAPI 调用失败，HTTP {response.status_code}。")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("SerpAPI 返回非 JSON 数据。") from exc

        if not isinstance(payload, dict):
            raise ValueError("SerpAPI 返回数据格式异常。")
        if payload.get("error"):
            raise ValueError(f"SerpAPI 返回错误：{payload['error']}")

        return payload


@dataclass
class GoogleLensTool:
    """
    整合上传、路径解析与 SerpAPI 调用的 Google Lens 工具。

    Args:
        client (GoogleLensClient): 已配置 API Key 的 SerpAPI 客户端。
        uploader (ReverseImageUploader): 本地图片上传器。
    """

    client: GoogleLensClient
    uploader: ReverseImageUploader

    def run(self, image_url_or_name: str) -> dict[str, Any]:
        """
        根据图片 URL 或本地文件名执行 Google Lens 搜索。

        Args:
            image_url_or_name (str): 图片的网络地址或本地文件名 / 路径。

        Returns:
            dict[str, Any]: 清洗后的查询结果数据；视觉匹配仅保留 title/source/link/position，
            关联内容仅保留 link/query。

        Raises:
            AssertionError: 当输入参数为空时抛出。
            ValueError: 当上传或 SerpAPI 调用失败时抛出。
            FileNotFoundError: 当本地文件找不到时抛出。
        """

        assert (
            isinstance(image_url_or_name, str) and image_url_or_name.strip()
        ), "image_url_or_name 不能为空。"

        target_url = self._prepare_image_url(image_url_or_name.strip())
        raw_payload = self.client.search(target_url)
        sanitized = self._sanitize_payload(raw_payload)
        sanitized["source_image_url"] = target_url
        return sanitized

    def _prepare_image_url(self, image_identifier: str) -> str:
        """
        根据输入判断是否需要上传本地图片并返回网络地址。

        Args:
            image_identifier (str): 图片标识，可能是 URL 或本地文件路径。

        Returns:
            str: 对应的可访问网络地址。
        """

        lowered = image_identifier.lower()
        if lowered.startswith(("http://", "https://")):
            return image_identifier

        path_candidate = Path(image_identifier).expanduser()
        if path_candidate.is_file():
            return self.uploader.upload(path_candidate)

        raise FileNotFoundError(f"本地图片不存在：{image_identifier}")

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        过滤 SerpAPI Google Lens 返回字段。

        Args:
            payload (dict[str, Any]): 原始的 SerpAPI 返回数据。

        Returns:
            dict[str, Any]: 已过滤的结果字典。
        """

        retained: dict[str, Any] = {}

        best_guesses = payload.get("best_guess_labels")
        if isinstance(best_guesses, list):
            guesses = [str(item).strip() for item in best_guesses if str(item).strip()]
            if guesses:
                retained["best_guess_labels"] = guesses

        knowledge_graph = payload.get("knowledge_graph")
        kg_items: list[dict[str, Any]] = []
        if isinstance(knowledge_graph, dict):
            knowledge_graph = [knowledge_graph]
        if isinstance(knowledge_graph, list):
            for item in knowledge_graph:
                if not isinstance(item, dict):
                    continue
                trimmed = {
                    key: item.get(key)
                    for key in ("title", "subtitle", "link", "source", "type")
                    if item.get(key)
                }
                if trimmed:
                    kg_items.append(trimmed)
        if kg_items:
            retained["knowledge_graph"] = kg_items

        visual_matches = payload.get("visual_matches")
        vm_items: list[dict[str, Any]] = []
        if isinstance(visual_matches, list):
            for item in visual_matches:
                if not isinstance(item, dict):
                    continue
                trimmed = {
                    key: item.get(key)
                    for key in ("title", "source", "link", "position")
                    if item.get(key)
                }
                if trimmed:
                    vm_items.append(trimmed)
        if vm_items:
            retained["visual_matches"] = vm_items[:5]  # 最多保留 5 条

        reverse_info = payload.get("reverse_image_search")
        if isinstance(reverse_info, dict):
            link = reverse_info.get("link")
            if link:
                retained["reverse_image_search_link"] = link

        related_content = payload.get("related_content")
        related_items: list[dict[str, Any]] = []
        if isinstance(related_content, list):
            for item in related_content:
                if not isinstance(item, dict):
                    continue
                trimmed = {
                    key: item.get(key)
                    for key in ("link", "query")
                    if item.get(key)
                }
                if trimmed:
                    related_items.append(trimmed)
        if related_items:
            retained["related_content"] = related_items

        return retained
