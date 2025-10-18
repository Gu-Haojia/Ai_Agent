"""
Google Reverse Image SerpAPI 客户端模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class GoogleReverseImageClient:
    """
    调用 SerpAPI Google 以图搜图接口的轻量客户端。

    Args:
        api_key (str): SerpAPI API Key。
        endpoint (str): SerpAPI 接口地址，默认 ``https://serpapi.com/search``。
        timeout_seconds (int): 请求超时时间（秒），默认 30。
    """

    api_key: str
    endpoint: str = "https://serpapi.com/search"
    timeout_seconds: int = 30

    def search(self, image_url: str, start: int | None = None) -> dict[str, Any]:
        """
        使用指定图片 URL 发起反向图片搜索。

        Args:
            image_url (str): 待搜索的图片 URL。
            start (int | None): 结果偏移量，用于分页；None 表示默认第一页。

        Returns:
            dict[str, Any]: SerpAPI 返回的 JSON 数据。

        Raises:
            AssertionError: 当 ``image_url`` 为空字符串时抛出。
            ValueError: 当网络请求失败、鉴权失败或 SerpAPI 返回错误时抛出。
        """

        assert isinstance(image_url, str) and image_url.strip(), "image_url 必须为非空字符串。"

        params = {
            "engine": "google_reverse_image",
            "image_url": image_url,
            "api_key": self.api_key,
        }

        if start is not None:
            assert isinstance(start, int) and start >= 0, "start 必须为非负整数。"
            params["start"] = start

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
        if "error" in payload:
            raise ValueError(f"SerpAPI 返回错误：{payload['error']}")

        return payload
