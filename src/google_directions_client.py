"""
Google Maps Directions SerpAPI 客户端模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class GoogleDirectionsClient:
    """调用 SerpAPI Google Maps Directions 接口的客户端。

    Args:
        api_key (str): SerpAPI API Key。
        endpoint (str): SerpAPI 接口地址，默认 ``https://serpapi.com/search``。
        timeout_seconds (int): HTTP 请求超时时间（秒），默认 30。
    """

    api_key: str
    endpoint: str = "https://serpapi.com/search"
    timeout_seconds: int = 30

    def search(
        self,
        start_addr: str,
        end_addr: str,
        time: str | None = None,
    ) -> dict[str, Any]:
        """查询两地之间的 Google Maps 路线。

        Args:
            start_addr (str): 起点地址。
            end_addr (str): 终点地址。
            time (str | None): SerpAPI 要求的时间参数，例如 ``depart_at:1698229538``。

        Returns:
            dict[str, Any]: SerpAPI 返回的 JSON 数据。

        Raises:
            AssertionError: 当地址参数为空时抛出。
            ValueError: 当网络请求失败或 SerpAPI 返回错误时抛出。
        """

        assert isinstance(start_addr, str) and start_addr.strip(), "start_addr 必须为非空字符串。"
        assert isinstance(end_addr, str) and end_addr.strip(), "end_addr 必须为非空字符串。"
        if time is not None:
            assert isinstance(time, str) and time.strip(), "time 参数必须为非空字符串。"

        normalized_time = time.strip() if isinstance(time, str) else None

        params = {
            "engine": "google_maps_directions",
            "start_addr": start_addr.strip(),
            "end_addr": end_addr.strip(),
            "hl": "ja",
            "travel_mode": "3",
            "distance_unit": "0",
            "api_key": self.api_key,
        }

        if normalized_time:
            params["time"] = normalized_time

        try:
            response = requests.get(self.endpoint, params=params, timeout=self.timeout_seconds)
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
