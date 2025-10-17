"""
Google Hotels 工具支持模块。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator


_SORT_BY_ALIASES: dict[str, str | None] = {
    "relevance": None,
    "price_low_to_high": "3",
    "price_high_to_low": "8",
    "most_reviewed": "13",
}


def _today_iso() -> str:
    """
    返回今天的 ISO8601 日期字符串。

    Returns:
        str: 形如 ``2024-09-09`` 的日期字符串。
    """

    return date.today().isoformat()


def _tomorrow_iso() -> str:
    """
    返回明天的 ISO8601 日期字符串。

    Returns:
        str: 形如 ``2024-09-10`` 的日期字符串。
    """

    return (date.today() + timedelta(days=1)).isoformat()


class GoogleHotelsRequest(BaseModel):
    """
    表示一次 Google Hotels 查询的参数集合。

    Args:
        query (str): 酒店或目的地关键词。
        check_in_date (str): 入住日期，默认今天。
        check_out_date (str): 离店日期，默认明天。
        adults (int): 入住成人数量，默认 1。
        hl (str): Google 语言代码，默认 ``zh-CN``。
        currency (str): 货币代码，默认 ``CNY``。
        sort_by (str | None): 排序策略，可选。
    """

    query: str = Field(..., description="酒店或目的地关键词，必须为非空字符串。")
    check_in_date: str = Field(
        default_factory=_today_iso,
        description="入住日期，ISO8601 格式，默认今天。",
    )
    check_out_date: str = Field(
        default_factory=_tomorrow_iso,
        description="离店日期，ISO8601 格式，默认明天。",
    )
    adults: int = Field(
        default=1,
        description="入住成人数量，必须为 >=1 的整数。",
    )
    hl: str = Field(
        default="zh-CN",
        description="Google 语言代码，建议传入 zh-CN。",
    )
    currency: str = Field(
        default="CNY",
        description="货币代码，例如 CNY、USD。",
    )
    sort_by: str | None = Field(
        default=None,
        description=(
            "排序策略，支持以下取值：``relevance``（默认）、``price_low_to_high``、``price_high_to_low``、``most_reviewed``。"
        ),
    )
    sort_code: str | None = Field(default=None, exclude=True)

    normalized_hl: str = Field(default="zh-CN", exclude=True)
    normalized_currency: str = Field(default="CNY", exclude=True)

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "GoogleHotelsRequest":
        """
        校验参数并生成规范化字段。

        Returns:
            GoogleHotelsRequest: 规范化后的自身实例。

        Raises:
            ValueError: 当输入参数不合法时抛出。
        """

        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query 必须为非空字符串。")
        self.query = self.query.strip()

        self.check_in_date = self._normalize_date(self.check_in_date, "check_in_date")
        self.check_out_date = self._normalize_date(
            self.check_out_date, "check_out_date"
        )

        check_in_obj = date.fromisoformat(self.check_in_date)
        check_out_obj = date.fromisoformat(self.check_out_date)
        if check_out_obj <= check_in_obj:
            raise ValueError("check_out_date 必须晚于 check_in_date。")

        if not isinstance(self.adults, int) or self.adults < 1:
            raise ValueError("adults 必须为 >=1 的整数。")

        hl_value = (self.hl or "").strip()
        if not hl_value:
            raise ValueError("hl 不可为空。")
        if not re.fullmatch(r"[a-z]{2}(-[a-z]{2})?", hl_value, re.IGNORECASE):
            raise ValueError("hl 必须为语言代码，例如 zh-CN。")
        self.normalized_hl = hl_value.lower()
        self.hl = self.normalized_hl

        currency_value = (self.currency or "").strip()
        if not currency_value or not re.fullmatch(r"[A-Z]{3}", currency_value):
            raise ValueError("currency 必须为 3 位大写货币代码，例如 CNY。")
        self.normalized_currency = currency_value.upper()
        self.currency = self.normalized_currency

        if self.sort_by is not None:
            sort_key = str(self.sort_by or "").strip().lower()
            self.sort_code = self._normalize_sort_by(sort_key)
            self.sort_by = sort_key
        else:
            self.sort_code = None

        return self

    @staticmethod
    def _normalize_date(value: str, field_name: str) -> str:
        """
        校验并格式化日期。

        Args:
            value (str): 原始日期字符串。
            field_name (str): 字段名，便于错误提示。

        Returns:
            str: 规范化后的日期。

        Raises:
            ValueError: 当日期格式非法时抛出。
        """

        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须为字符串。")
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} 不可为空。")
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} 必须符合 YYYY-MM-DD 格式。") from exc
        return parsed.isoformat()

    @staticmethod
    def _normalize_sort_by(value: str) -> str | None:
        """
        将用户输入的排序字段映射为 SerpAPI 支持的编码。

        Args:
            value (str): 归一化后的小写排序标识。

        Returns:
            str | None: SerpAPI 期望的排序编码，None 表示默认相关性。

        Raises:
            ValueError: 当取值不在支持范围内时抛出。
        """

        if not isinstance(value, str):
            raise ValueError("sort_by 必须为字符串。")
        key = value.strip().lower()
        if not key:
            raise ValueError("sort_by 不可为空字符串。")
        if key not in _SORT_BY_ALIASES:
            raise ValueError(
                "sort_by 仅支持 relevance、price_low_to_high、price_high_to_low、most_reviewed。"
            )
        return _SORT_BY_ALIASES[key]

    def to_params(self) -> dict[str, str]:
        """
        将请求对象转换为 SerpAPI 查询参数。

        Returns:
            dict[str, str]: 查询参数字典。
        """

        params: dict[str, str] = {
            "q": self.query,
            "hl": self.hl,
            "currency": self.currency,
            "check_in_date": self.check_in_date,
            "check_out_date": self.check_out_date,
            "adults": str(self.adults),
        }
        if self.sort_code:
            params["sort_by"] = self.sort_code
        return params


@dataclass
class GoogleHotelsClient:
    """
    Google Hotels SerpAPI 客户端。

    Args:
        api_key (str): SerpAPI API Key。
        endpoint (str): SerpAPI 接口地址，默认 https://serpapi.com/search。
        timeout (float): 请求超时时间，单位秒。
    """

    api_key: str
    endpoint: str = "https://serpapi.com/search"
    timeout: float = 20.0

    def __post_init__(self) -> None:
        """
        校验初始化参数。

        Raises:
            AssertionError: 当参数非法时抛出。
        """

        assert isinstance(self.api_key, str) and self.api_key.strip(), (
            "api_key 必须为非空字符串。"
        )
        assert isinstance(self.endpoint, str) and self.endpoint.startswith(
            "http"
        ), "endpoint 必须为合法 URL。"
        assert isinstance(self.timeout, (int, float)) and self.timeout > 0, (
            "timeout 必须为正数。"
        )
        self.api_key = self.api_key.strip()
        self.endpoint = self.endpoint.rstrip("/")
        self.timeout = float(self.timeout)

    def search(self, request: GoogleHotelsRequest) -> dict[str, Any]:
        """
        调用 SerpAPI 获取 Google Hotels 数据。

        Args:
            request (GoogleHotelsRequest): 已校验的请求对象。

        Returns:
            dict[str, Any]: 原始 JSON 响应。

        Raises:
            ValueError: 当网络请求失败或 SerpAPI 返回错误时抛出。
        """

        params = request.to_params()
        params["engine"] = "google_hotels"
        params["api_key"] = self.api_key

        try:
            response = requests.get(
                self.endpoint, params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ValueError("请求 SerpAPI 失败，请检查网络连接。") from exc

        if response.status_code == 401:
            raise ValueError("SerpAPI 鉴权失败：请确认 SERPAPI_API_KEY 是否正确。")
        if response.status_code == 403:
            raise ValueError("SerpAPI 拒绝访问：请检查订阅状态或配额限制。")
        if response.status_code != 200:
            raise ValueError(f"SerpAPI 调用失败，HTTP {response.status_code}。")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("SerpAPI 返回非 JSON 数据。") from exc

        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(f"SerpAPI 返回错误：{payload['error']}")

        if not isinstance(payload, dict):
            raise ValueError("SerpAPI 返回数据格式异常。")

        return payload


def sanitize_hotels_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    清理 Google Hotels 原始返回，移除冗余字段并裁剪属性列表。

    Args:
        payload (dict[str, Any]): 原始 JSON 数据。

    Returns:
        dict[str, Any]: 清理后的 JSON。
    """

    if not isinstance(payload, dict):
        return payload

    sanitized = copy.deepcopy(payload)
    for key in ("search_parameters", "brands", "ads"):
        sanitized.pop(key, None)

    properties = sanitized.get("properties")
    if isinstance(properties, list):
        sanitized["properties"] = properties[:20]
        for item in sanitized["properties"]:
            if isinstance(item, dict):
                item.pop("images", None)

    return sanitized


@dataclass
class GoogleHotelsConsoleFormatter:
    """
    将 Google Hotels 结果转为简化摘要，便于在控制台展示。

    Args:
        max_items (int): 输出的酒店条数上限。
    """

    max_items: int = field(default=5)

    def summarize(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        """
        生成酒店列表摘要。

        Args:
            payload (dict[str, Any]): SerpAPI 返回的原始数据。

        Returns:
            list[dict[str, str]]: 每项包含名称、评分、价格、地址与链接的摘要。
        """

        if not isinstance(payload, dict):
            return []
        properties = payload.get("properties")
        if not isinstance(properties, list):
            return []

        summary: list[dict[str, str]] = []
        for item in properties:
            if not isinstance(item, dict):
                continue
            name = self._extract_name(item)
            if not name:
                continue
            rating = self._extract_rating(item)
            price = self._extract_price(item)
            address = self._extract_address(item)
            link = self._extract_link(item)
            reviews = self._extract_review_count(item)
            summary.append(
                {
                    "name": name,
                    "rating": rating,
                    "reviews": reviews,
                    "price": price,
                    "address": address,
                    "link": link,
                }
            )
            if len(summary) >= self.max_items:
                break
        return summary

    @staticmethod
    def _extract_name(item: dict[str, Any]) -> str:
        """
        提取酒店名称。
        """

        name = item.get("name") or item.get("title")
        if isinstance(name, str):
            return name.strip()
        return ""

    @staticmethod
    def _extract_rating(item: dict[str, Any]) -> str:
        """
        提取评分文本。
        """

        rating = item.get("overall_rating") or item.get("rating")
        if rating is None:
            return "暂无评分"
        if isinstance(rating, (int, float)):
            return f"{rating:.1f}"
        return str(rating)

    @staticmethod
    def _extract_review_count(item: dict[str, Any]) -> str:
        """
        提取评论数量。
        """

        reviews = item.get("reviews") or item.get("reviews_summary")
        if isinstance(reviews, dict):
            count = reviews.get("review_count") or reviews.get("count")
            if isinstance(count, (int, float)):
                return str(int(count))
        if isinstance(reviews, list):
            return str(len(reviews))
        return "未知"

    @staticmethod
    def _extract_price(item: dict[str, Any]) -> str:
        """
        提取价格信息。
        """

        pricing_sources = [
            item.get("rate_per_night"),
            item.get("total_rate"),
            item.get("price"),
            item.get("price_summary"),
        ]
        for source in pricing_sources:
            if isinstance(source, dict):
                for key in ("lowest", "display_rate", "rate", "formatted"):
                    value = source.get(key)
                    if value:
                        return str(value)
            elif isinstance(source, str) and source.strip():
                return source.strip()
        return "暂无价格"

    @staticmethod
    def _extract_address(item: dict[str, Any]) -> str:
        """
        提取地址文本。
        """

        address = item.get("address")
        if isinstance(address, str) and address.strip():
            return address.strip()
        if isinstance(address, dict):
            parts = [
                address.get("street_address"),
                address.get("city"),
                address.get("region"),
                address.get("country"),
            ]
            joined = ", ".join(
                part.strip()
                for part in parts
                if isinstance(part, str) and part.strip()
            )
            if joined:
                return joined
        return "地址未提供"

    @staticmethod
    def _extract_link(item: dict[str, Any]) -> str:
        """
        提取酒店链接。
        """

        link = item.get("link") or item.get("hotel_link")
        if isinstance(link, str) and link.strip():
            return link.strip()
        hotel_links = item.get("hotel_links")
        if isinstance(hotel_links, list) and hotel_links:
            candidate = hotel_links[0]
            if isinstance(candidate, dict):
                link_value = candidate.get("link")
                if isinstance(link_value, str) and link_value.strip():
                    return link_value.strip()
        return "链接未提供"
