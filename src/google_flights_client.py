"""
Google Flights 工具支持模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator


_SORT_BY_ALIASES: dict[str, str | None] = {
    "top": "1",
    "top_flights": "1",
    "best": "1",
    "recommended": "1",
    "default": "1",
    "1": "1",
    "price": "2",
    "cheapest": "2",
    "low_price": "2",
    "2": "2",
    "departure_time": "3",
    "earliest_departure": "3",
    "depart_early": "3",
    "3": "3",
    "arrival_time": "4",
    "earliest_arrival": "4",
    "arrive_early": "4",
    "4": "4",
    "duration": "5",
    "shortest_duration": "5",
    "fastest": "5",
    "5": "5",
    "emissions": "6",
    "lowest_emissions": "6",
    "6": "6",
}
"""排序别名到 SerpAPI 枚举值的映射。"""

_TYPE_ALIASES: dict[str, str] = {
    "round_trip": "1",
    "roundtrip": "1",
    "return": "1",
    "往返": "1",
    "1": "1",
    "one_way": "2",
    "oneway": "2",
    "single": "2",
    "单程": "2",
    "2": "2",
}
"""航班类型别名到 SerpAPI 枚举值的映射（仅支持往返与单程）。"""


def _normalize_date(value: str, field_name: str) -> str:
    """
    校验并格式化日期。

    Args:
        value (str): 原始日期字符串。
        field_name (str): 字段名，便于错误提示。

    Returns:
        str: ISO8601 日期字符串。

    Raises:
        ValueError: 当输入为空或不符合 ``YYYY-MM-DD`` 格式时抛出。
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


class GoogleFlightsRequest(BaseModel):
    """
    表示一次 Google Flights 查询的参数集合。

    Args:
        departure_id (str): 出发机场或地点标识，可传 IATA 代码或 Freebase kgmid。
        arrival_id (str): 到达机场或地点标识，可传 IATA 代码或 Freebase kgmid。
        outbound_date (str): 去程日期，以 ``YYYY-MM-DD`` 表示。
        return_date (str | None): 返程日期，以 ``YYYY-MM-DD`` 表示；往返行程时必填。
        adults (int): 成人旅客数量，默认 1。
        sort_by (str | None): 排序策略，支持 ``top_flights``、``price``、``departure_time``、``arrival_time``、``duration``、``emissions`` 或对应枚举别名。
        trip_type (str): 行程类型，支持 ``round_trip``（默认）或 ``one_way``。
    """

    departure_id: str = Field(..., description="出发机场或地点标识。")
    arrival_id: str = Field(..., description="到达机场或地点标识。")
    outbound_date: str = Field(..., description="去程日期（YYYY-MM-DD）。")
    return_date: str | None = Field(
        None, description="返程日期（YYYY-MM-DD），往返行程必填。"
    )
    adults: int = Field(1, description="成人旅客数量，必须为正整数。")
    sort_by: str | None = Field(
        None,
        description=(
            "排序策略，支持 top_flights、price、departure_time、arrival_time、duration、emissions 或对应别名。"
        ),
    )
    trip_type: str = Field(
        "round_trip",
        description="行程类型，支持 round_trip 或 one_way，不支持 multi_city。",
    )

    sort_code: str | None = Field(default=None, exclude=True)
    type_code: str = Field(default="1", exclude=True)

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "GoogleFlightsRequest":
        """
        校验入参并生成内部使用的枚举值。

        Returns:
            GoogleFlightsRequest: 规范化后的请求对象。

        Raises:
            ValueError: 当任何字段不符合要求时抛出。
        """

        self.departure_id = self._normalize_location(self.departure_id, "departure_id")
        self.arrival_id = self._normalize_location(self.arrival_id, "arrival_id")
        self.outbound_date = _normalize_date(self.outbound_date, "outbound_date")

        if self.return_date is not None:
            self.return_date = _normalize_date(self.return_date, "return_date")

        if not isinstance(self.adults, int) or self.adults < 1:
            raise ValueError("adults 必须为 >=1 的整数。")

        self.type_code = self._normalize_type(self.trip_type)
        if self.type_code == "1" and self.return_date is None:
            raise ValueError("往返行程必须提供 return_date。")
        if self.type_code == "2" and self.return_date is None:
            # 单程行程不要求返程日期；保持 None
            pass

        if self.trip_type.strip().lower() in {"multi_city", "multi-city", "3"}:
            raise ValueError("当前工具不支持 multi city 行程。")

        if self.sort_by is not None:
            self.sort_code = self._normalize_sort(self.sort_by)
        else:
            self.sort_code = None

        return self

    @staticmethod
    def _normalize_location(value: str, field_name: str) -> str:
        """
        清洗机场或地点标识。

        Args:
            value (str): 原始标识。
            field_name (str): 字段名。

        Returns:
            str: 处理后的标识。

        Raises:
            ValueError: 当输入为空时抛出。
        """

        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须为字符串。")
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} 不可为空。")
        return text

    @staticmethod
    def _normalize_type(value: str) -> str:
        """
        将行程类型转换为 SerpAPI 枚举。

        Args:
            value (str): 用户输入的类型别名。

        Returns:
            str: SerpAPI 类型编码。

        Raises:
            ValueError: 当别名不受支持或为 multi city 时抛出。
        """

        if not isinstance(value, str):
            raise ValueError("trip_type 必须为字符串。")
        key = value.strip().lower()
        if key in {"multi_city", "multi-city", "3"}:
            raise ValueError("当前工具不支持 multi city 行程。")
        if key not in _TYPE_ALIASES:
            raise ValueError("trip_type 仅支持 round_trip 或 one_way。")
        return _TYPE_ALIASES[key]

    @staticmethod
    def _normalize_sort(value: str | int | float) -> str:
        """
        将排序方式转换为 SerpAPI 枚举。

        Args:
            value (str | int | float): 排序别名或数字编码。

        Returns:
            str: SerpAPI 排序枚举字符串。

        Raises:
            ValueError: 当取值不在支持范围内时抛出。
        """

        if isinstance(value, (int, float)):
            key = str(int(value))
        else:
            key = str(value or "").strip().lower()
        if not key:
            raise ValueError("sort_by 不可为空字符串。")
        if key not in _SORT_BY_ALIASES:
            raise ValueError(
                "sort_by 仅支持 top_flights、price、departure_time、arrival_time、duration、emissions 及对应别名。"
            )
        return _SORT_BY_ALIASES[key] or "1"

    def to_params(self) -> dict[str, str]:
        """
        转换为 SerpAPI 查询参数字典。

        Returns:
            dict[str, str]: 用于请求的参数字典。
        """

        params: dict[str, str] = {
            "departure_id": self.departure_id,
            "arrival_id": self.arrival_id,
            "outbound_date": self.outbound_date,
            "adults": str(self.adults),
            "type": self.type_code,
            "hl": "zh-CN",
            "currency": "CNY",
        }
        if self.return_date:
            params["return_date"] = self.return_date
        if self.sort_code and self.sort_code != "1":
            params["sort_by"] = self.sort_code
        return params


@dataclass
class GoogleFlightsClient:
    """
    Google Flights SerpAPI 客户端。

    Args:
        api_key (str): SerpAPI API Key。
        endpoint (str): 接口地址，默认 ``https://serpapi.com/search``。
        timeout (float): 请求超时时长，单位秒。
    """

    api_key: str
    endpoint: str = "https://serpapi.com/search"
    timeout: float = 20.0

    def __post_init__(self) -> None:
        """
        初始化客户端并校验参数。

        Raises:
            AssertionError: 当参数非法时抛出。
        """

        assert isinstance(self.api_key, str) and self.api_key.strip(), "api_key 不能为空。"
        assert isinstance(self.endpoint, str) and self.endpoint.startswith(
            "http"
        ), "endpoint 必须为合法 URL。"
        assert isinstance(self.timeout, (int, float)) and self.timeout > 0, (
            "timeout 必须为正数。"
        )
        self.api_key = self.api_key.strip()
        self.endpoint = self.endpoint.rstrip("/")
        self.timeout = float(self.timeout)

    def search(self, request: GoogleFlightsRequest) -> dict[str, Any]:
        """
        调用 SerpAPI 获取航班信息。

        Args:
            request (GoogleFlightsRequest): 已经校验过的请求参数对象。

        Returns:
            dict[str, Any]: 原始 JSON 响应。

        Raises:
            ValueError: 当网络请求失败或 SerpAPI 返回错误时抛出。
        """

        params = request.to_params()
        params["engine"] = "google_flights"
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


@dataclass
class GoogleFlightsConsoleFormatter:
    """
    将航班结果转为控制台摘要。

    Args:
        max_items (int): 提取航班数量上限。
    """

    max_items: int = field(default=5)

    def summarize(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """
        提取航班摘要列表，包含价格、航司、时刻与航段信息。

        Args:
            payload (dict[str, Any]): SerpAPI 返回的原始数据。

        Returns:
            list[dict[str, Any]]: 航班摘要列表。
        """

        if not isinstance(payload, dict):
            return []

        collected: list[dict[str, Any]] = []
        for key in ("best_flights", "other_flights"):
            flights = payload.get(key)
            if not isinstance(flights, list):
                continue
            for item in flights:
                summary = self._summarize_flight(item)
                if summary:
                    collected.append(summary)
                if len(collected) >= self.max_items:
                    return collected
        return collected

    @staticmethod
    def _summarize_flight(item: Any) -> dict[str, Any] | None:
        """
        汇总单条航班信息。

        Args:
            item (Any): 一条航班结果。

        Returns:
            dict[str, Any] | None: 航班摘要，若结构不符返回 None。
        """

        if not isinstance(item, dict):
            return None

        price_text = GoogleFlightsConsoleFormatter._extract_price(item)
        duration_text = GoogleFlightsConsoleFormatter._extract_duration(item)
        segments_data = item.get("flights")

        if not isinstance(segments_data, list) or not segments_data:
            return None

        segments: list[str] = []
        airlines: set[str] = set()
        for segment in segments_data:
            if not isinstance(segment, dict):
                continue
            airline_name = GoogleFlightsConsoleFormatter._extract_airline(segment)
            if airline_name:
                airlines.add(airline_name)
            segment_desc = GoogleFlightsConsoleFormatter._describe_segment(segment)
            if segment_desc:
                segments.append(segment_desc)

        if not segments:
            return None

        return {
            "price": price_text,
            "duration": duration_text,
            "airlines": ", ".join(sorted(airlines)) or "未知航司",
            "segments": segments,
        }

    @staticmethod
    def _extract_price(item: dict[str, Any]) -> str:
        """
        提取价格文本。
        """

        price = item.get("price")
        if isinstance(price, str) and price.strip():
            return price.strip()
        price_raw = item.get("price_raw")
        if isinstance(price_raw, (int, float)):
            return f"¥{price_raw}"
        return "价格未知"

    @staticmethod
    def _extract_duration(item: dict[str, Any]) -> str:
        """
        提取总时长信息。
        """

        duration = item.get("total_duration") or item.get("duration")
        if isinstance(duration, str) and duration.strip():
            return duration.strip()
        return "时长未知"

    @staticmethod
    def _extract_airline(segment: dict[str, Any]) -> str:
        """
        提取航司名称。
        """

        airline = segment.get("airline") or segment.get("carrier")
        if isinstance(airline, dict):
            name = airline.get("name") or airline.get("display_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(airline, str) and airline.strip():
            return airline.strip()
        return ""

    @staticmethod
    def _describe_segment(segment: dict[str, Any]) -> str:
        """
        生成单个航段描述。
        """

        dep_airport = segment.get("departure_airport") or {}
        arr_airport = segment.get("arrival_airport") or {}
        dep_code = GoogleFlightsConsoleFormatter._safe_airport_code(dep_airport)
        arr_code = GoogleFlightsConsoleFormatter._safe_airport_code(arr_airport)

        departure = segment.get("departure") or {}
        arrival = segment.get("arrival") or {}
        dep_time = GoogleFlightsConsoleFormatter._safe_time(departure)
        arr_time = GoogleFlightsConsoleFormatter._safe_time(arrival)

        segment_duration = segment.get("duration")
        duration_text = (
            segment_duration.strip()
            if isinstance(segment_duration, str) and segment_duration.strip()
            else ""
        )

        parts = [f"{dep_code}->{arr_code}", f"{dep_time}-{arr_time}"]
        if duration_text:
            parts.append(duration_text)
        return " ".join(parts).strip()

    @staticmethod
    def _safe_airport_code(airport: Any) -> str:
        """
        获取机场代码或名称。
        """

        if isinstance(airport, dict):
            code = airport.get("code")
            if isinstance(code, str) and code.strip():
                return code.strip()
            name = airport.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "未知机场"

    @staticmethod
    def _safe_time(moment: Any) -> str:
        """
        获取时间文本。
        """

        if isinstance(moment, dict):
            time_text = moment.get("time")
            if isinstance(time_text, str) and time_text.strip():
                return time_text.strip()
        return "时间未知"
