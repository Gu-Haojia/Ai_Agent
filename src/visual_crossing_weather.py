"""
Visual Crossing 天气工具支持模块。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualCrossingWeatherRequest(BaseModel):
    """
    表示一次 Visual Crossing 天气查询的参数集合。

    Args:
        location (str): 查询地点，支持自然语言地点或经纬度。
        start_time (str): 起始时间，允许仅传日期或带小时的日期时间。
        end_time (str | None): 结束时间，可为空；为空时按单次查询处理。
        hour (bool): 是否按小时粒度返回数据，False 表示仅返回按天数据。

    Returns:
        VisualCrossingWeatherRequest: 已验证并附加内部字段的请求对象。

    Raises:
        ValueError: 当参数组合不合法或时间格式不符合 ISO8601 时抛出。
    """

    location: str = Field(..., description="查询地点，支持城市、邮编或经纬度描述。")
    start_time: str = Field(
        ...,
        description="起始时间，ISO8601 日期或日期时间，例如 2024-05-01 或 2024-05-01T15:00。",
    )
    end_time: str | None = Field(
        None,
        description="结束时间，可为空，格式与 start_time 一致。",
    )
    hour: bool = Field(
        False,
        description="是否按小时粒度返回数据；False 时默认按天粒度。",
    )
    normalized_start: str = Field("", exclude=True)
    normalized_end: str | None = Field(None, exclude=True)
    use_hours: bool = Field(False, exclude=True)
    use_days: bool = Field(True, exclude=True)
    target_hour: int | None = Field(None, exclude=True)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "VisualCrossingWeatherRequest":
        """
        校验输入时间并生成内部规范化字段。

        Args:
            None

        Returns:
            VisualCrossingWeatherRequest: 自身实例。

        Raises:
            ValueError: 当输入无效时抛出。
        """

        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError("location 必须为非空字符串。")
        self.location = self.location.strip()

        norm_start, start_has_time, start_hour = self._normalize_time(
            self.start_time, "start_time"
        )
        self.normalized_start = norm_start

        if self.end_time is not None:
            norm_end, _has_time, _ = self._normalize_time(self.end_time, "end_time")
            self.normalized_end = norm_end
            if self._to_datetime(norm_end) < self._to_datetime(norm_start):
                raise ValueError("end_time 不可早于 start_time。")
        else:
            self.normalized_end = None

        self.use_hours = bool(self.hour)
        self.use_days = not self.use_hours

        if self.use_hours and self.end_time is None and start_has_time:
            self.target_hour = start_hour
        else:
            self.target_hour = None

        return self

    @staticmethod
    def _normalize_time(value: str, field_name: str) -> tuple[str, bool, int | None]:
        """
        将输入时间标准化为 Timeline API 接受的格式。

        Args:
            value (str): 原始输入字符串。
            field_name (str): 字段名称，用于错误提示。

        Returns:
            tuple[str, bool, int | None]: 规范化后的字符串、是否包含时间、小时值。

        Raises:
            ValueError: 当字符串无法解析为 ISO8601 日期或日期时间时抛出。
        """

        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须为字符串。")
        raw = value.strip()
        if not raw:
            raise ValueError(f"{field_name} 不可为空字符串。")
        iso_candidate = raw.replace(" ", "T")
        try:
            dt = datetime.fromisoformat(iso_candidate)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} 必须为 ISO8601 日期或日期时间，例如 2024-05-01 或 2024-05-01T15:00。"
            ) from exc
        has_time = ("T" in raw) or (":" in raw) or (" " in raw)
        if has_time:
            normalized = dt.strftime("%Y-%m-%dT%H:%M")
            hour_value = dt.hour
        else:
            normalized = dt.date().isoformat()
            hour_value = None
        return normalized, has_time, hour_value

    @staticmethod
    def _to_datetime(value: str) -> datetime:
        """
        将规范化后的日期或日期时间转换为 datetime 对象。

        Args:
            value (str): 规范化后的时间字符串。

        Returns:
            datetime: 对应的 datetime 对象。

        Raises:
            ValueError: 当字符串无法解析为 datetime 时抛出。
        """

        if "T" in value:
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(f"{value}T00:00")


class VisualCrossingWeatherClient:
    """
    Visual Crossing Timeline API 查询客户端。

    Args:
        api_key (str): Visual Crossing API Key。
        base_url (str): API 基础 URL，默认为官方 Timeline Endpoint。
        timeout (float): 请求超时时间，单位秒。

    Raises:
        AssertionError: 当参数不合法时抛出。
    """

    _DEFAULT_BASE_URL = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    )

    def __init__(
        self, api_key: str, base_url: str | None = None, timeout: float = 15.0
    ) -> None:
        """
        初始化客户端。

        Args:
            api_key (str): Visual Crossing API Key。
            base_url (str | None): API 基础 URL，可为空。
            timeout (float): 请求超时时间。

        Returns:
            None

        Raises:
            AssertionError: 当参数不符合要求时抛出。
        """

        assert isinstance(api_key, str) and api_key.strip(), "api_key 必须为非空字符串。"
        assert isinstance(timeout, (int, float)) and timeout > 0, "timeout 必须为正数。"
        self._api_key = api_key.strip()
        used_base = base_url or self._DEFAULT_BASE_URL
        assert isinstance(used_base, str) and used_base.strip(), "base_url 不可为空。"
        self._base_url = used_base.rstrip("/")
        self._timeout = float(timeout)

    def fetch(
        self,
        request: VisualCrossingWeatherRequest,
    ) -> dict[str, Any]:
        """
        执行天气查询请求。

        Args:
            request (VisualCrossingWeatherRequest): 已校验的请求参数。

        Returns:
            dict[str, Any]: Visual Crossing Timeline API 的 JSON 响应。

        Raises:
            RuntimeError: 当 HTTP 请求失败或返回格式异常时抛出。
        """

        url = self._compose_url(request)
        params = self._build_params(request)
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:  # pragma: no cover
            raise RuntimeError(f"调用 Visual Crossing API 失败: {exc}") from exc

        if response.status_code != 200:
            detail = ""
            try:
                detail = response.text[:200]
            except Exception:  # pragma: no cover
                detail = ""
            raise RuntimeError(
                f"Visual Crossing API 返回异常状态码: {response.status_code}，详情片段: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover
            raise RuntimeError("Visual Crossing API 返回内容非 JSON。") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Visual Crossing API 返回格式异常，应为 JSON 对象。")
        return data

    def _compose_url(self, request: VisualCrossingWeatherRequest) -> str:
        """
        构造 Timeline API 的完整请求 URL。

        Args:
            request (VisualCrossingWeatherRequest): 请求参数对象。

        Returns:
            str: 可直接使用的请求 URL。

        Raises:
            None
        """

        segments: list[str] = [self._base_url, quote(request.location)]
        segments.append(quote(request.normalized_start))
        if request.normalized_end:
            segments.append(quote(request.normalized_end))
        return "/".join(segments)

    def _build_params(self, request: VisualCrossingWeatherRequest) -> dict[str, Any]:
        """
        根据请求参数生成查询字符串。

        Args:
            request (VisualCrossingWeatherRequest): 请求参数对象。

        Returns:
            dict[str, Any]: 可用于 requests.get 的 params 字典。

        Raises:
            None
        """

        includes: list[str] = []
        if request.use_hours:
            includes.append("hours")
        if request.use_days:
            includes.append("days")

        params: dict[str, Any] = {
            "key": self._api_key,
            "unitGroup": "metric",
            "contentType": "json",
            "include": ",".join(includes) if includes else None,
        }
        return {k: v for k, v in params.items() if v is not None}


class VisualCrossingWeatherFormatter:
    """
    用于将 Visual Crossing 返回的天气数据整理为适合展示的文本。

    Args:
        None

    Raises:
        None
    """

    def format(
        self, request: VisualCrossingWeatherRequest, payload: dict[str, Any]
    ) -> str:
        """
        格式化天气响应结果。

        Args:
            request (VisualCrossingWeatherRequest): 用户请求参数。
            payload (dict[str, Any]): Visual Crossing API 的原始响应数据。

        Returns:
            str: 经过整理的 JSON 字符串，便于模型理解与复述。

        Raises:
            AssertionError: 当 payload 结构缺失关键字段时抛出。
        """

        assert isinstance(payload, dict), "payload 必须为字典。"
        result: dict[str, Any] = {
            "location": payload.get("resolvedAddress") or payload.get("address"),
            "timezone": payload.get("timezone"),
            "query": self._build_query_context(request),
        }

        current_conditions = payload.get("currentConditions")
        if isinstance(current_conditions, dict) and not request.use_hours:
            result["current"] = self._pick_fields(
                current_conditions,
                ["datetime", "temp", "feelslike", "humidity", "windspeed", "conditions"],
            )

        days_info = self._extract_days(request, payload.get("days"))
        if days_info:
            result["days"] = days_info

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _build_query_context(
        self, request: VisualCrossingWeatherRequest
    ) -> dict[str, Any]:
        """
        生成查询上下文信息，便于输出中说明查询范围。

        Args:
            request (VisualCrossingWeatherRequest): 用户请求参数。

        Returns:
            dict[str, Any]: 描述查询意图的字典。

        Raises:
            None
        """

        context = {
            "location": request.location,
            "start_time": request.normalized_start,
            "end_time": request.normalized_end,
            "granularity": "hour" if request.use_hours else "day",
        }
        if request.target_hour is not None:
            context["target_hour"] = request.target_hour
        return {k: v for k, v in context.items() if v is not None}

    def _extract_days(
        self,
        request: VisualCrossingWeatherRequest,
        days_payload: Any,
    ) -> list[dict[str, Any]]:
        """
        提取天级别数据并根据请求加上小时信息。

        Args:
            request (VisualCrossingWeatherRequest): 用户请求参数。
            days_payload (Any): API days 字段原始数据。

        Returns:
            list[dict[str, Any]]: 整理后的天级别数据列表。

        Raises:
            None
        """

        if not isinstance(days_payload, Iterable):
            return []

        days_result: list[dict[str, Any]] = []
        for day in days_payload:
            if not isinstance(day, dict):
                continue
            summary = self._pick_fields(
                day,
                [
                    "datetime",
                    "tempmax",
                    "tempmin",
                    "temp",
                    "feelslike",
                    "humidity",
                    "precip",
                    "precipprob",
                    "windspeed",
                    "conditions",
                    "description",
                ],
            )
            hours_data = self._extract_hours(request, day.get("hours"))
            if hours_data:
                summary["hours"] = hours_data
            days_result.append(summary)

        return days_result

    def _extract_hours(
        self, request: VisualCrossingWeatherRequest, hours_payload: Any
    ) -> list[dict[str, Any]]:
        """
        从天级别数据里筛选小时级别信息。

        Args:
            request (VisualCrossingWeatherRequest): 用户请求参数。
            hours_payload (Any): API hours 字段原始数据。

        Returns:
            list[dict[str, Any]]: 过滤后的小时数据列表。

        Raises:
            None
        """

        if not request.use_hours:
            return []

        if not isinstance(hours_payload, Iterable):
            return []

        filtered_hours: list[dict[str, Any]] = []
        for hour_item in hours_payload:
            if not isinstance(hour_item, dict):
                continue
            hour_entry = self._pick_fields(
                hour_item,
                [
                    "datetime",
                    "temp",
                    "feelslike",
                    "humidity",
                    "precip",
                    "precipprob",
                    "windspeed",
                    "conditions",
                ],
            )
            if request.target_hour is not None:
                hour_value = self._parse_hour_value(hour_entry.get("datetime"))
                if hour_value is None or hour_value != request.target_hour:
                    continue
            filtered_hours.append(hour_entry)

        return filtered_hours

    @staticmethod
    def _parse_hour_value(raw: Any) -> int | None:
        """
        尝试从字符串中解析小时数（0-23）。

        Args:
            raw (Any): 原始小时字段值。

        Returns:
            int | None: 若解析成功返回小时数，否则返回 None。

        Raises:
            None
        """

        if not isinstance(raw, str):
            return None
        value = raw.strip()
        if not value:
            return None
        candidate = value
        if "T" in candidate:
            candidate = candidate.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(candidate).hour
            except ValueError:
                tail = candidate.split("T")[-1]
                if tail:
                    candidate = tail
        if ":" in candidate:
            prefix = candidate.split(":")[0]
            if prefix[-2:].isdigit():
                return int(prefix[-2:])
        if candidate.isdigit():
            parsed = int(candidate)
            if 0 <= parsed <= 23:
                return parsed
        return None

    @staticmethod
    def _pick_fields(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        """
        从字典中提取需要的字段，过滤掉空值。

        Args:
            source (dict[str, Any]): 原始字段。
            keys (list[str]): 需要保留的键名列表。

        Returns:
            dict[str, Any]: 只保留指定键且值不为 None 的字典。

        Raises:
            None
        """

        return {k: source.get(k) for k in keys if source.get(k) is not None}


__all__ = [
    "VisualCrossingWeatherClient",
    "VisualCrossingWeatherFormatter",
    "VisualCrossingWeatherRequest",
]
