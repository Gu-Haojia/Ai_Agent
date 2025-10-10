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
        date (str | None): 指定日期（YYYY-MM-DD），为空时查询默认时间轴。
        start_date (str | None): 日期范围起始值（YYYY-MM-DD），需与 end_date 搭配使用。
        end_date (str | None): 日期范围结束值（YYYY-MM-DD），需与 start_date 搭配使用。
        datetime_text (str | None): ISO8601 日期时间（如 2024-05-01T15:00），用于查询特定小时。
        target_hour (int | None): 指定的小时（0-23），需与 date 或 datetime_text 搭配。
        include_day (bool): 是否在请求中包含 days 数据。
        include_hour (bool): 是否在请求中包含 hours 数据。
        include_current (bool): 是否在请求中包含 current 数据。
        unit_group (str): 单位组配置，可选 metric、us、uk、base。

    Returns:
        VisualCrossingWeatherRequest: 已验证的参数对象。

    Raises:
        ValueError: 当参数组合不合法或格式不正确时抛出。
    """

    location: str = Field(..., description="查询地点，支持城市、邮编或经纬度。")
    date: str | None = Field(
        None,
        description="单日查询日期，格式 YYYY-MM-DD。",
    )
    start_date: str | None = Field(
        None,
        alias="startDate",
        description="区间查询起始日期，格式 YYYY-MM-DD。",
    )
    end_date: str | None = Field(
        None,
        alias="endDate",
        description="区间查询结束日期，格式 YYYY-MM-DD。",
    )
    datetime_text: str | None = Field(
        None,
        alias="datetime",
        description="用于查询具体小时的 ISO8601 日期时间，例如 2024-05-01T15:00。",
    )
    target_hour: int | None = Field(
        None,
        alias="hour",
        ge=0,
        le=23,
        description="指定小时（0-23），需与 date 或 datetime_text 搭配使用。",
    )
    include_day: bool = Field(
        True, alias="day", description="是否包含天级别数据（days）。"
    )
    include_hour: bool = Field(
        False, alias="hourly", description="是否包含小时级别数据（hours）。"
    )
    include_current: bool = Field(
        False, alias="current", description="是否包含当前天气数据（current）。"
    )
    unit_group: str = Field(
        "metric",
        alias="unitGroup",
        description="单位组，可选 metric、us、uk、base，默认为 metric。",
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _validate_dates(self) -> "VisualCrossingWeatherRequest":
        """
        校验日期范围与小时参数组合的合法性。

        Args:
            None

        Returns:
            VisualCrossingWeatherRequest: 通过校验后的自身实例。

        Raises:
            ValueError: 当日期与小时参数组合不合法或格式错误时抛出。
        """

        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError("location 必须为非空字符串。")

        if self.start_date or self.end_date:
            if not self.start_date or not self.end_date:
                raise ValueError("start_date 与 end_date 必须同时提供。")
            if self.date:
                raise ValueError("使用日期区间时不应再传入单独的 date。")
            if self.datetime_text:
                raise ValueError("使用日期区间时不应传入 datetime_text。")

        if self.datetime_text:
            try:
                dt_value = datetime.fromisoformat(self.datetime_text)
            except ValueError as exc:
                raise ValueError(
                    "datetime_text 必须为 ISO8601 格式，例如 2024-05-01T15:00。"
                ) from exc
            normalized_date = dt_value.date().isoformat()
            self.date = self.date or normalized_date
            if self.target_hour is None:
                self.target_hour = dt_value.hour
            if not self.include_hour:
                self.include_hour = True

        if self.target_hour is not None and self.date is None:
            raise ValueError("指定 target_hour 时必须提供 date 或 datetime_text。")

        valid_units = {"metric", "us", "uk", "base"}
        if self.unit_group not in valid_units:
            raise ValueError(f"unit_group 仅支持 {sorted(valid_units)}。")

        if self.include_hour and not self.include_day and not (
            self.start_date or self.end_date or self.date
        ):
            raise ValueError("查询小时数据时必须指定 date 或日期区间。")

        return self


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

        segments: list[str] = [self._base_url, quote(request.location.strip())]

        if request.start_date and request.end_date:
            segments.append(quote(request.start_date))
            segments.append(quote(request.end_date))
        elif request.date:
            segments.append(quote(request.date))

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
        if request.include_day:
            includes.append("days")
        if request.include_hour or request.target_hour is not None:
            includes.append("hours")
        if request.include_current:
            includes.append("current")

        params: dict[str, Any] = {
            "key": self._api_key,
            "unitGroup": request.unit_group,
            "include": ",".join(includes) if includes else None,
            "contentType": "json",
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
        if request.include_current and isinstance(current_conditions, dict):
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
            "date": request.date,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "target_hour": request.target_hour,
            "unit_group": request.unit_group,
            "include_day": request.include_day,
            "include_hour": request.include_hour or request.target_hour is not None,
            "include_current": request.include_current,
        }
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

        if not request.include_hour and request.target_hour is None:
            return []

        if not isinstance(hours_payload, Iterable):
            return []

        filtered_hours: list[dict[str, Any]] = []
        for hour in hours_payload:
            if not isinstance(hour, dict):
                continue
            hour_entry = self._pick_fields(
                hour,
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
                if hour_value is None:
                    continue
                if hour_value != request.target_hour:
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
