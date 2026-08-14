"""
和风天气国内天气工具支持模块。
"""

from __future__ import annotations

import json
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


ChinaWeatherForecast = Literal["now", "24h", "72h", "3d", "7d"]


class ChinaWeatherRequest(BaseModel):
    """
    表示一次国内天气查询请求。

    Args:
        location (str): 中国境内城市或区县名称。
        adm (str | None): 可选的上级行政区关键词。
        forecast (ChinaWeatherForecast): 需要查询的天气范围。

    Returns:
        ChinaWeatherRequest: 校验后的请求对象。

    Raises:
        ValueError: 当地点为空或预报范围不受支持时抛出。
    """

    location: str = Field(
        ...,
        description=(
            "目标城市或区县的最小关键词，不包括上级行政区，例如苏州市或天宁区。"
        ),
    )
    adm: str | None = Field(
        None,
        description="可选的上级行政区关键词，例如江苏或常州。",
    )
    forecast: ChinaWeatherForecast = Field(
        "24h",
        description=(
            "天气范围，可选 now、24h、72h、3d、7d。now 包含当前实况和"
            "未来2小时分钟级降水摘要；24h、72h 仅包含逐小时预报，不包含当前实况。"
        ),
    )

    @field_validator("location")
    @classmethod
    def _validate_location(cls, value: str) -> str:
        """
        校验并清理地点名称。

        Args:
            value (str): 原始地点名称。

        Returns:
            str: 去除首尾空白后的地点名称。

        Raises:
            ValueError: 当地点为空时抛出。
        """

        if not isinstance(value, str) or not value.strip():
            raise ValueError("location 必须为非空字符串。")
        return value.strip()

    @field_validator("adm")
    @classmethod
    def _validate_adm(cls, value: str | None) -> str | None:
        """
        清理可选的上级行政区关键词。

        Args:
            value (str | None): 原始上级行政区关键词。

        Returns:
            str | None: 清理后的关键词，空字符串转换为 None。

        Raises:
            None
        """

        if value is None or not value.strip():
            return None
        return value.strip()


class ChinaWeatherClient:
    """
    和风天气 API 客户端。

    Args:
        api_host (str): 和风天气控制台分配的专属 API Host。
        api_key (str): 和风天气 API Key。
        timeout (float): 单次请求超时时间，单位秒。

    Raises:
        AssertionError: 当初始化参数不合法时抛出。
    """

    _WEATHER_PATHS: dict[ChinaWeatherForecast, str] = {
        "now": "/v7/weather/now",
        "24h": "/v7/weather/24h",
        "72h": "/v7/weather/72h",
        "3d": "/v7/weather/3d",
        "7d": "/v7/weather/7d",
    }

    def __init__(self, api_host: str, api_key: str, timeout: float = 15.0) -> None:
        """
        初始化和风天气客户端。

        Args:
            api_host (str): 专属 API Host，可包含 HTTPS 协议前缀。
            api_key (str): API Key。
            timeout (float): 单次请求超时时间，单位秒。

        Returns:
            None

        Raises:
            AssertionError: 当参数不符合要求时抛出。
        """

        assert isinstance(api_host, str) and api_host.strip(), (
            "api_host 必须为非空字符串。"
        )
        assert isinstance(api_key, str) and api_key.strip(), "api_key 必须为非空字符串。"
        assert isinstance(timeout, (int, float)) and timeout > 0, "timeout 必须为正数。"
        normalized_host = api_host.strip().rstrip("/")
        if not normalized_host.startswith("https://"):
            normalized_host = f"https://{normalized_host}"
        self._base_url = normalized_host
        self._api_key = api_key.strip()
        self._timeout = float(timeout)

    def fetch(self, request: ChinaWeatherRequest) -> dict[str, Any]:
        """
        解析地点并查询天气、分钟级降水与实时预警。

        Args:
            request (ChinaWeatherRequest): 已校验的国内天气请求。

        Returns:
            dict[str, Any]: 地点、天气、分钟级降水和预警组成的内部响应。

        Raises:
            RuntimeError: 当地点不存在、请求失败或响应结构异常时抛出。
        """

        assert isinstance(request, ChinaWeatherRequest), "request 类型无效。"
        location = self._fetch_location(request.location, request.adm)
        weather = self._fetch_weather(str(location["id"]), request.forecast)
        minutely = None
        if request.forecast == "now":
            minutely = self._fetch_minutely(
                latitude=str(location["lat"]),
                longitude=str(location["lon"]),
            )
        alerts = self._fetch_alerts(str(location["lat"]), str(location["lon"]))
        return {
            "location": location,
            "weather": weather,
            "minutely": minutely,
            "alerts": alerts,
        }

    def _fetch_location(
        self,
        location: str,
        adm: str | None,
    ) -> dict[str, Any]:
        """
        将中文地点解析为和风天气地点信息。

        Args:
            location (str): 中国境内城市或区县名称。
            adm (str | None): 可选的上级行政区关键词。

        Returns:
            dict[str, Any]: 排名第一的地点信息。

        Raises:
            RuntimeError: 当地点不存在或 GeoAPI 响应异常时抛出。
        """

        params: dict[str, Any] = {
            "location": location,
            "range": "cn",
            "number": 1,
            "lang": "zh",
        }
        if adm is not None:
            params["adm"] = adm
        payload = self._request_json("/geo/v2/city/lookup", params)
        self._validate_qweather_code(payload, "GeoAPI")
        locations = payload.get("location")
        if not isinstance(locations, list) or not locations:
            raise RuntimeError(f"和风天气 GeoAPI 未找到地点: {location}")
        resolved = locations[0]
        if not isinstance(resolved, dict):
            raise RuntimeError("和风天气 GeoAPI 地点格式异常。")
        for required_field in ("id", "lat", "lon", "name"):
            if not resolved.get(required_field):
                raise RuntimeError(
                    f"和风天气 GeoAPI 地点缺少字段: {required_field}"
                )
        return resolved

    def _fetch_weather(
        self,
        location_id: str,
        forecast: ChinaWeatherForecast,
    ) -> dict[str, Any]:
        """
        查询指定地点和范围的天气数据。

        Args:
            location_id (str): 和风天气 LocationID。
            forecast (ChinaWeatherForecast): 天气查询范围。

        Returns:
            dict[str, Any]: 和风天气原始天气响应。

        Raises:
            RuntimeError: 当天气接口响应异常时抛出。
        """

        payload = self._request_json(
            self._WEATHER_PATHS[forecast],
            {"location": location_id, "lang": "zh", "unit": "m"},
        )
        self._validate_qweather_code(payload, "天气 API")
        return payload

    def _fetch_alerts(self, latitude: str, longitude: str) -> list[dict[str, Any]]:
        """
        查询地点当前生效的气象预警。

        Args:
            latitude (str): 地点纬度。
            longitude (str): 地点经度。

        Returns:
            list[dict[str, Any]]: 当前预警列表，无预警时为空列表。

        Raises:
            RuntimeError: 当预警接口响应结构异常时抛出。
        """

        payload = self._request_json(
            f"/weatheralert/v1/current/{latitude}/{longitude}",
            {"localTime": "true", "lang": "zh"},
        )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("和风天气预警 API 缺少 metadata。")
        alerts = payload.get("alerts")
        if alerts is None and metadata.get("zeroResult") is True:
            return []
        if not isinstance(alerts, list):
            raise RuntimeError("和风天气预警 API 的 alerts 格式异常。")
        if not alerts and metadata.get("zeroResult") is not True:
            raise RuntimeError("和风天气预警 API 返回了无法识别的空结果。")
        if not all(isinstance(alert, dict) for alert in alerts):
            raise RuntimeError("和风天气预警 API 包含格式异常的预警。")
        return alerts

    def _fetch_minutely(
        self,
        latitude: str,
        longitude: str,
    ) -> dict[str, Any]:
        """
        查询未来两小时的分钟级降水预报。

        Args:
            latitude (str): 地点纬度。
            longitude (str): 地点经度。

        Returns:
            dict[str, Any]: 和风天气分钟级降水原始响应。

        Raises:
            RuntimeError: 当分钟级降水接口响应异常时抛出。
            ValueError: 当经纬度无法转换为数字时抛出。
        """

        coordinates = f"{float(longitude):.2f},{float(latitude):.2f}"
        payload = self._request_json(
            "/v7/minutely/5m",
            {"location": coordinates, "lang": "zh"},
        )
        self._validate_qweather_code(payload, "分钟级降水 API")
        return payload

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        发送带 API Key 的 GET 请求并解析 JSON。

        Args:
            path (str): API 请求路径。
            params (dict[str, Any]): 查询参数。

        Returns:
            dict[str, Any]: JSON 对象响应。

        Raises:
            RuntimeError: 当网络、HTTP 状态或 JSON 格式异常时抛出。
        """

        try:
            retrying = Retrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=2),
                retry=retry_if_exception_type(requests.RequestException),
                reraise=True,
            )
            response = retrying(
                requests.get,
                f"{self._base_url}{path}",
                headers={"X-QW-Api-Key": self._api_key},
                params=params,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover
            raise RuntimeError(f"调用和风天气 API 失败: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                "和风天气 API 返回异常状态码: "
                f"{response.status_code}，详情片段: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover
            raise RuntimeError("和风天气 API 返回内容非 JSON。") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("和风天气 API 返回格式异常，应为 JSON 对象。")
        return payload

    @staticmethod
    def _validate_qweather_code(payload: dict[str, Any], api_name: str) -> None:
        """
        验证带 code 字段的和风天气响应。

        Args:
            payload (dict[str, Any]): API JSON 响应。
            api_name (str): 用于错误消息的接口名称。

        Returns:
            None

        Raises:
            RuntimeError: 当响应 code 不是 200 时抛出。
        """

        code = payload.get("code")
        if code != "200":
            raise RuntimeError(f"和风天气{api_name}返回错误码: {code}")


class ChinaWeatherFormatter:
    """
    将和风天气响应压缩为适合 Agent 使用的 JSON。

    Args:
        None

    Raises:
        None
    """

    def format(self, request: ChinaWeatherRequest, payload: dict[str, Any]) -> str:
        """
        格式化天气与预警响应。

        Args:
            request (ChinaWeatherRequest): 用户的天气请求。
            payload (dict[str, Any]): 客户端返回的内部响应。

        Returns:
            str: 精简后的 JSON 字符串。

        Raises:
            AssertionError: 当响应缺少必要结构时抛出。
        """

        location = payload.get("location")
        weather = payload.get("weather")
        minutely = payload.get("minutely")
        alerts = payload.get("alerts")
        assert isinstance(location, dict), "location 必须为字典。"
        assert isinstance(weather, dict), "weather 必须为字典。"
        assert isinstance(alerts, list), "alerts 必须为列表。"

        result: dict[str, Any] = {
            "location": self._format_location(location),
            "updated_at": weather.get("updateTime"),
        }
        if request.forecast == "now":
            assert isinstance(minutely, dict), "实时天气响应缺少分钟级降水。"
            result["current"] = self._format_current(weather)
            result["next_2h_rain"] = self._format_minutely(minutely)
        elif request.forecast in ("24h", "72h"):
            result["hourly_fields"] = [
                "time",
                "weather",
                "temp_c",
                "rain_probability",
                "rain_mm",
                "humidity",
                "wind",
            ]
            result["hourly"] = self._format_hourly(weather)
        else:
            result["daily_fields"] = [
                "date",
                "day_weather",
                "night_weather",
                "temp_min_c",
                "temp_max_c",
                "rain_mm",
                "humidity",
                "wind",
            ]
            result["daily"] = self._format_daily(weather)
        result["alerts"] = self._format_alerts(alerts)
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _format_minutely(minutely: dict[str, Any]) -> dict[str, str]:
        """
        提取未来两小时分钟级降水摘要。

        Args:
            minutely (dict[str, Any]): 分钟级降水原始响应。

        Returns:
            dict[str, str]: 仅包含自然语言降水摘要的精简结果。

        Raises:
            AssertionError: 当分钟级降水摘要缺失时抛出。
        """

        summary = minutely.get("summary")
        assert isinstance(summary, str) and summary.strip(), (
            "分钟级降水响应缺少 summary。"
        )
        return {"summary": summary.strip()}

    def format_error(self, request: ChinaWeatherRequest, error: Exception) -> str:
        """
        将可预期的客户端错误转为结构化 JSON。

        Args:
            request (ChinaWeatherRequest): 用户的天气请求。
            error (Exception): 查询或格式化过程中抛出的明确错误。

        Returns:
            str: 结构化错误 JSON 字符串。

        Raises:
            None
        """

        query: dict[str, Any] = {
            "location": request.location,
            "forecast": request.forecast,
        }
        if request.adm is not None:
            query["adm"] = request.adm
        result = {
            "success": False,
            "error": {
                "code": "CHINA_WEATHER_REQUEST_FAILED",
                "message": str(error),
                "retryable": False,
            },
            "query": query,
        }
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def format_validation_error(error: ValidationError) -> str:
        """
        将工具入参校验错误转为结构化 JSON。

        Args:
            error (ValidationError): Pydantic 入参校验错误。

        Returns:
            str: 可直接返回给 Agent 的错误 JSON。

        Raises:
            None
        """

        result = {
            "success": False,
            "error": {
                "code": "CHINA_WEATHER_INVALID_INPUT",
                "message": str(error),
                "retryable": False,
            },
        }
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _format_location(location: dict[str, Any]) -> str:
        """
        组合并去重省、市、地点名称。

        Args:
            location (dict[str, Any]): GeoAPI 地点信息。

        Returns:
            str: 精简后的完整地点名称。

        Raises:
            AssertionError: 当地点名称为空时抛出。
        """

        parts: list[str] = []
        for field_name in ("adm1", "adm2", "name"):
            value = location.get(field_name)
            if isinstance(value, str) and value.strip() and value.strip() not in parts:
                parts.append(value.strip())
        assert parts, "地点名称不可为空。"
        return "".join(parts)

    def _format_current(self, weather: dict[str, Any]) -> dict[str, Any]:
        """
        提取实时天气关键指标。

        Args:
            weather (dict[str, Any]): 实时天气响应。

        Returns:
            dict[str, Any]: 精简的实时天气。

        Raises:
            AssertionError: 当响应缺少 now 字段时抛出。
        """

        current = weather.get("now")
        assert isinstance(current, dict), "实时天气响应缺少 now。"
        return {
            "weather": current.get("text"),
            "temp_c": self._to_number(current.get("temp")),
            "feels_like_c": self._to_number(current.get("feelsLike")),
            "humidity": self._to_number(current.get("humidity")),
            "rain_mm": self._to_number(current.get("precip")),
            "wind": self._format_wind(current.get("windDir"), current.get("windScale")),
        }

    def _format_hourly(self, weather: dict[str, Any]) -> list[list[Any]]:
        """
        将逐小时响应压缩为字段顺序固定的数组。

        Args:
            weather (dict[str, Any]): 逐小时天气响应。

        Returns:
            list[list[Any]]: 完整逐小时天气列表。

        Raises:
            AssertionError: 当 hourly 字段格式异常时抛出。
        """

        hourly = weather.get("hourly")
        assert isinstance(hourly, list), "逐小时天气响应缺少 hourly。"
        rows: list[list[Any]] = []
        for item in hourly:
            assert isinstance(item, dict), "hourly 元素必须为字典。"
            rows.append(
                [
                    item.get("fxTime"),
                    item.get("text"),
                    self._to_number(item.get("temp")),
                    self._to_number(item.get("pop")),
                    self._to_number(item.get("precip")),
                    self._to_number(item.get("humidity")),
                    self._format_wind(item.get("windDir"), item.get("windScale")),
                ]
            )
        return rows

    def _format_daily(self, weather: dict[str, Any]) -> list[list[Any]]:
        """
        将逐日响应压缩为字段顺序固定的数组。

        Args:
            weather (dict[str, Any]): 逐日天气响应。

        Returns:
            list[list[Any]]: 完整逐日天气列表。

        Raises:
            AssertionError: 当 daily 字段格式异常时抛出。
        """

        daily = weather.get("daily")
        assert isinstance(daily, list), "逐日天气响应缺少 daily。"
        rows: list[list[Any]] = []
        for item in daily:
            assert isinstance(item, dict), "daily 元素必须为字典。"
            day_wind = self._format_wind(
                item.get("windDirDay"), item.get("windScaleDay")
            )
            night_wind = self._format_wind(
                item.get("windDirNight"), item.get("windScaleNight")
            )
            wind = day_wind if day_wind == night_wind else f"{day_wind}/{night_wind}"
            rows.append(
                [
                    item.get("fxDate"),
                    item.get("textDay"),
                    item.get("textNight"),
                    self._to_number(item.get("tempMin")),
                    self._to_number(item.get("tempMax")),
                    self._to_number(item.get("precip")),
                    self._to_number(item.get("humidity")),
                    wind,
                ]
            )
        return rows

    @staticmethod
    def _format_alerts(alerts: list[Any]) -> list[list[Any]]:
        """
        压缩气象预警，只保留标题、严重程度和发布时间。

        Args:
            alerts (list[Any]): 预警 API 返回列表。

        Returns:
            list[list[Any]]: 精简预警列表。

        Raises:
            AssertionError: 当预警结构异常时抛出。
        """

        rows: list[list[Any]] = []
        for alert in alerts:
            assert isinstance(alert, dict), "预警元素必须为字典。"
            headline = alert.get("headline")
            assert isinstance(headline, str) and headline.strip(), "预警标题不可为空。"
            rows.append([headline, alert.get("severity"), alert.get("issuedTime")])
        return rows

    @staticmethod
    def _format_wind(direction: Any, scale: Any) -> str | None:
        """
        组合风向和风力等级。

        Args:
            direction (Any): 风向文字。
            scale (Any): 风力等级。

        Returns:
            str | None: 组合后的风向风力，无数据时为 None。

        Raises:
            AssertionError: 当字段类型异常时抛出。
        """

        if direction is None and scale is None:
            return None
        assert direction is None or isinstance(direction, str), "风向必须为字符串。"
        assert scale is None or isinstance(scale, str), "风力等级必须为字符串。"
        direction_text = direction or ""
        scale_text = f"{scale}级" if scale else ""
        return f"{direction_text}{scale_text}"

    @staticmethod
    def _to_number(value: Any) -> int | float | None:
        """
        将和风天气的数字字符串转换为数值。

        Args:
            value (Any): 原始数字字段。

        Returns:
            int | float | None: 转换后的数值，空值返回 None。

        Raises:
            AssertionError: 当字段类型不受支持时抛出。
            ValueError: 当字符串不是合法数字时抛出。
        """

        if value is None or value == "":
            return None
        assert isinstance(value, (str, int, float)), "天气数值字段类型异常。"
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric


class ChinaWeatherService:
    """
    在工具边界统一处理国内天气查询与格式化。

    Args:
        client (ChinaWeatherClient): 和风天气客户端。
        formatter (ChinaWeatherFormatter): 精简输出格式化器。

    Raises:
        AssertionError: 当依赖实例类型不正确时抛出。
    """

    def __init__(
        self,
        client: ChinaWeatherClient,
        formatter: ChinaWeatherFormatter,
    ) -> None:
        """
        初始化国内天气服务。

        Args:
            client (ChinaWeatherClient): 和风天气客户端。
            formatter (ChinaWeatherFormatter): 精简输出格式化器。

        Returns:
            None

        Raises:
            AssertionError: 当依赖实例类型不正确时抛出。
        """

        assert isinstance(client, ChinaWeatherClient), "client 类型无效。"
        assert isinstance(formatter, ChinaWeatherFormatter), "formatter 类型无效。"
        self._client = client
        self._formatter = formatter

    def query(self, request: ChinaWeatherRequest) -> str:
        """
        查询并格式化国内天气。

        Args:
            request (ChinaWeatherRequest): 已校验的国内天气请求。

        Returns:
            str: 精简天气数据或结构化错误 JSON。

        Raises:
            AssertionError: 当 request 类型不正确时抛出。
        """

        assert isinstance(request, ChinaWeatherRequest), "request 类型无效。"
        try:
            payload = self._client.fetch(request)
            return self._formatter.format(request, payload)
        except Exception as error:
            return self._formatter.format_error(request, error)


__all__ = [
    "ChinaWeatherClient",
    "ChinaWeatherFormatter",
    "ChinaWeatherRequest",
    "ChinaWeatherService",
]
