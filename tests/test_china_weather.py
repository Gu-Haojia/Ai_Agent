"""
和风天气国内天气工具单元测试。
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from pydantic import ValidationError

from src.china_weather import (
    ChinaWeatherClient,
    ChinaWeatherFormatter,
    ChinaWeatherRequest,
    ChinaWeatherService,
)


class ChinaWeatherTests(unittest.TestCase):
    """
    验证国内天气工具的请求、调用和精简格式。

    Args:
        unittest.TestCase: 标准测试基类。

    Returns:
        None

    Raises:
        None
    """

    def test_request_validates_location_and_forecast(self) -> None:
        """
        验证请求会清理地点并拒绝未知天气范围。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        request = ChinaWeatherRequest(
            location="  苏州市  ",
            adm="  江苏省  ",
            forecast="today",
        )
        self.assertEqual(request.location, "苏州市")
        self.assertEqual(request.adm, "江苏省")
        self.assertIsNone(ChinaWeatherRequest(location="上海", adm=" ").adm)
        self.assertEqual(ChinaWeatherRequest(location="上海").forecast, "today")
        self.assertEqual(
            ChinaWeatherRequest.model_json_schema()["properties"]["location"][
                "description"
            ],
            "目标城市或区县的最小关键词，不包括上级行政区，例如苏州市或天宁区。",
        )
        with self.assertRaises(ValidationError):
            ChinaWeatherRequest(location="苏州市", forecast="24h")

    def test_client_fetches_location_weather_and_alerts(self) -> None:
        """
        验证客户端依次请求地点、天气和新版预警接口。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        geo_response = self._response(
            {
                "code": "200",
                "location": [
                    {
                        "id": "101190401",
                        "name": "苏州",
                        "adm2": "苏州",
                        "adm1": "江苏省",
                        "lat": "31.30",
                        "lon": "120.58",
                    }
                ],
            }
        )
        weather_response = self._response(
            {
                "code": "200",
                "updateTime": "2026-08-15T00:33+08:00",
                "daily": [],
            }
        )
        alert_response = self._response(
            {
                "metadata": {"zeroResult": False},
                "alerts": [
                    {
                        "headline": "苏州市气象台发布暴雨橙色预警",
                        "severity": "severe",
                        "issuedTime": "2026-08-14T19:42+08:00",
                    }
                ],
            }
        )
        client = ChinaWeatherClient(
            api_host="weather.example.com",
            api_key="secret",
        )

        with mock.patch(
            "src.china_weather.requests.get",
            side_effect=[geo_response, weather_response, alert_response],
        ) as request_get:
            result = client.fetch(
                ChinaWeatherRequest(
                    location="苏州市",
                    adm="江苏",
                    forecast="7d",
                )
            )

        self.assertEqual(request_get.call_count, 3)
        self.assertTrue(
            request_get.call_args_list[0].args[0].endswith("/geo/v2/city/lookup")
        )
        self.assertTrue(
            request_get.call_args_list[1].args[0].endswith("/v7/weather/7d")
        )
        self.assertTrue(
            request_get.call_args_list[2].args[0].endswith(
                "/weatheralert/v1/current/31.30/120.58"
            )
        )
        self.assertEqual(
            request_get.call_args_list[0].kwargs["headers"],
            {"X-QW-Api-Key": "secret"},
        )
        self.assertEqual(
            request_get.call_args_list[0].kwargs["params"]["adm"],
            "江苏",
        )
        self.assertEqual(len(result["alerts"]), 1)

    def test_client_fetches_daily_and_hourly_weather_for_tomorrow(self) -> None:
        """
        验证明日查询会请求三日总览和未来72小时预报。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        geo_response = self._response(
            {
                "code": "200",
                "location": [
                    {
                        "id": "101191105",
                        "name": "天宁",
                        "lat": "31.78",
                        "lon": "119.96",
                    }
                ],
            }
        )
        daily_response = self._response(
            {"code": "200", "updateTime": "2026-08-15T08:00+08:00", "daily": []}
        )
        hourly_response = self._response(
            {
                "code": "200",
                "updateTime": "2026-08-15T09:00+08:00",
                "hourly": [],
            }
        )
        alert_response = self._response(
            {"metadata": {"zeroResult": True}, "alerts": []}
        )
        client = ChinaWeatherClient(api_host="weather.example.com", api_key="secret")

        with mock.patch(
            "src.china_weather.requests.get",
            side_effect=[
                geo_response,
                daily_response,
                hourly_response,
                alert_response,
            ],
        ) as request_get:
            result = client.fetch(
                ChinaWeatherRequest(
                    location="天宁",
                    adm="常州",
                    forecast="tomorrow",
                )
            )

        self.assertEqual(request_get.call_count, 4)
        self.assertTrue(
            request_get.call_args_list[1].args[0].endswith("/v7/weather/3d")
        )
        self.assertTrue(
            request_get.call_args_list[2].args[0].endswith("/v7/weather/72h")
        )
        self.assertEqual(result["weather"]["daily"], [])
        self.assertEqual(result["hourly_weather"]["hourly"], [])

    def test_client_accepts_zero_alert_result(self) -> None:
        """
        验证新版预警接口的零结果会转换为空列表。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        geo_response = self._response(
            {
                "code": "200",
                "location": [
                    {
                        "id": "101210201",
                        "name": "湖州",
                        "lat": "30.89",
                        "lon": "120.09",
                    }
                ],
            }
        )
        weather_response = self._response(
            {"code": "200", "updateTime": "2026-08-15T00:33+08:00", "now": {}}
        )
        minutely_response = self._response(
            {
                "code": "200",
                "updateTime": "2026-08-15T00:35+08:00",
                "summary": "未来2小时无降水",
                "minutely": [],
            }
        )
        alert_response = self._response(
            {"metadata": {"zeroResult": True}, "alerts": []}
        )
        client = ChinaWeatherClient(api_host="https://weather.example.com", api_key="x")

        with mock.patch(
            "src.china_weather.requests.get",
            side_effect=[
                geo_response,
                weather_response,
                minutely_response,
                alert_response,
            ],
        ) as request_get:
            result = client.fetch(
                ChinaWeatherRequest(location="湖州市", forecast="now")
            )

        self.assertEqual(request_get.call_count, 4)
        self.assertTrue(
            request_get.call_args_list[2].args[0].endswith("/v7/minutely/5m")
        )
        self.assertEqual(
            request_get.call_args_list[2].kwargs["params"]["location"],
            "120.09,30.89",
        )
        self.assertEqual(result["minutely"]["summary"], "未来2小时无降水")
        self.assertEqual(result["alerts"], [])

    def test_formatter_combines_tomorrow_overview_and_hourly_weather(self) -> None:
        """
        验证明日结果只保留明日总览和明日逐小时数据。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        formatter = ChinaWeatherFormatter()
        request = ChinaWeatherRequest(location="苏州市", forecast="tomorrow")
        payload = {
            "location": {"name": "苏州", "adm2": "苏州", "adm1": "江苏省"},
            "weather": {
                "updateTime": "2026-08-15T08:00+08:00",
                "daily": [
                    {"fxDate": "2026-08-15"},
                    {
                        "fxDate": "2026-08-16",
                        "textDay": "多云",
                        "textNight": "小雨",
                        "tempMin": "24",
                        "tempMax": "31",
                        "precip": "2.1",
                        "humidity": "88",
                        "vis": "18",
                        "uvIndex": "6",
                        "sunrise": "05:25",
                        "sunset": "18:43",
                        "windDirDay": "东风",
                        "windScaleDay": "1-3",
                        "windDirNight": "东风",
                        "windScaleNight": "1-3",
                    },
                ],
            },
            "hourly_weather": {
                "updateTime": "2026-08-15T09:00+08:00",
                "hourly": [
                    {
                        "fxTime": "2026-08-15T23:00+08:00",
                        "text": "阴",
                    },
                    {
                        "fxTime": "2026-08-16T00:00+08:00",
                        "text": "小雨",
                        "temp": "24",
                        "pop": "70",
                        "precip": "0.32",
                        "humidity": "97",
                        "windDir": "东风",
                        "windScale": "1-3",
                    },
                    {
                        "fxTime": "2026-08-16T01:00+08:00",
                        "text": "阴",
                        "temp": "24",
                        "pop": "30",
                        "precip": "0.0",
                        "humidity": "96",
                        "windDir": "东北风",
                        "windScale": "1-3",
                    },
                    {
                        "fxTime": "2026-08-17T00:00+08:00",
                        "text": "多云",
                    },
                ],
            },
            "alerts": [
                {
                    "headline": "暴雨橙色预警",
                    "severity": "severe",
                    "issuedTime": "2026-08-14T19:42+08:00",
                    "description": "不会进入精简结果的长文本",
                }
            ],
        }

        result = json.loads(formatter.format(request, payload))

        self.assertEqual(result["location"], "江苏省苏州")
        self.assertEqual(result["target_date"], "2026-08-16")
        self.assertEqual(result["daily"][0][0], "2026-08-16")
        self.assertEqual(result["hourly_updated_at"], "2026-08-15T09:00+08:00")
        self.assertEqual(len(result["hourly"]), 2)
        self.assertEqual(
            result["hourly"][0],
            [
                "2026-08-16T00:00+08:00",
                "小雨",
                24,
                70,
                0.32,
                97,
                "东风1-3级",
            ],
        )
        self.assertNotIn("current", result)
        self.assertNotIn("next_2h_rain", result)
        self.assertEqual(
            result["alerts"],
            [["暴雨橙色预警", "severe", "2026-08-14T19:42+08:00"]],
        )
        self.assertNotIn("description", formatter.format(request, payload))

    def test_formatter_selects_today_overview_and_hours(self) -> None:
        """
        验证今日结果只保留今日总览和今日逐小时数据。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        payload = {
            "location": {"name": "天宁", "adm2": "常州", "adm1": "江苏省"},
            "weather": {
                "updateTime": "2026-08-15T08:00+08:00",
                "daily": [
                    {"fxDate": "2026-08-15"},
                    {"fxDate": "2026-08-16"},
                ],
            },
            "hourly_weather": {
                "updateTime": "2026-08-15T09:00+08:00",
                "hourly": [
                    {"fxTime": "2026-08-15T10:00+08:00", "text": "多云"},
                    {"fxTime": "2026-08-16T00:00+08:00", "text": "小雨"},
                ],
            },
            "alerts": [],
        }

        result = json.loads(
            ChinaWeatherFormatter().format(
                ChinaWeatherRequest(location="天宁", forecast="today"),
                payload,
            )
        )

        self.assertEqual(result["target_date"], "2026-08-15")
        self.assertEqual(result["daily"][0][0], "2026-08-15")
        self.assertEqual(len(result["hourly"]), 1)
        self.assertEqual(result["hourly"][0][0], "2026-08-15T10:00+08:00")

    def test_formatter_supports_current_and_daily_forecasts(self) -> None:
        """
        验证实时与逐日响应只输出各自需要的指标。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        formatter = ChinaWeatherFormatter()
        location = {"name": "湖州", "adm2": "湖州", "adm1": "浙江省"}
        current = json.loads(
            formatter.format(
                ChinaWeatherRequest(location="湖州市", forecast="now"),
                {
                    "location": location,
                    "weather": {
                        "updateTime": "2026-08-15T00:33+08:00",
                        "now": {
                            "text": "阴",
                            "temp": "24",
                            "feelsLike": "27",
                            "humidity": "94",
                            "precip": "0.0",
                            "vis": "12",
                            "windDir": "西南风",
                            "windScale": "1-3",
                        },
                    },
                    "minutely": {
                        "code": "200",
                        "updateTime": "2026-08-15T00:35+08:00",
                        "summary": "35分钟后雨就停了",
                        "minutely": [],
                    },
                    "alerts": [],
                },
            )
        )
        daily = json.loads(
            formatter.format(
                ChinaWeatherRequest(location="湖州市", forecast="7d"),
                {
                    "location": location,
                    "weather": {
                        "updateTime": "2026-08-15T00:33+08:00",
                        "daily": [
                            {
                                "fxDate": "2026-08-15",
                                "textDay": "多云",
                                "textNight": "小雨",
                                "tempMin": "24",
                                "tempMax": "31",
                                "precip": "2.1",
                                "humidity": "88",
                                "vis": "18",
                                "uvIndex": "6",
                                "sunrise": "05:24",
                                "sunset": "18:44",
                                "windDirDay": "东风",
                                "windScaleDay": "1-3",
                                "windDirNight": "东风",
                                "windScaleNight": "1-3",
                            }
                        ],
                    },
                    "alerts": [],
                },
            )
        )

        self.assertEqual(current["current"]["feels_like_c"], 27)
        self.assertEqual(current["current"]["visibility_km"], 12)
        self.assertEqual(
            current["next_2h_rain"],
            {"summary": "35分钟后雨就停了"},
        )
        self.assertNotIn("hourly", current)
        self.assertEqual(
            set(current),
            {"location", "updated_at", "current", "next_2h_rain", "alerts"},
        )
        self.assertEqual(
            daily["daily"][0],
            [
                "2026-08-15",
                "多云",
                "小雨",
                24,
                31,
                2.1,
                88,
                "东风1-3级",
                18,
                6,
                "05:24",
                "18:44",
            ],
        )
        self.assertNotIn("current", daily)
        self.assertEqual(
            set(daily),
            {"location", "updated_at", "daily_fields", "daily", "alerts"},
        )

    def test_service_returns_structured_client_error(self) -> None:
        """
        验证客户端失败时服务返回精简结构化错误。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        client = ChinaWeatherClient(api_host="weather.example.com", api_key="x")
        formatter = ChinaWeatherFormatter()
        service = ChinaWeatherService(client=client, formatter=formatter)
        request = ChinaWeatherRequest(location="不存在的地点", forecast="today")

        with mock.patch.object(
            client,
            "fetch",
            side_effect=RuntimeError("和风天气 GeoAPI 未找到地点"),
        ):
            result = json.loads(service.query(request))

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "CHINA_WEATHER_REQUEST_FAILED")
        self.assertEqual(
            result["query"],
            {"location": "不存在的地点", "forecast": "today"},
        )

    def test_service_returns_structured_formatter_error(self) -> None:
        """
        验证响应格式异常时不会向工具边界抛出异常。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        client = ChinaWeatherClient(api_host="weather.example.com", api_key="x")
        formatter = ChinaWeatherFormatter()
        service = ChinaWeatherService(client=client, formatter=formatter)
        request = ChinaWeatherRequest(location="苏州市", forecast="today")

        with mock.patch.object(
            client,
            "fetch",
            return_value={"location": {}, "weather": {}, "alerts": []},
        ):
            result = json.loads(service.query(request))

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "CHINA_WEATHER_REQUEST_FAILED")

    def test_service_returns_unexpected_exception_to_agent(self) -> None:
        """
        验证未预期的普通异常也会结构化返回给 Agent。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        client = ChinaWeatherClient(api_host="weather.example.com", api_key="x")
        formatter = ChinaWeatherFormatter()
        service = ChinaWeatherService(client=client, formatter=formatter)
        request = ChinaWeatherRequest(location="苏州市", forecast="today")

        with mock.patch.object(
            client,
            "fetch",
            side_effect=KeyError("unexpected payload field"),
        ):
            result = json.loads(service.query(request))

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "CHINA_WEATHER_REQUEST_FAILED")
        self.assertIn("unexpected payload field", result["error"]["message"])

    def test_formatter_structures_validation_error(self) -> None:
        """
        验证工具入参校验错误可直接返回给 Agent。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        with self.assertRaises(ValidationError) as validation_context:
            ChinaWeatherRequest(location="苏州市", forecast="14d")

        result = json.loads(
            ChinaWeatherFormatter.format_validation_error(
                validation_context.exception
            )
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "CHINA_WEATHER_INVALID_INPUT")

    @staticmethod
    def _response(payload: dict[str, object]) -> mock.Mock:
        """
        创建 requests.get 使用的模拟响应。

        Args:
            payload (dict[str, object]): response.json 的返回对象。

        Returns:
            mock.Mock: 状态码为 200 的模拟响应。

        Raises:
            None
        """

        response = mock.Mock(status_code=200, text="")
        response.json.return_value = payload
        return response


if __name__ == "__main__":
    unittest.main()
