"""
Visual Crossing 天气工具单元测试（unittest 版本）。
"""

from __future__ import annotations

import json
import unittest

from src.visual_crossing_weather import (
    VisualCrossingWeatherClient,
    VisualCrossingWeatherFormatter,
    VisualCrossingWeatherRequest,
)


class VisualCrossingWeatherTests(unittest.TestCase):
    """
    针对 Visual Crossing 天气工具的数据准备与格式化行为的测试用例集合。

    Args:
        unittest.TestCase: 标准测试基类。

    Returns:
        None

    Raises:
        None
    """

    def test_request_datetime_parsing(self) -> None:
        """
        验证 datetime 字段可自动解析日期与小时。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        request = VisualCrossingWeatherRequest(
            location="Shanghai", datetime="2024-05-01T15:30", hourly=True
        )
        self.assertEqual(request.date, "2024-05-01")
        self.assertEqual(request.target_hour, 15)
        self.assertTrue(request.include_hour)

    def test_request_invalid_range(self) -> None:
        """
        确保日期区间参数缺失时抛出错误。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        with self.assertRaises(ValueError):
            VisualCrossingWeatherRequest(
                location="Beijing",
                startDate="2024-05-01",
            )

    def test_formatter_extract_hours_target(self) -> None:
        """
        验证格式化逻辑可筛选目标小时的数据。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        request = VisualCrossingWeatherRequest(
            location="New York",
            date="2024-05-01",
            hour=9,
            day=True,
            hourly=True,
        )
        payload = {
            "resolvedAddress": "New York, NY",
            "timezone": "America/New_York",
            "days": [
                {
                    "datetime": "2024-05-01",
                    "tempmax": 20.0,
                    "tempmin": 12.0,
                    "conditions": "Partially cloudy",
                    "hours": [
                        {
                            "datetime": "08:00:00",
                            "temp": 15.2,
                            "conditions": "Clear",
                        },
                        {
                            "datetime": "09:00:00",
                            "temp": 16.5,
                            "conditions": "Partially cloudy",
                        },
                    ],
                }
            ],
        }
        formatter = VisualCrossingWeatherFormatter()
        formatted = formatter.format(request, payload)
        data = json.loads(formatted)
        self.assertEqual(data["days"][0]["hours"][0]["datetime"], "09:00:00")
        self.assertAlmostEqual(data["days"][0]["hours"][0]["temp"], 16.5, places=1)

    def test_compose_url_with_range(self) -> None:
        """
        确保区间查询能正确拼接 URL。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        client = VisualCrossingWeatherClient(api_key="dummy")
        request = VisualCrossingWeatherRequest(
            location="Tokyo",
            startDate="2024-05-01",
            endDate="2024-05-05",
            day=True,
            hourly=False,
        )
        url = client._compose_url(request)
        self.assertTrue(url.endswith("/Tokyo/2024-05-01/2024-05-05"))


if __name__ == "__main__":
    unittest.main()
