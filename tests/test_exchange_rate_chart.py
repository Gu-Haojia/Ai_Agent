"""滚动汇率图核心的单元测试。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest import mock

import requests
from PIL import Image

from src.exchange_rate_chart import (
    CHART_MODES,
    ExchangeRateChartMode,
    ExchangeRateChartRenderer,
    IntradayRatePoint,
    RollingRateSeriesBuilder,
    TwelveDataIntradayClient,
)


def _response(payload: object, status_code: int = 200) -> mock.Mock:
    """
    构造 Twelve Data HTTP 响应替身。

    Args:
        payload (object): ``response.json()`` 返回内容。
        status_code (int): HTTP 状态码。

    Returns:
        mock.Mock: requests.Response 替身。

    Raises:
        AssertionError: 当状态码不是正数时抛出。
    """

    assert status_code > 0, "status_code 必须为正数。"
    response = mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _points() -> list[IntradayRatePoint]:
    """
    创建用于绘图测试的分时数据。

    Returns:
        list[IntradayRatePoint]: 三个按时间排列的数据点。
    """

    return [
        IntradayRatePoint(
            timestamp=datetime(2026, 7, 24, 0, 0),
            open_rate=Decimal("146.100"),
            high_rate=Decimal("146.200"),
            low_rate=Decimal("146.050"),
            close_rate=Decimal("146.150"),
        ),
        IntradayRatePoint(
            timestamp=datetime(2026, 7, 24, 12, 0),
            open_rate=Decimal("146.150"),
            high_rate=Decimal("146.420"),
            low_rate=Decimal("146.100"),
            close_rate=Decimal("146.380"),
        ),
        IntradayRatePoint(
            timestamp=datetime(2026, 7, 24, 23, 55),
            open_rate=Decimal("146.380"),
            high_rate=Decimal("146.500"),
            low_rate=Decimal("146.300"),
            close_rate=Decimal("146.440"),
        ),
    ]


def test_client_fetches_and_parses_intraday_points() -> None:
    """
    验证客户端请求参数和 OHLC 数据解析。

    Returns:
        None: 本测试无返回值。
    """

    response = _response(
        {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-07-24 00:00:00",
                    "open": "146.100",
                    "high": "146.200",
                    "low": "146.050",
                    "close": "146.150",
                }
            ],
        }
    )
    with mock.patch(
        "src.exchange_rate_chart.requests.get",
        return_value=response,
    ) as get:
        points = TwelveDataIntradayClient(api_key="secret").fetch(
            base_currency="usd",
            quote_currency="jpy",
            target_date=date(2026, 7, 24),
        )

    assert len(points) == 1
    assert points[0].close_rate == Decimal("146.150")
    request_params = get.call_args.kwargs["params"]
    assert request_params["symbol"] == "USD/JPY"
    assert request_params["date"] == "2026-07-24"
    assert request_params["order"] == "asc"


def test_renderer_writes_expected_png(tmp_path: Path) -> None:
    """
    验证绘图器生成指定尺寸的 PNG。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 本测试无返回值。
    """

    output_path = tmp_path / "chart.png"
    result = ExchangeRateChartRenderer().render(
        points=_points(),
        pair="USD / JPY",
        pair_note="美元兑日元",
        mode=CHART_MODES[0],
        timezone_label="Tokyo time",
        output_path=output_path,
    )

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.size == (1600, 900)
        assert image.format == "PNG"


def test_client_fetches_time_range() -> None:
    """
    验证客户端使用开始和结束时间查询滚动区间。

    Returns:
        None: 本测试无返回值。
    """

    response = _response(
        {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-07-24 12:00:00",
                    "open": "146.100",
                    "high": "146.200",
                    "low": "146.050",
                    "close": "146.150",
                }
            ],
        }
    )
    with mock.patch(
        "src.exchange_rate_chart.requests.get",
        return_value=response,
    ) as get:
        TwelveDataIntradayClient(api_key="secret").fetch_range(
            base_currency="USD",
            quote_currency="JPY",
            start_time=datetime(2026, 7, 23, 12, 0),
            end_time=datetime(2026, 7, 24, 12, 0),
            interval="5min",
        )

    request_params = get.call_args.kwargs["params"]
    assert request_params["start_date"] == "2026-07-23 12:00:00"
    assert request_params["end_date"] == "2026-07-24 12:00:00"
    assert "date" not in request_params


def test_client_parses_daily_date() -> None:
    """
    验证日线数据仅包含日期时仍能解析。

    Returns:
        None: 本测试无返回值。
    """

    response = _response(
        {
            "status": "ok",
            "values": [
                {
                    "datetime": "2026-07-24",
                    "open": "146.100",
                    "high": "146.200",
                    "low": "146.050",
                    "close": "146.150",
                }
            ],
        }
    )
    with mock.patch(
        "src.exchange_rate_chart.requests.get",
        return_value=response,
    ):
        points = TwelveDataIntradayClient(api_key="secret").fetch_range(
            base_currency="USD",
            quote_currency="JPY",
            start_time=datetime(2025, 7, 24),
            end_time=datetime(2026, 7, 24),
            interval="1day",
        )

    assert points[0].timestamp == datetime(2026, 7, 24)


def test_series_builder_forward_fills_missing_buckets() -> None:
    """
    验证滚动序列固定时间桶并沿用上一笔有效汇率。

    Returns:
        None: 本测试无返回值。
    """

    mode = ExchangeRateChartMode(
        name="Test",
        window=timedelta(minutes=20),
        bucket=timedelta(minutes=5),
        api_interval="5min",
        interval_label="5 分钟",
    )
    raw_points = [
        IntradayRatePoint(
            timestamp=datetime(2026, 7, 24, 0, 0),
            open_rate=Decimal("146.000"),
            high_rate=Decimal("146.000"),
            low_rate=Decimal("146.000"),
            close_rate=Decimal("146.000"),
        ),
        IntradayRatePoint(
            timestamp=datetime(2026, 7, 24, 0, 10),
            open_rate=Decimal("146.000"),
            high_rate=Decimal("146.100"),
            low_rate=Decimal("146.000"),
            close_rate=Decimal("146.100"),
        ),
    ]

    points = RollingRateSeriesBuilder().build(
        raw_points=raw_points,
        mode=mode,
        current_time=datetime(2026, 7, 24, 0, 20),
    )

    assert len(points) == 4
    assert [point.timestamp.minute for point in points] == [5, 10, 15, 20]
    assert [point.close_rate for point in points] == [
        Decimal("146.000"),
        Decimal("146.100"),
        Decimal("146.100"),
        Decimal("146.100"),
    ]


def test_renderer_supports_flat_weekend_series(tmp_path: Path) -> None:
    """
    验证休市期间的水平汇率序列可以正常绘制。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 本测试无返回值。
    """

    flat_points = [
        IntradayRatePoint(
            timestamp=datetime(2026, 7, 25, hour, 0),
            open_rate=Decimal("146.200"),
            high_rate=Decimal("146.200"),
            low_rate=Decimal("146.200"),
            close_rate=Decimal("146.200"),
        )
        for hour in (0, 12, 23)
    ]
    output_path = tmp_path / "flat.png"

    ExchangeRateChartRenderer().render(
        points=flat_points,
        pair="USD / JPY",
        pair_note="美元兑日元",
        mode=CHART_MODES[0],
        timezone_label="东京时间",
        output_path=output_path,
    )

    assert output_path.exists()
