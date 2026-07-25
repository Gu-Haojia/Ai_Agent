"""汇率分时图 Demo 的单元测试。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest import mock

import requests
from PIL import Image

from src.exchange_rate_chart_demo import (
    ExchangeRateChartRenderer,
    IntradayRatePoint,
    TwelveDataIntradayClient,
    latest_completed_weekday,
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
        "src.exchange_rate_chart_demo.requests.get",
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
        interval="5 min",
        timezone_label="Tokyo time",
        output_path=output_path,
    )

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.size == (1600, 900)
        assert image.format == "PNG"


def test_latest_completed_weekday_skips_weekend() -> None:
    """
    验证周一会选择上一个周五作为完整交易日。

    Returns:
        None: 本测试无返回值。
    """

    assert latest_completed_weekday(date(2026, 7, 27)) == date(2026, 7, 24)
