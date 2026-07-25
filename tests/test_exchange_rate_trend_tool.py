"""汇率趋势图片工具的定向测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest import mock

import requests

from src.exchange_rate_chart import TwelveDataIntradayClient
from src.exchange_rate_trend_tool import ExchangeRateTrendService


def _response(
    *,
    status_code: int = 200,
    payload: object | None = None,
) -> mock.Mock:
    """
    构造 Twelve Data HTTP 响应替身。

    Args:
        status_code (int): HTTP 状态码。
        payload (object | None): JSON 响应内容。

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


def _trend_payload() -> dict[str, object]:
    """
    创建包含窗口参考价和真实行情的 API 数据。

    Returns:
        dict[str, object]: Twelve Data 成功响应。
    """

    return {
        "status": "ok",
        "values": [
            {
                "datetime": "2026-07-24 22:45:00",
                "open": "146.100",
                "high": "146.100",
                "low": "146.100",
                "close": "146.100",
            },
            {
                "datetime": "2026-07-25 05:55:00",
                "open": "146.100",
                "high": "146.250",
                "low": "146.050",
                "close": "146.200",
            },
        ],
    }


def test_generate_returns_image_tool_compatible_result(tmp_path: Path) -> None:
    """
    验证成功结果包含生图工具需要的图片字段。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 本测试无返回值。
    """

    service = ExchangeRateTrendService(
        client=TwelveDataIntradayClient(api_key="secret"),
        output_dir=tmp_path,
        now_provider=lambda: datetime(2026, 7, 25, 22, 47),
    )
    with mock.patch(
        "src.exchange_rate_chart.requests.get",
        return_value=_response(payload=_trend_payload()),
    ) as get:
        result = service.generate("usd", "jpy", "day")

    assert result["success"] is True
    data = result["data"]
    assert isinstance(data, dict)
    assert data["mime_type"] == "image/png"
    text = str(data["text"])
    assert text.startswith("USD/JPY Day 汇率趋势：")
    assert "当前 146.200" in text
    assert "区间上涨 +0.100（+0.07%）" in text
    assert "最高 146.250，最低 146.050" in text
    assert "2026-07-24 22:50 至 2026-07-25 22:45（东京时间）" in text
    assert data["mode"] == "day"
    assert data["point_count"] == 288
    output_path = Path(str(data["path"]))
    assert output_path.is_file()
    assert output_path.parent == tmp_path.resolve()
    request_params = get.call_args.kwargs["params"]
    assert request_params["symbol"] == "USD/JPY"
    assert request_params["interval"] == "5min"
    assert request_params["end_date"] == "2026-07-25 22:45:00"


def test_generate_rejects_invalid_mode_without_api_request(tmp_path: Path) -> None:
    """
    验证不支持的模式直接返回结构化参数错误。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 本测试无返回值。
    """

    service = ExchangeRateTrendService(
        client=TwelveDataIntradayClient(api_key="secret"),
        output_dir=tmp_path,
    )
    with mock.patch("src.exchange_rate_chart.requests.get") as get:
        result = service.generate("USD", "JPY", "quarter")

    assert result["success"] is False
    error = result["error"]
    assert isinstance(error, dict)
    assert error["code"] == "INVALID_ARGUMENT"
    assert error["retryable"] is False
    get.assert_not_called()


def test_generate_returns_structured_api_error(tmp_path: Path) -> None:
    """
    验证行情 API 失败时返回可重试的结构化错误。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 本测试无返回值。
    """

    service = ExchangeRateTrendService(
        client=TwelveDataIntradayClient(api_key="secret"),
        output_dir=tmp_path,
        now_provider=lambda: datetime(2026, 7, 25, 22, 47),
    )
    with mock.patch(
        "src.exchange_rate_chart.requests.get",
        return_value=_response(status_code=503),
    ):
        result = service.generate("USD", "JPY", "week")

    assert result["success"] is False
    error = result["error"]
    assert isinstance(error, dict)
    assert error["code"] == "API_REQUEST_FAILED"
    assert error["retryable"] is True


def test_generate_rejects_same_currency_pair(tmp_path: Path) -> None:
    """
    验证相同货币代码不会生成无意义趋势图。

    Args:
        tmp_path (Path): pytest 临时目录。

    Returns:
        None: 本测试无返回值。
    """

    service = ExchangeRateTrendService(
        client=TwelveDataIntradayClient(api_key="secret"),
        output_dir=tmp_path,
    )

    result = service.generate("USD", "usd", "month")

    assert result["success"] is False
    error = result["error"]
    assert isinstance(error, dict)
    assert error["code"] == "INVALID_ARGUMENT"
