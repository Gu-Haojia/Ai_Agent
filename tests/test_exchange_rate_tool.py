"""最新汇率工具单元测试。"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import requests

from src.exchange_rate_tool import ExchangeRateClient


def _response(
    *,
    status_code: int = 200,
    payload: object | None = None,
) -> mock.Mock:
    """
    构造不访问网络的 HTTP 响应替身。

    Args:
        status_code (int): HTTP 状态码。
        payload (object | None): ``response.json()`` 的返回值。

    Returns:
        mock.Mock: requests.Response 的替身对象。

    Raises:
        AssertionError: 当状态码不是正整数时抛出。
    """

    assert status_code > 0, "status_code 必须是正整数。"
    response = mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    return response


class ExchangeRateClientTests(unittest.TestCase):
    """验证汇率客户端的参数校验、响应解析和结构化错误。"""

    def test_query_success_returns_structured_result(self) -> None:
        """
        验证成功响应包含换算结果和数据日期。

        Returns:
            None: 本测试无返回值。
        """

        response = _response(
            payload={
                "date": "2026-07-22",
                "base": "USD",
                "quote": "JPY",
                "rate": 158.2,
            }
        )
        with mock.patch(
            "src.exchange_rate_tool.requests.get", return_value=response
        ) as get:
            result = ExchangeRateClient().query("usd", "jpy", "100")

        self.assertTrue(result["success"])
        self.assertIsNotNone(result["data"])
        data = result["data"]
        assert isinstance(data, dict)
        self.assertEqual(data["base_currency"], "USD")
        self.assertEqual(data["quote_currency"], "JPY")
        self.assertEqual(data["amount"], "100")
        self.assertEqual(data["rate"], "158.2")
        self.assertEqual(data["converted_amount"], "15820")
        self.assertEqual(data["date"], "2026-07-22")
        get.assert_called_once_with(
            "https://api.frankfurter.dev/v2/rate/USD/JPY",
            timeout=10.0,
        )

    def test_query_invalid_argument_returns_structured_error(self) -> None:
        """
        验证非法货币代码不会抛出异常，而是返回参数错误。

        Returns:
            None: 本测试无返回值。
        """

        result = ExchangeRateClient().query("US", "JPY", "100")

        self.assertFalse(result["success"])
        error = result["error"]
        assert isinstance(error, dict)
        self.assertEqual(error["code"], "INVALID_ARGUMENT")
        self.assertFalse(error["retryable"])

    def test_query_nonpositive_amount_returns_structured_error(self) -> None:
        """
        验证非正金额会返回参数错误。

        Returns:
            None: 本测试无返回值。
        """

        result = ExchangeRateClient().query("USD", "JPY", "0")

        self.assertFalse(result["success"])
        error = result["error"]
        assert isinstance(error, dict)
        self.assertEqual(error["code"], "INVALID_ARGUMENT")

    def test_query_http_404_returns_rate_not_found(self) -> None:
        """
        验证 API 找不到货币对时返回明确错误代码。

        Returns:
            None: 本测试无返回值。
        """

        response = _response(status_code=404)
        with mock.patch("src.exchange_rate_tool.requests.get", return_value=response):
            result = ExchangeRateClient().query("USD", "ABC", "1")

        self.assertFalse(result["success"])
        error = result["error"]
        assert isinstance(error, dict)
        self.assertEqual(error["code"], "RATE_NOT_FOUND")
        self.assertFalse(error["retryable"])

    def test_query_network_error_returns_retryable_error(self) -> None:
        """
        验证网络异常会转换为可重试的结构化错误。

        Returns:
            None: 本测试无返回值。
        """

        with mock.patch(
            "src.exchange_rate_tool.requests.get",
            side_effect=requests.Timeout("timeout"),
        ):
            result = ExchangeRateClient().query("USD", "JPY", "1")

        self.assertFalse(result["success"])
        error = result["error"]
        assert isinstance(error, dict)
        self.assertEqual(error["code"], "API_REQUEST_FAILED")
        self.assertTrue(error["retryable"])

    def test_query_invalid_json_returns_structured_error(self) -> None:
        """
        验证 API 返回非 JSON 内容时不会向 Agent 抛出解析异常。

        Returns:
            None: 本测试无返回值。
        """

        response = _response()
        response.json.side_effect = ValueError("invalid json")
        with mock.patch("src.exchange_rate_tool.requests.get", return_value=response):
            result = ExchangeRateClient().query("USD", "JPY", "1")

        self.assertFalse(result["success"])
        error = result["error"]
        assert isinstance(error, dict)
        self.assertEqual(error["code"], "API_RESPONSE_INVALID")

    def test_query_missing_response_field_returns_structured_error(self) -> None:
        """
        验证 API 缺少必要字段时返回响应格式错误。

        Returns:
            None: 本测试无返回值。
        """

        response = _response(
            payload={
                "date": "2026-07-22",
                "base": "USD",
                "quote": "JPY",
            }
        )
        with mock.patch("src.exchange_rate_tool.requests.get", return_value=response):
            result = ExchangeRateClient().query("USD", "JPY", "1")

        self.assertFalse(result["success"])
        error = result["error"]
        assert isinstance(error, dict)
        self.assertEqual(error["code"], "API_RESPONSE_INVALID")

    def test_result_to_json_is_agent_readable(self) -> None:
        """
        验证结果可以序列化为中文 JSON 对象。

        Returns:
            None: 本测试无返回值。
        """

        result = ExchangeRateClient().query("US", "JPY", "1")
        payload = json.loads(ExchangeRateClient.to_json(result))

        self.assertFalse(payload["success"])
        self.assertIn("error", payload)
        self.assertIn("code", payload["error"])


if __name__ == "__main__":
    unittest.main()
