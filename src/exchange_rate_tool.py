"""最新汇率查询工具支持模块。"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field


_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class ExchangeRateToolInput(BaseModel):
    """
    定义 LangGraph 汇率工具接收的参数。

    Args:
        base_currency (str): 原始货币代码，例如 ``USD``。
        quote_currency (str): 目标货币代码，例如 ``JPY``。
        amount (str | int | float): 待换算金额。

    Returns:
        ExchangeRateToolInput: 工具入参对象。

    Raises:
        pydantic.ValidationError: 当工具参数类型无法解析时抛出。
    """

    base_currency: str = Field(
        default="",
        description="原始货币的三位代码，例如 USD。",
    )
    quote_currency: str = Field(
        default="",
        description="目标货币的三位代码，例如 JPY。",
    )
    amount: str | int | float = Field(
        default="1",
        description="待换算金额，例如 100 或 12.5。",
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class ExchangeRateClient:
    """
    Frankfurter 最新汇率 API 客户端。

    该客户端不缓存、不重试，也不使用备用数据源；预期错误统一返回 JSON
    对象，交给 Agent 继续处理。

    Args:
        base_url (str | None): API 基础地址，可为空以使用默认地址。
        timeout (float): HTTP 请求超时时间，单位为秒。

    Raises:
        AssertionError: 当 base_url 为空或 timeout 不是正数时抛出。
    """

    _DEFAULT_BASE_URL = "https://api.frankfurter.dev/v2"

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        """
        初始化汇率 API 客户端。

        Args:
            base_url (str | None): API 基础地址，可为空以使用默认地址。
            timeout (float): HTTP 请求超时时间，必须为正数。

        Returns:
            None

        Raises:
            AssertionError: 当配置参数不合法时抛出。
        """

        used_base_url = base_url or self._DEFAULT_BASE_URL
        assert isinstance(used_base_url, str) and used_base_url.strip(), (
            "base_url 必须是非空字符串。"
        )
        assert isinstance(timeout, (int, float)) and timeout > 0, (
            "timeout 必须是正数。"
        )
        self._base_url = used_base_url.rstrip("/")
        self._timeout = float(timeout)

    def query(
        self,
        base_currency: str,
        quote_currency: str,
        amount: str | int | float = "1",
    ) -> dict[str, Any]:
        """
        查询最新汇率并返回统一结构的结果。

        Args:
            base_currency (str): 原始货币代码。
            quote_currency (str): 目标货币代码。
            amount (str | int | float): 待换算金额，默认值为 1。

        Returns:
            dict[str, Any]: 包含 success、data 或 error 字段的结果对象。

        Raises:
            None: 可预期的参数、网络和 API 响应错误均转换为结构化结果。
        """

        try:
            normalized_base = self._normalize_currency(
                base_currency, "base_currency"
            )
            normalized_quote = self._normalize_currency(
                quote_currency, "quote_currency"
            )
            amount_decimal = Decimal(str(amount))
        except (InvalidOperation, TypeError, ValueError) as exc:
            return self._failure("INVALID_ARGUMENT", str(exc), False)

        if not amount_decimal.is_finite() or amount_decimal <= 0:
            return self._failure(
                "INVALID_ARGUMENT",
                "amount 必须是大于 0 的有限数字。",
                False,
            )

        url = f"{self._base_url}/rate/{normalized_base}/{normalized_quote}"
        try:
            response = requests.get(url, timeout=self._timeout)
        except requests.RequestException as exc:
            return self._failure(
                "API_REQUEST_FAILED",
                f"汇率服务请求失败：{exc}",
                True,
            )

        if response.status_code != 200:
            return self._http_failure(response.status_code)

        try:
            payload = response.json(parse_float=Decimal)
        except ValueError as exc:
            return self._failure(
                "API_RESPONSE_INVALID",
                f"汇率服务返回内容不是有效 JSON：{exc}",
                False,
            )
        if not isinstance(payload, dict):
            return self._failure(
                "API_RESPONSE_INVALID",
                "汇率服务返回格式不是 JSON 对象。",
                False,
            )

        return self._format_success(
            payload=payload,
            base_currency=normalized_base,
            quote_currency=normalized_quote,
            amount=amount_decimal,
        )

    @staticmethod
    def to_json(result: dict[str, Any]) -> str:
        """
        将结构化结果序列化为发送给 Agent 的 JSON 字符串。

        Args:
            result (dict[str, Any]): 汇率查询结果对象。

        Returns:
            str: 不带 ASCII 转义的 JSON 字符串。
        """

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _normalize_currency(value: str, field_name: str) -> str:
        """
        校验并规范化货币代码。

        Args:
            value (str): 原始货币代码。
            field_name (str): 字段名，用于生成错误信息。

        Returns:
            str: 三位大写货币代码。

        Raises:
            ValueError: 当输入不是三位货币代码时抛出。
        """

        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须是字符串。")
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(f"{field_name} 必须是三位货币代码，例如 USD。")
        return normalized

    @staticmethod
    def _http_failure(status_code: int) -> dict[str, Any]:
        """
        将 HTTP 错误状态码转换为结构化错误。

        Args:
            status_code (int): API 返回的 HTTP 状态码。

        Returns:
            dict[str, Any]: 统一格式的失败结果。
        """

        if status_code == 404:
            return ExchangeRateClient._failure(
                "RATE_NOT_FOUND",
                "未找到该货币对，请检查货币代码。",
                False,
            )
        return ExchangeRateClient._failure(
            "API_REQUEST_FAILED",
            f"汇率服务返回 HTTP {status_code}。",
            status_code == 429 or status_code >= 500,
        )

    @classmethod
    def _format_success(
        cls,
        payload: dict[str, Any],
        base_currency: str,
        quote_currency: str,
        amount: Decimal,
    ) -> dict[str, Any]:
        """
        校验 API 响应并构造成功结果。

        Args:
            payload (dict[str, Any]): API 返回的 JSON 对象。
            base_currency (str): 已规范化的原始货币代码。
            quote_currency (str): 已规范化的目标货币代码。
            amount (Decimal): 已校验的换算金额。

        Returns:
            dict[str, Any]: 成功结果或 API 响应格式错误。
        """

        required_fields = {"date", "base", "quote", "rate"}
        if not required_fields.issubset(payload):
            return cls._failure(
                "API_RESPONSE_INVALID",
                "汇率服务返回内容缺少必要字段。",
                False,
            )

        response_date = payload["date"]
        response_base = payload["base"]
        response_quote = payload["quote"]
        if not all(
            isinstance(value, str) and value.strip()
            for value in (response_date, response_base, response_quote)
        ):
            return cls._failure(
                "API_RESPONSE_INVALID",
                "汇率服务返回的日期或货币代码格式异常。",
                False,
            )
        if (
            response_base.upper() != base_currency
            or response_quote.upper() != quote_currency
        ):
            return cls._failure(
                "API_RESPONSE_INVALID",
                "汇率服务返回的货币对与请求不一致。",
                False,
            )

        try:
            rate = Decimal(str(payload["rate"]))
        except (InvalidOperation, TypeError, ValueError) as exc:
            return cls._failure(
                "API_RESPONSE_INVALID",
                f"汇率服务返回的 rate 不是有效数字：{exc}",
                False,
            )
        if not rate.is_finite() or rate <= 0:
            return cls._failure(
                "API_RESPONSE_INVALID",
                "汇率服务返回的 rate 不是正有限数字。",
                False,
            )

        return {
            "success": True,
            "data": {
                "base_currency": base_currency,
                "quote_currency": quote_currency,
                "amount": cls._decimal_text(amount),
                "rate": cls._decimal_text(rate),
                "converted_amount": cls._decimal_text(amount * rate),
                "date": response_date,
            },
            "error": None,
        }

    @staticmethod
    def _decimal_text(value: Decimal) -> str:
        """
        将 Decimal 转换为普通数字字符串。

        Args:
            value (Decimal): 待格式化的 Decimal 值。

        Returns:
            str: 不带科学计数法和无意义尾随零的数字字符串。
        """

        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _failure(code: str, message: str, retryable: bool) -> dict[str, Any]:
        """
        创建统一格式的失败结果。

        Args:
            code (str): 稳定的错误代码。
            message (str): 面向 Agent 的中文错误信息。
            retryable (bool): 是否适合稍后重试。

        Returns:
            dict[str, Any]: 包含 error 字段的失败结果。
        """

        return {
            "success": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        }


__all__ = ["ExchangeRateClient", "ExchangeRateToolInput"]
