"""为 Agent 生成滚动汇率趋势图片。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from src.exchange_rate_chart import (
    CHART_MODES,
    ExchangeRateChartMode,
    ExchangeRateChartRenderer,
    IntradayRatePoint,
    RollingRateSeriesBuilder,
    TwelveDataIntradayClient,
)


_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_MODE_BY_NAME = {mode.name.lower(): mode for mode in CHART_MODES}


class ExchangeRateTrendToolInput(BaseModel):
    """
    定义汇率趋势图工具的参数。

    Args:
        base_currency (str): 原始货币三位代码。
        quote_currency (str): 目标货币三位代码。
        mode (str): 图表模式，可为 day、week、month 或 year。

    Returns:
        ExchangeRateTrendToolInput: 工具入参对象。
    """

    base_currency: str = Field(
        default="",
        description="原始货币的三位代码，例如 USD。",
    )
    quote_currency: str = Field(
        default="",
        description="目标货币的三位代码，例如 JPY。",
    )
    mode: str = Field(
        default="day",
        description="趋势范围，仅支持 day、week、month、year。",
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class ExchangeRateTrendService:
    """
    查询滚动行情并将趋势图写入指定 generated 目录。

    Args:
        client (TwelveDataIntradayClient): Twelve Data 客户端。
        output_dir (Path): 图片输出目录。
        timezone_name (str): 图表使用的 IANA 时区。
        now_provider (Callable[[], datetime] | None): 当前时间提供器。

    Raises:
        AssertionError: 当依赖或配置不合法时抛出。
    """

    def __init__(
        self,
        client: TwelveDataIntradayClient,
        output_dir: Path,
        timezone_name: str = "Asia/Tokyo",
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        """
        初始化汇率趋势服务。

        Args:
            client (TwelveDataIntradayClient): Twelve Data 客户端。
            output_dir (Path): 图片输出目录。
            timezone_name (str): 图表使用的 IANA 时区。
            now_provider (Callable[[], datetime] | None): 当前时间提供器。

        Returns:
            None

        Raises:
            AssertionError: 当依赖或输出目录不合法时抛出。
        """

        assert isinstance(client, TwelveDataIntradayClient), "client 类型不正确。"
        assert isinstance(output_dir, Path) and output_dir.is_dir(), (
            "output_dir 必须是已存在的目录。"
        )
        assert isinstance(timezone_name, str) and timezone_name.strip(), (
            "timezone_name 不能为空。"
        )
        ZoneInfo(timezone_name)
        assert now_provider is None or callable(now_provider), (
            "now_provider 必须可调用。"
        )
        self._client = client
        self._output_dir = output_dir.resolve()
        self._timezone_name = timezone_name
        self._timezone = ZoneInfo(timezone_name)
        self._now_provider = now_provider or self._current_time
        self._series_builder = RollingRateSeriesBuilder()
        self._renderer = ExchangeRateChartRenderer()

    def generate(
        self,
        base_currency: str,
        quote_currency: str,
        mode: str,
    ) -> dict[str, Any]:
        """
        生成指定货币对和模式的趋势图。

        Args:
            base_currency (str): 原始货币三位代码。
            quote_currency (str): 目标货币三位代码。
            mode (str): day、week、month 或 year。

        Returns:
            dict[str, Any]: 成功时包含图片信息，失败时包含结构化错误。

        Raises:
            None: 可预期的参数、API、数据和文件错误均结构化返回。
        """

        try:
            normalized_base = self._normalize_currency(
                base_currency,
                "base_currency",
            )
            normalized_quote = self._normalize_currency(
                quote_currency,
                "quote_currency",
            )
            chart_mode = self._normalize_mode(mode)
        except (TypeError, ValueError) as exc:
            return self._failure("INVALID_ARGUMENT", str(exc), False)

        if normalized_base == normalized_quote:
            return self._failure(
                "INVALID_ARGUMENT",
                "base_currency 与 quote_currency 不能相同。",
                False,
            )

        current_time = self._normalized_current_time()
        end_time = self._series_builder.floor_time(
            current_time,
            chart_mode.bucket,
        )
        query_start = end_time - chart_mode.window - timedelta(days=7)
        try:
            raw_points = self._client.fetch_range(
                base_currency=normalized_base,
                quote_currency=normalized_quote,
                start_time=query_start,
                end_time=end_time,
                interval=chart_mode.api_interval,
                timezone_name=self._timezone_name,
            )
        except RuntimeError as exc:
            return self._failure("API_REQUEST_FAILED", str(exc), True)

        try:
            points = self._series_builder.build(
                raw_points=raw_points,
                mode=chart_mode,
                current_time=current_time,
            )
        except RuntimeError as exc:
            return self._failure("NO_REFERENCE_RATE", str(exc), False)

        output_path = self._output_path(
            normalized_base,
            normalized_quote,
            chart_mode,
        )
        description = self._description(
            base_currency=normalized_base,
            quote_currency=normalized_quote,
            mode=chart_mode,
            points=points,
        )
        try:
            rendered_path = self._renderer.render(
                points=points,
                pair=f"{normalized_base} / {normalized_quote}",
                pair_note="",
                mode=chart_mode,
                timezone_label="东京时间",
                output_path=output_path,
            )
        except OSError as exc:
            return self._failure("IMAGE_RENDER_FAILED", str(exc), False)

        return {
            "success": True,
            "data": {
                "path": str(rendered_path.resolve()),
                "mime_type": "image/png",
                "text": description,
                "base_currency": normalized_base,
                "quote_currency": normalized_quote,
                "mode": chart_mode.name.lower(),
                "point_count": len(points),
                "start_time": points[0].timestamp.isoformat(),
                "end_time": points[-1].timestamp.isoformat(),
            },
            "error": None,
        }

    @staticmethod
    def to_json(result: dict[str, Any]) -> str:
        """
        将趋势图结果序列化为 Agent 可读 JSON。

        Args:
            result (dict[str, Any]): 趋势图生成结果。

        Returns:
            str: 不进行 ASCII 转义的 JSON 字符串。
        """

        return json.dumps(result, ensure_ascii=False)

    def _current_time(self) -> datetime:
        """
        获取图表时区内的当前时间。

        Returns:
            datetime: 带时区的当前时间。
        """

        return datetime.now(self._timezone)

    def _normalized_current_time(self) -> datetime:
        """
        获取不带时区但已转换为图表时区的当前时间。

        Returns:
            datetime: 图表时区的本地时间。

        Raises:
            AssertionError: 当时间提供器返回值不是 datetime 时抛出。
        """

        current_time = self._now_provider()
        assert isinstance(current_time, datetime), (
            "now_provider 必须返回 datetime。"
        )
        if current_time.tzinfo is not None:
            return current_time.astimezone(self._timezone).replace(tzinfo=None)
        return current_time

    def _output_path(
        self,
        base_currency: str,
        quote_currency: str,
        mode: ExchangeRateChartMode,
    ) -> Path:
        """
        创建不会覆盖已有图片的输出路径。

        Args:
            base_currency (str): 已规范化的原始货币代码。
            quote_currency (str): 已规范化的目标货币代码。
            mode (ExchangeRateChartMode): 图表模式。

        Returns:
            Path: generated 目录中的唯一 PNG 路径。
        """

        filename = (
            f"exchange-rate-{base_currency.lower()}-"
            f"{quote_currency.lower()}-{mode.name.lower()}-"
            f"{uuid.uuid4().hex}.png"
        )
        return self._output_dir / filename

    @staticmethod
    def _description(
        base_currency: str,
        quote_currency: str,
        mode: ExchangeRateChartMode,
        points: Sequence[IntradayRatePoint],
    ) -> str:
        """
        生成可随图片发送的关键行情摘要。

        Args:
            base_currency (str): 已规范化的原始货币代码。
            quote_currency (str): 已规范化的目标货币代码。
            mode (ExchangeRateChartMode): 图表模式。
            points (Sequence[IntradayRatePoint]): 已规范化的汇率序列。

        Returns:
            str: 包含当前值、涨跌、极值和时间范围的中文描述。

        Raises:
            AssertionError: 当汇率序列为空时抛出。
        """

        assert points, "生成图片描述需要汇率数据。"
        opening = points[0].open_rate
        latest = points[-1].close_rate
        change = latest - opening
        change_percent = change / opening * Decimal("100")
        if change > 0:
            trend_text = f"上涨 +{change:.3f}（+{change_percent:.2f}%）"
        elif change < 0:
            trend_text = f"下跌 {change:.3f}（{change_percent:.2f}%）"
        else:
            trend_text = "持平 0.000（0.00%）"

        high = max(point.high_rate for point in points)
        low = min(point.low_rate for point in points)
        if mode.name == "Day":
            start_text = points[0].timestamp.strftime("%Y-%m-%d %H:%M")
            end_text = points[-1].timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            start_text = points[0].timestamp.strftime("%Y-%m-%d")
            end_text = points[-1].timestamp.strftime("%Y-%m-%d")
        return (
            f"{base_currency}/{quote_currency} {mode.name} 汇率趋势："
            f"当前 {latest:.3f}，区间{trend_text}；"
            f"最高 {high:.3f}，最低 {low:.3f}；"
            f"{start_text} 至 {end_text}（东京时间）。"
        )

    @staticmethod
    def _normalize_currency(value: str, field_name: str) -> str:
        """
        校验并规范化三位货币代码。

        Args:
            value (str): 原始货币代码。
            field_name (str): 用于错误提示的字段名。

        Returns:
            str: 三位大写货币代码。

        Raises:
            TypeError: 当货币代码不是字符串时抛出。
            ValueError: 当货币代码不是三位字母时抛出。
        """

        if not isinstance(value, str):
            raise TypeError(f"{field_name} 必须是字符串。")
        normalized = value.strip().upper()
        if not _CURRENCY_PATTERN.fullmatch(normalized):
            raise ValueError(f"{field_name} 必须是三位货币代码，例如 USD。")
        return normalized

    @staticmethod
    def _normalize_mode(value: str) -> ExchangeRateChartMode:
        """
        校验并返回图表模式。

        Args:
            value (str): 原始模式名称。

        Returns:
            ExchangeRateChartMode: 匹配的图表模式。

        Raises:
            TypeError: 当模式不是字符串时抛出。
            ValueError: 当模式不受支持时抛出。
        """

        if not isinstance(value, str):
            raise TypeError("mode 必须是字符串。")
        normalized = value.strip().lower()
        chart_mode = _MODE_BY_NAME.get(normalized)
        if chart_mode is None:
            raise ValueError("mode 仅支持 day、week、month、year。")
        return chart_mode

    @staticmethod
    def _failure(
        code: str,
        message: str,
        retryable: bool,
    ) -> dict[str, Any]:
        """
        构造统一的结构化错误。

        Args:
            code (str): 机器可读错误代码。
            message (str): 中文错误说明。
            retryable (bool): 是否适合稍后重试。

        Returns:
            dict[str, Any]: 失败结果。
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


__all__ = ["ExchangeRateTrendService", "ExchangeRateTrendToolInput"]
