"""使用 Twelve Data 滚动汇率数据绘制现代风格 PNG Demo。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageFont


_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class IntradayRatePoint:
    """
    表示一条分时汇率 OHLC 数据。

    Args:
        timestamp (datetime): 数据点时间。
        open_rate (Decimal): 开盘汇率。
        high_rate (Decimal): 最高汇率。
        low_rate (Decimal): 最低汇率。
        close_rate (Decimal): 收盘汇率。

    Returns:
        IntradayRatePoint: 不可变的分时汇率数据对象。
    """

    timestamp: datetime
    open_rate: Decimal
    high_rate: Decimal
    low_rate: Decimal
    close_rate: Decimal


@dataclass(frozen=True)
class ExchangeRateChartMode:
    """
    定义滚动汇率图的时间范围和采样粒度。

    Args:
        name (str): 模式名称。
        window (timedelta): 向前滚动的时间范围。
        bucket (timedelta): 单个时间桶的长度。
        api_interval (str): Twelve Data 查询粒度。
        interval_label (str): 图表展示的粒度文本。
        output_name (str): Demo 图片文件名。

    Returns:
        ExchangeRateChartMode: 不可变的图表模式配置。
    """

    name: str
    window: timedelta
    bucket: timedelta
    api_interval: str
    interval_label: str
    output_name: str

    @property
    def bucket_count(self) -> int:
        """
        计算模式对应的固定时间桶数量。

        Returns:
            int: 时间桶数量。

        Raises:
            AssertionError: 当时间范围不能被粒度整除时抛出。
        """

        window_seconds = int(self.window.total_seconds())
        bucket_seconds = int(self.bucket.total_seconds())
        assert bucket_seconds > 0, "时间桶长度必须为正数。"
        assert window_seconds % bucket_seconds == 0, (
            "时间范围必须能够被时间桶长度整除。"
        )
        return window_seconds // bucket_seconds


CHART_MODES = (
    ExchangeRateChartMode(
        name="Day",
        window=timedelta(hours=24),
        bucket=timedelta(minutes=5),
        api_interval="5min",
        interval_label="5 分钟",
        output_name="exchange_rate_chart_day.png",
    ),
    ExchangeRateChartMode(
        name="Week",
        window=timedelta(days=7),
        bucket=timedelta(minutes=30),
        api_interval="30min",
        interval_label="30 分钟",
        output_name="exchange_rate_chart_week.png",
    ),
    ExchangeRateChartMode(
        name="Month",
        window=timedelta(days=30),
        bucket=timedelta(hours=2),
        api_interval="2h",
        interval_label="2 小时",
        output_name="exchange_rate_chart_month.png",
    ),
    ExchangeRateChartMode(
        name="Year",
        window=timedelta(days=365),
        bucket=timedelta(days=1),
        api_interval="1day",
        interval_label="1 天",
        output_name="exchange_rate_chart_year.png",
    ),
)


class TwelveDataIntradayClient:
    """
    查询 Twelve Data 外汇分时数据。

    Args:
        api_key (str): Twelve Data API Key。
        base_url (str): Twelve Data API 地址。
        timeout (float): 请求超时时间，单位为秒。

    Raises:
        AssertionError: 当配置参数不合法时抛出。
    """

    _DEFAULT_BASE_URL = "https://api.twelvedata.com/time_series"
    _SUPPORTED_INTERVALS = {
        "1min",
        "5min",
        "15min",
        "30min",
        "45min",
        "1h",
        "2h",
        "4h",
        "8h",
        "1day",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        """
        初始化 Twelve Data 客户端。

        Args:
            api_key (str): Twelve Data API Key。
            base_url (str | None): API 地址，可为空以使用默认地址。
            timeout (float): 请求超时时间，必须为正数。

        Returns:
            None

        Raises:
            AssertionError: 当 API Key、地址或超时时间不合法时抛出。
        """

        used_base_url = base_url or self._DEFAULT_BASE_URL
        assert isinstance(api_key, str) and api_key.strip(), "api_key 不能为空。"
        assert isinstance(used_base_url, str) and used_base_url.strip(), (
            "base_url 不能为空。"
        )
        assert isinstance(timeout, (int, float)) and timeout > 0, (
            "timeout 必须为正数。"
        )
        self._api_key = api_key.strip()
        self._base_url = used_base_url.strip()
        self._timeout = float(timeout)

    def fetch(
        self,
        base_currency: str,
        quote_currency: str,
        target_date: date,
        interval: str = "5min",
        timezone_name: str = "Asia/Tokyo",
    ) -> list[IntradayRatePoint]:
        """
        查询指定日期的外汇分时数据。

        Args:
            base_currency (str): 原始货币代码，例如 USD。
            quote_currency (str): 目标货币代码，例如 JPY。
            target_date (date): 查询日期。
            interval (str): 数据粒度，默认 5min。
            timezone_name (str): 响应时间使用的 IANA 时区。

        Returns:
            list[IntradayRatePoint]: 按时间升序排列的数据点。

        Raises:
            AssertionError: 当查询参数不合法时抛出。
            RuntimeError: 当 API 请求失败或响应格式异常时抛出。
        """

        normalized_base = self._normalize_currency(base_currency)
        normalized_quote = self._normalize_currency(quote_currency)
        assert isinstance(target_date, date), "target_date 必须是 date。"
        assert interval in self._SUPPORTED_INTERVALS, "interval 不受支持。"
        assert isinstance(timezone_name, str) and timezone_name.strip(), (
            "timezone_name 不能为空。"
        )

        params = {
            "symbol": f"{normalized_base}/{normalized_quote}",
            "interval": interval,
            "date": target_date.isoformat(),
            "timezone": timezone_name,
            "order": "asc",
            "outputsize": 5000,
            "apikey": self._api_key,
        }
        return self._request_points(params)

    def fetch_range(
        self,
        base_currency: str,
        quote_currency: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
        timezone_name: str = "Asia/Tokyo",
    ) -> list[IntradayRatePoint]:
        """
        查询指定时间范围内的外汇数据。

        Args:
            base_currency (str): 原始货币代码，例如 USD。
            quote_currency (str): 目标货币代码，例如 JPY。
            start_time (datetime): 查询开始时间。
            end_time (datetime): 查询结束时间。
            interval (str): 数据粒度。
            timezone_name (str): 响应时间使用的 IANA 时区。

        Returns:
            list[IntradayRatePoint]: 按时间升序排列的数据点。

        Raises:
            AssertionError: 当查询参数不合法时抛出。
            RuntimeError: 当 API 请求失败或响应格式异常时抛出。
        """

        normalized_base = self._normalize_currency(base_currency)
        normalized_quote = self._normalize_currency(quote_currency)
        assert isinstance(start_time, datetime), "start_time 必须是 datetime。"
        assert isinstance(end_time, datetime), "end_time 必须是 datetime。"
        assert start_time < end_time, "start_time 必须早于 end_time。"
        assert interval in self._SUPPORTED_INTERVALS, "interval 不受支持。"
        assert isinstance(timezone_name, str) and timezone_name.strip(), (
            "timezone_name 不能为空。"
        )

        params = {
            "symbol": f"{normalized_base}/{normalized_quote}",
            "interval": interval,
            "start_date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone_name,
            "order": "asc",
            "outputsize": 5000,
            "apikey": self._api_key,
        }
        return self._request_points(params)

    def _request_points(self, params: dict[str, object]) -> list[IntradayRatePoint]:
        """
        请求并解析 Twelve Data 时间序列。

        Args:
            params (dict[str, object]): API 查询参数。

        Returns:
            list[IntradayRatePoint]: 按时间升序排列的数据点。

        Raises:
            RuntimeError: 当 API 请求失败或响应格式异常时抛出。
        """

        try:
            response = requests.get(
                self._base_url,
                params=params,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Twelve Data 请求失败：{exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(f"Twelve Data 返回 HTTP {response.status_code}。")
        try:
            payload = response.json(parse_float=Decimal)
        except ValueError as exc:
            raise RuntimeError("Twelve Data 返回内容不是有效 JSON。") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Twelve Data 返回格式不是 JSON 对象。")
        if payload.get("status") != "ok":
            message = str(payload.get("message") or "未知错误")
            raise RuntimeError(f"Twelve Data 查询失败：{message}")

        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Twelve Data 未返回任何分时汇率数据。")

        points: list[IntradayRatePoint] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise RuntimeError(f"第 {index} 条分时数据不是 JSON 对象。")
            try:
                point = self._parse_point(value)
            except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"第 {index} 条分时数据格式异常。") from exc
            points.append(point)
        return points

    @staticmethod
    def _normalize_currency(value: str) -> str:
        """
        规范化三位货币代码。

        Args:
            value (str): 原始货币代码。

        Returns:
            str: 三位大写货币代码。

        Raises:
            AssertionError: 当货币代码格式不正确时抛出。
        """

        assert isinstance(value, str), "货币代码必须是字符串。"
        normalized = value.strip().upper()
        assert _CURRENCY_PATTERN.fullmatch(normalized), (
            "货币代码必须是三位字母，例如 USD。"
        )
        return normalized

    @staticmethod
    def _parse_point(value: dict[str, Any]) -> IntradayRatePoint:
        """
        将 API 数据转换为分时汇率对象。

        Args:
            value (dict[str, Any]): API 返回的一条 OHLC 数据。

        Returns:
            IntradayRatePoint: 完成类型转换的数据点。

        Raises:
            InvalidOperation: 当汇率不是有效数字时抛出。
            KeyError: 当必要字段缺失时抛出。
            ValueError: 当时间或汇率关系不合法时抛出。
        """

        point = IntradayRatePoint(
            timestamp=TwelveDataIntradayClient._parse_timestamp(
                str(value["datetime"])
            ),
            open_rate=Decimal(str(value["open"])),
            high_rate=Decimal(str(value["high"])),
            low_rate=Decimal(str(value["low"])),
            close_rate=Decimal(str(value["close"])),
        )
        rates = (
            point.open_rate,
            point.high_rate,
            point.low_rate,
            point.close_rate,
        )
        if not all(rate.is_finite() and rate > 0 for rate in rates):
            raise ValueError("OHLC 汇率必须是正有限数。")
        if point.high_rate < max(point.open_rate, point.close_rate, point.low_rate):
            raise ValueError("最高汇率低于其它 OHLC 值。")
        if point.low_rate > min(point.open_rate, point.close_rate, point.high_rate):
            raise ValueError("最低汇率高于其它 OHLC 值。")
        return point

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        """
        解析 Twelve Data 的分时或日线时间。

        Args:
            value (str): API 返回的时间文本。

        Returns:
            datetime: 不带时区的本地行情时间。

        Raises:
            ValueError: 当时间格式不是日线或分时格式时抛出。
        """

        if len(value) == 10:
            return datetime.strptime(value, "%Y-%m-%d")
        if len(value) == 19:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        raise ValueError("行情时间格式不受支持。")


class RollingRateSeriesBuilder:
    """将原始行情整理为固定长度的滚动汇率序列。"""

    def build(
        self,
        raw_points: Sequence[IntradayRatePoint],
        mode: ExchangeRateChartMode,
        current_time: datetime,
    ) -> list[IntradayRatePoint]:
        """
        按图表模式建立固定时间桶并向前填充缺失行情。

        Args:
            raw_points (Sequence[IntradayRatePoint]): 包含窗口前参考价的行情。
            mode (ExchangeRateChartMode): 图表模式。
            current_time (datetime): 当前本地时间。

        Returns:
            list[IntradayRatePoint]: 固定数量的连续汇率时间桶。

        Raises:
            AssertionError: 当参数或数据顺序不合法时抛出。
            RuntimeError: 当窗口开始前不存在参考汇率时抛出。
        """

        assert isinstance(mode, ExchangeRateChartMode), "mode 类型不正确。"
        assert isinstance(current_time, datetime), "current_time 必须是 datetime。"
        assert raw_points, "raw_points 不能为空。"
        ordered_points = sorted(raw_points, key=lambda point: point.timestamp)
        assert list(raw_points) == ordered_points, "raw_points 必须按时间升序排列。"

        end_time = self.floor_time(current_time, mode.bucket)
        window_start = end_time - mode.window
        point_index = 0
        current_rate: Decimal | None = None
        while (
            point_index < len(ordered_points)
            and ordered_points[point_index].timestamp <= window_start
        ):
            current_rate = ordered_points[point_index].close_rate
            point_index += 1
        if current_rate is None:
            raise RuntimeError("滚动窗口开始前不存在可用于填充的参考汇率。")

        normalized_points: list[IntradayRatePoint] = []
        previous_bucket_end = window_start
        for bucket_index in range(1, mode.bucket_count + 1):
            bucket_end = window_start + mode.bucket * bucket_index
            bucket_points: list[IntradayRatePoint] = []
            while (
                point_index < len(ordered_points)
                and ordered_points[point_index].timestamp <= bucket_end
            ):
                point = ordered_points[point_index]
                if point.timestamp > previous_bucket_end:
                    bucket_points.append(point)
                point_index += 1

            if bucket_points:
                first_point = bucket_points[0]
                last_point = bucket_points[-1]
                current_rate = last_point.close_rate
                normalized_points.append(
                    IntradayRatePoint(
                        timestamp=bucket_end,
                        open_rate=first_point.open_rate,
                        high_rate=max(point.high_rate for point in bucket_points),
                        low_rate=min(point.low_rate for point in bucket_points),
                        close_rate=current_rate,
                    )
                )
            else:
                normalized_points.append(
                    IntradayRatePoint(
                        timestamp=bucket_end,
                        open_rate=current_rate,
                        high_rate=current_rate,
                        low_rate=current_rate,
                        close_rate=current_rate,
                    )
                )
            previous_bucket_end = bucket_end

        assert len(normalized_points) == mode.bucket_count, (
            "生成的时间桶数量不符合模式配置。"
        )
        return normalized_points

    @staticmethod
    def floor_time(value: datetime, bucket: timedelta) -> datetime:
        """
        将时间向下取整到指定时间桶边界。

        Args:
            value (datetime): 需要取整的时间。
            bucket (timedelta): 时间桶长度。

        Returns:
            datetime: 向下取整后的时间。

        Raises:
            AssertionError: 当时间桶长度不合法时抛出。
        """

        assert isinstance(value, datetime), "value 必须是 datetime。"
        bucket_seconds = int(bucket.total_seconds())
        assert 0 < bucket_seconds <= 86400, "时间桶长度必须在一天以内。"
        assert 86400 % bucket_seconds == 0, "时间桶必须能够整除一天。"
        seconds_since_midnight = (
            value.hour * 3600 + value.minute * 60 + value.second
        )
        floored_seconds = seconds_since_midnight - (
            seconds_since_midnight % bucket_seconds
        )
        return value.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            seconds=floored_seconds
        )


class ExchangeRateChartRenderer:
    """
    使用 Pillow 绘制现代白底滚动汇率图。

    Args:
        width (int): 图片宽度。
        height (int): 图片高度。

    Raises:
        AssertionError: 当图片尺寸过小时抛出。
    """

    _UP_COLOR = (224, 58, 64)
    _DOWN_COLOR = (24, 157, 96)

    def __init__(self, width: int = 1600, height: int = 900) -> None:
        """
        初始化绘图器。

        Args:
            width (int): 图片宽度。
            height (int): 图片高度。

        Returns:
            None

        Raises:
            AssertionError: 当图片尺寸小于设计要求时抛出。
        """

        assert width >= 1200, "width 至少为 1200。"
        assert height >= 700, "height 至少为 700。"
        self._width = width
        self._height = height

    def render(
        self,
        points: Sequence[IntradayRatePoint],
        pair: str,
        pair_note: str,
        mode: ExchangeRateChartMode,
        timezone_label: str,
        output_path: Path,
    ) -> Path:
        """
        将分时数据绘制并保存为 PNG。

        Args:
            points (Sequence[IntradayRatePoint]): 按时间排列的汇率点。
            pair (str): 图表展示的货币对。
            pair_note (str): 货币对的中文小字注解。
            mode (ExchangeRateChartMode): 图表模式。
            timezone_label (str): 时区展示文本。
            output_path (Path): PNG 输出路径。

        Returns:
            Path: 已写入的图片路径。

        Raises:
            AssertionError: 当数据不足、顺序错误或输出格式不正确时抛出。
        """

        assert len(points) >= 2, "绘图至少需要两个数据点。"
        assert isinstance(mode, ExchangeRateChartMode), "mode 类型不正确。"
        ordered_points = sorted(points, key=lambda point: point.timestamp)
        assert list(points) == ordered_points, "points 必须按时间升序排列。"
        assert output_path.suffix.lower() == ".png", "输出文件必须是 PNG。"

        image = self._create_background()
        draw = ImageDraw.Draw(image)
        self._draw_header(
            draw=draw,
            points=ordered_points,
            pair=pair,
            pair_note=pair_note,
            mode=mode,
            timezone_label=timezone_label,
        )
        self._draw_chart(image=image, draw=draw, points=ordered_points)
        self._draw_footer(draw=draw)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
        return output_path

    def _create_background(self) -> Image.Image:
        """
        创建带轻微层次的白色背景。

        Returns:
            Image.Image: RGBA 图片画布。
        """

        image = Image.new("RGBA", (self._width, self._height))
        draw = ImageDraw.Draw(image)
        top = (255, 255, 255)
        bottom = (246, 249, 253)
        for y in range(self._height):
            ratio = y / max(self._height - 1, 1)
            color = tuple(
                round(top[index] + (bottom[index] - top[index]) * ratio)
                for index in range(3)
            )
            draw.line((0, y, self._width, y), fill=(*color, 255))

        accent = Image.new("RGBA", image.size, (0, 0, 0, 0))
        accent_draw = ImageDraw.Draw(accent)
        accent_draw.ellipse(
            (self._width - 460, -280, self._width + 220, 400),
            fill=(255, 105, 105, 20),
        )
        accent_draw.ellipse(
            (-260, 520, 330, 1110),
            fill=(39, 190, 120, 16),
        )
        accent = accent.filter(ImageFilter.GaussianBlur(110))
        return Image.alpha_composite(image, accent)

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        points: Sequence[IntradayRatePoint],
        pair: str,
        pair_note: str,
        mode: ExchangeRateChartMode,
        timezone_label: str,
    ) -> None:
        """
        绘制标题、当前汇率和统计卡片。

        Args:
            draw (ImageDraw.ImageDraw): Pillow 绘图对象。
            points (Sequence[IntradayRatePoint]): 汇率数据点。
            pair (str): 货币对。
            pair_note (str): 货币对的中文小字注解。
            mode (ExchangeRateChartMode): 图表模式。
            timezone_label (str): 时区展示文本。

        Returns:
            None
        """

        title_font = self._font(56)
        subtitle_font = self._font(25)
        rate_font = self._font(66)
        change_font = self._font(28)

        draw.text(
            (82, 62),
            pair,
            font=title_font,
            fill=(27, 38, 57, 255),
            stroke_width=1,
        )
        range_text = self._range_text(points)
        subtitle = (
            f"{pair_note}  ·  {mode.name}  ·  {mode.interval_label}  ·  "
            f"{range_text}  ·  {timezone_label}"
        )
        draw.text(
            (84, 123),
            subtitle,
            font=subtitle_font,
            fill=(101, 116, 139, 255),
        )

        latest = points[-1].close_rate
        opening = points[0].open_rate
        change = latest - opening
        change_percent = change / opening * Decimal("100")
        positive = change >= 0
        trend_rgb = self._UP_COLOR if positive else self._DOWN_COLOR
        change_color = (*trend_rgb, 255)
        sign = "+" if positive else ""
        draw.text(
            (1518, 55),
            self._rate_text(latest),
            font=rate_font,
            fill=(24, 35, 52, 255),
            anchor="ra",
            stroke_width=1,
        )
        draw.text(
            (1518, 124),
            f"{sign}{change:.3f}  {sign}{change_percent:.2f}%",
            font=change_font,
            fill=change_color,
            anchor="ra",
        )

        high = max(point.high_rate for point in points)
        low = min(point.low_rate for point in points)
        stats = (
            ("起始", opening),
            ("区间最高", high),
            ("区间最低", low),
        )
        card_x = 915
        for label, value in stats:
            self._draw_stat_card(draw, card_x, 170, label, value)
            card_x += 205

    def _draw_stat_card(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        label: str,
        value: Decimal,
    ) -> None:
        """
        绘制单个汇率统计卡片。

        Args:
            draw (ImageDraw.ImageDraw): Pillow 绘图对象。
            x (int): 卡片左上角横坐标。
            y (int): 卡片左上角纵坐标。
            label (str): 统计字段标题。
            value (Decimal): 汇率值。

        Returns:
            None
        """

        draw.rounded_rectangle(
            (x, y, x + 184, y + 74),
            radius=16,
            fill=(255, 255, 255, 255),
            outline=(220, 227, 237, 255),
            width=1,
        )
        draw.text(
            (x + 16, y + 14),
            label,
            font=self._font(17),
            fill=(111, 126, 149, 255),
        )
        draw.text(
            (x + 16, y + 39),
            self._rate_text(value),
            font=self._font(28),
            fill=(35, 48, 68, 255),
            stroke_width=1,
        )

    def _draw_chart(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        points: Sequence[IntradayRatePoint],
    ) -> None:
        """
        绘制坐标网格、面积和汇率折线。

        Args:
            image (Image.Image): RGBA 图片画布。
            draw (ImageDraw.ImageDraw): Pillow 绘图对象。
            points (Sequence[IntradayRatePoint]): 汇率数据点。

        Returns:
            None
        """

        left, top, right, bottom = 120, 300, 1515, 770
        low = min(point.low_rate for point in points)
        high = max(point.high_rate for point in points)
        spread = high - low
        if spread == 0:
            padding = max(high * Decimal("0.0005"), Decimal("0.001"))
        else:
            padding = spread * Decimal("0.12")
        y_min = low - padding
        y_max = high + padding
        start_time = points[0].timestamp
        end_time = points[-1].timestamp
        total_seconds = (end_time - start_time).total_seconds()
        assert total_seconds > 0, "绘图数据时间范围必须大于零。"

        label_font = self._font(19)
        for index in range(6):
            ratio = index / 5
            y = round(top + (bottom - top) * ratio)
            draw.line(
                (left, y, right, y),
                fill=(218, 225, 235, 255),
                width=1,
            )
            rate = y_max - (y_max - y_min) * Decimal(str(ratio))
            draw.text(
                (left - 14, y),
                self._rate_text(rate),
                font=label_font,
                fill=(112, 126, 148, 255),
                anchor="rm",
            )

        for index in range(7):
            ratio = index / 6
            x = round(left + (right - left) * ratio)
            draw.line(
                (x, top, x, bottom),
                fill=(231, 236, 243, 255),
                width=1,
            )
            tick_time = start_time + (end_time - start_time) * ratio
            duration_days = total_seconds / 86400
            if duration_days <= 2:
                tick_label = tick_time.strftime("%H:%M")
            elif duration_days <= 45:
                tick_label = tick_time.strftime("%m-%d")
            else:
                tick_label = tick_time.strftime("%Y-%m")
            draw.text(
                (x, bottom + 20),
                tick_label,
                font=label_font,
                fill=(112, 126, 148, 255),
                anchor="ma",
            )

        line_points: list[tuple[int, int]] = []
        for point in points:
            x_ratio = (point.timestamp - start_time).total_seconds() / total_seconds
            y_ratio = float((y_max - point.close_rate) / (y_max - y_min))
            x = round(left + (right - left) * x_ratio)
            y = round(top + (bottom - top) * y_ratio)
            line_points.append((x, y))

        positive = points[-1].close_rate >= points[0].open_rate
        trend_rgb = self._UP_COLOR if positive else self._DOWN_COLOR

        area_mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(area_mask)
        mask_draw.polygon(
            [*line_points, (right, bottom), (left, bottom)],
            fill=255,
        )
        area_gradient = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(area_gradient)
        for y in range(top, bottom + 1):
            ratio = (y - top) / max(bottom - top, 1)
            alpha = round(88 * (1 - ratio) + 8)
            gradient_draw.line(
                (left, y, right, y),
                fill=(*trend_rgb, alpha),
            )
        image.alpha_composite(
            Image.composite(
                area_gradient,
                Image.new("RGBA", image.size, (0, 0, 0, 0)),
                area_mask,
            )
        )

        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        glow_draw.line(
            line_points,
            fill=(*trend_rgb, 72),
            width=7,
            joint="curve",
        )
        glow = glow.filter(ImageFilter.GaussianBlur(4))
        image.alpha_composite(glow)

        draw = ImageDraw.Draw(image)
        draw.line(
            line_points,
            fill=(*trend_rgb, 255),
            width=4,
            joint="curve",
        )
        latest_x, latest_y = line_points[-1]
        draw.ellipse(
            (latest_x - 11, latest_y - 11, latest_x + 11, latest_y + 11),
            fill=(255, 255, 255, 255),
            outline=(*trend_rgb, 255),
            width=4,
        )

    def _draw_footer(
        self,
        draw: ImageDraw.ImageDraw,
    ) -> None:
        """
        绘制图表脚注。

        Args:
            draw (ImageDraw.ImageDraw): Pillow 绘图对象。

        Returns:
            None
        """

        footer_font = self._font(18)
        draw.text(
            (90, 850),
            "数据来源：Twelve Data  ·  使用 OHLC 收盘价",
            font=footer_font,
            fill=(120, 133, 153, 255),
        )

    @staticmethod
    def _range_text(points: Sequence[IntradayRatePoint]) -> str:
        """
        根据时间跨度生成紧凑的区间文本。

        Args:
            points (Sequence[IntradayRatePoint]): 汇率时间序列。

        Returns:
            str: 图表标题区展示的时间范围。

        Raises:
            AssertionError: 当数据点少于两个时抛出。
        """

        assert len(points) >= 2, "生成时间范围至少需要两个数据点。"
        start_time = points[0].timestamp
        end_time = points[-1].timestamp
        duration = end_time - start_time
        if duration <= timedelta(days=2):
            return (
                f"{start_time:%m-%d %H:%M} — "
                f"{end_time:%m-%d %H:%M}"
            )
        return f"{start_time:%Y-%m-%d} — {end_time:%Y-%m-%d}"

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """
        创建支持中文的系统字体。

        Args:
            size (int): 字体像素大小。

        Returns:
            ImageFont.FreeTypeFont | ImageFont.ImageFont: 中文字体对象。

        Raises:
            AssertionError: 当字号不是正数或字体文件不存在时抛出。
        """

        assert size > 0, "字体大小必须为正数。"
        font_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
        assert font_path.exists(), f"中文字体不存在：{font_path}"
        return ImageFont.truetype(str(font_path), size=size)

    @staticmethod
    def _rate_text(value: Decimal) -> str:
        """
        将汇率格式化为三位小数。

        Args:
            value (Decimal): 汇率值。

        Returns:
            str: 三位小数汇率文本。
        """

        return f"{value:.3f}"


def latest_completed_weekday(today: date) -> date:
    """
    返回今天之前最近的工作日。

    Args:
        today (date): 当前日期。

    Returns:
        date: 今天之前最近的周一至周五日期。

    Raises:
        AssertionError: 当 today 不是 date 时抛出。
    """

    assert isinstance(today, date), "today 必须是 date。"
    candidate = today - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def build_demo() -> list[Path]:
    """
    拉取 USD/JPY 行情并生成四种滚动模式的 Demo 图片。

    Returns:
        list[Path]: Day、Week、Month、Year 模式的 PNG 路径。

    Raises:
        AssertionError: 当环境变量未配置时抛出。
        RuntimeError: 当 API 查询或图片生成失败时抛出。
    """

    load_dotenv(Path.cwd() / ".env")
    api_key = os.environ.get("TWELVE_API_KEY", "").strip()
    assert api_key, "缺少 TWELVE_API_KEY 环境变量。"

    timezone_name = "Asia/Tokyo"
    current_time = datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None)
    client = TwelveDataIntradayClient(api_key=api_key)
    series_builder = RollingRateSeriesBuilder()
    renderer = ExchangeRateChartRenderer()
    result_paths: list[Path] = []
    latest_rate: Decimal | None = None

    for mode in CHART_MODES:
        end_time = series_builder.floor_time(current_time, mode.bucket)
        query_start = end_time - mode.window - timedelta(days=7)
        raw_points = client.fetch_range(
            base_currency="USD",
            quote_currency="JPY",
            start_time=query_start,
            end_time=end_time,
            interval=mode.api_interval,
            timezone_name=timezone_name,
        )
        points = series_builder.build(
            raw_points=raw_points,
            mode=mode,
            current_time=current_time,
        )
        if latest_rate is None:
            latest_rate = points[-1].close_rate
        else:
            last_point = points[-1]
            points[-1] = IntradayRatePoint(
                timestamp=last_point.timestamp,
                open_rate=last_point.open_rate,
                high_rate=max(last_point.high_rate, latest_rate),
                low_rate=min(last_point.low_rate, latest_rate),
                close_rate=latest_rate,
            )
        output_path = Path("docs/assets") / mode.output_name
        result_path = renderer.render(
            points=points,
            pair="USD / JPY",
            pair_note="美元兑日元",
            mode=mode,
            timezone_label="东京时间",
            output_path=output_path,
        )
        result_paths.append(result_path)
        print(
            f"Generated {result_path} with {len(points)} "
            f"{mode.name} points."
        )

    return result_paths


if __name__ == "__main__":
    build_demo()
