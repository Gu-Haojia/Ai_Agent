"""使用 Twelve Data 分时汇率绘制现代风格 PNG Demo。"""

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
        assert interval in {"1min", "5min", "15min", "30min", "45min", "1h"}, (
            "interval 不受支持。"
        )
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
            timestamp=datetime.strptime(
                str(value["datetime"]),
                "%Y-%m-%d %H:%M:%S",
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


class ExchangeRateChartRenderer:
    """
    使用 Pillow 绘制现代深色分时汇率图。

    Args:
        width (int): 图片宽度。
        height (int): 图片高度。

    Raises:
        AssertionError: 当图片尺寸过小时抛出。
    """

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
        interval: str,
        timezone_label: str,
        output_path: Path,
    ) -> Path:
        """
        将分时数据绘制并保存为 PNG。

        Args:
            points (Sequence[IntradayRatePoint]): 按时间排列的汇率点。
            pair (str): 图表展示的货币对。
            interval (str): 数据粒度。
            timezone_label (str): 时区展示文本。
            output_path (Path): PNG 输出路径。

        Returns:
            Path: 已写入的图片路径。

        Raises:
            AssertionError: 当数据不足、顺序错误或输出格式不正确时抛出。
        """

        assert len(points) >= 2, "绘图至少需要两个数据点。"
        ordered_points = sorted(points, key=lambda point: point.timestamp)
        assert list(points) == ordered_points, "points 必须按时间升序排列。"
        assert output_path.suffix.lower() == ".png", "输出文件必须是 PNG。"

        image = self._create_background()
        draw = ImageDraw.Draw(image)
        self._draw_header(
            draw=draw,
            points=ordered_points,
            pair=pair,
            interval=interval,
            timezone_label=timezone_label,
        )
        self._draw_chart(image=image, draw=draw, points=ordered_points)
        self._draw_footer(draw=draw, points=ordered_points)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
        return output_path

    def _create_background(self) -> Image.Image:
        """
        创建带纵向渐变的深色背景。

        Returns:
            Image.Image: RGBA 图片画布。
        """

        image = Image.new("RGBA", (self._width, self._height))
        draw = ImageDraw.Draw(image)
        top = (8, 14, 28)
        bottom = (17, 26, 47)
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
            fill=(27, 185, 255, 55),
        )
        accent = accent.filter(ImageFilter.GaussianBlur(110))
        return Image.alpha_composite(image, accent)

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        points: Sequence[IntradayRatePoint],
        pair: str,
        interval: str,
        timezone_label: str,
    ) -> None:
        """
        绘制标题、当前汇率和统计卡片。

        Args:
            draw (ImageDraw.ImageDraw): Pillow 绘图对象。
            points (Sequence[IntradayRatePoint]): 汇率数据点。
            pair (str): 货币对。
            interval (str): 数据粒度。
            timezone_label (str): 时区展示文本。

        Returns:
            None
        """

        title_font = self._font(48)
        subtitle_font = self._font(21)
        rate_font = self._font(58)
        change_font = self._font(23)
        badge_font = self._font(16)

        draw.text(
            (82, 62),
            pair,
            font=title_font,
            fill=(245, 249, 255, 255),
            stroke_width=1,
        )
        subtitle = (
            f"Intraday exchange rate  ·  {interval}  ·  "
            f"{points[0].timestamp:%Y-%m-%d}  ·  {timezone_label}"
        )
        draw.text(
            (84, 123),
            subtitle,
            font=subtitle_font,
            fill=(139, 156, 186, 255),
        )

        draw.rounded_rectangle(
            (82, 171, 173, 207),
            radius=18,
            fill=(18, 91, 126, 255),
            outline=(67, 202, 255, 210),
            width=1,
        )
        draw.text(
            (127, 189),
            "REAL DATA",
            font=badge_font,
            fill=(92, 215, 255, 255),
            anchor="mm",
        )

        latest = points[-1].close_rate
        opening = points[0].open_rate
        change = latest - opening
        change_percent = change / opening * Decimal("100")
        positive = change >= 0
        change_color = (
            (51, 214, 159, 255) if positive else (255, 107, 122, 255)
        )
        sign = "+" if positive else ""
        draw.text(
            (1518, 55),
            self._rate_text(latest),
            font=rate_font,
            fill=(247, 250, 255, 255),
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
            ("OPEN", opening),
            ("DAY HIGH", high),
            ("DAY LOW", low),
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
            fill=(21, 34, 57, 255),
            outline=(76, 100, 136, 180),
            width=1,
        )
        draw.text(
            (x + 16, y + 14),
            label,
            font=self._font(14),
            fill=(124, 143, 176, 255),
        )
        draw.text(
            (x + 16, y + 39),
            self._rate_text(value),
            font=self._font(24),
            fill=(229, 238, 251, 255),
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

        left, top, right, bottom = 90, 300, 1515, 770
        low = min(point.low_rate for point in points)
        high = max(point.high_rate for point in points)
        spread = high - low
        assert spread > 0, "绘图数据必须存在价格波动。"
        padding = spread * Decimal("0.12")
        y_min = low - padding
        y_max = high + padding
        start_time = points[0].timestamp
        end_time = points[-1].timestamp
        total_seconds = (end_time - start_time).total_seconds()
        assert total_seconds > 0, "绘图数据时间范围必须大于零。"

        label_font = self._font(16)
        for index in range(6):
            ratio = index / 5
            y = round(top + (bottom - top) * ratio)
            draw.line(
                (left, y, right, y),
                fill=(139, 161, 197, 28),
                width=1,
            )
            rate = y_max - (y_max - y_min) * Decimal(str(ratio))
            draw.text(
                (left - 14, y),
                self._rate_text(rate),
                font=label_font,
                fill=(112, 132, 165, 255),
                anchor="rm",
            )

        for index in range(7):
            ratio = index / 6
            x = round(left + (right - left) * ratio)
            draw.line(
                (x, top, x, bottom),
                fill=(139, 161, 197, 20),
                width=1,
            )
            tick_time = start_time + (end_time - start_time) * ratio
            rounded_minutes = round(
                (
                    tick_time.hour * 60
                    + tick_time.minute
                    + tick_time.second / 60
                )
                / 5
            ) * 5
            rounded_minutes = min(rounded_minutes, 23 * 60 + 55)
            tick_label = (
                f"{rounded_minutes // 60:02d}:{rounded_minutes % 60:02d}"
            )
            draw.text(
                (x, bottom + 20),
                tick_label,
                font=label_font,
                fill=(112, 132, 165, 255),
                anchor="ma",
            )

        line_points: list[tuple[int, int]] = []
        for point in points:
            x_ratio = (point.timestamp - start_time).total_seconds() / total_seconds
            y_ratio = float((y_max - point.close_rate) / (y_max - y_min))
            x = round(left + (right - left) * x_ratio)
            y = round(top + (bottom - top) * y_ratio)
            line_points.append((x, y))

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
            alpha = round(92 * (1 - ratio))
            gradient_draw.line(
                (left, y, right, y),
                fill=(38, 196, 255, alpha),
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
            fill=(28, 194, 255, 165),
            width=14,
            joint="curve",
        )
        glow = glow.filter(ImageFilter.GaussianBlur(12))
        image.alpha_composite(glow)

        draw = ImageDraw.Draw(image)
        draw.line(
            line_points,
            fill=(81, 214, 255, 255),
            width=4,
            joint="curve",
        )
        latest_x, latest_y = line_points[-1]
        draw.ellipse(
            (latest_x - 11, latest_y - 11, latest_x + 11, latest_y + 11),
            fill=(11, 24, 44, 255),
            outline=(90, 220, 255, 255),
            width=4,
        )

    def _draw_footer(
        self,
        draw: ImageDraw.ImageDraw,
        points: Sequence[IntradayRatePoint],
    ) -> None:
        """
        绘制图表脚注。

        Args:
            draw (ImageDraw.ImageDraw): Pillow 绘图对象。
            points (Sequence[IntradayRatePoint]): 汇率数据点。

        Returns:
            None
        """

        footer_font = self._font(15)
        draw.text(
            (90, 850),
            "Source: Twelve Data  ·  OHLC close values",
            font=footer_font,
            fill=(98, 119, 152, 255),
        )
        draw.text(
            (1515, 850),
            f"{len(points)} data points  ·  Deterministic render",
            font=footer_font,
            fill=(98, 119, 152, 255),
            anchor="ra",
        )

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """
        创建 Pillow 内置字体。

        Args:
            size (int): 字体像素大小。

        Returns:
            ImageFont.FreeTypeFont | ImageFont.ImageFont: 可用于绘图的字体。

        Raises:
            AssertionError: 当字号不是正数时抛出。
        """

        assert size > 0, "字体大小必须为正数。"
        return ImageFont.load_default(size=size)

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


def build_demo() -> Path:
    """
    拉取最近工作日的 USD/JPY 分时数据并生成 Demo 图片。

    Returns:
        Path: 生成的 PNG 路径。

    Raises:
        AssertionError: 当环境变量未配置时抛出。
        RuntimeError: 当 API 查询或图片生成失败时抛出。
    """

    load_dotenv(Path.cwd() / ".env")
    api_key = os.environ.get("TWELVE_API_KEY", "").strip()
    assert api_key, "缺少 TWELVE_API_KEY 环境变量。"

    timezone_name = "Asia/Tokyo"
    target_date = latest_completed_weekday(
        datetime.now(ZoneInfo(timezone_name)).date()
    )
    client = TwelveDataIntradayClient(api_key=api_key)
    points = client.fetch(
        base_currency="USD",
        quote_currency="JPY",
        target_date=target_date,
        interval="5min",
        timezone_name=timezone_name,
    )
    renderer = ExchangeRateChartRenderer()
    output_path = Path("docs/assets/exchange_rate_chart_demo.png")
    result_path = renderer.render(
        points=points,
        pair="USD / JPY",
        interval="5 min",
        timezone_label="Tokyo time",
        output_path=output_path,
    )
    print(
        f"Generated {result_path} with {len(points)} points "
        f"for {target_date.isoformat()}."
    )
    return result_path


if __name__ == "__main__":
    build_demo()
