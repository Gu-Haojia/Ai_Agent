"""Token 消费静态图表测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.token_usage_chart import (
    CHART_FONT_FAMILY,
    TokenUsageChartBuilder,
    TokenUsageChartRenderer,
    TokenUsageGranularity,
)
from src.token_usage_logger import (
    TokenUsageRecord,
    TokenUsageReport,
    TokenUsageSummary,
)

JST = timezone(timedelta(hours=9))


def _record(
    recorded_at: datetime,
    model_name: str,
    total_tokens: int,
) -> TokenUsageRecord:
    """构造图表测试使用的 Token 记录。

    Args:
        recorded_at (datetime): 调用完成时间。
        model_name (str): 模型名称。
        total_tokens (int): 总 Token 数。

    Returns:
        TokenUsageRecord: 测试记录。

    Raises:
        None: 本函数不主动抛出异常。
    """
    output_tokens = total_tokens // 10
    return TokenUsageRecord(
        recorded_at=recorded_at,
        model_name=model_name,
        total_tokens=total_tokens,
        input_tokens=total_tokens - output_tokens,
        cache_read=total_tokens // 2,
        output_tokens=output_tokens,
    )


def _report(records: tuple[TokenUsageRecord, ...]) -> TokenUsageReport:
    """根据测试记录构造 Token 报告。

    Args:
        records (tuple[TokenUsageRecord, ...]): 测试记录。

    Returns:
        TokenUsageReport: 汇总后的测试报告。

    Raises:
        AssertionError: 当记录为空时抛出。
    """
    assert records, "测试报告必须包含记录"
    ordered = tuple(sorted(records, key=lambda record: record.recorded_at))
    summary = TokenUsageSummary(
        start_time=ordered[0].recorded_at,
        total_tokens=sum(record.total_tokens for record in ordered),
        input_tokens=sum(record.input_tokens for record in ordered),
        cache_read=sum(record.cache_read for record in ordered),
        output_tokens=sum(record.output_tokens for record in ordered),
    )
    return TokenUsageReport(summary, ordered, ordered[-1].recorded_at)


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        (timedelta(days=3), TokenUsageGranularity.HOUR),
        (timedelta(days=30), TokenUsageGranularity.DAY),
        (timedelta(days=190), TokenUsageGranularity.WEEK),
        (timedelta(days=720), TokenUsageGranularity.MONTH),
    ],
)
def test_chart_builder_selects_granularity_from_span(
    span: timedelta,
    expected: TokenUsageGranularity,
) -> None:
    """验证图表粒度由实际数据跨度决定。

    Args:
        span (timedelta): 测试数据跨度。
        expected (TokenUsageGranularity): 预期粒度。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    start = datetime(2026, 1, 1, tzinfo=JST)

    result = TokenUsageChartBuilder.select_granularity(start, start + span)

    assert result is expected


def test_chart_builder_aggregates_daily_model_usage() -> None:
    """验证图表构建器按日与模型聚合记录。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    start = datetime(2026, 8, 1, 10, tzinfo=JST)
    report = _report(
        (
            _record(start, "flash", 100),
            _record(start + timedelta(hours=2), "pro", 40),
            _record(start + timedelta(days=8), "flash", 200),
        )
    )

    chart = TokenUsageChartBuilder().build(report)

    assert chart.granularity is TokenUsageGranularity.DAY
    assert chart.model_names == ("flash", "pro")
    assert chart.model_totals == (300, 40)
    assert len(chart.points) == 9
    assert chart.points[0].model_tokens == (100, 40)
    assert all(point.total_tokens == 0 for point in chart.points[1:-1])
    assert chart.points[-1].model_tokens == (200, 0)
    assert chart.points[-1].average_tokens == pytest.approx(200 / 7)


def test_chart_builder_fills_missing_hour_buckets_with_zero() -> None:
    """验证缺失小时补零并参与连续四小时均线。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    start = datetime(2026, 8, 11, 1, tzinfo=JST)
    report = _report(
        (
            _record(start, "flash", 100),
            _record(start + timedelta(hours=3), "flash", 300),
        )
    )

    chart = TokenUsageChartBuilder().build(report)

    assert chart.granularity is TokenUsageGranularity.HOUR
    assert [point.total_tokens for point in chart.points] == [100, 0, 0, 300]
    assert chart.points[-1].average_tokens == 100


@pytest.mark.parametrize(
    ("start", "end", "expected_granularity", "expected_count"),
    [
        (
            datetime(2026, 1, 5, tzinfo=JST),
            datetime(2026, 4, 13, tzinfo=JST),
            TokenUsageGranularity.WEEK,
            15,
        ),
        (
            datetime(2025, 1, 1, tzinfo=JST),
            datetime(2026, 7, 1, tzinfo=JST),
            TokenUsageGranularity.MONTH,
            19,
        ),
    ],
)
def test_chart_builder_fills_week_and_month_buckets(
    start: datetime,
    end: datetime,
    expected_granularity: TokenUsageGranularity,
    expected_count: int,
) -> None:
    """验证周和月粒度同样补齐中间的空时间桶。

    Args:
        start (datetime): 第一条记录时间。
        end (datetime): 最后一条记录时间。
        expected_granularity (TokenUsageGranularity): 预期聚合粒度。
        expected_count (int): 首尾之间应生成的时间桶数量。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    chart = TokenUsageChartBuilder().build(
        _report(
            (
                _record(start, "flash", 100),
                _record(end, "flash", 200),
            )
        )
    )

    assert chart.granularity is expected_granularity
    assert len(chart.points) == expected_count
    assert all(point.total_tokens == 0 for point in chart.points[1:-1])


def test_chart_renderer_embeds_local_d3_and_fixed_font() -> None:
    """验证静态页面使用本地 D3、指定字体与实心饼图。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    start = datetime(2026, 8, 10, 10, tzinfo=JST)
    chart = TokenUsageChartBuilder().build(
        _report((_record(start, "flash", 100),))
    )

    html = TokenUsageChartRenderer()._render_html(chart)

    assert '<script src="' not in html
    assert f'font-family:"{CHART_FONT_FAMILY}"' in html
    assert "innerRadius(0)" in html
    assert "自动粒度" in html
    assert '.token-value{color:#9aa0a6' in html
    assert 'attr("class","token-value").text(d=>compact(d.value))' in html


def test_chart_renderer_outputs_fixed_png() -> None:
    """验证 Chromium 能够生成固定尺寸 PNG。

    Returns:
        None: 测试完成后不返回额外值。

    Raises:
        None: 预期行为由断言验证。
    """
    start = datetime(2026, 8, 10, 10, tzinfo=JST)
    report = _report((_record(start, "flash", 100),))

    image = TokenUsageChartRenderer().render_to_png_bytes(report)

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert int.from_bytes(image[16:20], "big") == 1024
    assert int.from_bytes(image[20:24], "big") == 634
