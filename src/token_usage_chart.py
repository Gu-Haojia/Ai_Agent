"""将 Token 使用日志渲染为固定横版 PNG 图表。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from src.token_usage_logger import TokenUsageReport

CHART_WIDTH = 1024
CHART_HEIGHT = 634
CHART_FONT_FAMILY = "Noto Sans CJK SC"
D3_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "token_usage_chart"
    / "d3.v7.9.0.min.js"
)
MODEL_COLORS = ("#4da3f5", "#ff9557", "#7c6fe8", "#f0bf44", "#4dc4a6")
MAX_VISIBLE_MODELS = 4


class TokenUsageGranularity(str, Enum):
    """Token 趋势图支持的时间聚合粒度。"""

    HOUR = "小时"
    DAY = "日"
    WEEK = "周"
    MONTH = "月"


@dataclass(frozen=True)
class TokenUsageChartPoint:
    """单个时间桶的趋势图数据。

    Args:
        bucket_start (datetime): 时间桶起始时间。
        model_tokens (tuple[int, ...]): 各模型的 Token 数。
        total_tokens (int): 时间桶总 Token 数。
        average_tokens (float | None): 移动平均 Token 数。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None: 数据类初始化不主动抛出异常。
    """

    bucket_start: datetime
    model_tokens: tuple[int, ...]
    total_tokens: int
    average_tokens: float | None


@dataclass(frozen=True)
class TokenUsageChartData:
    """静态 Token 图表所需的完整数据。

    Args:
        granularity (TokenUsageGranularity): 时间聚合粒度。
        average_window (int): 移动平均窗口长度。
        model_names (tuple[str, ...]): 图中模型名称。
        model_totals (tuple[int, ...]): 各模型累计 Token 数。
        points (tuple[TokenUsageChartPoint, ...]): 时间趋势点。
        start_time (datetime): 图表起始时间。
        end_time (datetime): 图表结束时间。
        total_tokens (int): 总 Token 数。
        input_tokens (int): 输入 Token 数。
        cache_read (int): 缓存命中 Token 数。
        output_tokens (int): 输出 Token 数。

    Returns:
        None: 数据类初始化不返回额外值。

    Raises:
        None: 数据类初始化不主动抛出异常。
    """

    granularity: TokenUsageGranularity
    average_window: int
    model_names: tuple[str, ...]
    model_totals: tuple[int, ...]
    points: tuple[TokenUsageChartPoint, ...]
    start_time: datetime
    end_time: datetime
    total_tokens: int
    input_tokens: int
    cache_read: int
    output_tokens: int


class TokenUsageChartBuilder:
    """将 Token 日志报告聚合为静态图表数据。"""

    def build(self, report: TokenUsageReport) -> TokenUsageChartData:
        """按报告实际跨度聚合图表数据。

        Args:
            report (TokenUsageReport): 同一日志快照中的 Token 报告。

        Returns:
            TokenUsageChartData: 可供图表渲染的数据。

        Raises:
            AssertionError: 当报告没有记录或汇总时间不完整时抛出。
        """
        assert report.records, "Token 报告没有可绘制记录"
        assert report.summary.start_time is not None, "Token 报告缺少开始时间"
        assert report.end_time is not None, "Token 报告缺少结束时间"
        granularity = self.select_granularity(
            report.summary.start_time,
            report.end_time,
        )
        average_window = {
            TokenUsageGranularity.HOUR: 4,
            TokenUsageGranularity.DAY: 7,
            TokenUsageGranularity.WEEK: 4,
            TokenUsageGranularity.MONTH: 3,
        }[granularity]
        model_names = self._select_models(report)
        model_index = {name: index for index, name in enumerate(model_names)}
        buckets: dict[datetime, list[int]] = {}
        for record in report.records:
            bucket_start = self._bucket_start(record.recorded_at, granularity)
            values = buckets.setdefault(bucket_start, [0] * len(model_names))
            model_name = record.model_name or "未知模型"
            index = model_index.get(model_name, len(model_names) - 1)
            values[index] += record.total_tokens
        first_bucket = min(buckets)
        last_bucket = max(buckets)
        ordered: list[tuple[datetime, list[int]]] = []
        bucket_start = first_bucket
        while bucket_start <= last_bucket:
            ordered.append(
                (
                    bucket_start,
                    buckets.get(bucket_start, [0] * len(model_names)),
                )
            )
            bucket_start = self._next_bucket(bucket_start, granularity)
        totals = [sum(values) for _, values in ordered]
        points: list[TokenUsageChartPoint] = []
        for index, (bucket_start, values) in enumerate(ordered):
            window = totals[max(0, index - average_window + 1) : index + 1]
            average = (
                sum(window) / average_window
                if len(window) == average_window
                else None
            )
            points.append(
                TokenUsageChartPoint(
                    bucket_start=bucket_start,
                    model_tokens=tuple(values),
                    total_tokens=totals[index],
                    average_tokens=average,
                )
            )
        model_totals = tuple(
            sum(point.model_tokens[index] for point in points)
            for index in range(len(model_names))
        )
        summary = report.summary
        return TokenUsageChartData(
            granularity=granularity,
            average_window=average_window,
            model_names=model_names,
            model_totals=model_totals,
            points=tuple(points),
            start_time=summary.start_time,
            end_time=report.end_time,
            total_tokens=summary.total_tokens,
            input_tokens=summary.input_tokens,
            cache_read=summary.cache_read,
            output_tokens=summary.output_tokens,
        )

    @staticmethod
    def select_granularity(
        start_time: datetime,
        end_time: datetime,
    ) -> TokenUsageGranularity:
        """根据实际数据跨度选择聚合粒度。

        Args:
            start_time (datetime): 数据开始时间。
            end_time (datetime): 数据结束时间。

        Returns:
            TokenUsageGranularity: 自动选择的时间粒度。

        Raises:
            AssertionError: 当时间缺少时区或结束时间早于开始时间时抛出。
        """
        assert start_time.tzinfo is not None, "开始时间必须包含时区"
        assert end_time.tzinfo is not None, "结束时间必须包含时区"
        assert end_time >= start_time, "结束时间不得早于开始时间"
        span = end_time - start_time
        if span <= timedelta(days=3):
            return TokenUsageGranularity.HOUR
        if span <= timedelta(days=90):
            return TokenUsageGranularity.DAY
        if span <= timedelta(days=540):
            return TokenUsageGranularity.WEEK
        return TokenUsageGranularity.MONTH

    @staticmethod
    def _select_models(report: TokenUsageReport) -> tuple[str, ...]:
        """选择消费最高的模型并将其余模型合并为“其他”。

        Args:
            report (TokenUsageReport): Token 日志报告。

        Returns:
            tuple[str, ...]: 按消费量降序排列的图表模型名称。

        Raises:
            AssertionError: 当报告没有记录时抛出。
        """
        totals: defaultdict[str, int] = defaultdict(int)
        for record in report.records:
            totals[record.model_name or "未知模型"] += record.total_tokens
        ordered = sorted(totals, key=lambda name: (-totals[name], name))
        assert ordered, "Token 报告没有模型数据"
        if len(ordered) <= MAX_VISIBLE_MODELS:
            return tuple(ordered)
        return tuple(ordered[:MAX_VISIBLE_MODELS] + ["其他"])

    @staticmethod
    def _bucket_start(
        recorded_at: datetime,
        granularity: TokenUsageGranularity,
    ) -> datetime:
        """计算记录所属时间桶的起始时间。

        Args:
            recorded_at (datetime): Token 记录时间。
            granularity (TokenUsageGranularity): 目标聚合粒度。

        Returns:
            datetime: 保留原时区的时间桶起始时间。

        Raises:
            AssertionError: 当记录时间缺少时区时抛出。
        """
        assert recorded_at.tzinfo is not None, "Token 记录时间必须包含时区"
        if granularity is TokenUsageGranularity.HOUR:
            return recorded_at.replace(minute=0, second=0, microsecond=0)
        if granularity is TokenUsageGranularity.DAY:
            return recorded_at.replace(hour=0, minute=0, second=0, microsecond=0)
        if granularity is TokenUsageGranularity.WEEK:
            day_start = recorded_at.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            return day_start - timedelta(days=day_start.weekday())
        return recorded_at.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    @staticmethod
    def _next_bucket(
        bucket_start: datetime,
        granularity: TokenUsageGranularity,
    ) -> datetime:
        """返回当前时间桶之后的下一个连续时间桶。

        Args:
            bucket_start (datetime): 当前时间桶起始时间。
            granularity (TokenUsageGranularity): 时间聚合粒度。

        Returns:
            datetime: 下一个时间桶起始时间。

        Raises:
            AssertionError: 当时间缺少时区时抛出。
        """
        assert bucket_start.tzinfo is not None, "时间桶必须包含时区"
        if granularity is TokenUsageGranularity.HOUR:
            return bucket_start + timedelta(hours=1)
        if granularity is TokenUsageGranularity.DAY:
            return bucket_start + timedelta(days=1)
        if granularity is TokenUsageGranularity.WEEK:
            return bucket_start + timedelta(weeks=1)
        year = bucket_start.year + bucket_start.month // 12
        month = bucket_start.month % 12 + 1
        return bucket_start.replace(year=year, month=month)


class TokenUsageChartRenderer:
    """使用本地 D3 与无头 Chromium 生成 Token 图表 PNG。"""

    def __init__(
        self,
        builder: TokenUsageChartBuilder | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        """初始化 Token 图表渲染器。

        Args:
            builder (TokenUsageChartBuilder | None): 图表数据构建器。
            timeout_ms (int): Chromium 渲染超时毫秒数。

        Returns:
            None: 初始化不返回额外值。

        Raises:
            AssertionError: 当超时不为正数时抛出。
        """
        assert timeout_ms > 0, "timeout_ms 必须大于 0"
        self._builder = builder or TokenUsageChartBuilder()
        self._timeout_ms = timeout_ms

    def render_to_png_bytes(self, report: TokenUsageReport) -> bytes:
        """将 Token 日志报告渲染为固定尺寸 PNG。

        Args:
            report (TokenUsageReport): 同一日志快照中的 Token 报告。

        Returns:
            bytes: 1024×634 PNG 图片字节。

        Raises:
            AssertionError: 当数据、字体或截图结果不符合预期时抛出。
            OSError: 当本地 D3 文件无法读取时抛出。
            RuntimeError: 当 Playwright 缺失或 Chromium 渲染失败时抛出。
        """
        try:
            from playwright.sync_api import (
                Error as PlaywrightError,
                TimeoutError as PlaywrightTimeoutError,
                sync_playwright,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("生成 Token 图表需要安装 Playwright") from exc
        chart_data = self._builder.build(report)
        html = self._render_html(chart_data)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page(
                        viewport={"width": CHART_WIDTH, "height": CHART_HEIGHT},
                        device_scale_factor=1,
                    )
                    page.set_content(
                        html,
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms,
                    )
                    page.evaluate(
                        f'document.fonts.load("12px \\"{CHART_FONT_FAMILY}\\"")'
                    )
                    font_ready = page.evaluate(
                        f'document.fonts.check("12px \\"{CHART_FONT_FAMILY}\\"")'
                    )
                    assert font_ready is True, "Token 图表字体加载失败"
                    page.wait_for_function(
                        "window.__TOKEN_USAGE_CHART_READY__ === true",
                        timeout=self._timeout_ms,
                    )
                    image_bytes = page.locator("#token-usage-chart").screenshot(
                        omit_background=False
                    )
                finally:
                    browser.close()
        except (PlaywrightError, PlaywrightTimeoutError) as exc:
            raise RuntimeError(f"Token 图表截图失败：{exc}") from exc
        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n"), (
            "Token 图表截图不是有效 PNG"
        )
        return image_bytes

    def _render_html(self, chart: TokenUsageChartData) -> str:
        """生成包含本地 D3 的完整静态图表 HTML。

        Args:
            chart (TokenUsageChartData): 已聚合的图表数据。

        Returns:
            str: 可直接交给 Chromium 渲染的 HTML。

        Raises:
            OSError: 当本地 D3 文件无法读取时抛出。
        """
        d3_source = D3_PATH.read_text(encoding="utf-8")
        payload = json.dumps(
            self._serialize_chart(chart),
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("</", "<\\/")
        return _HTML_TEMPLATE.replace("__D3_SOURCE__", d3_source).replace(
            "__CHART_DATA__",
            payload,
        )

    @staticmethod
    def _serialize_chart(chart: TokenUsageChartData) -> dict[str, Any]:
        """将图表数据转换为可嵌入 HTML 的字典。

        Args:
            chart (TokenUsageChartData): 已聚合的图表数据。

        Returns:
            dict[str, Any]: JSON 可序列化的图表数据。

        Raises:
            None: 本方法不主动抛出异常。
        """
        date_format = "%Y.%m.%d"
        if chart.granularity is TokenUsageGranularity.HOUR:
            point_format = (
                "%H:00"
                if chart.start_time.date() == chart.end_time.date()
                else "%m/%d %H"
            )
        elif chart.granularity is TokenUsageGranularity.MONTH:
            point_format = "%Y/%m"
        else:
            point_format = "%m/%d"
        date_range = chart.start_time.strftime(date_format)
        if chart.start_time.date() != chart.end_time.date():
            date_range += f" — {chart.end_time.strftime(date_format)}"
        models = [
            {
                "name": name,
                "total": chart.model_totals[index],
                "color": MODEL_COLORS[index],
            }
            for index, name in enumerate(chart.model_names)
        ]
        return {
            "granularity": chart.granularity.value,
            "averageLabel": f"{chart.average_window} {chart.granularity.value}均线",
            "dateRange": date_range,
            "metrics": {
                "total": chart.total_tokens,
                "input": chart.input_tokens,
                "cache": chart.cache_read,
                "output": chart.output_tokens,
            },
            "models": models,
            "points": [
                {
                    "label": point.bucket_start.strftime(point_format),
                    "models": list(point.model_tokens),
                    "total": point.total_tokens,
                    "average": point.average_tokens,
                }
                for point in chart.points
            ],
        }


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box}html,body{margin:0;width:1024px;height:634px;background:#fff;color:#1D2939;font-family:"Noto Sans CJK SC",sans-serif}body{overflow:hidden}#token-usage-chart{display:grid;grid-template-rows:auto auto auto 250px auto;width:1024px;height:634px;padding:22px 24px 16px;background:#fff}.header{display:flex;align-items:flex-start;justify-content:space-between}.title{font-size:21px;font-weight:650;letter-spacing:-.02em;margin:0 0 5px}.subtitle,.period,.metric-label,.metric-detail,.legend,.panel-list{color:#667085}.subtitle,.period{font-size:12px;margin:0}.period{padding-top:4px;font-variant-numeric:tabular-nums}.metrics{display:grid;grid-template-columns:repeat(4,1fr);margin:10px 0 5px;padding:10px 0;background:#f7f9fc;border:1px solid #eaecf0;border-radius:12px}.metric{position:relative;padding:0 16px}.metric:not(:first-child)::before{content:"";position:absolute;left:0;top:50%;width:1px;height:34px;background:#e4e7ec;transform:translateY(-50%)}.metric:first-child{padding-left:16px}.metric-label{display:block;font-size:11px;font-weight:500;letter-spacing:.06em;margin-bottom:3px}.metric-value{display:block;font-size:23px;font-weight:650;font-variant-numeric:tabular-nums}.metric-detail{font-size:11px}.legend{display:flex;flex-wrap:wrap;gap:7px 20px;min-height:23px;font-size:12px}.legend-item{display:inline-flex;align-items:center;gap:7px}.swatch{display:inline-block;width:16px;height:3px}.trend{width:100%;height:250px}.trend svg{display:block;width:100%;height:100%}.chart-frame{fill:transparent;stroke:none}.grid line{stroke:#e9edf2;stroke-opacity:1}.grid path,.axis path{display:none}.axis line{stroke:transparent}.axis text{fill:#667085;font-size:12px}.axis-title{fill:#344054;font-size:12px;font-weight:500}.bottom{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;padding-top:0}.pie-panel{display:grid;grid-template-columns:150px 1fr;align-items:center;background:#fafbfc;border:1px solid #eef0f3;border-radius:12px;padding:6px 10px}.pie{width:150px;height:132px}.panel-title{font-size:14px;font-weight:600;margin:0 0 10px}.panel-list{display:grid;gap:7px;font-size:12px}.panel-row{display:grid;grid-template-columns:10px minmax(0,1fr) 52px 42px;gap:7px;align-items:center}.dot{width:8px;height:8px}.token-value{color:#9aa0a6;text-align:right;font-variant-numeric:tabular-nums}.value{color:#1D2939;font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<main id="token-usage-chart">
  <header class="header"><div><h1 class="title">Token 消费概览</h1><p class="subtitle"></p></div><p class="period"></p></header>
  <section class="metrics"></section>
  <section class="legend"></section>
  <section class="trend"><svg role="img" aria-label="Token 消费时间趋势"></svg></section>
  <section class="bottom">
    <div class="pie-panel"><svg class="pie model-pie" role="img" aria-label="模型消费占比"></svg><div><h2 class="panel-title">模型消费占比</h2><div class="panel-list model-list"></div></div></div>
    <div class="pie-panel"><svg class="pie composition-pie" role="img" aria-label="Token 构成"></svg><div><h2 class="panel-title">Token 构成</h2><div class="panel-list composition-list"></div></div></div>
  </section>
</main>
<script>__D3_SOURCE__</script>
<script>
(()=>{
const data=__CHART_DATA__;
const compact=v=>d3.format(".3~s")(v).replace("G","B");
const percent=(v,total)=>`${(v/Math.max(total,1)*100).toFixed(1)}%`;
document.querySelector(".subtitle").textContent=`自动粒度：${data.granularity}`;
document.querySelector(".period").textContent=data.dateRange;
const m=data.metrics;
const metricRows=[
  ["总消耗",m.total,""],
  ["输入",m.input,`占总量 ${percent(m.input,m.total)}`],
  ["缓存命中",m.cache,`缓存命中率 ${percent(m.cache,m.input)}`],
  ["输出",m.output,`占总量 ${percent(m.output,m.total)}`]
];
d3.select(".metrics").selectAll("div").data(metricRows).join("div").attr("class","metric").html(d=>`<span class="metric-label">${d[0]}</span><span class="metric-value">${compact(d[1])}</span>${d[2]?`<span class="metric-detail">${d[2]}</span>`:""}`);
const legendData=[...data.models,{name:data.averageLabel,color:"#ef6eb2"}];
const legend=d3.select(".legend").selectAll("span.legend-item").data(legendData).join("span").attr("class","legend-item");
legend.append("span").attr("class","swatch").style("background",d=>d.color);
legend.append("span").text(d=>d.name);
const svg=d3.select(".trend svg"),width=976,height=250,margin={top:12,right:18,bottom:42,left:70},left=margin.left,right=width-margin.right,top=margin.top,bottom=height-margin.bottom;
svg.attr("viewBox",`0 0 ${width} ${height}`);
const x=d3.scaleBand().domain(d3.range(data.points.length)).range([left,right]).paddingInner(.22).paddingOuter(.08);
const maxValue=d3.max(data.points,d=>Math.max(d.total,d.average||0))||1;
const y=d3.scaleLinear().domain([0,maxValue*1.1]).nice().range([bottom,top]);
svg.append("rect").attr("class","chart-frame").attr("x",left).attr("y",top).attr("width",right-left).attr("height",bottom-top);
svg.append("g").attr("class","grid").attr("transform",`translate(${left},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(right-left)).tickFormat(""));
svg.append("g").attr("class","axis").attr("transform",`translate(${left},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(compact));
const tickCount=Math.min(7,data.points.length),tickIndexes=d3.range(tickCount).map(i=>Math.round(i*(data.points.length-1)/Math.max(1,tickCount-1)));
svg.append("g").attr("class","axis").attr("transform",`translate(0,${bottom})`).call(d3.axisBottom(x).tickValues(tickIndexes).tickFormat(i=>data.points[i].label));
svg.append("text").attr("class","axis-title").attr("transform",`translate(18,${(top+bottom)/2}) rotate(-90)`).attr("text-anchor","middle").text(`Tokens / ${data.granularity}`);
svg.append("text").attr("class","axis-title").attr("x",(left+right)/2).attr("y",height-4).attr("text-anchor","middle").text("时间");
const bars=svg.append("g");
data.points.forEach((point,pointIndex)=>{let yTop=bottom;point.models.forEach((value,modelIndex)=>{const h=bottom-y(value);yTop-=h;bars.append("rect").attr("x",x(pointIndex)).attr("y",yTop).attr("width",x.bandwidth()).attr("height",h).attr("fill",data.models[modelIndex].color).attr("fill-opacity",.84)})});
const averageLine=d3.line().defined(d=>d.average!==null).x((d,i)=>x(i)+x.bandwidth()/2).y(d=>y(d.average)).curve(d3.curveMonotoneX);
svg.append("path").datum(data.points).attr("d",averageLine).attr("fill","none").attr("stroke","#ef6eb2").attr("stroke-width",2.5);
function drawPie(svgSelector,listSelector,values){const pieSvg=d3.select(svgSelector).attr("viewBox","0 0 150 132"),radius=50,group=pieSvg.append("g").attr("transform","translate(75,66)"),pie=d3.pie().sort(null).value(d=>d.value)(values),arc=d3.arc().innerRadius(0).outerRadius(radius);group.selectAll("path").data(pie).join("path").attr("d",arc).attr("fill",d=>d.data.color).attr("stroke","#fff").attr("stroke-width",2);const total=d3.sum(values,d=>d.value),rows=d3.select(listSelector).selectAll("div").data(values).join("div").attr("class","panel-row");rows.append("span").attr("class","dot").style("background",d=>d.color);rows.append("span").text(d=>d.name);rows.append("span").attr("class","token-value").text(d=>compact(d.value));rows.append("span").attr("class","value").text(d=>percent(d.value,total))}
drawPie(".model-pie",".model-list",data.models.map(d=>({name:d.name,value:d.total,color:d.color})));
drawPie(".composition-pie",".composition-list",[
  {name:"非缓存输入",value:Math.max(0,m.input-m.cache),color:"#62ca80"},
  {name:"缓存命中",value:m.cache,color:"#9b7be3"},
  {name:"输出",value:m.output,color:"#37b5a5"}
]);
window.__TOKEN_USAGE_CHART_READY__=true;
})();
</script>
</body>
</html>"""


TOKEN_USAGE_CHART_RENDERER = TokenUsageChartRenderer()
