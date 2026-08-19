"""Eventernote 活动搜索与详情工具。"""

from __future__ import annotations

import json
import math
import re
from calendar import monthrange
from datetime import datetime
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

EVENTERNOTE_BASE_URL = "https://www.eventernote.com"
EVENTERNOTE_SEARCH_URL = f"{EVENTERNOTE_BASE_URL}/events/search"
EVENTERNOTE_ACTOR_SEARCH_URL = f"{EVENTERNOTE_BASE_URL}/actors/search"
EVENTERNOTE_PLACE_SEARCH_URL = f"{EVENTERNOTE_BASE_URL}/places/search"
TOOL_PAGE_SIZE = 20
UPSTREAM_PAGE_SIZE = 30
REQUEST_TIMEOUT_SECONDS = 10
SEARCH_TYPES = {"event", "actor", "place"}
DateFilter = tuple[int, int | None, int | None]


class EventernoteError(Exception):
    """表示可返回给 Agent 的 Eventernote 业务错误。

    Args:
        code (str): 稳定错误码。
        message (str): 错误说明。

    Raises:
        AssertionError: 当错误码或错误说明为空时抛出。
    """

    def __init__(self, code: str, message: str) -> None:
        """初始化 Eventernote 业务错误。

        Args:
            code (str): 稳定错误码。
            message (str): 错误说明。

        Returns:
            None: 本方法仅初始化异常。

        Raises:
            AssertionError: 当错误码或错误说明为空时抛出。
        """
        assert code.strip(), "code 不能为空"
        assert message.strip(), "message 不能为空"
        super().__init__(message)
        self.code = code


class EventernoteClient:
    """请求并解析 Eventernote 活动数据。

    Args:
        session (requests.Session | None): 可选 HTTP 会话。

    Raises:
        AssertionError: 当 session 类型无效时抛出。
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        """初始化 Eventernote 客户端。

        Args:
            session (requests.Session | None): 可选 HTTP 会话，主要用于测试。

        Returns:
            None: 本方法仅初始化客户端。

        Raises:
            AssertionError: 当 session 类型无效时抛出。
        """
        assert session is None or isinstance(
            session, requests.Session
        ), "session 类型无效"
        self._session = session or requests.Session()

    def search(
        self,
        query: str,
        type: Literal["event", "actor", "place"] = "event",
        date: str | None = None,
        page: int = 1,
    ) -> dict[str, Any]:
        """按活动关键词、出演者或会场搜索 Eventernote 活动。

        Args:
            query (str): 搜索关键词；仅按活动日期查询时可为空。
            type (Literal["event", "actor", "place"]): 关键词类型。
            date (str | None): 可选的 ``YYYY``、``YYYY-MM`` 或
                ``YYYY-MM-DD`` 日期。
            page (int): 从 1 开始的 Tool 页码。

        Returns:
            dict[str, Any]: 包含分页信息和活动 ID、名称的成功结果。

        Raises:
            EventernoteError: 当参数、请求或页面解析失败时抛出。
        """
        normalized_query = query.strip()
        query_date = self._parse_date(date)
        if type not in SEARCH_TYPES:
            raise EventernoteError(
                "invalid_argument",
                "type 必须是 event、actor 或 place。",
            )
        if not normalized_query and (type != "event" or query_date is None):
            raise EventernoteError(
                "invalid_argument",
                "query 不能为空；仅 event 类型可在传入 date 时使用空关键词。",
            )
        if page <= 0:
            raise EventernoteError(
                "invalid_argument",
                "page 必须为正整数。",
            )
        if type == "event":
            return self._search_event_keyword(
                normalized_query,
                query_date,
                page,
            )
        return self._search_entity_events(
            normalized_query,
            type,
            query_date,
            page,
        )

    def _search_event_keyword(
        self,
        query: str,
        date: DateFilter | None,
        page: int,
    ) -> dict[str, Any]:
        """使用官网活动关键词入口搜索活动。

        Args:
            query (str): 活动关键词。
            date (DateFilter | None): 可选活动日期条件。
            page (int): 从 1 开始的 Tool 页码。

        Returns:
            dict[str, Any]: 每页二十条的活动搜索结果。

        Raises:
            EventernoteError: 当请求、分页或页面解析失败时抛出。
        """

        source_page = ((page - 1) * TOOL_PAGE_SIZE) // UPSTREAM_PAGE_SIZE + 1
        offset = ((page - 1) * TOOL_PAGE_SIZE) % UPSTREAM_PAGE_SIZE
        params: dict[str, str | int] = {
            "keyword": query,
            "page": source_page,
        }
        if date is not None:
            year, month, day = date
            params["year"] = year
            if month is not None:
                params["month"] = month
            if day is not None:
                params["day"] = day

        html = self._get(EVENTERNOTE_SEARCH_URL, params)
        total_items, source_items = self._parse_search_page(html)
        total_pages = math.ceil(total_items / TOOL_PAGE_SIZE)
        if total_pages > 0 and page > total_pages:
            raise EventernoteError(
                "page_out_of_range",
                f"page 超出范围，当前总页数为 {total_pages}。",
            )
        items = source_items[offset : offset + TOOL_PAGE_SIZE]
        expected_count = min(
            TOOL_PAGE_SIZE,
            max(total_items - (page - 1) * TOOL_PAGE_SIZE, 0),
        )
        if len(items) < expected_count:
            next_params = {**params, "page": source_page + 1}
            next_html = self._get(EVENTERNOTE_SEARCH_URL, next_params)
            _, next_items = self._parse_search_page(next_html)
            items.extend(next_items[: expected_count - len(items)])
        return self._search_result(page, total_pages, items)

    def _search_entity_events(
        self,
        query: str,
        type: Literal["actor", "place"],
        date: DateFilter | None,
        page: int,
    ) -> dict[str, Any]:
        """按出演者或会场的精准关系搜索活动。

        Args:
            query (str): 出演者或会场关键词。
            type (Literal["actor", "place"]): 关键词类型。
            date (DateFilter | None): 可选活动日期条件。
            page (int): 从 1 开始的 Tool 页码。

        Returns:
            dict[str, Any]: 每页二十条的精准关系活动结果。

        Raises:
            EventernoteError: 当实体不存在、请求或页面解析失败时抛出。
        """
        events_url, matched_name = self._resolve_entity_events_url(query, type)
        if date is not None:
            result = self._search_entity_events_by_date(events_url, date, page)
            result[type] = matched_name
            return result

        html = self._get(
            events_url,
            {"limit": TOOL_PAGE_SIZE, "page": page},
        )
        total_items, items = self._parse_search_page(html)
        total_pages = math.ceil(total_items / TOOL_PAGE_SIZE)
        if total_pages > 0 and page > total_pages:
            raise EventernoteError(
                "page_out_of_range",
                f"page 超出范围，当前总页数为 {total_pages}。",
            )
        result = self._search_result(page, total_pages, items)
        result[type] = matched_name
        return result

    def _resolve_entity_events_url(
        self,
        query: str,
        type: Literal["actor", "place"],
    ) -> tuple[str, str]:
        """解析官网排序第一的出演者或会场活动列表地址。

        Args:
            query (str): 出演者或会场关键词。
            type (Literal["actor", "place"]): 关键词类型。

        Returns:
            tuple[str, str]: 精准关系活动列表地址和匹配实体名称。

        Raises:
            EventernoteError: 当没有匹配实体或搜索页无法解析时抛出。
        """
        search_url = (
            EVENTERNOTE_ACTOR_SEARCH_URL
            if type == "actor"
            else EVENTERNOTE_PLACE_SEARCH_URL
        )
        html = self._get(search_url, {"keyword": query})
        soup = BeautifulSoup(html, "html.parser")
        path_pattern = (
            re.compile(r"/actors/[^/]+/\d+")
            if type == "actor"
            else re.compile(r"/places/\d+")
        )
        for link in soup.select("ul.gb_actors_list > li > a[href]"):
            href = str(link.get("href") or "")
            if path_pattern.fullmatch(href):
                name_node = link.find(string=True, recursive=False)
                matched_name = str(name_node).strip()
                return f"{EVENTERNOTE_BASE_URL}{href}/events", matched_name
        entity_name = "Actor" if type == "actor" else "Place"
        raise EventernoteError(
            "not_found",
            f"没有找到匹配的 {entity_name}。",
        )

    def _search_entity_events_by_date(
        self,
        events_url: str,
        date: DateFilter,
        page: int,
    ) -> dict[str, Any]:
        """按日期扫描精准关系活动列表并在越过目标日期后停止。

        Args:
            events_url (str): 出演者或会场活动列表地址。
            date (DateFilter): 目标活动日期条件。
            page (int): 从 1 开始的 Tool 页码。

        Returns:
            dict[str, Any]: 本地日期过滤后的活动结果。

        Raises:
            EventernoteError: 当请求、分页或页面解析失败时抛出。
        """
        year, month, day = date
        if day is not None:
            start_date = end_date = f"{year:04d}-{month:02d}-{day:02d}"
        elif month is not None:
            start_date = f"{year:04d}-{month:02d}-01"
            end_date = (
                f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}"
            )
        else:
            start_date = f"{year:04d}-01-01"
            end_date = f"{year:04d}-12-31"
        matched_items: list[dict[str, int | str | None]] = []
        source_page = 1
        source_total_pages = 1
        reached_older_event = False
        while source_page <= source_total_pages and not reached_older_event:
            html = self._get(
                events_url,
                {"limit": TOOL_PAGE_SIZE, "page": source_page},
            )
            total_items, items = self._parse_search_page(html)
            source_total_pages = math.ceil(total_items / TOOL_PAGE_SIZE)
            for item in items:
                item_date = item["date"]
                if (
                    item_date is not None
                    and start_date <= item_date <= end_date
                ):
                    matched_items.append(item)
                elif item_date is not None and item_date < start_date:
                    reached_older_event = True
                    break
            source_page += 1

        total_pages = math.ceil(len(matched_items) / TOOL_PAGE_SIZE)
        if total_pages > 0 and page > total_pages:
            raise EventernoteError(
                "page_out_of_range",
                f"page 超出范围，当前总页数为 {total_pages}。",
            )
        offset = (page - 1) * TOOL_PAGE_SIZE
        items = matched_items[offset : offset + TOOL_PAGE_SIZE]
        return self._search_result(page, total_pages, items)

    def _search_result(
        self,
        page: int,
        total_pages: int,
        items: list[dict[str, int | str | None]],
    ) -> dict[str, Any]:
        """构造统一活动搜索结果。

        Args:
            page (int): 当前页码。
            total_pages (int): 总页数。
            items (list[dict[str, int | str | None]]): 当前页活动。

        Returns:
            dict[str, Any]: 统一分页结果。

        Raises:
            AssertionError: 当页码或总页数无效时抛出。
        """
        assert page > 0, "page 必须为正整数"
        assert total_pages >= 0, "total_pages 不能为负数"
        return {
            "ok": True,
            "page": {
                "current": page,
                "total": total_pages,
                "size": TOOL_PAGE_SIZE,
            },
            "items": items,
        }

    def get(self, event_id: int) -> dict[str, Any]:
        """获取 Eventernote 活动核心详情。

        Args:
            event_id (int): Eventernote 活动 ID。

        Returns:
            dict[str, Any]: 扁平化会场和出演者后的活动详情。

        Raises:
            EventernoteError: 当参数、请求或页面解析失败时抛出。
        """
        if event_id <= 0:
            raise EventernoteError(
                "invalid_argument",
                "id 必须为正整数。",
            )
        url = f"{EVENTERNOTE_BASE_URL}/events/{event_id}"
        html = self._get(url)
        return {
            "ok": True,
            "data": self._parse_event_page(html, event_id, url),
        }

    def _parse_date(self, value: str | None) -> DateFilter | None:
        """解析必须从年份开始的可选日期条件。

        Args:
            value (str | None): 可选日期文本。

        Returns:
            DateFilter | None: 年、可选月、可选日；未传入时返回 None。

        Raises:
            EventernoteError: 当日期格式或日期值非法时抛出。
        """
        if value is None:
            return None
        match = re.fullmatch(
            r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?",
            value,
        )
        if match is None:
            raise EventernoteError(
                "invalid_date",
                "date 必须是 YYYY、YYYY-MM 或 YYYY-MM-DD 格式的有效日期。",
            )
        year = int(match.group(1))
        month = int(match.group(2)) if match.group(2) else None
        day = int(match.group(3)) if match.group(3) else None
        try:
            datetime(year, month or 1, day or 1)
        except ValueError as exc:
            raise EventernoteError(
                "invalid_date",
                "date 必须是 YYYY、YYYY-MM 或 YYYY-MM-DD 格式的有效日期。",
            ) from exc
        return year, month, day

    def _get(
        self,
        url: str,
        params: dict[str, str | int] | None = None,
    ) -> str:
        """发送 Eventernote GET 请求。

        Args:
            url (str): 请求地址。
            params (dict[str, str | int] | None): 可选查询参数。

        Returns:
            str: 响应 HTML。

        Raises:
            EventernoteError: 当网络失败或上游返回错误状态时抛出。
        """
        try:
            response = self._session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise EventernoteError(
                "network_error",
                f"Eventernote 请求失败：{exc}",
            ) from exc
        if response.status_code == 404:
            raise EventernoteError("not_found", "指定的 Event 不存在。")
        if response.status_code != 200:
            raise EventernoteError(
                "upstream_error",
                f"Eventernote 返回 HTTP {response.status_code}。",
            )
        return response.text

    def _parse_search_page(
        self,
        html: str,
    ) -> tuple[int, list[dict[str, int | str | None]]]:
        """解析活动搜索页。

        Args:
            html (str): 活动搜索页 HTML。

        Returns:
            tuple[int, list[dict[str, int | str | None]]]: 总结果数和当前
            官网页活动。

        Raises:
            EventernoteError: 当搜索页结构无法解析时抛出。
        """
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(" ", strip=True)
        no_results_text = (
            "指定された条件での検索結果が見つかりませんでした。"
        )
        if no_results_text in page_text:
            return 0, []
        count_match = re.search(
            r"([\d,]+)件(?:のイベントが)?見つかりました",
            page_text,
        )
        if count_match is None:
            raise EventernoteError(
                "parse_error",
                "Eventernote 搜索结果总数无法解析。",
            )
        total_items = int(count_match.group(1).replace(",", ""))
        items: list[dict[str, int | str | None]] = []
        for link in soup.select(
            '.gb_event_list .event h4 a[href^="/events/"]'
        ):
            href = str(link.get("href") or "")
            id_match = re.fullmatch(r"/events/(\d+)", href)
            if id_match is None:
                raise EventernoteError(
                    "parse_error",
                    "Eventernote 活动 ID 无法解析。",
                )
            event_row = link.find_parent("li", class_="clearfix")
            date_element = (
                event_row.select_one(".date p") if event_row else None
            )
            date_match = re.search(
                r"\d{4}-\d{2}-\d{2}",
                date_element.get_text(" ", strip=True)
                if date_element
                else "",
            )
            items.append(
                {
                    "id": int(id_match.group(1)),
                    "name": link.get_text(" ", strip=True),
                    "date": date_match.group(0) if date_match else None,
                }
            )
        if total_items > 0 and not items:
            raise EventernoteError(
                "parse_error",
                "Eventernote 活动列表无法解析。",
            )
        return total_items, items

    def _parse_event_page(
        self,
        html: str,
        event_id: int,
        url: str,
    ) -> dict[str, Any]:
        """解析活动详情页的核心字段。

        Args:
            html (str): 活动详情页 HTML。
            event_id (int): Eventernote 活动 ID。
            url (str): 活动详情页地址。

        Returns:
            dict[str, Any]: 结构化活动摘要。

        Raises:
            EventernoteError: 当活动详情页结构无法解析时抛出。
        """
        soup = BeautifulSoup(html, "html.parser")
        title = soup.select_one(".gb_events_detail_title h2")
        if title is None:
            raise EventernoteError(
                "parse_error",
                "Eventernote 活动详情无法解析。",
            )
        rows: dict[str, Any] = {}
        for row in soup.select(".gb_events_info_table tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) >= 2:
                rows[cells[0].get_text(" ", strip=True)] = cells[1]

        date_text = self._cell_text(rows, "開催日時")
        time_text = self._cell_text(rows, "時間")
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
        weekday_match = re.search(r"\(([^)]+)\)", date_text)
        place_cell = rows.get("開催場所")
        place_link = place_cell.select_one('a[href^="/places/"]') if place_cell else None
        actor_cell = rows.get("出演者")
        actors = (
            [
                link.get_text(" ", strip=True)
                for link in actor_cell.select('a[href^="/actors/"]')
            ]
            if actor_cell
            else []
        )
        link_cell = rows.get("関連リンク")
        related_links = (
            [str(link.get("href")) for link in link_cell.select("a[href]")]
            if link_cell
            else []
        )
        return {
            "id": event_id,
            "name": title.get_text(" ", strip=True),
            "date": date_match.group(0) if date_match else None,
            "weekday": weekday_match.group(1) if weekday_match else None,
            "open_time": self._parse_time(time_text, "開場"),
            "start_time": self._parse_time(time_text, "開演"),
            "end_time": self._parse_time(time_text, "終演"),
            "place": place_link.get_text(" ", strip=True) if place_link else None,
            "actors": actors,
            "related_links": related_links,
            "url": url,
        }

    def _cell_text(self, rows: dict[str, Any], label: str) -> str:
        """读取详情表格单元格文本。

        Args:
            rows (dict[str, Any]): 表格标题到单元格的映射。
            label (str): 目标表格标题。

        Returns:
            str: 规范化后的单元格文本。

        Raises:
            AssertionError: 当 label 为空时抛出。
        """
        assert label.strip(), "label 不能为空"
        cell = rows.get(label)
        return cell.get_text(" ", strip=True) if cell else ""

    def _parse_time(self, value: str, label: str) -> str | None:
        """从活动时间文本中解析指定时间。

        Args:
            value (str): 活动时间文本。
            label (str): ``開場``、``開演`` 或 ``終演``。

        Returns:
            str | None: ``HH:MM`` 时间；缺失时返回 None。

        Raises:
            AssertionError: 当 label 不受支持时抛出。
        """
        assert label in {"開場", "開演", "終演"}, "不支持的时间标签"
        match = re.search(rf"{label}\s*([0-9]{{1,2}}:[0-9]{{2}})", value)
        return match.group(1) if match else None


def _json_result(payload: dict[str, Any]) -> str:
    """序列化 Tool 结构化结果。

    Args:
        payload (dict[str, Any]): 待序列化结果。

    Returns:
        str: 保留日文字符的 JSON 字符串。

    Raises:
        AssertionError: 当 payload 不是字典时抛出。
    """
    assert isinstance(payload, dict), "payload 必须是字典"
    return json.dumps(payload, ensure_ascii=False)


def _error_result(error: EventernoteError) -> str:
    """将 Eventernote 错误转换为结构化结果。

    Args:
        error (EventernoteError): 待转换错误。

    Returns:
        str: 结构化失败 JSON。

    Raises:
        AssertionError: 当 error 类型无效时抛出。
    """
    assert isinstance(error, EventernoteError), "error 类型无效"
    return _json_result(
        {
            "ok": False,
            "error": {
                "code": error.code,
                "message": str(error),
            },
        }
    )


def _validation_error_result(error: Exception) -> str:
    """将 LangChain 参数校验错误转换为结构化结果。

    Args:
        error (Exception): LangChain 参数校验错误。

    Returns:
        str: 结构化参数错误 JSON。

    Raises:
        AssertionError: 当 error 不是异常时抛出。
    """
    assert isinstance(error, Exception), "error 必须是异常"
    return _error_result(
        EventernoteError("invalid_argument", "Tool 参数格式不正确。")
    )


def build_eventernote_tools(
    client: EventernoteClient | None = None,
) -> tuple[BaseTool, BaseTool]:
    """创建 Eventernote 搜索和详情 Tool。

    Args:
        client (EventernoteClient | None): 可选客户端，主要用于测试注入。

    Returns:
        tuple[BaseTool, BaseTool]: 搜索 Tool 和详情 Tool。

    Raises:
        AssertionError: 当 client 类型无效时抛出。
    """
    assert client is None or isinstance(
        client, EventernoteClient
    ), "client 类型无效"
    eventernote_client = client or EventernoteClient()

    @tool("eventernote_search")
    def eventernote_search(
        query: str,
        type: Literal["event", "actor", "place"] = "event",
        date: str | None = None,
        page: int = 1,
    ) -> str:
        """搜索 Eventernote 活动，每页返回二十条活动 ID、名称和日期。

        query 只支持 Eventernote 收录的源语言名称。不要先把活动名、
        出演者名或会场名翻译成其他语言。event 执行活动全文关键词搜索；
        actor 和 place 先取官网匹配度最高的实体，再查询其精准关联活动。
        仅 event 按日期查询时允许 query 为空字符串。

        Args:
            query (str): 使用 Eventernote 源语言名称的活动、出演者或
                会场关键词，不要传入翻译后的专名。
            type (Literal["event", "actor", "place"]): 关键词类型。
            date (str | None): 可选的 ``YYYY``、``YYYY-MM`` 或
                ``YYYY-MM-DD`` 活动日期。
            page (int): 从 1 开始的页码。

        Returns:
            str: 包含当前页、总页数和活动列表的 JSON；actor/place
            搜索还包含实际匹配的同名顶层键，失败时返回错误 JSON。

        Raises:
            Exception: 未预期的程序错误原样抛出。
        """
        try:
            return _json_result(
                eventernote_client.search(query, type, date, page)
            )
        except EventernoteError as exc:
            return _error_result(exc)

    @tool("eventernote_get")
    def eventernote_get(id: int) -> str:
        """根据 Event ID 获取 Eventernote 活动核心摘要。

        Args:
            id (int): eventernote_search 返回的正整数 Event ID。

        Returns:
            str: 会场与出演者仅保留名称的活动摘要 JSON；失败时返回
            错误 JSON。

        Raises:
            Exception: 未预期的程序错误原样抛出。
        """
        try:
            return _json_result(eventernote_client.get(id))
        except EventernoteError as exc:
            return _error_result(exc)

    eventernote_search.handle_validation_error = _validation_error_result
    eventernote_get.handle_validation_error = _validation_error_result
    return eventernote_search, eventernote_get


eventernote_search, eventernote_get = build_eventernote_tools()

__all__ = [
    "EventernoteClient",
    "EventernoteError",
    "build_eventernote_tools",
    "eventernote_get",
    "eventernote_search",
]
