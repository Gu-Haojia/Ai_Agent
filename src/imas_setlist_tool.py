"""
imas-db Setlist 搜索与详情查询工具。

搜索工具每次读取活动索引并返回全部匹配候选；详情工具只接受搜索结果
中的精确候选 ID。实现不缓存、不落盘，也不推断页面未提供的数据。
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from langchain_core.tools import tool
import requests


IMAS_SETLIST_INDEX_URL = "https://imas-db.jp/song/event/"
IMAS_SETLIST_CANDIDATE_PREFIX = "imas-setlist:"
IMAS_SETLIST_USER_AGENT = "LangGraph-ImasSetlistTool/1.0"

HttpGet = Callable[..., requests.Response]


class ImasSetlistToolError(RuntimeError):
    """表示可安全暴露给 Agent 的 imas-db Setlist 查询错误。"""

    def __init__(self, error_code: str, message: str) -> None:
        """
        初始化带稳定错误码的工具异常。

        Args:
            error_code (str): 供 Agent 判断失败类型的稳定错误码。
            message (str): 面向 Agent 的错误说明。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当错误码或错误说明为空时抛出。
        """
        assert error_code.strip(), "error_code 不能为空"
        assert message.strip(), "message 不能为空"
        super().__init__(message)
        self.error_code = error_code.strip()


class ImasSetlistClient:
    """请求并解析 imas-db Setlist 索引和详情页。"""

    def __init__(
        self,
        timeout_seconds: float = 15.0,
        http_get: HttpGet | None = None,
    ) -> None:
        """
        初始化无缓存客户端。

        Args:
            timeout_seconds (float): 单次 HTTP 请求超时秒数。
            http_get (HttpGet | None): 可注入的 HTTP GET 调用函数。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当超时秒数不大于零时抛出。
        """
        assert timeout_seconds > 0, "timeout_seconds 必须大于 0"
        self._timeout_seconds = timeout_seconds
        self._http_get = http_get or requests.get

    def search(self, query: str) -> dict[str, object]:
        """
        搜索并返回全部匹配候选。

        Args:
            query (str): 活动名称、年份、品牌或场次关键词。

        Returns:
            dict[str, object]: 搜索状态及匹配候选。

        Raises:
            AssertionError: 当查询词或索引结构非法时抛出。
            ImasSetlistToolError: 当索引页请求失败时抛出。
        """
        normalized_query = query.strip()
        assert normalized_query, "query 不能为空"
        assert len(normalized_query) <= 200, "query 长度不能超过 200"
        terms = self._query_terms(normalized_query)
        candidates = self._load_candidates()
        matched = [
            candidate
            for candidate in candidates
            if self._matches(str(candidate["search_text"]), terms)
        ]
        public_candidates = [
            {
                "candidate_id": candidate["candidate_id"],
                "title": candidate["title"],
                "day": candidate["day"],
            }
            for candidate in matched
        ]
        return {
            "status": "success" if public_candidates else "not_found",
            "query": normalized_query,
            "candidates": public_candidates,
        }

    def get(self, candidate_id: str) -> dict[str, object]:
        """
        校验候选 ID 后返回对应 Setlist。

        Args:
            candidate_id (str): 搜索工具返回的精确候选 ID。

        Returns:
            dict[str, object]: 活动标题、日期、曲目及来源链接。

        Raises:
            AssertionError: 当候选 ID、索引或详情结构非法时抛出。
            ImasSetlistToolError: 当候选不存在或页面请求失败时抛出。
        """
        normalized_id = candidate_id.strip()
        assert re.fullmatch(
            rf"{re.escape(IMAS_SETLIST_CANDIDATE_PREFIX)}[A-Za-z0-9_.-]+",
            normalized_id,
        ), "candidate_id 格式非法"
        candidates = self._load_candidates()
        candidate = next(
            (
                item
                for item in candidates
                if item["candidate_id"] == normalized_id
            ),
            None,
        )
        if candidate is None:
            raise ImasSetlistToolError(
                "candidate_not_found",
                "candidate_id 不存在，请先调用 imas_setlist_search。",
            )

        source_url = str(candidate["source_url"])
        self._assert_detail_url(source_url)
        title, tracks = self._parse_detail(self._fetch_html(source_url))
        return {
            "status": "success",
            "title": title,
            "day": candidate["day"],
            "tracks": tracks,
            "source_url": source_url,
        }

    def _load_candidates(self) -> list[dict[str, str]]:
        """
        每次重新请求索引并解析候选。

        Returns:
            list[dict[str, str]]: 当前索引页全部 Setlist 候选。

        Raises:
            AssertionError: 当索引结构不符合预期时抛出。
            ImasSetlistToolError: 当索引请求失败时抛出。
        """
        soup = BeautifulSoup(
            self._fetch_html(IMAS_SETLIST_INDEX_URL),
            "html.parser",
        )
        main = soup.select_one("main")
        assert isinstance(main, Tag), "索引页缺少 main 元素"
        candidates = [
            candidate
            for anchor in main.select("a[href]")
            if (candidate := self._candidate_from_anchor(anchor)) is not None
        ]
        assert candidates, "索引页未解析到任何 Setlist 候选"
        return candidates

    def _candidate_from_anchor(self, anchor: Tag) -> dict[str, str] | None:
        """
        将一个索引链接转换为候选。

        Args:
            anchor (Tag): 索引页中的链接元素。

        Returns:
            dict[str, str] | None: 候选字典或非 Setlist 链接的 None。

        Raises:
            AssertionError: 当 Setlist 链接缺少必要结构时抛出。
        """
        source_url = urljoin(
            IMAS_SETLIST_INDEX_URL,
            str(anchor.get("href", "")).strip(),
        )
        parsed_url = urlparse(source_url)
        match = re.fullmatch(
            r"/song/event/([A-Za-z0-9_.-]+)\.html",
            parsed_url.path,
        )
        if (
            parsed_url.scheme != "https"
            or parsed_url.netloc != "imas-db.jp"
            or match is None
        ):
            return None

        anchor_text = self._clean_text(anchor.get_text(" ", strip=True))
        anchor_title = self._clean_text(str(anchor.get("title", "")))
        if "楽曲まとめ" in anchor_text or "楽曲と収録CD" in anchor_title:
            return None

        item = anchor.find_parent("li")
        assert isinstance(item, Tag), "Setlist 详情链接必须位于 li 元素中"
        parent_item = self._parent_item(item)
        parent_text = self._direct_item_text(parent_item)
        item_text = self._direct_item_text(item)
        title = anchor_title or self._clean_text(
            f"{parent_text} {anchor_text}"
        )
        assert title, "Setlist 候选标题不能为空"

        date_node = item.find("small", class_="date", recursive=False)
        day = ""
        if isinstance(date_node, Tag):
            day = self._normalize_day(date_node.get_text(" ", strip=True))

        slug = match.group(1)
        candidate_id = f"{IMAS_SETLIST_CANDIDATE_PREFIX}{slug}"
        return {
            "candidate_id": candidate_id,
            "title": title,
            "day": day,
            "source_url": source_url,
            "search_text": self._clean_text(
                " ".join(
                    (
                        candidate_id,
                        title,
                        parent_text,
                        item_text,
                        anchor_text,
                        day,
                    )
                )
            ),
        }

    def _parse_detail(
        self,
        html: str,
    ) -> tuple[str, list[dict[str, str | None]]]:
        """
        解析详情页标题及所有有序号曲目。

        Args:
            html (str): imas-db Setlist 详情 HTML。

        Returns:
            tuple[str, list[dict[str, str | None]]]: 标题和曲目列表。

        Raises:
            AssertionError: 当详情页或曲目结构不符合预期时抛出。
        """
        soup = BeautifulSoup(html, "html.parser")
        title_node = soup.select_one("h1#page_title")
        assert isinstance(title_node, Tag), "详情页缺少 h1#page_title"
        title = self._clean_text(title_node.get_text(" ", strip=True))
        assert title, "详情页标题不能为空"

        tables = soup.select("table.tracklist")
        assert tables, "详情页缺少 table.tracklist"
        tracks: list[dict[str, str | None]] = []
        for table in tables:
            headers = [
                self._clean_text(node.get_text(" ", strip=True))
                for node in table.select("thead th")
            ]
            assert headers == ["No.", "楽曲", "演者"], (
                f"Setlist 表头不符合预期：{headers}"
            )
            for row in table.select("tbody tr"):
                track = self._track_from_row(row)
                if track is not None:
                    tracks.append(track)
        assert tracks, "详情页未解析到任何有序号曲目"
        return title, tracks

    def _track_from_row(
        self,
        row: Tag,
    ) -> dict[str, str | None] | None:
        """
        解析一行曲目，无序号说明行返回 None。

        Args:
            row (Tag): tracklist 表格中的 tr 元素。

        Returns:
            dict[str, str | None] | None: 最小曲目结构或 None。

        Raises:
            AssertionError: 当有序号曲目缺少标题或演者时抛出。
        """
        cells = row.find_all("td", recursive=False)
        if len(cells) != 3:
            return None
        no = self._clean_text(cells[0].get_text(" ", strip=True))
        if not no:
            return None

        title_cell = BeautifulSoup(str(cells[1]), "html.parser").find("td")
        assert isinstance(title_cell, Tag), "曲目标题单元格结构非法"
        brands = [
            self._clean_text(node.get_text(" ", strip=True))
            for node in title_cell.select("small.badge")
        ]
        brand = " / ".join(value for value in brands if value) or None
        for node in title_cell.select("small.badge, .visually-hidden"):
            node.decompose()
        title = self._clean_text(title_cell.get_text(" ", strip=True))
        performers = self._clean_text(cells[2].get_text(" ", strip=True))
        performers = re.sub(r"\s+,\s*", ", ", performers)
        performers = re.sub(r"\s+\)", ")", performers)
        assert title, f"第 {no} 首曲目缺少标题"
        assert performers, f"第 {no} 首曲目缺少演者"
        return {
            "no": no,
            "title": title,
            "brand": brand,
            "performers": performers,
        }

    def _fetch_html(self, url: str) -> str:
        """
        请求并校验一个 HTML 页面。

        Args:
            url (str): 需要获取的页面 URL。

        Returns:
            str: 按 UTF-8 解码的 HTML。

        Raises:
            AssertionError: 当响应内容类型或正文非法时抛出。
            ImasSetlistToolError: 当网络或 HTTP 响应失败时抛出。
        """
        try:
            response = self._http_get(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": IMAS_SETLIST_USER_AGENT,
                },
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise ImasSetlistToolError(
                "upstream_timeout",
                "imas-db Setlist 页面请求超时。",
            ) from exc
        except requests.RequestException as exc:
            raise ImasSetlistToolError(
                "upstream_http_error",
                f"imas-db Setlist 页面请求失败：{exc}",
            ) from exc

        if response.status_code != 200:
            raise ImasSetlistToolError(
                "upstream_http_error",
                f"imas-db Setlist 页面返回 HTTP {response.status_code}。",
            )
        content_type = (response.headers.get("Content-Type") or "").lower()
        assert "text/html" in content_type, "imas-db 响应必须是 HTML"
        response.encoding = "utf-8"
        assert response.text.strip(), "imas-db 响应正文不能为空"
        return response.text

    @staticmethod
    def _parent_item(item: Tag) -> Tag | None:
        """
        查找当前场次所属的父活动列表项。

        Args:
            item (Tag): 当前链接所在列表项。

        Returns:
            Tag | None: 父活动列表项；顶层活动返回 None。

        Raises:
            None: 本方法不主动抛出异常。
        """
        parent_list = item.find_parent("ul")
        if not isinstance(parent_list, Tag):
            return None
        parent_item = parent_list.find_parent("li")
        return parent_item if isinstance(parent_item, Tag) else None

    @staticmethod
    def _direct_item_text(item: Tag | None) -> str:
        """
        提取列表项自身文本，不混入嵌套场次。

        Args:
            item (Tag | None): 待提取的列表项。

        Returns:
            str: 清洗后的自身文本。

        Raises:
            None: 本方法不主动抛出异常。
        """
        if item is None:
            return ""
        cloned = BeautifulSoup(str(item), "html.parser").find("li")
        if not isinstance(cloned, Tag):
            return ""
        for nested_list in cloned.find_all(["ul", "ol"]):
            nested_list.decompose()
        return ImasSetlistClient._clean_text(
            cloned.get_text(" ", strip=True)
        )

    @staticmethod
    def _query_terms(query: str) -> tuple[str, ...]:
        """
        将查询拆为标准化的全部包含匹配词。

        Args:
            query (str): 已清理首尾空白的查询词。

        Returns:
            tuple[str, ...]: 非空标准化查询词。

        Raises:
            AssertionError: 当标准化后无有效内容时抛出。
        """
        terms = tuple(
            compact
            for raw in re.split(r"\s+", unicodedata.normalize("NFKC", query))
            if (compact := ImasSetlistClient._compact_text(raw))
        )
        assert terms, "query 标准化后不能为空"
        return terms

    @staticmethod
    def _matches(search_text: str, terms: tuple[str, ...]) -> bool:
        """
        判断候选是否包含全部查询词。

        Args:
            search_text (str): 候选内部搜索文本。
            terms (tuple[str, ...]): 标准化查询词。

        Returns:
            bool: True 表示全部查询词命中。

        Raises:
            AssertionError: 当查询词为空时抛出。
        """
        assert terms, "terms 不能为空"
        candidate = ImasSetlistClient._compact_text(search_text)
        return all(term in candidate for term in terms)

    @staticmethod
    def _compact_text(value: str) -> str:
        """
        标准化全半角与大小写，并去除空格和标点。

        Args:
            value (str): 待标准化文本。

        Returns:
            str: 仅保留字母、数字和中日韩文字的紧凑文本。

        Raises:
            None: 本方法不主动抛出异常。
        """
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return "".join(character for character in normalized if character.isalnum())

    @staticmethod
    def _normalize_day(value: str) -> str:
        """
        去除日期文本的列表装饰符。

        Args:
            value (str): 页面日期原文。

        Returns:
            str: 清理后的日期文本。

        Raises:
            None: 本方法不主动抛出异常。
        """
        day = ImasSetlistClient._clean_text(value).removeprefix("-").strip()
        if day.startswith("(") and day.endswith(")"):
            day = day[1:-1].strip()
        return day

    @staticmethod
    def _clean_text(value: str) -> str:
        """
        折叠文本中的连续空白。

        Args:
            value (str): 原始文本。

        Returns:
            str: 清洗后的文本。

        Raises:
            None: 本方法不主动抛出异常。
        """
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _assert_detail_url(source_url: str) -> None:
        """
        校验详情 URL 仅指向 imas-db Setlist 目录。

        Args:
            source_url (str): 待校验绝对 URL。

        Returns:
            None: 校验通过时无返回值。

        Raises:
            AssertionError: 当 URL 主机或路径不符合要求时抛出。
        """
        parsed = urlparse(source_url)
        assert parsed.scheme == "https", "详情页必须使用 HTTPS"
        assert parsed.netloc == "imas-db.jp", "详情页主机必须是 imas-db.jp"
        assert re.fullmatch(
            r"/song/event/[A-Za-z0-9_.-]+\.html",
            parsed.path,
        ), "详情页路径不合法"
        assert not parsed.query and not parsed.fragment, "详情页不能包含查询或锚点"


def _failure_payload(
    action: str,
    input_value: str,
    error: ImasSetlistToolError | AssertionError,
) -> str:
    """
    将工具异常转换为稳定失败 JSON。

    Args:
        action (str): search 或 get。
        input_value (str): 当前查询词或候选 ID。
        error (ImasSetlistToolError | AssertionError): 待转换异常。

    Returns:
        str: 结构化失败 JSON。

    Raises:
        AssertionError: 当动作或输入值为空时抛出。
    """
    assert action.strip(), "action 不能为空"
    assert input_value.strip(), "input_value 不能为空"
    error_code = (
        error.error_code
        if isinstance(error, ImasSetlistToolError)
        else "invalid_response"
    )
    return json.dumps(
        {
            "status": "failed",
            "action": action,
            "input": input_value,
            "error": error_code,
            "message": str(error),
        },
        ensure_ascii=False,
    )


@tool("imas_setlist_search")
def imas_setlist_search(query: str) -> str:
    """
    搜索偶像大师（THE IDOLM@STER／アイマス）企划的演唱会歌单候选。

    当用户询问偶像大师各品牌或企划的演唱会、Live、活动歌单
    （Setlist）、曲目表、演唱曲目，或“某场演出唱了什么”时，先调用
    本工具。可使用活动名、公演名、年份、DAY 或品牌关键词搜索
    imas-db；本工具返回全部匹配候选，每项只包含 candidate_id、title
    和 day。

    获取某一场演出的具体歌单时，必须把返回的 candidate_id 原样传给
    ``imas_setlist_get``，不得自行猜测候选 ID。

    Args:
        query (str): 非空搜索词，例如活动名、年份、品牌或 DAY。

    Returns:
        str: JSON 字符串形式的全部匹配候选。

    Raises:
        AssertionError: 当 query 为空或过长时抛出。
    """
    normalized_query = query.strip()
    assert normalized_query, "query 不能为空"
    assert len(normalized_query) <= 200, "query 长度不能超过 200"
    try:
        result = ImasSetlistClient().search(normalized_query)
    except (ImasSetlistToolError, AssertionError) as exc:
        return _failure_payload("search", normalized_query, exc)
    return json.dumps(result, ensure_ascii=False)


@tool("imas_setlist_get")
def imas_setlist_get(candidate_id: str) -> str:
    """
    获取指定偶像大师演唱会、Live 或活动的完整歌单（Setlist）。

    仅在 ``imas_setlist_search`` 找到目标场次后调用，并原样传入搜索
    结果中的 candidate_id。返回该场演出的活动标题、日期，以及按演出
    顺序排列的曲目表；每首曲目仅包含序号 no、歌名 title、偶像大师
    品牌或企划 brand、演唱者 performers，同时附上 imas-db 来源链接。

    适合回答“这场偶像大师演出有哪些歌”“第几首是什么歌”“谁演唱了
    哪首歌”等需要精确歌单内容的问题。

    Args:
        candidate_id (str): 搜索工具返回的精确候选 ID。

    Returns:
        str: JSON 字符串形式的活动与曲目。

    Raises:
        AssertionError: 当 candidate_id 为空或过长时抛出。
    """
    normalized_id = candidate_id.strip()
    assert normalized_id, "candidate_id 不能为空"
    assert len(normalized_id) <= 200, "candidate_id 长度不能超过 200"
    try:
        result = ImasSetlistClient().get(normalized_id)
    except (ImasSetlistToolError, AssertionError) as exc:
        return _failure_payload("get", normalized_id, exc)
    return json.dumps(result, ensure_ascii=False)
