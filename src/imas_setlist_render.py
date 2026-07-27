"""
imas-db Setlist 原始表格清洗与固定样式单图渲染。

图片后端保留页面中全部 Setlist 表格行和可见文本，只移除链接、事件属性
及不参与截图的元数据。渲染布局使用本地固定 CSS，颜色优先采用详情页指定
的 imas-db 官方颜色表。
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, Tag

from src.imas_setlist_tool import ImasSetlistClient, ImasSetlistPageSource

if TYPE_CHECKING:
    from playwright.sync_api import Page


IMAS_SETLIST_CSS_WIDTH: Final[int] = 640
IMAS_SETLIST_DEVICE_SCALE_FACTOR: Final[int] = 2
IMAS_SETLIST_PIXEL_WIDTH: Final[int] = (
    IMAS_SETLIST_CSS_WIDTH * IMAS_SETLIST_DEVICE_SCALE_FACTOR
)
IMAS_SETLIST_ALLOWED_TAGS: Final[frozenset[str]] = frozenset(
    {
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "a",
        "span",
        "small",
        "br",
        "ruby",
        "rb",
        "rt",
        "rp",
        "ol",
        "li",
        "p",
    }
)
IMAS_SETLIST_ALLOWED_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {"class", "colspan", "rowspan"}
)
IMAS_SETLIST_PALETTE_STYLESHEET_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/css/imas.css",
        "/css/imas.min.css",
    }
)


def _is_trusted_palette_stylesheet_url(stylesheet_url: str) -> bool:
    """
    判断颜色表 URL 是否属于允许访问的 imas-db 官方路径。

    Args:
        stylesheet_url (str): 待校验的绝对 URL。

    Returns:
        bool: 仅当 URL 使用官方 HTTPS 主机和固定 CSS 路径时为真。

    Raises:
        None: 本函数不主动抛出异常。
    """
    parsed = urlparse(stylesheet_url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "imas-db.jp"
        and parsed.path in IMAS_SETLIST_PALETTE_STYLESHEET_PATHS
        and not parsed.fragment
    )


IMAS_SETLIST_FIXED_CSS: Final[str] = """
:root {
  --setlist-pink: #ff74b8;
  --setlist-border: #e4e4e4;
  --brand-general: #747488;
  --brand-idolmaster: #ff74b8;
  --brand-765as: #f34f6d;
  --brand-cinderella: #2681c8;
  --brand-million: #ffc30b;
  --brand-sidem: #0fbe94;
  --brand-shinycolors: #8dbbff;
  --brand-gakuen: #f39800;
  --brand-dearlystars: orange;
  --brand-valiv: #656a75;
  --cinderella-cute: #ef2782;
  --cinderella-cool: #006aff;
  --cinderella-passion: #f49207;
  --million-princess: #ea3f83;
  --million-fairy: #275cf6;
  --million-angel: #f5be41;
  --million-vocal: #ff3366;
  --million-dance: #0099cc;
  --million-visual: #ff9900;
  --sidem-physical: #e71b27;
  --sidem-intelli: #438fcd;
  --sidem-mental: #f9be00;
  --collaboration-nijisanji: rgb(45, 75, 112);
  --collaboration-hololive: #27c7ff;
  --asobinotes: #101010;
  --denonbu: #151412;
  --susanoo-magic: #036eb6;
  --namco-taiko: #eb1e06;
  --namco-katamari: #8eb538;
  --lovelive: #e50080;
  --lovelive-sunshine: #19b1f6;
  --lovelive-nijigasaki: #f8b656;
  --lovelive-superstar: #da57d8;
  --lovelive-musical: #c40035;
  --lovelive-hasunosora: #fb8a9b;
}
* {
  box-sizing: border-box;
}
html,
body {
  width: 640px;
  margin: 0;
  padding: 0;
  background: var(--setlist-pink);
  font-family: "Noto Sans CJK JP", sans-serif;
}
#setlist-card {
  width: 640px;
  overflow: hidden;
  color: #202124;
  background: #fff;
  border-right: 2px solid var(--setlist-pink);
  border-left: 2px solid var(--setlist-pink);
}
.setlist-header {
  padding: 12px 12px 9px;
  color: #fff;
  background: var(--setlist-pink);
  border-bottom: 2px solid #fff;
  font-size: 20px;
  font-weight: 700;
  line-height: 1.2;
}
.event-meta {
  padding: 13px 16px 11px;
  background: #fff;
  border-bottom: 1px solid #cfcfcf;
}
.event-meta h1 {
  margin: 0 0 6px;
  color: #555;
  font-size: 20px;
  font-weight: 700;
  line-height: 1.35;
  text-align: center;
  text-shadow: 1px 1px #eee;
}
.event-meta h1:last-child {
  margin-bottom: 0;
}
.event-meta p {
  margin: 0;
  color: #666;
  font-size: 11px;
  line-height: 1.45;
  text-align: center;
}
.setlist-content {
  background: #fff;
}
.visually-hidden {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}
table.tracklist {
  --tracklist-number-width: 3.4rem;
  display: block;
  width: 100%;
  margin: 0;
  font-size: 18px;
  line-height: 1.42;
  border-collapse: collapse;
}
table.tracklist thead {
  display: none;
}
table.tracklist tbody {
  display: block;
}
table.tracklist tbody tr {
  display: grid;
  grid-template-columns: var(--tracklist-number-width) 1fr;
  gap: 0 .5rem;
  width: 100%;
  margin: 0;
  padding: 8px 5px 7px 0;
  border-bottom: 1px solid var(--setlist-border);
  vertical-align: top;
}
table.tracklist td,
table.tracklist th {
  display: inline-block;
  width: auto;
  margin: 0;
  padding: 0;
  grid-column: auto;
}
table.tracklist tr td:first-child,
table.tracklist tr th:first-child {
  padding-left: 4px;
  color: #292929;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
table.tracklist tr td:first-child::after {
  content: ".";
}
table.tracklist tr td:first-child:empty::after {
  content: "";
}
table.tracklist tr.extra td:first-child::before {
  content: "* ";
}
table.tracklist tr td:nth-child(2),
table.tracklist tr th:nth-child(2) {
  color: #202124;
  font-size: 19px;
  line-height: 1.35;
  text-shadow: 1px 1px #ddd;
}
table.tracklist tr td:nth-child(3),
table.tracklist tr th:nth-child(3) {
  grid-column: 2;
}
table.tracklist tr td:nth-child(3) {
  padding-top: 2px;
  color: #303134;
  font-size: 16px;
  line-height: 1.45;
}
table.tracklist tr.part-header {
  grid-template-columns: 1fr;
  padding: 5px 8px;
  color: #606060;
  background: #f4f4f4;
  font-size: 17px;
  font-weight: 700;
  text-align: center;
  text-shadow: 1px 1px #fff;
}
table.tracklist tr.part-header td,
table.tracklist tr.part-header th {
  grid-column: 1;
  width: 100%;
  text-align: center;
}
table.tracklist tr.part-header td::before,
table.tracklist tr.part-header td::after {
  content: "";
}
table.tracklist tr.part-header:empty {
  min-height: 8px;
  padding: 0;
}
table.tracklist a {
  color: inherit;
  text-decoration: none;
}
table.tracklist ol {
  margin: 3px 0 0;
  padding-left: 1.4em;
}
table.tracklist .notes,
table.tracklist .caption,
table.tracklist .additional {
  color: #666;
  font-size: 88.8%;
}
.setlist-content > .notes {
  margin: 0;
  padding: 10px 16px 11px;
  color: #666;
  border-bottom: 1px solid var(--setlist-border);
  font-size: 16px;
  font-weight: 400;
  line-height: 1.45;
}
.badge {
  display: inline-block;
  margin-left: 3px;
  padding: .32em .58em;
  color: #fff;
  border-radius: 7px;
  font-size: 13px;
  font-weight: 700;
  line-height: 1;
  text-align: center;
  white-space: nowrap;
  vertical-align: 2px;
  text-shadow: none;
}
.bg-imas-brand-idolmaster { background-color: var(--brand-idolmaster) !important; }
.bg-imas-brand-765as { background-color: var(--brand-765as) !important; }
.bg-imas-brand-cinderella { background-color: var(--brand-cinderella) !important; }
.bg-imas-brand-million { background-color: var(--brand-million) !important; color: #202124; }
.bg-imas-brand-sidem { background-color: var(--brand-sidem) !important; }
.bg-imas-brand-shinycolors { background-color: var(--brand-shinycolors) !important; color: #202124; }
.bg-imas-brand-gakuen { background-color: var(--brand-gakuen) !important; }
.bg-imas-brand-dearlystars,
.bg-imas-brand-876pro { background-color: var(--brand-dearlystars) !important; }
.bg-imas-brand-valiv { background-color: var(--brand-valiv) !important; }
.bg-imas-brand-961pro,
.bg-imas-brand-others,
.bg-imas-brand-xenoglossia { background-color: var(--brand-general) !important; }
.bg-nijisanji { background-color: var(--collaboration-nijisanji) !important; }
.bg-hololive { background-color: var(--collaboration-hololive) !important; }
.bg-asobinotes-brand-asobinotes { background-color: var(--asobinotes) !important; }
.bg-asobinotes-brand-denonbu { background-color: var(--denonbu) !important; }
.bg-asobinotes-brand-susanoo-magic { background-color: var(--susanoo-magic) !important; }
.bg-namco-brand-taiko { background-color: var(--namco-taiko) !important; }
.bg-namco-brand-katamaridamacy { background-color: var(--namco-katamari) !important; }
.bg-lovelive-brand-lovelive,
.bg-lovelive-brand-muse { background-color: var(--lovelive) !important; }
.bg-lovelive-brand-sunshine { background-color: var(--lovelive-sunshine) !important; }
.bg-lovelive-brand-nijigasaki { background-color: var(--lovelive-nijigasaki) !important; color: #202124; }
.bg-lovelive-brand-superstar { background-color: var(--lovelive-superstar) !important; }
.bg-lovelive-brand-musical { background-color: var(--lovelive-musical) !important; }
.bg-lovelive-brand-hasunosora { background-color: var(--lovelive-hasunosora) !important; }
.idol-name {
  display: inline;
  border-bottom: 2px solid var(--brand-general);
}
.idol-name[data-brand-id="0"] { border-color: var(--brand-idolmaster); }
.idol-name[data-brand-id="1"] { border-color: var(--brand-765as); }
.idol-name[data-brand-id="3"] { border-color: var(--brand-dearlystars); }
.idol-name[data-brand-id="4"] { border-color: var(--brand-cinderella); }
.idol-name[data-brand-id="5"] { border-color: var(--brand-million); }
.idol-name[data-brand-id="6"] { border-color: var(--brand-sidem); }
.idol-name[data-brand-id="8"] { border-color: var(--brand-shinycolors); }
.idol-name[data-brand-id="10"] { border-color: var(--brand-valiv); }
.idol-name[data-brand-id="11"] { border-color: var(--brand-gakuen); }
.idol-name[data-cinderella-attr="1"] { border-color: var(--cinderella-cute); }
.idol-name[data-cinderella-attr="2"] { border-color: var(--cinderella-cool); }
.idol-name[data-cinderella-attr="3"] { border-color: var(--cinderella-passion); }
.idol-name[data-million-attr="1"] { border-color: var(--million-princess); }
.idol-name[data-million-attr="2"] { border-color: var(--million-fairy); }
.idol-name[data-million-attr="3"] { border-color: var(--million-angel); }
.idol-name[data-million-gree-attr="1"] { border-color: var(--million-vocal); }
.idol-name[data-million-gree-attr="2"] { border-color: var(--million-dance); }
.idol-name[data-million-gree-attr="3"] { border-color: var(--million-visual); }
.idol-name[data-sidem-attr="1"] { border-color: var(--sidem-physical); }
.idol-name[data-sidem-attr="2"] { border-color: var(--sidem-intelli); }
.idol-name[data-sidem-attr="3"] { border-color: var(--sidem-mental); }
.idol-name[data-nijisanji] {
  border-bottom-style: dashed;
  border-color: var(--collaboration-nijisanji);
}
.idol-name[data-hololive] {
  border-bottom-style: dashed;
  border-color: var(--collaboration-hololive);
}
.setlist-footer {
  padding: 4px 10px 5px;
  color: #aaa;
  background: #fff;
  border-top: 1px solid #eee;
  font-size: 8px;
  line-height: 1.2;
  text-align: right;
}
"""


@dataclass(frozen=True, slots=True)
class ImasSetlistRenderDocument:
    """
    表示已经安全清洗、可直接渲染的 Setlist 文档。

    Args:
        candidate_id (str): 搜索工具返回的候选 ID。
        title (str): 详情页活动标题。
        day (str): 索引页活动日期，可为空。
        source_url (str): 详情页来源 URL，仅用于内部追踪。
        palette_stylesheet_url (str): 详情页指定的官方颜色表 URL。
        theme_brand_id (int): 官网歌单区块声明的主题品牌 ID。
        tables_html (str): 清洗后的全部 Setlist 表格 HTML。
        table_count (int): 表格数量。
        row_count (int): 表格行总数。
        warnings (tuple[str, ...]): 不影响渲染的兼容警告。
        venue (str): 官方活动页提供的场馆，可为空。
    """

    candidate_id: str
    title: str
    day: str
    source_url: str
    palette_stylesheet_url: str
    theme_brand_id: int
    tables_html: str
    table_count: int
    row_count: int
    warnings: tuple[str, ...]
    venue: str = ""


@dataclass(frozen=True, slots=True)
class ImasSetlistRenderedImage:
    """
    表示单张 Setlist PNG 及其结构化元信息。

    Args:
        candidate_id (str): 搜索工具返回的候选 ID。
        title (str): 详情页活动标题。
        day (str): 索引页活动日期，可为空。
        png_bytes (bytes): 单张 PNG 图片内容。
        table_count (int): 渲染的表格数量。
        row_count (int): 渲染的表格行总数。
        warnings (tuple[str, ...]): 不影响渲染的兼容警告。
    """

    candidate_id: str
    title: str
    day: str
    png_bytes: bytes
    table_count: int
    row_count: int
    warnings: tuple[str, ...]


class ImasSetlistRenderError(RuntimeError):
    """表示可安全暴露给 Agent 的 Setlist 图片渲染错误。"""

    def __init__(self, error_code: str, message: str) -> None:
        """
        初始化图片渲染错误。

        Args:
            error_code (str): 稳定错误码。
            message (str): 面向 Agent 的错误说明。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当错误码或说明为空时抛出。
        """
        assert error_code.strip(), "error_code 不能为空"
        assert message.strip(), "message 不能为空"
        super().__init__(message)
        self.error_code = error_code.strip()


class ImasSetlistDocumentParser:
    """将详情页转换为保留全部可见表格内容的安全渲染文档。"""

    def parse(
        self,
        source: ImasSetlistPageSource,
    ) -> ImasSetlistRenderDocument:
        """
        清洗详情页中的全部 Setlist 表格。

        Args:
            source (ImasSetlistPageSource): 已通过候选校验的详情页来源。

        Returns:
            ImasSetlistRenderDocument: 可安全嵌入本地模板的文档。

        Raises:
            ImasSetlistRenderError: 当页面没有可显示的 Setlist 内容时抛出。
        """
        soup = BeautifulSoup(source.html, "html.parser")
        title_node = soup.select_one("h1#page_title")
        if not isinstance(title_node, Tag) or not title_node.get_text(
            strip=True
        ):
            raise ImasSetlistRenderError(
                "setlist_title_missing",
                "Setlist 详情页缺少活动标题。",
            )
        title = self._clean_text(title_node.get_text(" ", strip=True))
        palette_stylesheet_url = self._resolve_palette_stylesheet_url(
            soup,
            source.source_url,
        )
        warnings: list[str] = []

        tables = soup.select("table.tracklist")
        if not tables:
            raise ImasSetlistRenderError(
                "setlist_table_missing",
                "Setlist 页面没有可渲染的曲目表。",
            )
        theme_brand_id = self._resolve_theme_brand_id(tables)
        sanitized_content: list[str] = []
        row_count = 0
        for table in tables:
            cleaned, table_warnings = self._sanitize_table(table)
            warnings.extend(table_warnings)
            row_count += len(cleaned.select("tr"))
            sanitized_content.append(str(cleaned))
            cleaned_notes, note_warnings = (
                self._sanitize_following_notes(table)
            )
            warnings.extend(note_warnings)
            sanitized_content.extend(str(note) for note in cleaned_notes)
        if row_count <= 0:
            raise ImasSetlistRenderError(
                "setlist_rows_missing",
                "Setlist 页面没有可渲染的表格行。",
            )
        return ImasSetlistRenderDocument(
            candidate_id=source.candidate_id,
            title=title,
            day=source.day.strip(),
            source_url=source.source_url,
            palette_stylesheet_url=palette_stylesheet_url,
            theme_brand_id=theme_brand_id,
            tables_html="\n".join(sanitized_content),
            table_count=len(tables),
            row_count=row_count,
            warnings=tuple(dict.fromkeys(warnings)),
            venue=source.venue.strip(),
        )

    def _sanitize_table(self, table: Tag) -> tuple[Tag, tuple[str, ...]]:
        """
        清洗单张表格，并保留未知标签中的可见文字。

        Args:
            table (Tag): 原始 tracklist 表格。

        Returns:
            tuple[Tag, tuple[str, ...]]: 清洗后的表格与兼容警告。

        Raises:
            ImasSetlistRenderError: 当表格克隆失败或文字被意外丢失时抛出。
        """
        source_text = self._clean_text(table.get_text(" ", strip=True))
        cloned_soup = BeautifulSoup(str(table), "html.parser")
        cloned = cloned_soup.select_one("table")
        if not isinstance(cloned, Tag):
            raise ImasSetlistRenderError(
                "setlist_table_invalid",
                "Setlist 表格无法安全克隆。",
            )
        for comment in cloned.find_all(
            string=lambda value: isinstance(value, Comment)
        ):
            comment.extract()

        warnings: list[str] = []
        for node in list(cloned.find_all(True)):
            if node.name not in IMAS_SETLIST_ALLOWED_TAGS:
                warnings.append(f"unwrapped_tag:{node.name}")
                node.unwrap()
                continue
            node.attrs = {
                key: value
                for key, value in node.attrs.items()
                if (
                    key in IMAS_SETLIST_ALLOWED_ATTRIBUTES
                    or key.startswith("data-")
                )
            }
        cleaned_text = self._clean_text(cloned.get_text(" ", strip=True))
        if cleaned_text != source_text:
            raise ImasSetlistRenderError(
                "setlist_text_changed",
                "Setlist 清洗前后可见文字不一致。",
            )
        return cloned, tuple(dict.fromkeys(warnings))

    def _sanitize_following_notes(
        self,
        table: Tag,
    ) -> tuple[list[Tag], tuple[str, ...]]:
        """
        清洗紧邻当前歌单表格的全部官网注解段落。

        Args:
            table (Tag): 原始 tracklist 表格。

        Returns:
            tuple[list[Tag], tuple[str, ...]]: 安全注解及兼容警告。

        Raises:
            ImasSetlistRenderError: 当注解无法安全克隆或文字变化时抛出。
        """
        cleaned_notes: list[Tag] = []
        warnings: list[str] = []
        for sibling in table.next_siblings:
            if isinstance(sibling, Comment):
                continue
            if isinstance(sibling, str):
                if not sibling.strip():
                    continue
                break
            if not isinstance(sibling, Tag):
                break
            if (
                sibling.name != "p"
                or "notes" not in sibling.get("class", [])
            ):
                break
            cleaned, note_warnings = self._sanitize_note(sibling)
            cleaned_notes.append(cleaned)
            warnings.extend(note_warnings)
        return cleaned_notes, tuple(dict.fromkeys(warnings))

    def _sanitize_note(self, note: Tag) -> tuple[Tag, tuple[str, ...]]:
        """
        清洗单条官网注解，并保留其中全部可见文字。

        Args:
            note (Tag): 原始 p.notes 注解。

        Returns:
            tuple[Tag, tuple[str, ...]]: 清洗后的注解与兼容警告。

        Raises:
            ImasSetlistRenderError: 当注解克隆失败或可见文字变化时抛出。
        """
        source_text = self._clean_text(note.get_text(" ", strip=True))
        cloned_soup = BeautifulSoup(str(note), "html.parser")
        cloned = cloned_soup.select_one("p.notes")
        if not isinstance(cloned, Tag):
            raise ImasSetlistRenderError(
                "setlist_note_invalid",
                "Setlist 注解无法安全克隆。",
            )
        for comment in cloned.find_all(
            string=lambda value: isinstance(value, Comment)
        ):
            comment.extract()

        warnings: list[str] = []
        for node in [cloned, *list(cloned.find_all(True))]:
            if node.name not in IMAS_SETLIST_ALLOWED_TAGS:
                warnings.append(f"unwrapped_tag:{node.name}")
                node.unwrap()
                continue
            node.attrs = {
                key: value
                for key, value in node.attrs.items()
                if (
                    key in IMAS_SETLIST_ALLOWED_ATTRIBUTES
                    or key.startswith("data-")
                )
            }
        cleaned_text = self._clean_text(cloned.get_text(" ", strip=True))
        if cleaned_text != source_text:
            raise ImasSetlistRenderError(
                "setlist_note_text_changed",
                "Setlist 注解清洗前后可见文字不一致。",
            )
        return cloned, tuple(dict.fromkeys(warnings))

    @staticmethod
    def _resolve_palette_stylesheet_url(
        soup: BeautifulSoup,
        source_url: str,
    ) -> str:
        """
        解析并校验详情页声明的 imas-db 官方颜色表地址。

        Args:
            soup (BeautifulSoup): 详情页 DOM。
            source_url (str): 已校验的详情页 URL。

        Returns:
            str: 绝对 HTTPS 官方颜色表 URL。

        Raises:
            ImasSetlistRenderError: 当页面没有可信颜色表时抛出。
        """
        for stylesheet in soup.select(
            'link[rel~="stylesheet"][href]'
        ):
            href = stylesheet.get("href")
            if not isinstance(href, str) or not href.strip():
                continue
            stylesheet_url = urljoin(source_url, href.strip())
            if _is_trusted_palette_stylesheet_url(stylesheet_url):
                return stylesheet_url
        raise ImasSetlistRenderError(
            "setlist_palette_stylesheet_missing",
            "Setlist 详情页缺少可信的 imas-db 官方颜色表。",
        )

    @staticmethod
    def _resolve_theme_brand_id(tables: list[Tag]) -> int:
        """
        读取官网包裹歌单区块声明的统一主题品牌 ID。

        Args:
            tables (list[Tag]): 详情页中的全部 Setlist 表格。

        Returns:
            int: 官网颜色表可识别的品牌 ID。

        Raises:
            ImasSetlistRenderError: 当品牌 ID 缺失、非法或不一致时抛出。
        """
        brand_ids: list[int] = []
        for table in tables:
            section = table.find_parent(class_="section")
            if not isinstance(section, Tag):
                raise ImasSetlistRenderError(
                    "setlist_theme_brand_missing",
                    "Setlist 表格缺少官网主题区块。",
                )
            raw_brand_id = section.get("data-brand-id")
            if raw_brand_id is None:
                brand_ids.append(0)
                continue
            if (
                not isinstance(raw_brand_id, str)
                or not raw_brand_id.isascii()
                or not raw_brand_id.isdigit()
            ):
                raise ImasSetlistRenderError(
                    "setlist_theme_brand_invalid",
                    "Setlist 官网主题区块包含非法品牌 ID。",
                )
            brand_id = int(raw_brand_id)
            if not 0 <= brand_id <= 255:
                raise ImasSetlistRenderError(
                    "setlist_theme_brand_invalid",
                    "Setlist 官网主题品牌 ID 超出允许范围。",
                )
            brand_ids.append(brand_id)
        if len(set(brand_ids)) != 1:
            raise ImasSetlistRenderError(
                "setlist_theme_brand_inconsistent",
                "同一张 Setlist 图片包含不一致的官网主题品牌。",
            )
        return brand_ids[0]

    @staticmethod
    def _clean_text(value: str) -> str:
        """
        折叠连续空白以比较可见文字。

        Args:
            value (str): 原始文字。

        Returns:
            str: 折叠空白后的文字。

        Raises:
            None: 本方法不主动抛出异常。
        """
        return " ".join(value.split())


class ImasSetlistHtmlRenderer:
    """将安全文档放入已经确认的固定 1280px 图片模板。"""

    def render(self, document: ImasSetlistRenderDocument) -> str:
        """
        生成不依赖远程资源的完整 HTML。

        Args:
            document (ImasSetlistRenderDocument): 安全渲染文档。

        Returns:
            str: 可交给 Chromium 截图的 HTML。

        Raises:
            AssertionError: 当文档缺少必要内容时抛出。
        """
        assert document.title.strip(), "渲染标题不能为空"
        assert document.tables_html.strip(), "渲染表格不能为空"
        metadata = tuple(
            value
            for value in (document.day, document.venue)
            if value
        )
        metadata_html = (
            f"<p>{'　·　'.join(html.escape(value) for value in metadata)}</p>"
            if metadata
            else ""
        )
        return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <style>
{IMAS_SETLIST_FIXED_CSS}
  </style>
</head>
<body>
  <article
    id="setlist-card"
    data-theme-brand-id="{document.theme_brand_id}"
  >
    <header class="setlist-header">♪ セットリスト</header>
    <section class="event-meta">
      <h1>{html.escape(document.title)}</h1>
      {metadata_html}
    </section>
    <section class="setlist-content">{document.tables_html}</section>
    <footer class="setlist-footer">
      © imas-db.jp　天海春香Agent 生成
    </footer>
  </article>
</body>
</html>"""


IMAS_SETLIST_APPLY_PALETTE_SCRIPT: Final[str] = """
async ({stylesheetUrl, themeBrandId, timeoutMs}) => {
  const themeProbeStyle = document.createElement("style");
  themeProbeStyle.textContent = (
    ".section { border-color: rgb(1, 2, 3); }"
  );
  document.head.appendChild(themeProbeStyle);
  const themeProbe = document.createElement("div");
  themeProbe.className = "section";
  themeProbe.dataset.brandId = String(themeBrandId);
  themeProbe.hidden = true;
  document.body.appendChild(themeProbe);

  const stylesheet = document.createElement("link");
  stylesheet.rel = "stylesheet";
  stylesheet.href = stylesheetUrl;

  const loadStatus = await new Promise((resolve) => {
    let settled = false;
    const finish = (status) => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timerId);
      resolve(status);
    };
    stylesheet.addEventListener(
      "load",
      () => finish("success"),
      {once: true}
    );
    stylesheet.addEventListener(
      "error",
      () => finish("load_failed"),
      {once: true}
    );
    const timerId = window.setTimeout(
      () => finish("load_failed"),
      timeoutMs
    );
    document.head.appendChild(stylesheet);
  });
  if (loadStatus !== "success") {
    stylesheet.remove();
    themeProbe.remove();
    themeProbeStyle.remove();
    return {status: loadStatus};
  }

  const isColor = (value) => CSS.supports("color", value);
  const isBorderStyle = (value) => (
    CSS.supports("border-bottom-style", value)
  );
  let invalidValueCount = 0;
  const themeColor = (
    window.getComputedStyle(themeProbe).borderTopColor
  );
  if (
    !isColor(themeColor)
    || themeColor === "rgb(1, 2, 3)"
  ) {
    invalidValueCount += 1;
  } else {
    document.documentElement.style.setProperty(
      "--setlist-pink",
      themeColor,
      "important"
    );
  }
  const idolNames = Array.from(
    document.querySelectorAll(".idol-name")
  );
  for (const element of idolNames) {
    const computed = window.getComputedStyle(element);
    if (
      !isColor(computed.borderBottomColor)
      || !isBorderStyle(computed.borderBottomStyle)
    ) {
      invalidValueCount += 1;
      continue;
    }
    element.style.setProperty(
      "border-bottom-color",
      computed.borderBottomColor,
      "important"
    );
    element.style.setProperty(
      "border-bottom-style",
      computed.borderBottomStyle,
      "important"
    );
  }

  const badges = Array.from(document.querySelectorAll(".badge"));
  for (const element of badges) {
    const computed = window.getComputedStyle(element);
    if (
      !isColor(computed.backgroundColor)
      || !isColor(computed.color)
    ) {
      invalidValueCount += 1;
      continue;
    }
    element.style.setProperty(
      "background-color",
      computed.backgroundColor,
      "important"
    );
    element.style.setProperty(
      "color",
      computed.color,
      "important"
    );
  }

  stylesheet.remove();
  themeProbe.remove();
  themeProbeStyle.remove();
  await new Promise((resolve) => {
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(resolve);
    });
  });
  if (invalidValueCount > 0) {
    return {status: "invalid_color"};
  }
  return {
    status: "success",
    theme_color: themeColor,
    idol_name_count: idolNames.length,
    badge_count: badges.length,
  };
}
"""


class BrowserImasSetlistPaletteApplier:
    """通过官方 CSS 计算颜色并固化允许的视觉属性。"""

    def apply(
        self,
        page: Page,
        stylesheet_url: str,
        theme_brand_id: int,
        timeout_ms: int,
    ) -> None:
        """
        临时加载官方颜色表，并将计算结果写入安全内联样式。

        Args:
            page (Page): 已载入本地固定模板的 Playwright 页面。
            stylesheet_url (str): 已通过白名单校验的官方颜色表 URL。
            theme_brand_id (int): 官网歌单区块声明的品牌 ID。
            timeout_ms (int): 样式表加载超时毫秒数。

        Returns:
            None: 成功时颜色已固化，官方样式表已移除。

        Raises:
            AssertionError: 当地址为空或超时不为正数时抛出。
            ImasSetlistRenderError: 当颜色表地址不可信或无法应用时抛出。
        """
        assert stylesheet_url.strip(), "stylesheet_url 不能为空"
        assert 0 <= theme_brand_id <= 255, (
            "theme_brand_id 必须位于 0 到 255"
        )
        assert timeout_ms > 0, "timeout_ms 必须大于 0"
        if not _is_trusted_palette_stylesheet_url(stylesheet_url):
            raise ImasSetlistRenderError(
                "setlist_palette_stylesheet_untrusted",
                "Setlist 颜色表地址不属于允许的 imas-db 官方路径。",
            )
        result: dict[str, object] = page.evaluate(
            IMAS_SETLIST_APPLY_PALETTE_SCRIPT,
            {
                "stylesheetUrl": stylesheet_url,
                "themeBrandId": theme_brand_id,
                "timeoutMs": timeout_ms,
            },
        )
        if result["status"] != "success":
            raise ImasSetlistRenderError(
                "setlist_palette_stylesheet_failed",
                "imas-db 官方颜色表加载或颜色解析失败。",
            )


class BrowserImasSetlistRenderer:
    """使用无头 Chromium 将固定模板截图为单张 PNG。"""

    def __init__(
        self,
        html_renderer: ImasSetlistHtmlRenderer | None = None,
        palette_applier: BrowserImasSetlistPaletteApplier | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        """
        初始化浏览器渲染器。

        Args:
            html_renderer (ImasSetlistHtmlRenderer | None): HTML 渲染器。
            palette_applier (BrowserImasSetlistPaletteApplier | None):
                官网颜色应用器。
            timeout_ms (int): 单次页面渲染超时毫秒数。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当超时不为正数时抛出。
        """
        assert timeout_ms > 0, "timeout_ms 必须大于 0"
        self._html_renderer = html_renderer or ImasSetlistHtmlRenderer()
        self._palette_applier = (
            palette_applier or BrowserImasSetlistPaletteApplier()
        )
        self._timeout_ms = timeout_ms

    def render_to_png_bytes(
        self,
        document: ImasSetlistRenderDocument,
    ) -> bytes:
        """
        将文档渲染为一张 1280px 宽的 PNG。

        Args:
            document (ImasSetlistRenderDocument): 安全渲染文档。

        Returns:
            bytes: PNG 图片字节。

        Raises:
            ImasSetlistRenderError: 当 Playwright 不可用或截图失败时抛出。
        """
        try:
            from playwright.sync_api import (
                Error as PlaywrightError,
                TimeoutError as PlaywrightTimeoutError,
                sync_playwright,
            )
        except ModuleNotFoundError as exc:
            raise ImasSetlistRenderError(
                "playwright_missing",
                "生成 Setlist 图片需要安装 Playwright。",
            ) from exc

        render_html = self._html_renderer.render(document)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page(
                        viewport={
                            "width": IMAS_SETLIST_CSS_WIDTH,
                            "height": 1200,
                        },
                        device_scale_factor=(
                            IMAS_SETLIST_DEVICE_SCALE_FACTOR
                        ),
                    )
                    page.set_content(
                        render_html,
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms,
                    )
                    self._palette_applier.apply(
                        page,
                        document.palette_stylesheet_url,
                        document.theme_brand_id,
                        self._timeout_ms,
                    )
                    page.evaluate("document.fonts.ready")
                    image_bytes = page.locator("#setlist-card").screenshot(
                        omit_background=False
                    )
                finally:
                    browser.close()
        except (PlaywrightError, PlaywrightTimeoutError) as exc:
            raise ImasSetlistRenderError(
                "setlist_screenshot_failed",
                f"Setlist 图片截图失败：{exc}",
            ) from exc
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ImasSetlistRenderError(
                "setlist_png_invalid",
                "Setlist 截图没有生成有效 PNG。",
            )
        return image_bytes


class ImasSetlistImageService:
    """组合无缓存详情抓取、DOM 清洗与单图渲染。"""

    def __init__(
        self,
        client: ImasSetlistClient | None = None,
        document_parser: ImasSetlistDocumentParser | None = None,
        browser_renderer: BrowserImasSetlistRenderer | None = None,
    ) -> None:
        """
        初始化图片服务。

        Args:
            client (ImasSetlistClient | None): 无缓存页面客户端。
            document_parser (ImasSetlistDocumentParser | None): DOM 解析器。
            browser_renderer (BrowserImasSetlistRenderer | None): PNG 渲染器。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self._client = client or ImasSetlistClient()
        self._document_parser = (
            document_parser or ImasSetlistDocumentParser()
        )
        self._browser_renderer = (
            browser_renderer or BrowserImasSetlistRenderer()
        )

    def render(self, candidate_id: str) -> ImasSetlistRenderedImage:
        """
        获取候选详情并生成一张 PNG。

        Args:
            candidate_id (str): 搜索工具返回的精确候选 ID。

        Returns:
            ImasSetlistRenderedImage: PNG 和结构统计。

        Raises:
            ImasSetlistToolError: 当页面请求或候选校验失败时抛出。
            ImasSetlistRenderError: 当页面或截图无法渲染时抛出。
        """
        source = self._client.fetch_source(candidate_id)
        document = self._document_parser.parse(source)
        png_bytes = self._browser_renderer.render_to_png_bytes(document)
        return ImasSetlistRenderedImage(
            candidate_id=document.candidate_id,
            title=document.title,
            day=document.day,
            png_bytes=png_bytes,
            table_count=document.table_count,
            row_count=document.row_count,
            warnings=document.warnings,
        )
