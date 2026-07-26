"""
imas-db Setlist 搜索、详情解析与工具包装单元测试。
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import requests

from src.imas_setlist_tool import (
    IMAS_SETLIST_INDEX_URL,
    ImasSetlistClient,
    ImasSetlistToolError,
    build_imas_setlist_get_tool,
    imas_setlist_get,
    imas_setlist_search,
)


INDEX_HTML = """
<html>
  <body>
    <main>
      <h2>2026年</h2>
      <ul>
        <li data-brand-ids="1,4">
          <span class="badge">MR</span>
          IDOL WORLD SUPER FESTIVAL 2026
          <span class="badge">765AS</span>
          <ul>
            <li>
              <a
                href="./idolmaster_iwsf_day2.html"
                title="IDOL WORLD SUPER FESTIVAL 2026 第二公演 -ZESSYOU-"
              >第二公演 -ZESSYOU-</a>
              <small class="date">- 2026/07/25(土)</small>
            </li>
            <li>
              <a
                href="./idolmaster_iwsf_day3.html"
                title="IDOL WORLD SUPER FESTIVAL 2026 第三公演 -KYOUMEI-"
              >第三公演 -KYOUMEI-</a>
              <small class="date">- 2026/07/26(日)</small>
            </li>
          </ul>
        </li>
        <li>
          <a href="./solo_live.html">如月千早武道館単独公演「OathONE」</a>
          <small class="date">- 2026/01/24(土)・25(日)</small>
          <span class="badge">765AS</span>
        </li>
        <li>
          THE IDOLM@STER MILLION LIVE! 3rdLIVE TOUR
          <ul>
            <li>
              <a
                href="./million_3rdlive_songs.html"
                title="楽曲と収録CDの一覧"
              >楽曲まとめ</a>
            </li>
          </ul>
        </li>
      </ul>
    </main>
  </body>
</html>
"""

MOIW_INDEX_HTML = """
<html>
  <body>
    <main>
      <ul>
        <li>
          <a
            href="./idolmaster_idolworld2025_day2.html"
            title="THE IDOLM@STER M@STERS OF IDOL WORLD 2025 DAY2"
          >M@STERS OF IDOL WORLD 2025 DAY2</a>
          <small class="date">- 2025/12/14(日)</small>
        </li>
      </ul>
    </main>
  </body>
</html>
"""


DETAIL_HTML = """
<html>
  <body>
    <h1 id="page_title">
      IDOL WORLD SUPER FESTIVAL 2026 -KYOUMEI- [DAY3]
    </h1>
    <table class="tracklist">
      <thead>
        <tr><th>No.</th><th>楽曲</th><th>演者</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>
            <a href="/song/detail/1.html">キラメキラリ</a>
            <span class="visually-hidden">(</span>
            <small class="badge">765AS</small>
            <span class="visually-hidden">)</span>
          </td>
          <td>
            <span class="idol-name">高槻やよい</span>,
            <span class="idol-name">日高愛</span>
          </td>
        </tr>
        <tr><th colspan="3">アイ・クライマックスメドレー</th></tr>
        <tr>
          <td></td>
          <td>(告知映像)</td>
          <td></td>
        </tr>
        <tr>
          <td>2</td>
          <td>新曲 (short ver.)</td>
          <td>出演者A 出演者B</td>
        </tr>
      </tbody>
    </table>
  </body>
</html>
"""

NON_STANDARD_DETAIL_HTML = """
<html>
  <body>
    <h1 id="page_title">既存特殊形式公演</h1>
    <table class="tracklist">
      <thead>
        <tr><th>No.</th><th>内容/楽曲</th><th>演者</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>
            初
            <small class="badge bg-imas-brand-gakuen">学園</small>
          </td>
        </tr>
        <tr>
          <td>10〜25</td>
          <td colspan="2">
            会場の全員で歌唱するユニット曲メドレー
            <ol>
              <li>運命光年</li>
              <li>We're the one</li>
            </ol>
          </td>
        </tr>
        <tr class="part-header">
          <th colspan="3">【第二部】</th>
        </tr>
        <tr class="part-header">
          <td colspan="3">「みんな元気!!!!!」メドレー ここまで</td>
        </tr>
        <tr>
          <td>26</td>
          <td>終曲</td>
          <td><ruby><rb>特殊演者</rb><rp>(</rp><rt>とくしゅ</rt><rp>)</rp></ruby></td>
        </tr>
      </tbody>
    </table>
  </body>
</html>
"""


def _response(
    text: str,
    status_code: int = 200,
    content_type: str = "text/html",
) -> mock.Mock:
    """
    构造固定 HTML 内容的 requests.Response 替身。

    Args:
        text (str): 响应正文。
        status_code (int): HTTP 状态码。
        content_type (str): Content-Type 响应头。

    Returns:
        mock.Mock: 可用于 HTTP 客户端测试的响应替身。

    Raises:
        AssertionError: 当状态码不是正整数时抛出。
    """
    assert status_code > 0, "status_code 必须为正整数"
    response = mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}
    response.text = text
    response.encoding = None
    return response


class ImasSetlistIndexParserTests(unittest.TestCase):
    """验证活动索引页能够生成最小候选结构。"""

    def test_parse_builds_full_nested_titles_and_excludes_summaries(self) -> None:
        """
        嵌套场次应使用完整 title，乐曲汇总链接不应成为候选。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当解析结果与预期不一致时由断言抛出。
        """
        client = ImasSetlistClient(
            http_get=mock.Mock(return_value=_response(INDEX_HTML))
        )

        candidates = client._load_candidates()

        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            candidates[1]["candidate_id"],
            "imas-setlist:idolmaster_iwsf_day3",
        )
        self.assertEqual(
            candidates[1]["title"],
            "IDOL WORLD SUPER FESTIVAL 2026 第三公演 -KYOUMEI-",
        )
        self.assertEqual(candidates[1]["day"], "2026/07/26(日)")
        self.assertEqual(
            candidates[2]["title"],
            "如月千早武道館単独公演「OathONE」",
        )

    def test_search_candidate_dict_has_only_agreed_fields(self) -> None:
        """
        搜索候选对外只应包含候选 ID、标题和日期。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当返回字段超出约定时由断言抛出。
        """
        client = ImasSetlistClient(
            http_get=mock.Mock(return_value=_response(INDEX_HTML))
        )

        result = client.search("iwsf day2")

        self.assertEqual(set(result["candidates"][0]), {
            "candidate_id",
            "title",
            "day",
        })

    def test_search_supports_moiw_year_and_short_day_aliases(self) -> None:
        """
        MOIW25 d2 应命中 M@STERS OF IDOL WORLD 2025 DAY2。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当常用简称无法命中时由断言抛出。
        """
        client = ImasSetlistClient(
            http_get=mock.Mock(return_value=_response(MOIW_INDEX_HTML))
        )

        result = client.search("MOIW25 d2")

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["candidates"][0]["candidate_id"],
            "imas-setlist:idolmaster_idolworld2025_day2",
        )


class ImasSetlistPageParserTests(unittest.TestCase):
    """验证详情页只解析有序号的最小曲目字段。"""

    def test_parse_tracks_skips_unnumbered_rows(self) -> None:
        """
        Medley 标题与无序号说明行应被跳过。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当解析结果与预期不一致时由断言抛出。
        """
        client = ImasSetlistClient()

        title, tracks = client._parse_detail(DETAIL_HTML)

        self.assertEqual(
            title,
            "IDOL WORLD SUPER FESTIVAL 2026 -KYOUMEI- [DAY3]",
        )
        self.assertEqual(
            tracks,
            [
                {
                    "no": "1",
                    "title": "キラメキラリ",
                    "brand": "765AS",
                    "performers": "高槻やよい, 日高愛",
                },
                {
                    "no": "2",
                    "title": "新曲 (short ver.)",
                    "brand": None,
                    "performers": "出演者A 出演者B",
                },
            ],
        )

    def test_parse_tracks_rejects_changed_headers(self) -> None:
        """
        表头变化时应显式失败，不能猜测新结构。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 被测解析器应按预期抛出。
        """
        changed_html = DETAIL_HTML.replace("<th>演者</th>", "<th>出演</th>")

        with self.assertRaisesRegex(AssertionError, "表头"):
            ImasSetlistClient()._parse_detail(changed_html)

    def test_parse_tracks_accepts_existing_non_standard_rows(self) -> None:
        """
        合法特殊曲目不得丢失，段落标题不得误判为曲目。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当现有特殊结构未被完整转换时由断言抛出。
        """
        _, tracks = ImasSetlistClient()._parse_detail(
            NON_STANDARD_DETAIL_HTML
        )

        self.assertEqual(
            tracks,
            [
                {
                    "no": "1",
                    "title": "初",
                    "brand": "学園",
                    "performers": "",
                },
                {
                    "no": "10〜25",
                    "title": (
                        "会場の全員で歌唱するユニット曲メドレー "
                        "運命光年 We're the one"
                    ),
                    "brand": None,
                    "performers": "",
                },
                {
                    "no": "26",
                    "title": "終曲",
                    "brand": None,
                    "performers": "特殊演者",
                },
            ],
        )


class ImasSetlistServiceTests(unittest.TestCase):
    """验证无缓存搜索和精确候选详情流程。"""

    def test_search_returns_all_matching_candidates_without_limit(self) -> None:
        """
        广义查询应返回全部匹配候选，并保留页面顺序。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当搜索结果与预期不一致时由断言抛出。
        """
        http_get = mock.Mock(return_value=_response(INDEX_HTML))
        client = ImasSetlistClient(http_get=http_get)

        result = client.search("iwsf")

        self.assertEqual(result["status"], "success")
        candidates = result["candidates"]
        self.assertIsInstance(candidates, list)
        self.assertEqual(len(candidates), 2)
        http_get.assert_called_once_with(
            IMAS_SETLIST_INDEX_URL,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "LangGraph-ImasSetlistTool/1.0",
            },
            timeout=15.0,
        )

    def test_search_matches_all_compacted_query_terms(self) -> None:
        """
        搜索应忽略大小写、空格与标点，并要求全部词命中。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当匹配结果与预期不一致时由断言抛出。
        """
        client = ImasSetlistClient(
            http_get=mock.Mock(return_value=_response(INDEX_HTML))
        )

        result = client.search("IWSF DAY3")

        self.assertEqual(
            result["candidates"],
            [
                {
                    "candidate_id": "imas-setlist:idolmaster_iwsf_day3",
                    "title": (
                        "IDOL WORLD SUPER FESTIVAL 2026 "
                        "第三公演 -KYOUMEI-"
                    ),
                    "day": "2026/07/26(日)",
                }
            ],
        )

    def test_search_requests_index_again_for_each_call(self) -> None:
        """
        连续搜索不得使用缓存，每次都应重新请求索引页。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当请求次数不符合预期时由断言抛出。
        """
        http_get = mock.Mock(return_value=_response(INDEX_HTML))
        client = ImasSetlistClient(http_get=http_get)

        client.search("iwsf")
        client.search("oathone")

        self.assertEqual(http_get.call_count, 2)

    def test_get_validates_index_then_fetches_detail(self) -> None:
        """
        Get 应先重新校验索引，再抓取精确候选详情。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当请求顺序或返回结构不符合预期时由断言抛出。
        """
        detail_url = (
            "https://imas-db.jp/song/event/idolmaster_iwsf_day3.html"
        )
        http_get = mock.Mock(
            side_effect=[
                _response(INDEX_HTML),
                _response(DETAIL_HTML),
            ]
        )
        client = ImasSetlistClient(http_get=http_get)

        result = client.get("imas-setlist:idolmaster_iwsf_day3")

        self.assertEqual(
            result,
            {
                "status": "success",
                "title": (
                    "IDOL WORLD SUPER FESTIVAL 2026 -KYOUMEI- [DAY3]"
                ),
                "day": "2026/07/26(日)",
                "tracks": [
                    {
                        "no": "1",
                        "title": "キラメキラリ",
                        "brand": "765AS",
                        "performers": "高槻やよい, 日高愛",
                    },
                    {
                        "no": "2",
                        "title": "新曲 (short ver.)",
                        "brand": None,
                        "performers": "出演者A 出演者B",
                    },
                ],
                "source_url": detail_url,
            },
        )
        self.assertEqual(http_get.call_count, 2)
        self.assertEqual(http_get.call_args_list[1].args[0], detail_url)

    def test_get_rejects_candidate_missing_from_current_index(self) -> None:
        """
        当前索引不存在的候选不得直接请求详情页。

        Returns:
            None: 测试方法无返回值。

        Raises:
            ImasSetlistToolError: 被测服务应按预期抛出。
        """
        http_get = mock.Mock(return_value=_response(INDEX_HTML))
        client = ImasSetlistClient(http_get=http_get)

        with self.assertRaisesRegex(
            ImasSetlistToolError,
            "请先调用 imas_setlist_search",
        ):
            client.get("imas-setlist:not_in_index")

        self.assertEqual(http_get.call_count, 1)


class ImasSetlistToolWrapperTests(unittest.TestCase):
    """验证两个 LangChain 工具的公开参数和错误输出。"""

    def test_tool_descriptions_expose_idolmaster_setlist_intent(self) -> None:
        """
        工具描述应包含偶像大师、演唱会与歌单等意图检索关键词。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当工具描述缺少关键意图词时由断言抛出。
        """
        for current_tool in (imas_setlist_search, imas_setlist_get):
            self.assertIn("偶像大师", current_tool.description)
            self.assertIn("演唱会", current_tool.description)
            self.assertIn("歌单", current_tool.description)
            self.assertIn("Setlist", current_tool.description)
        self.assertIn(
            "imas_setlist_get",
            imas_setlist_search.description,
        )
        self.assertIn(
            "imas_setlist_search",
            imas_setlist_get.description,
        )

    def test_search_tool_exposes_only_query_parameter(self) -> None:
        """
        搜索工具应只要求 query，不暴露 limit 或候选 ID。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当工具 Schema 不符合预期时由断言抛出。
        """
        schema = imas_setlist_search.tool_call_schema.model_json_schema()

        self.assertEqual(set(schema["properties"]), {"query"})
        self.assertEqual(schema["required"], ["query"])

    def test_get_tool_exposes_only_candidate_id_parameter(self) -> None:
        """
        Get 工具应要求 candidate_id，并提供默认关闭的图片参数。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当工具 Schema 不符合预期时由断言抛出。
        """
        schema = imas_setlist_get.tool_call_schema.model_json_schema()

        self.assertEqual(
            set(schema["properties"]),
            {"candidate_id", "render_image"},
        )
        self.assertEqual(schema["required"], ["candidate_id"])
        self.assertFalse(schema["properties"]["render_image"]["default"])

    def test_search_tool_returns_service_payload(self) -> None:
        """
        搜索包装层应原样返回服务的结构化结果。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当包装结果与预期不一致时由断言抛出。
        """
        expected = {
            "status": "success",
            "query": "iwsf day3",
            "candidates": [
                {
                    "candidate_id": "imas-setlist:idolmaster_iwsf_day3",
                    "title": "IWSF DAY3",
                    "day": "2026/07/26(日)",
                }
            ],
        }
        with mock.patch.object(
            ImasSetlistClient,
            "search",
            return_value=expected,
        ) as search:
            output = imas_setlist_search.invoke({"query": "iwsf day3"})

        self.assertEqual(json.loads(output), expected)
        search.assert_called_once_with("iwsf day3")

    def test_get_tool_returns_structured_failure(self) -> None:
        """
        上游错误应转换为带稳定错误码的失败 JSON。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当错误结构不符合预期时由断言抛出。
        """
        with mock.patch.object(
            ImasSetlistClient,
            "get",
            side_effect=ImasSetlistToolError(
                "upstream_timeout",
                "imas-db Setlist 页面请求超时。",
            ),
        ):
            output = imas_setlist_get.invoke(
                {"candidate_id": "imas-setlist:idolmaster_iwsf_day3"}
            )

        payload = json.loads(output)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "upstream_timeout")

    def test_get_tool_image_mode_sends_image_and_returns_status_only(self) -> None:
        """
        图片模式应写入图片回调，且不向模型重复返回曲目数据。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当图片模式状态或回调不符合预期时抛出。
        """
        rendered = mock.Mock()
        rendered.candidate_id = "imas-setlist:idolmaster_iwsf_day3"
        rendered.title = "IWSF DAY3"
        rendered.day = "2026/07/26(日)"
        rendered.warnings = ()
        image_sink = mock.Mock()
        get_tool = build_imas_setlist_get_tool(image_sink=image_sink)
        with mock.patch(
            "src.imas_setlist_render.ImasSetlistImageService.render",
            return_value=rendered,
        ) as render:
            output = get_tool.invoke(
                {
                    "candidate_id": rendered.candidate_id,
                    "render_image": True,
                }
            )

        payload = json.loads(output)
        self.assertEqual(
            payload,
            {
                "status": "rendered",
                "candidate_id": rendered.candidate_id,
                "title": "IWSF DAY3",
                "day": "2026/07/26(日)",
                "image_count": 1,
                "warnings": [],
            },
        )
        self.assertNotIn("tracks", payload)
        image_sink.assert_called_once_with(rendered)
        render.assert_called_once_with(rendered.candidate_id)


if __name__ == "__main__":
    unittest.main()
