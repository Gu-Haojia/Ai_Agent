"""Eventernote 活动搜索与详情 Tool 测试。"""

import json
from unittest.mock import Mock

import requests

from src.eventernote_tool import EventernoteClient, build_eventernote_tools


def _response(html: str, status_code: int = 200) -> Mock:
    """创建模拟 HTTP 响应。

    Args:
        html (str): 响应 HTML。
        status_code (int): HTTP 状态码。

    Returns:
        Mock: 模拟响应。

    Raises:
        AssertionError: 当参数类型非法时抛出。
    """
    assert isinstance(html, str), "html 必须是字符串"
    assert isinstance(status_code, int), "status_code 必须是整数"
    response = Mock()
    response.text = html
    response.status_code = status_code
    return response


def _search_page(total: int, count: int, start_id: int = 1001) -> str:
    """创建最小活动搜索页 HTML。

    Args:
        total (int): 搜索结果总数。
        count (int): 当前官网页活动数。
        start_id (int): 当前官网页首个活动 ID。

    Returns:
        str: 活动搜索页 HTML。

    Raises:
        AssertionError: 当数量为负数时抛出。
    """
    assert total >= 0 and count >= 0, "数量不能为负数"
    assert start_id > 0, "start_id 必须为正整数"
    items = "".join(
        '<li class="clearfix"><div class="event"><h4>'
        f'<a href="/events/{start_id + index}">活动 {start_id + index}</a>'
        "</h4></div></li>"
        for index in range(count)
    )
    return (
        f'<p class="t2">{total}件のイベントが見つかりました。</p>'
        f'<div class="gb_event_list"><ul>{items}</ul></div>'
    )


def _event_page() -> str:
    """创建最小活动详情页 HTML。

    Returns:
        str: 活动详情页 HTML。

    Raises:
        AssertionError: 本函数不抛出异常。
    """
    return """
    <div class="gb_events_detail_title"><h2>测试活动</h2></div>
    <div class="gb_events_info_table"><table>
      <tr><td>開催日時</td><td>2026-08-19 (水)</td></tr>
      <tr><td>時間</td><td>開場 17:00 開演 18:00 終演 20:30</td></tr>
      <tr><td>開催場所</td><td><a href="/places/36">新宿LOFT</a></td></tr>
      <tr><td>出演者</td><td>
        <a href="/actors/A/1">出演者A</a>
        <a href="/actors/B/2">出演者B</a>
      </td></tr>
      <tr><td>関連リンク</td><td>
        <a href="https://example.com/event">公式ページ</a>
      </td></tr>
      <tr><td>Twitterハッシュタグ</td><td>#不要返回</td></tr>
    </table></div>
    """


class TestEventernoteSearchTool:
    """验证 Eventernote 活动搜索 Tool。"""

    def test_client_uses_optional_date_and_default_page(self) -> None:
        """客户端应与 Tool 一样允许省略日期和页码。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            return_value=_response(_search_page(1, 1))
        )
        client = EventernoteClient(session)

        result = client.search(query="活动")

        assert result["page"] == {"current": 1, "total": 1, "size": 20}

    def test_returns_twenty_items_with_page_metadata(self) -> None:
        """搜索应按每页二十条返回当前页和总页数。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            return_value=_response(_search_page(40, 30))
        )
        search_tool, _ = build_eventernote_tools(EventernoteClient(session))

        result = json.loads(
            search_tool.invoke({"query": "活动", "date": None, "page": 1})
        )

        assert result["ok"] is True
        assert result["page"] == {"current": 1, "total": 2, "size": 20}
        assert len(result["items"]) == 20
        assert result["items"][0] == {"id": 1001, "name": "活动 1001"}
        params = session.get.call_args.kwargs["params"]  # type: ignore[attr-defined]
        assert params["page"] == 1

    def test_combines_upstream_pages_when_tool_page_crosses_boundary(self) -> None:
        """第二个 Tool 页应拼接官网第一页和第二页。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            side_effect=[
                _response(_search_page(40, 30)),
                _response(_search_page(40, 10, start_id=1031)),
            ]
        )
        search_tool, _ = build_eventernote_tools(EventernoteClient(session))

        result = json.loads(
            search_tool.invoke({"query": "活动", "date": None, "page": 2})
        )

        assert result["page"] == {"current": 2, "total": 2, "size": 20}
        assert len(result["items"]) == 20
        assert result["items"][0] == {"id": 1021, "name": "活动 1021"}
        assert result["items"][-1] == {"id": 1040, "name": "活动 1040"}
        requested_pages = [
            call.kwargs["params"]["page"]
            for call in session.get.call_args_list
        ]
        assert requested_pages == [1, 2]

    def test_accepts_empty_query_with_date(self) -> None:
        """精确日期搜索应允许空关键词。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            return_value=_response(
                "<p>指定された条件での検索結果が見つかりませんでした。</p>"
            )
        )
        search_tool, _ = build_eventernote_tools(EventernoteClient(session))

        result = json.loads(
            search_tool.invoke(
                {"query": "", "date": "2026-08-19", "page": 1}
            )
        )

        assert result == {
            "ok": True,
            "page": {"current": 1, "total": 0, "size": 20},
            "items": [],
        }

    def test_returns_structured_invalid_date_error(self) -> None:
        """非法日期应返回结构化错误。"""
        search_tool, _ = build_eventernote_tools()

        result = json.loads(
            search_tool.invoke(
                {"query": "", "date": "2026-02-30", "page": 1}
            )
        )

        assert result["ok"] is False
        assert result["error"]["code"] == "invalid_date"

    def test_returns_structured_validation_error(self) -> None:
        """LangChain 参数校验失败也应返回结构化错误。"""
        search_tool, _ = build_eventernote_tools()

        result = json.loads(search_tool.invoke({"page": 1}))

        assert result == {
            "ok": False,
            "error": {
                "code": "invalid_argument",
                "message": "Tool 参数格式不正确。",
            },
        }


class TestEventernoteGetTool:
    """验证 Eventernote 活动详情 Tool。"""

    def test_returns_flat_event_summary_without_hashtag(self) -> None:
        """详情应将会场和出演者扁平化并排除 Hashtag。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            return_value=_response(_event_page())
        )
        _, get_tool = build_eventernote_tools(EventernoteClient(session))

        result = json.loads(get_tool.invoke({"id": 480723}))

        assert result == {
            "ok": True,
            "data": {
                "id": 480723,
                "name": "测试活动",
                "date": "2026-08-19",
                "weekday": "水",
                "open_time": "17:00",
                "start_time": "18:00",
                "end_time": "20:30",
                "place": "新宿LOFT",
                "actors": ["出演者A", "出演者B"],
                "related_links": ["https://example.com/event"],
                "url": "https://www.eventernote.com/events/480723",
            },
        }
        assert "hashtag" not in result["data"]

    def test_returns_structured_not_found_error(self) -> None:
        """不存在的 Event 应返回结构化错误。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            return_value=_response("", 404)
        )
        _, get_tool = build_eventernote_tools(EventernoteClient(session))

        result = json.loads(get_tool.invoke({"id": 999999999}))

        assert result == {
            "ok": False,
            "error": {
                "code": "not_found",
                "message": "指定的 Event 不存在。",
            },
        }

    def test_returns_structured_network_error(self) -> None:
        """网络失败应返回结构化错误。"""
        session = requests.Session()
        session.get = Mock(  # type: ignore[method-assign]
            side_effect=requests.ConnectionError("连接中断")
        )
        _, get_tool = build_eventernote_tools(EventernoteClient(session))

        result = json.loads(get_tool.invoke({"id": 480723}))

        assert result["ok"] is False
        assert result["error"] == {
            "code": "network_error",
            "message": "Eventernote 请求失败：连接中断",
        }
