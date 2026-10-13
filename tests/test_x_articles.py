"""验证 X Articles 在链接拉取、截图和翻译流程中的兼容行为。"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from src.x_monitor import XAPIClient, XMonitorManager
from src.x_monitor_media import XPostImagePayloadBuilder, compose_x_media_message
from src.x_monitor_render import (
    BrowserTweetRenderer,
    XTweetPayloadParser,
    render_tweet_html,
)
from src.x_monitor_translate import (
    TRANSLATION_MODE_ENV,
    XRenderedTweetTextTranslator,
)

ARTICLE_LINK = (
    "https://x.com/6000_doll/status/2109870891039670497"
    "?s=46&t=9XKw5b3s-b3q94SXnCWcZA"
)
ARTICLE_TITLE = "ラジオ冒頭部分（文字起こし）"
ARTICLE_BODY = "皆さん、お久しぶりです。\n\n番組冒頭のお時間をお借りします。\n最後までお読みください。"
ARTICLE_TEXT = f"{ARTICLE_TITLE}\n\n{ARTICLE_BODY}"


@pytest.fixture
def article_payload() -> dict[str, Any]:
    """构造与用户链接同结构的文章响应，不包含真实文章内容。

    Returns:
        dict[str, Any]: 含文章标题、全文及摘要的 X API 响应。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return {
        "data": [
            {
                "id": "2109870891039670497",
                "author_id": "10",
                "text": "https://t.co/article",
                "entities": {
                    "urls": [
                        {
                            "url": "https://t.co/article",
                            "expanded_url": "https://x.com/i/article/2109867478704963584",
                        }
                    ]
                },
                "article": {
                    "title": ARTICLE_TITLE,
                    "plain_text": ARTICLE_BODY,
                    "preview_text": "仅有摘要",
                    "entities": {},
                },
            }
        ],
        "includes": {
            "users": [{"id": "10", "name": "Test Author", "username": "6000_doll"}]
        },
    }


def test_lookup_and_timeline_request_article_field() -> None:
    """单条推文和监控时间线均须显式请求完整文章字段。

    Returns:
        None: 请求字段正确时通过。

    Raises:
        AssertionError: 任一请求缺少 article 字段时抛出。
    """
    client = XAPIClient(bearer_token="test-token")
    with mock.patch.object(client, "_request_json", return_value={}) as request:
        client.get_tweet_by_id("2109870891039670497")
        client.get_user_posts("10")

    assert request.call_count == 2
    for call in request.call_args_list:
        assert "article" in call.kwargs["params"]["tweet.fields"].split(",")


def test_article_parser_uses_title_and_full_body(article_payload: dict[str, Any]) -> None:
    """截图正文应包含文章标题及全文，并优先于摘要和普通长推文。

    Args:
        article_payload (dict[str, Any]): 文章测试响应。

    Returns:
        None: 标题和全文完整保留时通过。

    Raises:
        AssertionError: 文章内容丢失或被其他字段覆盖时抛出。
    """
    article_payload["data"][0]["note_tweet"] = {"text": "普通长推文"}
    tweet = XTweetPayloadParser().parse(article_payload)[0]

    assert tweet.text == ARTICLE_TEXT
    assert tweet.author.username == "6000_doll"
    document = render_tweet_html(tweet)
    assert ARTICLE_TITLE in document
    assert ARTICLE_BODY in document
    assert "仅有摘要" not in document
    assert "https://t.co/article" not in document


def test_article_uses_own_url_entities_and_escapes_html(
    article_payload: dict[str, Any],
) -> None:
    """文章 URL 应使用文章实体，并安全渲染标题和正文中的 HTML 字符。

    Args:
        article_payload (dict[str, Any]): 文章测试响应。

    Returns:
        None: 链接清洗和 HTML 转义正确时通过。

    Raises:
        AssertionError: 使用推文实体或输出未转义 HTML 时抛出。
    """
    article_payload["data"][0]["article"] = {
        "title": "A &amp; B",
        "plain_text": "&lt;script&gt; https://t.co/article",
        "entities": {
            "urls": [
                {
                    "url": "https://t.co/article",
                    "expanded_url": "https://example.com/source",
                }
            ]
        },
    }
    tweet = XTweetPayloadParser().parse(article_payload)[0]

    assert tweet.text == "A & B\n\n<script> https://example.com/source"
    document = render_tweet_html(tweet)
    assert "A &amp; B" in document
    assert "&lt;script&gt;" in document
    assert "<script>" not in document
    assert "x.com/i/article/" not in document


def test_fetch_shared_article_link_returns_full_text(
    article_payload: dict[str, Any],
) -> None:
    """用户提供的分享链接应在管理器结果中返回文章全文。

    Args:
        article_payload (dict[str, Any]): 文章测试响应。

    Returns:
        None: 分享参数不影响拉取且结果包含全文时通过。

    Raises:
        AssertionError: 推文 ID、作者或文章正文解析错误时抛出。
    """
    client = mock.Mock(spec=XAPIClient)
    client.get_tweet_by_id.return_value = article_payload
    result = XMonitorManager(client=client).fetch_link(ARTICLE_LINK)

    client.get_tweet_by_id.assert_called_once_with("2109870891039670497")
    assert result.text == ARTICLE_TEXT
    assert result.username == "6000_doll"
    assert result.source_payload is article_payload


def test_quoted_article_is_parsed(article_payload: dict[str, Any]) -> None:
    """引用文章的普通推文也应在引用区域展示文章内容。

    Args:
        article_payload (dict[str, Any]): 文章测试响应。

    Returns:
        None: 引用文章包含全文时通过。

    Raises:
        AssertionError: 引用区域缺失文章正文时抛出。
    """
    article_payload["includes"]["tweets"] = article_payload["data"]
    article_payload["data"] = [
        {
            "id": "2",
            "author_id": "10",
            "text": "推荐阅读",
            "referenced_tweets": [{"type": "quoted", "id": "2109870891039670497"}],
        }
    ]
    tweet = XTweetPayloadParser().parse(article_payload)[0]

    assert tweet.text == "推荐阅读"
    assert tweet.quote is not None
    assert tweet.quote.tweet.text == ARTICLE_TEXT
    assert ARTICLE_BODY in render_tweet_html(tweet)


@pytest.mark.parametrize("mode", ["none", "translated", "bilingual"])
def test_article_flows_through_image_builder_and_translation(
    article_payload: dict[str, Any], mode: str,
) -> None:
    """文章全文应按现有翻译模式传入截图器，并组成 OneBot 图片消息。

    Args:
        article_payload (dict[str, Any]): 文章测试响应。
        mode (str): 推文翻译模式。

    Returns:
        None: 各模式均使用全文并生成图片消息时通过。

    Raises:
        AssertionError: 翻译输入、渲染正文或图片消息错误时抛出。
    """
    result = XMonitorManager()._parse_posts("6000_doll", article_payload)[0]
    translator = mock.Mock()
    translator.translate_texts.return_value = ["中文标题\n\n完整译文"]
    renderer = mock.Mock(spec=BrowserTweetRenderer)
    renderer.render_to_png_bytes.return_value = b"png"
    builder = XPostImagePayloadBuilder(
        renderer=renderer,
        text_translator=XRenderedTweetTextTranslator(translator=translator),
    )
    with mock.patch.dict(
        "os.environ", {TRANSLATION_MODE_ENV: mode, "X_MONITOR_LEGACY_MEDIA": "0"}
    ):
        message = compose_x_media_message("article", [result], renderer=builder.render)

    tweet = renderer.render_to_png_bytes.call_args.args[0]
    if mode == "none":
        translator.translate_texts.assert_not_called()
        assert tweet.text == ARTICLE_TEXT
    else:
        translator.translate_texts.assert_called_once_with([ARTICLE_TEXT])
        if mode == "translated":
            assert tweet.text == "中文标题\n\n完整译文"
            assert tweet.is_translated_text
        else:
            assert tweet.text == ARTICLE_TEXT
            assert tweet.translation_text == "中文标题\n\n完整译文"
    assert message[0]["type"] == "image"
    assert message[0]["data"]["file"] == "base64://cG5n"


@pytest.mark.parametrize(
    "article",
    [
        {"title": ARTICLE_TITLE},
        {"title": ARTICLE_TITLE, "preview_text": "仅有摘要"},
        {"title": ARTICLE_TITLE, "plain_text": " "},
        {"title": ARTICLE_TITLE, "plain_text": ["错误类型"]},
        {"plain_text": ARTICLE_BODY},
        "错误类型",
    ],
)
def test_incomplete_article_fails_explicitly(
    article_payload: dict[str, Any], article: object,
) -> None:
    """文章字段缺失或类型异常时须报错，不能用摘要或链接代替全文。

    Args:
        article_payload (dict[str, Any]): 文章测试响应。
        article (object): 不完整或格式错误的文章对象。

    Returns:
        None: 解析器和管理器均明确拒绝异常响应时通过。

    Raises:
        AssertionError: 异常文章被静默接受时抛出。
    """
    article_payload["data"][0]["article"] = article
    with pytest.raises(AssertionError, match="article"):
        XTweetPayloadParser().parse(article_payload)
    with pytest.raises(AssertionError, match="article"):
        XMonitorManager()._parse_posts("6000_doll", article_payload)
