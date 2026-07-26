"""
imas-db Setlist 单图渲染数据保真与固定样式测试。
"""

from __future__ import annotations

import base64
from pathlib import Path
import unittest
from unittest import mock

from image_storage import GeneratedImage, ImageStorageManager
from sql_agent_cli_stream_plus import SQLCheckpointAgentStreamingPlus
from src.imas_setlist_render import (
    BrowserImasSetlistPaletteApplier,
    BrowserImasSetlistRenderer,
    ImasSetlistDocumentParser,
    ImasSetlistHtmlRenderer,
    ImasSetlistImageService,
    ImasSetlistRenderedImage,
    ImasSetlistRenderError,
)
from src.imas_setlist_tool import ImasSetlistClient, ImasSetlistPageSource


SOURCE_HTML = """
<html>
  <head>
    <link
      rel="stylesheet"
      href="/css/imas.min.css?v=20260516"
    >
  </head>
  <body>
    <h1 id="page_title">特殊形式公演</h1>
    <div class="section" data-brand-id="8">
      <table class="tracklist" style="--tracklist-title-width:25rem">
        <thead>
          <tr><th>No.</th><th>内容/楽曲</th><th>演者</th></tr>
        </thead>
        <tbody>
          <tr onclick="alert(1)">
            <td>1</td>
            <td>
              <a href="https://example.com/song">楽曲名</a>
              <small class="badge bg-nijisanji">NIJISANJI</small>
            </td>
            <td>
              <span
                class="idol-name"
                data-brand-id="255"
                data-nijisanji="valz-01"
                title="不要属性"
              >演者A</span>
              <ruby><rb>演者B</rb><rp>(</rp><rt>えんじゃ</rt><rp>)</rp></ruby>
              <em>特殊注记</em>
            </td>
          </tr>
          <tr class="part-header"><th colspan="3">【第二部】</th></tr>
          <tr>
            <td>2〜4</td>
            <td colspan="2"><ol><li>曲A</li><li>曲B</li></ol></td>
          </tr>
        </tbody>
      </table>
      <p class="notes">
        (*) 29-31曲目はアンコール
        <a href="https://example.com/note">注記リンク</a>
        <em>補足</em>
      </p>
      <p class="advertisement">広告</p>
    </div>
  </body>
</html>
"""


def _source(html: str = SOURCE_HTML) -> ImasSetlistPageSource:
    """
    构造渲染测试使用的页面来源。

    Args:
        html (str): 详情页 HTML。

    Returns:
        ImasSetlistPageSource: 固定候选元信息与 HTML。

    Raises:
        AssertionError: 当 html 为空时抛出。
    """
    assert html.strip(), "html 不能为空"
    return ImasSetlistPageSource(
        candidate_id="imas-setlist:special",
        candidate_title="索引标题",
        day="",
        source_url="https://imas-db.jp/song/event/special.html",
        html=html,
    )


class ImasSetlistDocumentParserTests(unittest.TestCase):
    """验证图片解析保留现有表格结构并清除危险属性。"""

    def test_parse_preserves_visible_content_and_render_attributes(self) -> None:
        """
        特殊行、Ruby、列表、合作属性和全部可见文字应被保留。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当清洗结果丢失数据或保留危险属性时抛出。
        """
        document = ImasSetlistDocumentParser().parse(_source())

        self.assertEqual(document.title, "特殊形式公演")
        self.assertEqual(document.day, "")
        self.assertEqual(
            document.palette_stylesheet_url,
            "https://imas-db.jp/css/imas.min.css?v=20260516",
        )
        self.assertEqual(document.theme_brand_id, 8)
        self.assertEqual(document.table_count, 1)
        self.assertEqual(document.row_count, 4)
        self.assertIn("<ruby>", document.tables_html)
        self.assertIn("<ol>", document.tables_html)
        self.assertIn('data-nijisanji="valz-01"', document.tables_html)
        self.assertIn('data-brand-id="255"', document.tables_html)
        self.assertIn("特殊注记", document.tables_html)
        self.assertIn(
            '<p class="notes">',
            document.tables_html,
        )
        self.assertIn(
            "(*) 29-31曲目はアンコール",
            document.tables_html,
        )
        self.assertIn("注記リンク", document.tables_html)
        self.assertIn("補足", document.tables_html)
        self.assertNotIn("広告", document.tables_html)
        self.assertNotIn("onclick", document.tables_html)
        self.assertNotIn("href=", document.tables_html)
        self.assertNotIn("title=", document.tables_html)
        self.assertNotIn("<em>", document.tables_html)

    def test_parse_rejects_missing_detail_title(self) -> None:
        """
        详情页缺少标题时应显式失败，不得静默借用索引标题。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当解析器未返回预期错误码时抛出。
        """
        source = _source(
            SOURCE_HTML.replace(
                '<h1 id="page_title">特殊形式公演</h1>',
                "",
            )
        )

        with self.assertRaises(ImasSetlistRenderError) as context:
            ImasSetlistDocumentParser().parse(source)

        self.assertEqual(
            context.exception.error_code,
            "setlist_title_missing",
        )

    def test_parse_rejects_missing_official_palette_stylesheet(self) -> None:
        """
        详情页缺少官方颜色表时应显式失败，不得使用过时的本地颜色。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当解析器未返回预期错误码时抛出。
        """
        source = _source(
            SOURCE_HTML.replace(
                '<link\n      rel="stylesheet"\n'
                '      href="/css/imas.min.css?v=20260516"\n'
                "    >",
                "",
            )
        )

        with self.assertRaises(ImasSetlistRenderError) as context:
            ImasSetlistDocumentParser().parse(source)

        self.assertEqual(
            context.exception.error_code,
            "setlist_palette_stylesheet_missing",
        )

    def test_parse_rejects_external_palette_stylesheet(self) -> None:
        """
        颜色表只允许来自 imas-db.jp 的固定 CSS 路径。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当外部样式表被解析器接受时抛出。
        """
        source = _source(
            SOURCE_HTML.replace(
                "/css/imas.min.css?v=20260516",
                "https://example.com/css/imas.min.css",
            )
        )

        with self.assertRaises(ImasSetlistRenderError) as context:
            ImasSetlistDocumentParser().parse(source)

        self.assertEqual(
            context.exception.error_code,
            "setlist_palette_stylesheet_missing",
        )

    def test_parse_uses_general_theme_for_unbranded_section(self) -> None:
        """
        多品牌官网区块没有单一品牌 ID 时应使用综合企划色。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当解析器没有返回综合企划品牌 ID 时抛出。
        """
        source = _source(
            SOURCE_HTML.replace(' data-brand-id="8"', "", 1)
        )

        document = ImasSetlistDocumentParser().parse(source)

        self.assertEqual(document.theme_brand_id, 0)


class ImasSetlistHtmlRendererTests(unittest.TestCase):
    """验证已经确认的单张 1280px 图片样式。"""

    def test_render_uses_fixed_width_and_compact_copyright(self) -> None:
        """
        HTML 应保持 640 CSS 像素和低调版权标记，不输出来源 URL。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当固定样式或版权文案变化时抛出。
        """
        document = ImasSetlistDocumentParser().parse(_source())

        rendered = ImasSetlistHtmlRenderer().render(document)

        self.assertIn("width: 640px", rendered)
        self.assertIn(
            'font-family: "Noto Sans CJK JP", sans-serif;',
            rendered,
        )
        self.assertIn('data-theme-brand-id="8"', rendered)
        self.assertIn(".setlist-content > .notes", rendered)
        self.assertIn(
            "(*) 29-31曲目はアンコール",
            rendered,
        )
        self.assertIn("© imas-db.jp", rendered)
        self.assertIn("天海春香Agent 生成", rendered)
        self.assertNotIn("[by imas-db.jp]", rendered)
        self.assertNotIn(document.source_url, rendered)
        self.assertNotIn("https://imas-db.jp/css/", rendered)


class BrowserImasSetlistPaletteApplierTests(unittest.TestCase):
    """验证官网颜色仅被固化到允许的视觉属性。"""

    def test_apply_uses_official_stylesheet_and_timeout(self) -> None:
        """
        颜色应用器应把官网样式地址和超时传入浏览器脚本。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当浏览器调用参数或结果不符合预期时抛出。
        """
        page = mock.Mock()
        page.evaluate.return_value = {
            "status": "success",
            "idol_name_count": 2,
            "badge_count": 1,
        }
        stylesheet_url = (
            "https://imas-db.jp/css/imas.min.css?v=20260516"
        )

        BrowserImasSetlistPaletteApplier().apply(
            page,
            stylesheet_url,
            8,
            12_000,
        )

        page.evaluate.assert_called_once()
        script, arguments = page.evaluate.call_args.args
        self.assertIn("border-bottom-color", script)
        self.assertIn("background-color", script)
        self.assertIn("--setlist-pink", script)
        self.assertIn('className = "section"', script)
        self.assertIn("stylesheet.remove()", script)
        self.assertEqual(arguments["stylesheetUrl"], stylesheet_url)
        self.assertEqual(arguments["themeBrandId"], 8)
        self.assertEqual(arguments["timeoutMs"], 12_000)

    def test_apply_reports_stylesheet_load_failure(self) -> None:
        """
        官网颜色表加载失败时应返回稳定错误，不得静默使用本地颜色。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当错误类型或错误码不符合预期时抛出。
        """
        page = mock.Mock()
        page.evaluate.return_value = {"status": "load_failed"}

        with self.assertRaises(ImasSetlistRenderError) as context:
            BrowserImasSetlistPaletteApplier().apply(
                page,
                "https://imas-db.jp/css/imas.min.css?v=20260516",
                8,
                12_000,
            )

        self.assertEqual(
            context.exception.error_code,
            "setlist_palette_stylesheet_failed",
        )

    def test_apply_rejects_external_stylesheet_before_browser_call(
        self,
    ) -> None:
        """
        浏览器网络入口应再次拒绝非官方颜色表地址。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当外部地址被加载或错误码不正确时抛出。
        """
        page = mock.Mock()

        with self.assertRaises(ImasSetlistRenderError) as context:
            BrowserImasSetlistPaletteApplier().apply(
                page,
                "https://example.com/css/imas.min.css",
                8,
                12_000,
            )

        self.assertEqual(
            context.exception.error_code,
            "setlist_palette_stylesheet_untrusted",
        )
        page.evaluate.assert_not_called()


class ImasSetlistImageServiceTests(unittest.TestCase):
    """验证详情抓取、文档解析和 PNG 渲染的组合流程。"""

    def test_render_returns_png_and_metadata(self) -> None:
        """
        服务应返回 PNG、标题、日期与结构统计。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当组合结果缺少必要信息时抛出。
        """
        client = mock.Mock(spec=ImasSetlistClient)
        client.fetch_source.return_value = _source()
        renderer = mock.Mock(spec=BrowserImasSetlistRenderer)
        renderer.render_to_png_bytes.return_value = b"\x89PNG\r\n\x1a\nmock"
        service = ImasSetlistImageService(
            client=client,
            browser_renderer=renderer,
        )

        result = service.render("imas-setlist:special")

        self.assertEqual(result.png_bytes, b"\x89PNG\r\n\x1a\nmock")
        self.assertEqual(result.title, "特殊形式公演")
        self.assertEqual(result.day, "")
        self.assertEqual(result.table_count, 1)
        self.assertEqual(result.row_count, 4)
        client.fetch_source.assert_called_once_with("imas-setlist:special")
        renderer.render_to_png_bytes.assert_called_once()


class ImasSetlistAgentImageQueueTests(unittest.TestCase):
    """验证图片模式复用 Agent 现有生成图片发送队列。"""

    def test_store_image_encodes_png_once_and_queues_result(self) -> None:
        """
        Setlist PNG 应保存一次，并作为单张 GeneratedImage 入队。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当保存参数或队列结果不符合预期时抛出。
        """
        agent = object.__new__(SQLCheckpointAgentStreamingPlus)
        manager = mock.Mock(spec=ImageStorageManager)
        stored = GeneratedImage(
            path=Path("/tmp/setlist.png"),
            mime_type="image/png",
            prompt="imas-db Setlist: 特殊形式公演",
        )
        manager.save_generated_image.return_value = stored
        agent._image_manager = manager
        agent._generated_images = []
        rendered = ImasSetlistRenderedImage(
            candidate_id="imas-setlist:special",
            title="特殊形式公演",
            day="",
            png_bytes=b"\x89PNG\r\n\x1a\nmock",
            table_count=1,
            row_count=4,
            warnings=(),
        )

        result = agent._store_imas_setlist_image(rendered)

        manager.save_generated_image.assert_called_once_with(
            base64.b64encode(rendered.png_bytes).decode("ascii"),
            "imas-db Setlist: 特殊形式公演",
            "image/png",
        )
        self.assertEqual(agent._generated_images, [stored])
        self.assertEqual(result, stored)

    def test_store_image_reports_queue_write_failure(self) -> None:
        """
        图片落盘失败时应抛出可转换为工具状态的渲染错误。

        Returns:
            None: 测试方法无返回值。

        Raises:
            AssertionError: 当错误类型或错误码不符合预期时抛出。
        """
        agent = object.__new__(SQLCheckpointAgentStreamingPlus)
        manager = mock.Mock(spec=ImageStorageManager)
        manager.save_generated_image.side_effect = OSError("disk full")
        agent._image_manager = manager
        agent._generated_images = []
        rendered = ImasSetlistRenderedImage(
            candidate_id="imas-setlist:special",
            title="特殊形式公演",
            day="",
            png_bytes=b"\x89PNG\r\n\x1a\nmock",
            table_count=1,
            row_count=4,
            warnings=(),
        )

        with self.assertRaises(ImasSetlistRenderError) as context:
            agent._store_imas_setlist_image(rendered)

        self.assertEqual(
            context.exception.error_code,
            "image_queue_write_failed",
        )
        self.assertEqual(agent._generated_images, [])


if __name__ == "__main__":
    unittest.main()
