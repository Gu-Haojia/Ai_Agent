"""
Meru 监控图片抓取与群发工具。

实现目标：
- 与现有 ImageStorage / ReverseImageUploader 解耦；
- 仅在内存中下载图片、合成商品链接二维码并转成 base64；
- 提供最小接口给 QQ 机器人使用。
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Callable, Optional, Sequence
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import qrcode
import requests
from PIL import Image, ImageOps

from src.meru_monitor import MeruSearchResult

MessagePayload = Sequence[dict[str, dict[str, str]]] | str


class MeruProductImageRenderer:
    """在商品图右下角合成链接二维码，全程使用内存处理。"""

    def render_to_jpeg_bytes(self, image_data: bytes, product_url: str) -> bytes:
        """生成仅包含商品图和二维码的 JPEG。

        Args:
            image_data (bytes): 下载的商品图片。
            product_url (str): 二维码对应的商品链接。

        Returns:
            bytes: 带二维码的 JPEG 图片。

        Raises:
            AssertionError: 当链接非法或图片比例无法容纳二维码时抛出。
            PIL.UnidentifiedImageError: 当图片无法识别时抛出。
            OSError: 当图片解码或编码失败时抛出。
        """
        assert product_url.startswith(("https://", "http://")), "商品链接必须为 HTTP URL"
        with Image.open(io.BytesIO(image_data)) as source:
            image = ImageOps.contain(
                ImageOps.exif_transpose(source),
                (640, 640),
                method=Image.Resampling.BILINEAR,
            ).convert("RGB")
        qr = qrcode.QRCode(box_size=4, border=4)
        qr.add_data(product_url)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").get_image()
        padding = 12
        assert (
            qr_image.width + 2 * padding <= image.width
            and qr_image.height + 2 * padding <= image.height
        ), "商品图比例无法容纳二维码"
        image.paste(
            qr_image,
            (image.width - qr_image.width - padding, image.height - qr_image.height - padding),
        )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90, subsampling=0)
        return output.getvalue()


def _send_group_msg(
    api_base: str, group_id: int, message: MessagePayload, access_token: str = ""
) -> None:
    """
    轻量版 OneBot send_group_msg，用于外挂功能内部调用。

    Args:
        api_base (str): OneBot HTTP API 基地址。
        group_id (int): 群号。
        message (MessagePayload): 文本或消息段列表。
        access_token (str): API Token。
    """
    url = urljoin(api_base.rstrip("/") + "/", "send_group_msg")
    payload = {"group_id": group_id, "message": message}
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"send_group_msg HTTP {resp.status}")


def _download_image(
    url: str, timeout: int = 12, max_bytes: int = 900_000
) -> bytes:
    """
    下载商品图片，返回尚未进行 base64 编码的字节。

    Args:
        url (str): 图片 URL。
        timeout (int): 超时时间（秒）。
        max_bytes (int): 允许的最大体积。

    Returns:
        bytes: 下载的商品图片。

    Raises:
        ValueError: 当类型非法或体积超限。
        requests.RequestException: 网络异常。
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        raise ValueError(f"非法图片类型: {mime or 'unknown'}")
    data = resp.content
    if len(data) > max_bytes:
        raise ValueError(f"图片过大，大小 {len(data)} bytes")
    return data


def compose_meru_media_message(
    text: str,
    items: Sequence[MeruSearchResult],
    at_qq: Optional[int] = None,
    fetcher: Optional[Callable[[str], bytes]] = None,
    max_images: int = 5,
) -> MessagePayload:
    """
    生成包含商品二维码图片的 OneBot 消息段列表（base64 内联）。

    Args:
        text (str): 文本内容。
        items (Sequence[MeruSearchResult]): 新品列表。
        at_qq (Optional[int]): 可选的 @ 目标。
        fetcher (Optional[Callable[[str], bytes]]): 可注入图片下载器。
        max_images (int): 最多附图数量。

    Returns:
        MessagePayload: 可直接发送的消息体。

    Raises:
        AssertionError: 当文本为空、商品缺少链接或图片、数量超过上限时抛出。
        ValueError: 当下载的图片类型非法或体积超限时抛出。
        requests.RequestException: 当图片下载失败时抛出。
        OSError: 当图片解码或编码失败时抛出。
    """
    assert text.strip(), "文本不可为空"
    assert max_images > 0, "max_images 必须大于 0"
    assert len(items) <= max_images, "商品数量超过图片上限，无法为每件商品附加二维码"
    product_images: list[tuple[MeruSearchResult, str]] = []
    for item in items:
        assert item.url.startswith(("https://", "http://")), "商品缺少有效链接"
        image_url = next(
            (url for url in item.image_urls if url.startswith(("https://", "http://"))),
            None,
        )
        assert image_url is not None, f"商品 {item.item_id} 缺少图片，无法附加二维码"
        product_images.append((item, image_url))
    segments: list[dict[str, dict[str, str]]] = []
    if at_qq is not None:
        segments.append({"type": "at", "data": {"qq": str(int(at_qq))}})
    segments.append({"type": "text", "data": {"text": text}})
    fetch = fetcher or _download_image
    renderer = MeruProductImageRenderer()
    ts = int(time.time())
    for idx, (item, url) in enumerate(product_images, 1):
        image_data = renderer.render_to_jpeg_bytes(fetch(url), item.url)
        b64 = base64.b64encode(image_data).decode("ascii")
        segments.append(
            {
                "type": "image",
                "data": {
                    "file": f"base64://{b64}",
                    "name": f"meru_{ts}_{idx}.jpg",
                    "cache": "0",
                },
            }
        )
    return segments


def send_meru_message_with_images(
    api_base: str,
    group_id: int,
    access_token: str,
    text: str,
    items: Sequence[MeruSearchResult],
    at_qq: Optional[int] = None,
    fetcher: Optional[Callable[[str], bytes]] = None,
    max_images: int = 5,
) -> None:
    """
    发送携带图片的 Meru 消息。

    Args:
        api_base (str): OneBot API 基地址。
        group_id (int): 目标群号。
        access_token (str): API Token。
        text (str): 文本内容。
        items (Sequence[MeruSearchResult]): 商品列表。
        at_qq (Optional[int]): @ 目标。
        fetcher (Optional[Callable[[str], bytes]]): 自定义下载器。
        max_images (int): 附图上限。
    """
    payload = compose_meru_media_message(
        text, items, at_qq=at_qq, fetcher=fetcher, max_images=max_images
    )
    _send_group_msg(api_base, group_id, payload, access_token)
