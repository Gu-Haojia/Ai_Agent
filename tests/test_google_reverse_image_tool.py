"""
Google 反向搜图上传器单元测试。
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import requests

from src.google_reverse_image_tool import ReverseImageUploader


def _response(
    *,
    status_code: int = 200,
    payload: object | None = None,
    text: str = "",
) -> mock.Mock:
    """
    构造固定内容的 requests.Response 替身。

    Args:
        status_code (int): HTTP 状态码。
        payload (object | None): ``response.json()`` 返回的数据。
        text (str): ``response.text`` 返回的文本。

    Returns:
        mock.Mock: 可用于上传器测试的响应替身。

    Raises:
        AssertionError: 当状态码不是正整数时抛出。
    """

    assert status_code > 0, "status_code 必须为正整数"
    response = mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.text = text
    return response


def test_upload_resolves_signed_tmpfiles_download_url(tmp_path: Path) -> None:
    """上传成功后应从落地页提取 tmpfiles 当前使用的签名下载链接。"""

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-image")
    landing_url = "https://tmpfiles.org/file-id/image.jpg"
    signed_url = "https://tmpfiles.org/dl/1234567890.signature/file-id/image.jpg"
    upload_response = _response(
        payload={"status": "success", "data": {"url": landing_url}}
    )
    landing_response = _response(
        text=(
            "<html><body>"
            f'<a class="download" href="{signed_url}">Download</a>'
            "</body></html>"
        )
    )

    with (
        mock.patch(
            "src.google_reverse_image_tool.requests.post",
            return_value=upload_response,
        ) as post,
        mock.patch(
            "src.google_reverse_image_tool.requests.get",
            return_value=landing_response,
        ) as get,
    ):
        result = ReverseImageUploader().upload(image_path)

    assert result == signed_url
    post.assert_called_once()
    get.assert_called_once_with(
        landing_url,
        headers={"User-Agent": "curl/8.1.2", "Accept": "text/html"},
        timeout=30,
    )


def test_upload_rejects_landing_page_without_download_link(tmp_path: Path) -> None:
    """tmpfiles 页面结构不符合预期时应显式报错而不是返回无效链接。"""

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-image")
    landing_url = "https://tmpfiles.org/file-id/image.jpg"
    upload_response = _response(
        payload={"status": "success", "data": {"url": landing_url}}
    )
    landing_response = _response(text="<html><body>missing link</body></html>")

    with (
        mock.patch(
            "src.google_reverse_image_tool.requests.post",
            return_value=upload_response,
        ),
        mock.patch(
            "src.google_reverse_image_tool.requests.get",
            return_value=landing_response,
        ),
        pytest.raises(ValueError, match="缺少下载链接"),
    ):
        ReverseImageUploader().upload(image_path)
