"""
图像存储与生成管理模块。

提供统一的入站图像下载存储、模型生成图像落盘、
以及向多模态模型投喂数据所需的 Base64 数据等能力。
"""

from __future__ import annotations

import base64
import mimetypes
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

__all__ = [
    "StoredImage",
    "GeneratedImage",
    "ImageStorageManager",
]


@dataclass(frozen=True)
class StoredImage:
    """本地化保存后的入站图像信息。"""

    path: Path
    mime_type: str
    base64_data: str

    def data_url(self) -> str:
        """返回 data URL 形式的图像字符串。

        Returns:
            str: 形如 ``data:image/png;base64,xxxx`` 的字符串。

        Raises:
            AssertionError: 当图像缺失 MIME 类型或 Base64 内容时抛出。
        """
        assert self.mime_type and self.base64_data, "存储后的图像缺少必要信息"
        return f"data:{self.mime_type};base64,{self.base64_data}"


@dataclass(frozen=True)
class GeneratedImage:
    """模型生成的图像落盘信息。"""

    path: Path
    mime_type: str
    prompt: str


class ImageStorageManager:
    """
    负责管理 QQ Bot 入站/出站图像的本地存储。

    - 所有图像将保存在 base_dir 下的 incoming/generated 子目录；
    - 入站图像保存后提供 Base64 编码，便于向多模态模型提供数据；
    - 生成图像调用 OpenAI Images API，落盘后返回路径供 Bot 发送 CQ 图片。
    """

    def __init__(self, base_dir: str, image_model: Optional[str] = None) -> None:
        assert isinstance(base_dir, str) and base_dir.strip(), "base_dir 不能为空"
        self._base_dir = Path(base_dir).expanduser().resolve()
        self._incoming_dir = self._base_dir / "incoming"
        self._generated_dir = self._base_dir / "generated"
        self._incoming_dir.mkdir(parents=True, exist_ok=True)
        self._generated_dir.mkdir(parents=True, exist_ok=True)
        self._image_model = image_model or os.environ.get("IMAGE_MODEL_NAME", "gpt-image-1")
        self._lock = threading.Lock()
        self._http_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    @staticmethod
    def _infer_mime(data: bytes, fallback: Optional[str]) -> str:
        """推断图像 MIME 类型。

        Args:
            data (bytes): 图像二进制数据。
            fallback (Optional[str]): 备用的 MIME 字符串，例如 HTTP 头中的 ``Content-Type``。

        Returns:
            str: 标准 MIME 类型字符串。

        Raises:
            AssertionError: 当无法识别图像类型时抛出。
        """
        import imghdr

        guessed = imghdr.what(None, data)
        if guessed:
            if guessed == "jpeg":
                return "image/jpeg"
            if guessed == "png":
                return "image/png"
            if guessed == "gif":
                return "image/gif"
            if guessed == "webp":
                return "image/webp"
        if fallback:
            return fallback
        raise AssertionError("无法识别图像 MIME 类型")

    def _write_bytes(self, directory: Path, data: bytes, suffix: str) -> Path:
        """将二进制图像写入指定目录。

        Args:
            directory (Path): 目标目录。
            data (bytes): 图像数据。
            suffix (str): 文件扩展名（包含 ``.`` ）。

        Returns:
            Path: 写入后的绝对路径。

        Raises:
            AssertionError: 当目录不存在时抛出。
        """
        assert directory.is_dir(), "目标目录不存在"
        file_name = f"{uuid.uuid4().hex}{suffix}"
        path = directory / file_name
        with self._lock:
            path.write_bytes(data)
        return path

    def save_remote_image(self, url: str, filename_hint: Optional[str] = None) -> StoredImage:
        """
        下载并保存远程图像。

        Args:
            url (str): 图像下载地址。
            filename_hint (Optional[str]): 原始文件名提示，若存在用于决定默认扩展名。

        Returns:
            StoredImage: 本地化后的图像信息。

        Raises:
            AssertionError: 当 URL 无效或无法确定扩展名时抛出。
            RuntimeError: 当网络请求失败或返回内容为空时抛出。
        """
        assert url and url.startswith("http"), "仅支持通过 HTTP(S) 下载图像"
        resp = requests.get(url, timeout=15, headers=self._http_headers)
        if resp.status_code != 200:
            raise RuntimeError(f"下载图像失败，HTTP {resp.status_code}")
        data = resp.content
        if not data:
            raise RuntimeError("下载到的图像为空")
        content_type = resp.headers.get("Content-Type") or ""
        mime = self._infer_mime(data, content_type.split(";")[0] if content_type else None)
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(mime)
        if not suffix:
            if mime and mime.startswith("image/"):
                guessed = mimetypes.guess_extension(mime, strict=False)
                if guessed:
                    suffix = guessed
            if not suffix and filename_hint and "." in filename_hint:
                suffix = "." + filename_hint.rsplit(".", 1)[-1]
            if not suffix:
                # 默认兜底为 jpg，保证写盘成功
                suffix = ".jpg"
        path = self._write_bytes(self._incoming_dir, data, suffix)
        b64 = base64.b64encode(data).decode("ascii")
        return StoredImage(path=path, mime_type=mime, base64_data=b64)

    def save_generated_image(self, b64_data: str, prompt: str, mime_type: str) -> GeneratedImage:
        """
        将模型生成的 Base64 图像保存到磁盘。

        Args:
            b64_data (str): Base64 编码内容。
            prompt (str): 生成时使用的提示。
            mime_type (str): 图像 MIME 类型。

        Returns:
            GeneratedImage: 包含本地路径与元数据的对象。

        Raises:
            AssertionError: 当入参不完整或图像类型不受支持时抛出。
        """
        assert b64_data and prompt and mime_type, "生成图像入参不完整"
        data = base64.b64decode(b64_data)
        suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(mime_type)
        if not suffix:
            raise AssertionError(f"不支持的生成图像类型: {mime_type}")
        path = self._write_bytes(self._generated_dir, data, suffix)
        return GeneratedImage(path=path, mime_type=mime_type, prompt=prompt)

    def save_base64_image(
        self, b64_data: str, mime_type: str = "image/jpeg"
    ) -> StoredImage:
        """
        将 Base64 图像写入 incoming 目录，并返回存储信息。

        Args:
            b64_data (str): Base64 编码内容。
            mime_type (str): 图像 MIME 类型，默认 "image/jpeg"。

        Returns:
            StoredImage: 保存后的图像信息。

        Raises:
            AssertionError: 当 Base64 数据为空时抛出。
        """
        assert b64_data, "Base64 数据不能为空"
        data = base64.b64decode(b64_data)
        suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(mime_type, ".jpg")
        path = self._write_bytes(self._incoming_dir, data, suffix)
        encoded = base64.b64encode(data).decode("ascii")
        return StoredImage(path=path, mime_type=mime_type, base64_data=encoded)

    def generate_image_via_openai(self, prompt: str, size: str = "1024x1024") -> GeneratedImage:
        """
        使用 OpenAI Images API 生成图像并保存。

        Args:
            prompt (str): 图像描述。
            size (str): 输出尺寸，默认 1024x1024。

        Returns:
            GeneratedImage: 生成后的图像信息。

        Raises:
            AssertionError: 当提示为空时抛出。
            RuntimeError: 当 OpenAI 接口未返回图像或缺少必要字段时抛出。
        """
        from openai import OpenAI

        assert prompt.strip(), "prompt 不能为空"
        client = OpenAI()
        response = client.images.generate(
            model=self._image_model,
            prompt=prompt.strip(),
            size=size,
            response_format="b64_json",
        )
        if not response.data:
            raise RuntimeError("未收到图像生成结果")
        item = response.data[0]
        b64_data = getattr(item, "b64_json", None)
        mime_type = getattr(item, "mime_type", None) or "image/png"
        if not b64_data:
            raise RuntimeError("生成结果缺少 Base64 数据")
        return self.save_generated_image(b64_data, prompt.strip(), mime_type)
