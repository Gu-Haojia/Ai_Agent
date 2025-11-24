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
from io import BytesIO
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from google import genai
from google.genai import types
from google.genai import errors

__all__ = [
    "StoredImage",
    "GeneratedImage",
    "ImageStorageManager",
]

# Gemini 接口需要的参考图像结构：(mime_type, base64 数据)
GeminiReferenceImage = tuple[str, str]


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

    def _generate_url_candidates(self, url: str) -> list[str]:
        """根据已知规则生成优先尝试的下载地址列表。"""
        parsed = urlparse(url)
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        candidates: list[str] = []

        # Twitter / X：优先尝试原始尺寸
        if "pbs.twimg.com" in netloc and path:
            trimmed_path = path
            replaced = False
            for suffix in [
                "_400x400",
                "_200x200",
                "_133x133",
                "_96x96",
                "_bigger",
                "_normal",
                "_mini",
            ]:
                if suffix in trimmed_path:
                    trimmed_path = trimmed_path.replace(suffix, "")
                    replaced = True
            if replaced:
                base = parsed._replace(path=trimmed_path, query="")
                candidates.append(urlunparse(base._replace(query="format=jpg&name=orig")))
                candidates.append(urlunparse(base._replace(query="name=orig")))
                candidates.append(urlunparse(base._replace(query="name=large")))

        # 去除常见缩略图参数
        if query:
            original_pairs = parse_qsl(query, keep_blank_values=True)
            filtered_pairs = [
                (k, v)
                for k, v in original_pairs
                if k not in {"imageView", "thumbnail", "w", "h", "width", "height", "quality"}
            ]
            if filtered_pairs != original_pairs:
                filtered_url = urlunparse(
                    parsed._replace(query=urlencode(filtered_pairs, doseq=True))
                )
                candidates.append(filtered_url)
            candidates.append(urlunparse(parsed._replace(query="")))

        candidates.append(url)

        seen: set[str] = set()
        ordered: list[str] = []
        for item in candidates:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered

    def is_generated_path(self, candidate: str) -> bool:
        """
        判断给定路径是否位于生成图像目录内。

        Args:
            candidate (str): 待判断的路径字符串，支持 ``file://`` 前缀。

        Returns:
            bool: 当路径指向 ``generated`` 目录或其子项时返回 ``True``，否则返回 ``False``。

        Raises:
            AssertionError: 当传入参数类型不是字符串时抛出。
        """

        assert isinstance(candidate, str), "candidate 必须为字符串"
        normalized = candidate.strip()
        if not normalized:
            return False
        if normalized.startswith("file://"):
            normalized = normalized[len("file://") :]
        target_path = Path(normalized).expanduser()
        try:
            target_resolved = target_path.resolve(strict=False)
            generated_resolved = self._generated_dir.resolve(strict=False)
        except Exception:
            return False
        return generated_resolved == target_resolved or generated_resolved in target_resolved.parents

    def save_remote_image(
        self, url: str, filename_hint: Optional[str] = None
    ) -> Optional[StoredImage]:
        """
        下载并保存远程图像。

        Args:
            url (str): 图像下载地址。
            filename_hint (Optional[str]): 原始文件名提示，若存在用于决定默认扩展名。

        Returns:
            Optional[StoredImage]: 本地化后的图像信息；当目标为 GIF 等不支持类型时返回 ``None``。

        Raises:
            AssertionError: 当 URL 无效或无法确定扩展名时抛出。
            RuntimeError: 当网络请求失败或返回内容为空时抛出。
        """
        assert url and url.startswith("http"), "仅支持通过 HTTP(S) 下载图像"
        last_error = ""
        data = b""
        mime = ""
        chosen_url = ""
        for candidate in self._generate_url_candidates(url):
            try:
                resp = requests.get(candidate, timeout=15, headers=self._http_headers)
            except Exception as exc:
                last_error = str(exc)
                continue
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                continue
            if not resp.content:
                last_error = "空响应"
                continue
            content_type = resp.headers.get("Content-Type") or ""
            try:
                mime = self._infer_mime(
                    resp.content,
                    content_type.split(";")[0] if content_type else None,
                )
            except AssertionError as err:
                last_error = str(err)
                continue
            if mime == "image/gif":
                # GIF 动图不参与多模态分析，直接忽略并结束
                return None
            data = resp.content
            chosen_url = candidate
            break

        if not data:
            raise RuntimeError(f"下载图像失败：{last_error or '未知原因'}")

        if not filename_hint:
            filename_hint = os.path.basename(urlparse(chosen_url).path)
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

    def load_stored_image(self, filename: str) -> StoredImage:
        """
        根据文件名读取已保存的图像。

        优先在 incoming 目录下查找，其次尝试 generated 目录；
        读取成功后重新封装为 StoredImage，包含最新的 Base64 数据。

        Args:
            filename (str): 图像文件名，仅支持纯文件名，不允许包含路径分隔符。

        Returns:
            StoredImage: 匹配到的图像信息。

        Raises:
            AssertionError: 当文件名无效或文件不存在时抛出。
        """

        assert isinstance(filename, str) and filename.strip(), "filename 不能为空"
        normalized = filename.strip()
        assert Path(normalized).name == normalized, "filename 不允许包含路径"

        def _read_from(path: Path) -> StoredImage:
            data_bytes = path.read_bytes()
            guessed = mimetypes.guess_type(str(path))[0]
            mime = (
                guessed
                if guessed and guessed.startswith("image/")
                else self._infer_mime(data_bytes, guessed)
            )
            encoded = base64.b64encode(data_bytes).decode("ascii")
            return StoredImage(path=path, mime_type=mime, base64_data=encoded)

        for directory in (self._incoming_dir, self._generated_dir):
            candidate = directory / normalized
            if candidate.is_file():
                return _read_from(candidate)

        raise AssertionError(f"找不到已保存的图像文件: {normalized}")

    def resolve_image_path(self, filename: str) -> Path:
        """
        根据文件名获取图像的绝对路径。

        Args:
            filename (str): 图像文件名，仅支持纯文件名。

        Returns:
            Path: 匹配到的本地图像绝对路径。

        Raises:
            AssertionError: 当文件名为空或包含路径分隔符时抛出。
            FileNotFoundError: 当未找到对应图像时抛出。
        """

        assert isinstance(filename, str) and filename.strip(), "filename 不能为空"
        normalized = filename.strip()
        assert Path(normalized).name == normalized, "filename 不允许包含路径"

        for directory in (self._incoming_dir, self._generated_dir):
            candidate = directory / normalized
            if candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(f"找不到图像文件: {normalized}")

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

    def generate_image_via_gemini(
        self,
        prompt: str,
        *,
        aspect_ratio: Optional[str] = None,
        reference_images: Optional[Sequence[GeminiReferenceImage]] = None,
        model: Optional[str] = None,
        timeout: int = 180,
    ) -> GeneratedImage:
        """
        使用 Gemini API 生成或编辑图像，并将结果保存到本地。

        Args:
            prompt (str): 图像生成或编辑的文本描述。
            aspect_ratio (Optional[str]): 输出比例，可选 ``"1:1"``、``"2:3"``、``"3:2"``、``"3:4"``、``"4:3"``、``"9:16"``、``"16:9"``。
                当用户未指定比例时请勿传入参数，传入 ``None`` 表示不指定比例。
            reference_images (Optional[Sequence[GeminiReferenceImage]]):
                参考图像列表，每项为 ``(mime_type, base64_data)``。
            model (Optional[str]): Gemini 图像模型名称，默认 ``gemini-2.5-flash-image``。
            timeout (int): HTTP 请求超时时间（秒）。

        Returns:
            GeneratedImage: 保存后的图像信息。

        Raises:
            AssertionError: 当参数非法或缺少 API 密钥时抛出。
            ValueError: 当 HTTP 请求失败时抛出。
            RuntimeError: 当多次请求仍失败或响应缺少图像数据、被判定为违规内容时抛出。
        """

        assert isinstance(prompt, str) and prompt.strip(), "prompt 不能为空"
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        )
        assert api_key, "缺少 Gemini API Key，请设置 GOOGLE_API_KEY 或 GEMINI_API_KEY"

        model_name = model or os.environ.get(
            "GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"
        )

        ratio = aspect_ratio.strip() if isinstance(aspect_ratio, str) else None
        allowed_ratio = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"}
        if ratio:
            assert ratio in allowed_ratio, f"aspect ratio 仅支持 {sorted(allowed_ratio)}"

        client = genai.Client()

        http_options = None
        if timeout is not None:
            http_options = types.HttpOptions(timeout=int(timeout * 1000))

        image_cfg = None
        if ratio:
            image_cfg = types.ImageConfig(aspect_ratio=ratio)

        config = (
            types.GenerateContentConfig(
                image_config=image_cfg,
                http_options=http_options,
            )
            if (image_cfg is not None or http_options is not None)
            else None
        )

        prompt_clean = prompt.strip()

        parts: list[types.Part] = []
        if reference_images:
            for mime_type, data_b64 in reference_images:
                assert (
                    isinstance(mime_type, str)
                    and mime_type.startswith("image/")
                ), "参考图像必须提供合法的 MIME 类型"
                assert (
                    isinstance(data_b64, str) and data_b64.strip()
                ), "参考图像 Base64 数据不能为空"
                try:
                    image_bytes = base64.b64decode(data_b64.strip(), validate=True)
                except Exception as exc:
                    raise ValueError("参考图像 Base64 数据无效") from exc
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            data=image_bytes, mime_type=mime_type.strip()
                        )
                    )
                )

        parts.append(types.Part(text=prompt_clean))
        contents = [types.Content(role="user", parts=parts)]

        max_attempts = 2
        response = None
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                break
            except errors.ClientError as exc:
                status = getattr(exc, "status_code", None)
                if status and int(status) >= 500 and attempt + 1 < max_attempts:
                    print(
                        f"警告：Gemini 图像生成 HTTP {status}，准备重试…",
                        flush=True,
                    )
                    time.sleep(1)
                    continue
                raise

        if response is None:
            raise RuntimeError("Gemini 图像生成接口无响应")

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            print(f"ERROR: 完整 Gemini 响应内容: {response}", flush=True)
            raise RuntimeError("Gemini 响应缺少候选结果")

        first = candidates[0]
        finish_reason = getattr(first, "finish_reason", None)
        if finish_reason in {
            types.FinishReason.PROHIBITED_CONTENT,
            types.FinishReason.SAFETY,
            types.FinishReason.IMAGE_SAFETY,
            types.FinishReason.BLOCKLIST,
        }:
            finish_message = getattr(first, "finish_message", None) or "Gemini 判定请求涉及受限内容"
            raise RuntimeError(f"Gemini 拒绝生成图像：{finish_message}")

        content = getattr(first, "content", None)
        parts_data = getattr(content, "parts", None) if content else None
        if not parts_data:
            print(f"ERROR: 完整 Gemini 响应内容: {response}", flush=True)
            raise RuntimeError("Gemini 响应缺少内容片段")

        inline_data = None
        text_outputs: list[str] = []
        for part in parts_data:
            if getattr(part, "text", None):
                text_outputs.append(str(part.text))
            blob = getattr(part, "inline_data", None)
            if blob and getattr(blob, "data", None) and getattr(blob, "mime_type", ""):
                inline_data = blob
                break

        if not inline_data:
            print(f"ERROR: 完整 Gemini 响应内容: {response}", flush=True)
            raise RuntimeError("Gemini 响应未返回图像数据")

        raw_bytes = inline_data.data
        if isinstance(raw_bytes, str):
            try:
                raw_bytes = base64.b64decode(raw_bytes, validate=True)
            except Exception:
                raw_bytes = raw_bytes.encode("utf-8")
        assert isinstance(raw_bytes, (bytes, bytearray)), "返回的图像数据格式异常"

        mime_type = getattr(inline_data, "mime_type", None) or "image/png"

        cleaned_bytes = raw_bytes
        cleaned_mime = mime_type
        try:
            from PIL import Image

            with Image.open(BytesIO(raw_bytes)) as im:
                # 强制去除元数据，转存为 PNG，避免下游预览因 EXIF/JFIF 差异产生异常。
                mode = "RGBA" if im.mode == "RGBA" else "RGB"
                converted = im.convert(mode)
                buf = BytesIO()
                converted.save(buf, format="PNG")
                cleaned_bytes = buf.getvalue()
                cleaned_mime = "image/png"
        except Exception:
            pass

        b64_data = base64.b64encode(cleaned_bytes).decode("ascii")

        prompt_text = " ".join(text_outputs).strip() or prompt_clean

        return self.save_generated_image(b64_data, prompt_text, cleaned_mime)
