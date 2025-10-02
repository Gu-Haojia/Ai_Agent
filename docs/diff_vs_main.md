# 与 `main` 的差异行级注释 (自动生成)

基线：main@d738feb；当前：fix-restore-clean-lines@a46f4c6

| 文件 | 序号 | 修改 | 注释 |
| --- | --- | --- | --- |
| `image_storage.py` | 1 | `--- /dev/null` | Diff 标记：旧文件路径 |
| `image_storage.py` | 2 | `+++ b/image_storage.py` | Diff 标记：新文件路径 |
| `image_storage.py` | 3 | `+"""` | 文档字符串内容 |
| `image_storage.py` | 4 | `+图像存储与生成管理模块。` | 文档字符串内容 |
| `image_storage.py` | 5 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 6 | `+提供统一的入站图像下载存储、模型生成图像落盘、` | 文档字符串内容 |
| `image_storage.py` | 7 | `+以及向多模态模型投喂数据所需的 Base64 数据等能力。` | 文档字符串内容 |
| `image_storage.py` | 8 | `+"""` | 文档字符串内容 |
| `image_storage.py` | 9 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 10 | `+from __future__ import annotations` | 按需导入对象 |
| `image_storage.py` | 11 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 12 | `+import base64` | 导入标准库/三方模块 |
| `image_storage.py` | 13 | `+import mimetypes` | 导入标准库/三方模块 |
| `image_storage.py` | 14 | `+import os` | 导入标准库/三方模块 |
| `image_storage.py` | 15 | `+import threading` | 导入标准库/三方模块 |
| `image_storage.py` | 16 | `+import uuid` | 导入标准库/三方模块 |
| `image_storage.py` | 17 | `+from dataclasses import dataclass` | 按需导入对象 |
| `image_storage.py` | 18 | `+from pathlib import Path` | 按需导入对象 |
| `image_storage.py` | 19 | `+from typing import Optional` | 按需导入对象 |
| `image_storage.py` | 20 | `+from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse` | 按需导入对象 |
| `image_storage.py` | 21 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 22 | `+import requests` | 导入标准库/三方模块 |
| `image_storage.py` | 23 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 24 | `+__all__ = [` | 赋值/构建数据 |
| `image_storage.py` | 25 | `+    "StoredImage",` | 字符串字面量 |
| `image_storage.py` | 26 | `+    "GeneratedImage",` | 字符串字面量 |
| `image_storage.py` | 27 | `+    "ImageStorageManager",` | 字符串字面量 |
| `image_storage.py` | 28 | `+]` | 集合结构定义 |
| `image_storage.py` | 29 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 30 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 31 | `+@dataclass(frozen=True)` | 装饰器，修饰定义 |
| `image_storage.py` | 32 | `+class StoredImage:` | 定义类 StoredImage: |
| `image_storage.py` | 33 | `+    """本地化保存后的入站图像信息。"""` | 文档字符串内容 |
| `image_storage.py` | 34 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 35 | `+    path: Path` | 逻辑实现 |
| `image_storage.py` | 36 | `+    mime_type: str` | 逻辑实现 |
| `image_storage.py` | 37 | `+    base64_data: str` | Base64 处理 |
| `image_storage.py` | 38 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 39 | `+    def data_url(self) -> str:` | 定义函数 data_url |
| `image_storage.py` | 40 | `+        """返回 data URL 形式的图像字符串。` | 文档字符串内容 |
| `image_storage.py` | 41 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 42 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 43 | `+            str: 形如 \`\`data:image/png;base64,xxxx\`\` 的字符串。` | 文档字符串内容 |
| `image_storage.py` | 44 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 45 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 46 | `+            AssertionError: 当图像缺失 MIME 类型或 Base64 内容时抛出。` | 文档字符串内容 |
| `image_storage.py` | 47 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 48 | `+        assert self.mime_type and self.base64_data, "存储后的图像缺少必要信息"` | 断言前置条件 |
| `image_storage.py` | 49 | `+        return f"data:{self.mime_type};base64,{self.base64_data}"` | 返回处理结果 |
| `image_storage.py` | 50 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 51 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 52 | `+@dataclass(frozen=True)` | 装饰器，修饰定义 |
| `image_storage.py` | 53 | `+class GeneratedImage:` | 定义类 GeneratedImage: |
| `image_storage.py` | 54 | `+    """模型生成的图像落盘信息。"""` | 文档字符串内容 |
| `image_storage.py` | 55 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 56 | `+    path: Path` | 逻辑实现 |
| `image_storage.py` | 57 | `+    mime_type: str` | 逻辑实现 |
| `image_storage.py` | 58 | `+    prompt: str` | 逻辑实现 |
| `image_storage.py` | 59 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 60 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 61 | `+class ImageStorageManager:` | 定义类 ImageStorageManager: |
| `image_storage.py` | 62 | `+    """` | 文档字符串内容 |
| `image_storage.py` | 63 | `+    负责管理 QQ Bot 入站/出站图像的本地存储。` | 文档字符串内容 |
| `image_storage.py` | 64 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 65 | `+    - 所有图像将保存在 base_dir 下的 incoming/generated 子目录；` | 文档字符串内容 |
| `image_storage.py` | 66 | `+    - 入站图像保存后提供 Base64 编码，便于向多模态模型提供数据；` | 文档字符串内容 |
| `image_storage.py` | 67 | `+    - 生成图像调用 OpenAI Images API，落盘后返回路径供 Bot 发送 CQ 图片。` | 文档字符串内容 |
| `image_storage.py` | 68 | `+    """` | 文档字符串内容 |
| `image_storage.py` | 69 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 70 | `+    def __init__(self, base_dir: str, image_model: Optional[str] = None) -> None:` | 定义函数 __init__ |
| `image_storage.py` | 71 | `+        assert isinstance(base_dir, str) and base_dir.strip(), "base_dir 不能为空"` | 断言前置条件 |
| `image_storage.py` | 72 | `+        self._base_dir = Path(base_dir).expanduser().resolve()` | 实例属性操作 |
| `image_storage.py` | 73 | `+        self._incoming_dir = self._base_dir / "incoming"` | 实例属性操作 |
| `image_storage.py` | 74 | `+        self._generated_dir = self._base_dir / "generated"` | 实例属性操作 |
| `image_storage.py` | 75 | `+        self._incoming_dir.mkdir(parents=True, exist_ok=True)` | 实例属性操作 |
| `image_storage.py` | 76 | `+        self._generated_dir.mkdir(parents=True, exist_ok=True)` | 实例属性操作 |
| `image_storage.py` | 77 | `+        self._image_model = image_model or os.environ.get("IMAGE_MODEL_NAME", "gpt-image-1")` | 实例属性操作 |
| `image_storage.py` | 78 | `+        self._lock = threading.Lock()` | 实例属性操作 |
| `image_storage.py` | 79 | `+        self._http_headers = {` | 实例属性操作 |
| `image_storage.py` | 80 | `+            "User-Agent": (` | 字符串字面量 |
| `image_storage.py` | 81 | `+                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "` | 字符串字面量 |
| `image_storage.py` | 82 | `+                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"` | 字符串字面量 |
| `image_storage.py` | 83 | `+            ),` | 逻辑实现 |
| `image_storage.py` | 84 | `+            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",` | 字符串字面量 |
| `image_storage.py` | 85 | `+            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",` | 字符串字面量 |
| `image_storage.py` | 86 | `+        }` | 逻辑实现 |
| `image_storage.py` | 87 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 88 | `+    @staticmethod` | 装饰器，修饰定义 |
| `image_storage.py` | 89 | `+    def _infer_mime(data: bytes, fallback: Optional[str]) -> str:` | 定义函数 _infer_mime |
| `image_storage.py` | 90 | `+        """推断图像 MIME 类型。` | 文档字符串内容 |
| `image_storage.py` | 91 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 92 | `+        Args:` | 文档字符串内容 |
| `image_storage.py` | 93 | `+            data (bytes): 图像二进制数据。` | 文档字符串内容 |
| `image_storage.py` | 94 | `+            fallback (Optional[str]): 备用的 MIME 字符串，例如 HTTP 头中的 \`\`Content-Type\`\`。` | 文档字符串内容 |
| `image_storage.py` | 95 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 96 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 97 | `+            str: 标准 MIME 类型字符串。` | 文档字符串内容 |
| `image_storage.py` | 98 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 99 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 100 | `+            AssertionError: 当无法识别图像类型时抛出。` | 文档字符串内容 |
| `image_storage.py` | 101 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 102 | `+        import imghdr` | 导入标准库/三方模块 |
| `image_storage.py` | 103 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 104 | `+        guessed = imghdr.what(None, data)` | 赋值/构建数据 |
| `image_storage.py` | 105 | `+        if guessed:` | 条件判断 |
| `image_storage.py` | 106 | `+            if guessed == "jpeg":` | 条件判断 |
| `image_storage.py` | 107 | `+                return "image/jpeg"` | 返回处理结果 |
| `image_storage.py` | 108 | `+            if guessed == "png":` | 条件判断 |
| `image_storage.py` | 109 | `+                return "image/png"` | 返回处理结果 |
| `image_storage.py` | 110 | `+            if guessed == "gif":` | 条件判断 |
| `image_storage.py` | 111 | `+                return "image/gif"` | 返回处理结果 |
| `image_storage.py` | 112 | `+            if guessed == "webp":` | 条件判断 |
| `image_storage.py` | 113 | `+                return "image/webp"` | 返回处理结果 |
| `image_storage.py` | 114 | `+        if fallback:` | 条件判断 |
| `image_storage.py` | 115 | `+            return fallback` | 返回处理结果 |
| `image_storage.py` | 116 | `+        raise AssertionError("无法识别图像 MIME 类型")` | 抛出异常 |
| `image_storage.py` | 117 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 118 | `+    def _write_bytes(self, directory: Path, data: bytes, suffix: str) -> Path:` | 定义函数 _write_bytes |
| `image_storage.py` | 119 | `+        """将二进制图像写入指定目录。` | 文档字符串内容 |
| `image_storage.py` | 120 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 121 | `+        Args:` | 文档字符串内容 |
| `image_storage.py` | 122 | `+            directory (Path): 目标目录。` | 文档字符串内容 |
| `image_storage.py` | 123 | `+            data (bytes): 图像数据。` | 文档字符串内容 |
| `image_storage.py` | 124 | `+            suffix (str): 文件扩展名（包含 \`\`.\`\` ）。` | 文档字符串内容 |
| `image_storage.py` | 125 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 126 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 127 | `+            Path: 写入后的绝对路径。` | 文档字符串内容 |
| `image_storage.py` | 128 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 129 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 130 | `+            AssertionError: 当目录不存在时抛出。` | 文档字符串内容 |
| `image_storage.py` | 131 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 132 | `+        assert directory.is_dir(), "目标目录不存在"` | 断言前置条件 |
| `image_storage.py` | 133 | `+        file_name = f"{uuid.uuid4().hex}{suffix}"` | 赋值/构建数据 |
| `image_storage.py` | 134 | `+        path = directory / file_name` | 赋值/构建数据 |
| `image_storage.py` | 135 | `+        with self._lock:` | 上下文管理 |
| `image_storage.py` | 136 | `+            path.write_bytes(data)` | 逻辑实现 |
| `image_storage.py` | 137 | `+        return path` | 返回处理结果 |
| `image_storage.py` | 138 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 139 | `+    def _generate_url_candidates(self, url: str) -> list[str]:` | 定义函数 _generate_url_candidates |
| `image_storage.py` | 140 | `+        """根据已知规则生成优先尝试的下载地址列表。"""` | 文档字符串内容 |
| `image_storage.py` | 141 | `+        parsed = urlparse(url)` | 赋值/构建数据 |
| `image_storage.py` | 142 | `+        netloc = (parsed.netloc or "").lower()` | 赋值/构建数据 |
| `image_storage.py` | 143 | `+        path = parsed.path or ""` | 赋值/构建数据 |
| `image_storage.py` | 144 | `+        query = parsed.query or ""` | 赋值/构建数据 |
| `image_storage.py` | 145 | `+        candidates: list[str] = []` | 集合结构定义 |
| `image_storage.py` | 146 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 147 | `+        # Twitter / X：优先尝试原始尺寸` | 注释，补充说明 |
| `image_storage.py` | 148 | `+        if "pbs.twimg.com" in netloc and path:` | 条件判断 |
| `image_storage.py` | 149 | `+            trimmed_path = path` | 赋值/构建数据 |
| `image_storage.py` | 150 | `+            replaced = False` | 赋值/构建数据 |
| `image_storage.py` | 151 | `+            for suffix in [` | 遍历集合 |
| `image_storage.py` | 152 | `+                "_400x400",` | 字符串字面量 |
| `image_storage.py` | 153 | `+                "_200x200",` | 字符串字面量 |
| `image_storage.py` | 154 | `+                "_133x133",` | 字符串字面量 |
| `image_storage.py` | 155 | `+                "_96x96",` | 字符串字面量 |
| `image_storage.py` | 156 | `+                "_bigger",` | 字符串字面量 |
| `image_storage.py` | 157 | `+                "_normal",` | 字符串字面量 |
| `image_storage.py` | 158 | `+                "_mini",` | 字符串字面量 |
| `image_storage.py` | 159 | `+            ]:` | 逻辑实现 |
| `image_storage.py` | 160 | `+                if suffix in trimmed_path:` | 条件判断 |
| `image_storage.py` | 161 | `+                    trimmed_path = trimmed_path.replace(suffix, "")` | 赋值/构建数据 |
| `image_storage.py` | 162 | `+                    replaced = True` | 赋值/构建数据 |
| `image_storage.py` | 163 | `+            if replaced:` | 条件判断 |
| `image_storage.py` | 164 | `+                base = parsed._replace(path=trimmed_path, query="")` | 赋值/构建数据 |
| `image_storage.py` | 165 | `+                candidates.append(urlunparse(base._replace(query="format=jpg&name=orig")))` | 向列表追加 |
| `image_storage.py` | 166 | `+                candidates.append(urlunparse(base._replace(query="name=orig")))` | 向列表追加 |
| `image_storage.py` | 167 | `+                candidates.append(urlunparse(base._replace(query="name=large")))` | 向列表追加 |
| `image_storage.py` | 168 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 169 | `+        # 去除常见缩略图参数` | 注释，补充说明 |
| `image_storage.py` | 170 | `+        if query:` | 条件判断 |
| `image_storage.py` | 171 | `+            original_pairs = parse_qsl(query, keep_blank_values=True)` | 赋值/构建数据 |
| `image_storage.py` | 172 | `+            filtered_pairs = [` | 赋值/构建数据 |
| `image_storage.py` | 173 | `+                (k, v)` | 逻辑实现 |
| `image_storage.py` | 174 | `+                for k, v in original_pairs` | 遍历集合 |
| `image_storage.py` | 175 | `+                if k not in {"imageView", "thumbnail", "w", "h", "width", "height", "quality"}` | 条件判断 |
| `image_storage.py` | 176 | `+            ]` | 集合结构定义 |
| `image_storage.py` | 177 | `+            if filtered_pairs != original_pairs:` | 条件判断 |
| `image_storage.py` | 178 | `+                filtered_url = urlunparse(` | 赋值/构建数据 |
| `image_storage.py` | 179 | `+                    parsed._replace(query=urlencode(filtered_pairs, doseq=True))` | 赋值/构建数据 |
| `image_storage.py` | 180 | `+                )` | 逻辑实现 |
| `image_storage.py` | 181 | `+                candidates.append(filtered_url)` | 向列表追加 |
| `image_storage.py` | 182 | `+            candidates.append(urlunparse(parsed._replace(query="")))` | 向列表追加 |
| `image_storage.py` | 183 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 184 | `+        candidates.append(url)` | 向列表追加 |
| `image_storage.py` | 185 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 186 | `+        seen: set[str] = set()` | 赋值/构建数据 |
| `image_storage.py` | 187 | `+        ordered: list[str] = []` | 集合结构定义 |
| `image_storage.py` | 188 | `+        for item in candidates:` | 遍历集合 |
| `image_storage.py` | 189 | `+            if item and item not in seen:` | 条件判断 |
| `image_storage.py` | 190 | `+                ordered.append(item)` | 向列表追加 |
| `image_storage.py` | 191 | `+                seen.add(item)` | 逻辑实现 |
| `image_storage.py` | 192 | `+        return ordered` | 返回处理结果 |
| `image_storage.py` | 193 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 194 | `+    def save_remote_image(self, url: str, filename_hint: Optional[str] = None) -> StoredImage:` | 定义函数 save_remote_image |
| `image_storage.py` | 195 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 196 | `+        下载并保存远程图像。` | 文档字符串内容 |
| `image_storage.py` | 197 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 198 | `+        Args:` | 文档字符串内容 |
| `image_storage.py` | 199 | `+            url (str): 图像下载地址。` | 文档字符串内容 |
| `image_storage.py` | 200 | `+            filename_hint (Optional[str]): 原始文件名提示，若存在用于决定默认扩展名。` | 文档字符串内容 |
| `image_storage.py` | 201 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 202 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 203 | `+            StoredImage: 本地化后的图像信息。` | 文档字符串内容 |
| `image_storage.py` | 204 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 205 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 206 | `+            AssertionError: 当 URL 无效或无法确定扩展名时抛出。` | 文档字符串内容 |
| `image_storage.py` | 207 | `+            RuntimeError: 当网络请求失败或返回内容为空时抛出。` | 文档字符串内容 |
| `image_storage.py` | 208 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 209 | `+        assert url and url.startswith("http"), "仅支持通过 HTTP(S) 下载图像"` | 断言前置条件 |
| `image_storage.py` | 210 | `+        last_error = ""` | 赋值/构建数据 |
| `image_storage.py` | 211 | `+        data = b""` | 赋值/构建数据 |
| `image_storage.py` | 212 | `+        mime = ""` | 赋值/构建数据 |
| `image_storage.py` | 213 | `+        chosen_url = ""` | 赋值/构建数据 |
| `image_storage.py` | 214 | `+        for candidate in self._generate_url_candidates(url):` | 遍历集合 |
| `image_storage.py` | 215 | `+            try:` | 异常捕获块 |
| `image_storage.py` | 216 | `+                resp = requests.get(candidate, timeout=15, headers=self._http_headers)` | 发起 HTTP 请求 |
| `image_storage.py` | 217 | `+            except Exception as exc:` | 异常处理 |
| `image_storage.py` | 218 | `+                last_error = str(exc)` | 赋值/构建数据 |
| `image_storage.py` | 219 | `+                continue` | 逻辑实现 |
| `image_storage.py` | 220 | `+            if resp.status_code != 200:` | 条件判断 |
| `image_storage.py` | 221 | `+                last_error = f"HTTP {resp.status_code}"` | 赋值/构建数据 |
| `image_storage.py` | 222 | `+                continue` | 逻辑实现 |
| `image_storage.py` | 223 | `+            if not resp.content:` | 条件判断 |
| `image_storage.py` | 224 | `+                last_error = "空响应"` | 赋值/构建数据 |
| `image_storage.py` | 225 | `+                continue` | 逻辑实现 |
| `image_storage.py` | 226 | `+            content_type = resp.headers.get("Content-Type") or ""` | 赋值/构建数据 |
| `image_storage.py` | 227 | `+            try:` | 异常捕获块 |
| `image_storage.py` | 228 | `+                mime = self._infer_mime(resp.content, content_type.split(";")[0] if content_type else None)` | 字符串拆分 |
| `image_storage.py` | 229 | `+            except AssertionError as err:` | 异常处理 |
| `image_storage.py` | 230 | `+                last_error = str(err)` | 赋值/构建数据 |
| `image_storage.py` | 231 | `+                continue` | 逻辑实现 |
| `image_storage.py` | 232 | `+            data = resp.content` | 赋值/构建数据 |
| `image_storage.py` | 233 | `+            chosen_url = candidate` | 赋值/构建数据 |
| `image_storage.py` | 234 | `+            break` | 逻辑实现 |
| `image_storage.py` | 235 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 236 | `+        if not data:` | 条件判断 |
| `image_storage.py` | 237 | `+            raise RuntimeError(f"下载图像失败：{last_error or '未知原因'}")` | 抛出异常 |
| `image_storage.py` | 238 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 239 | `+        if not filename_hint:` | 条件判断 |
| `image_storage.py` | 240 | `+            filename_hint = os.path.basename(urlparse(chosen_url).path)` | 赋值/构建数据 |
| `image_storage.py` | 241 | `+        suffix = {` | 赋值/构建数据 |
| `image_storage.py` | 242 | `+            "image/jpeg": ".jpg",` | 字符串字面量 |
| `image_storage.py` | 243 | `+            "image/png": ".png",` | 字符串字面量 |
| `image_storage.py` | 244 | `+            "image/gif": ".gif",` | 字符串字面量 |
| `image_storage.py` | 245 | `+            "image/webp": ".webp",` | 字符串字面量 |
| `image_storage.py` | 246 | `+        }.get(mime)` | 逻辑实现 |
| `image_storage.py` | 247 | `+        if not suffix:` | 条件判断 |
| `image_storage.py` | 248 | `+            if mime and mime.startswith("image/"):` | 条件判断 |
| `image_storage.py` | 249 | `+                guessed = mimetypes.guess_extension(mime, strict=False)` | 赋值/构建数据 |
| `image_storage.py` | 250 | `+                if guessed:` | 条件判断 |
| `image_storage.py` | 251 | `+                    suffix = guessed` | 赋值/构建数据 |
| `image_storage.py` | 252 | `+            if not suffix and filename_hint and "." in filename_hint:` | 条件判断 |
| `image_storage.py` | 253 | `+                suffix = "." + filename_hint.rsplit(".", 1)[-1]` | 字符串拆分 |
| `image_storage.py` | 254 | `+            if not suffix:` | 条件判断 |
| `image_storage.py` | 255 | `+                # 默认兜底为 jpg，保证写盘成功` | 注释，补充说明 |
| `image_storage.py` | 256 | `+                suffix = ".jpg"` | 赋值/构建数据 |
| `image_storage.py` | 257 | `+        path = self._write_bytes(self._incoming_dir, data, suffix)` | 赋值/构建数据 |
| `image_storage.py` | 258 | `+        b64 = base64.b64encode(data).decode("ascii")` | Base64 处理 |
| `image_storage.py` | 259 | `+        return StoredImage(path=path, mime_type=mime, base64_data=b64)` | 返回处理结果 |
| `image_storage.py` | 260 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 261 | `+    def save_generated_image(self, b64_data: str, prompt: str, mime_type: str) -> GeneratedImage:` | 定义函数 save_generated_image |
| `image_storage.py` | 262 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 263 | `+        将模型生成的 Base64 图像保存到磁盘。` | 文档字符串内容 |
| `image_storage.py` | 264 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 265 | `+        Args:` | 文档字符串内容 |
| `image_storage.py` | 266 | `+            b64_data (str): Base64 编码内容。` | 文档字符串内容 |
| `image_storage.py` | 267 | `+            prompt (str): 生成时使用的提示。` | 文档字符串内容 |
| `image_storage.py` | 268 | `+            mime_type (str): 图像 MIME 类型。` | 文档字符串内容 |
| `image_storage.py` | 269 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 270 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 271 | `+            GeneratedImage: 包含本地路径与元数据的对象。` | 文档字符串内容 |
| `image_storage.py` | 272 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 273 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 274 | `+            AssertionError: 当入参不完整或图像类型不受支持时抛出。` | 文档字符串内容 |
| `image_storage.py` | 275 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 276 | `+        assert b64_data and prompt and mime_type, "生成图像入参不完整"` | 断言前置条件 |
| `image_storage.py` | 277 | `+        data = base64.b64decode(b64_data)` | Base64 处理 |
| `image_storage.py` | 278 | `+        suffix = {` | 赋值/构建数据 |
| `image_storage.py` | 279 | `+            "image/png": ".png",` | 字符串字面量 |
| `image_storage.py` | 280 | `+            "image/jpeg": ".jpg",` | 字符串字面量 |
| `image_storage.py` | 281 | `+            "image/gif": ".gif",` | 字符串字面量 |
| `image_storage.py` | 282 | `+            "image/webp": ".webp",` | 字符串字面量 |
| `image_storage.py` | 283 | `+        }.get(mime_type)` | 逻辑实现 |
| `image_storage.py` | 284 | `+        if not suffix:` | 条件判断 |
| `image_storage.py` | 285 | `+            raise AssertionError(f"不支持的生成图像类型: {mime_type}")` | 抛出异常 |
| `image_storage.py` | 286 | `+        path = self._write_bytes(self._generated_dir, data, suffix)` | 赋值/构建数据 |
| `image_storage.py` | 287 | `+        return GeneratedImage(path=path, mime_type=mime_type, prompt=prompt)` | 返回处理结果 |
| `image_storage.py` | 288 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 289 | `+    def save_base64_image(` | 定义函数 save_base64_image |
| `image_storage.py` | 290 | `+        self, b64_data: str, mime_type: str = "image/jpeg"` | Base64 处理 |
| `image_storage.py` | 291 | `+    ) -> StoredImage:` | 处理图片结构 |
| `image_storage.py` | 292 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 293 | `+        将 Base64 图像写入 incoming 目录，并返回存储信息。` | 文档字符串内容 |
| `image_storage.py` | 294 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 295 | `+        Args:` | 文档字符串内容 |
| `image_storage.py` | 296 | `+            b64_data (str): Base64 编码内容。` | 文档字符串内容 |
| `image_storage.py` | 297 | `+            mime_type (str): 图像 MIME 类型，默认 "image/jpeg"。` | 文档字符串内容 |
| `image_storage.py` | 298 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 299 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 300 | `+            StoredImage: 保存后的图像信息。` | 文档字符串内容 |
| `image_storage.py` | 301 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 302 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 303 | `+            AssertionError: 当 Base64 数据为空时抛出。` | 文档字符串内容 |
| `image_storage.py` | 304 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 305 | `+        assert b64_data, "Base64 数据不能为空"` | 断言前置条件 |
| `image_storage.py` | 306 | `+        data = base64.b64decode(b64_data)` | Base64 处理 |
| `image_storage.py` | 307 | `+        suffix = {` | 赋值/构建数据 |
| `image_storage.py` | 308 | `+            "image/png": ".png",` | 字符串字面量 |
| `image_storage.py` | 309 | `+            "image/jpeg": ".jpg",` | 字符串字面量 |
| `image_storage.py` | 310 | `+            "image/gif": ".gif",` | 字符串字面量 |
| `image_storage.py` | 311 | `+            "image/webp": ".webp",` | 字符串字面量 |
| `image_storage.py` | 312 | `+        }.get(mime_type, ".jpg")` | 逻辑实现 |
| `image_storage.py` | 313 | `+        path = self._write_bytes(self._incoming_dir, data, suffix)` | 赋值/构建数据 |
| `image_storage.py` | 314 | `+        encoded = base64.b64encode(data).decode("ascii")` | Base64 处理 |
| `image_storage.py` | 315 | `+        return StoredImage(path=path, mime_type=mime_type, base64_data=encoded)` | 返回处理结果 |
| `image_storage.py` | 316 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 317 | `+    def generate_image_via_openai(self, prompt: str, size: str = "1024x1024") -> GeneratedImage:` | 定义函数 generate_image_via_openai |
| `image_storage.py` | 318 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 319 | `+        使用 OpenAI Images API 生成图像并保存。` | 文档字符串内容 |
| `image_storage.py` | 320 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 321 | `+        Args:` | 文档字符串内容 |
| `image_storage.py` | 322 | `+            prompt (str): 图像描述。` | 文档字符串内容 |
| `image_storage.py` | 323 | `+            size (str): 输出尺寸，默认 1024x1024。` | 文档字符串内容 |
| `image_storage.py` | 324 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 325 | `+        Returns:` | 文档字符串内容 |
| `image_storage.py` | 326 | `+            GeneratedImage: 生成后的图像信息。` | 文档字符串内容 |
| `image_storage.py` | 327 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 328 | `+        Raises:` | 文档字符串内容 |
| `image_storage.py` | 329 | `+            AssertionError: 当提示为空时抛出。` | 文档字符串内容 |
| `image_storage.py` | 330 | `+            RuntimeError: 当 OpenAI 接口未返回图像或缺少必要字段时抛出。` | 文档字符串内容 |
| `image_storage.py` | 331 | `+        """` | 文档字符串内容 |
| `image_storage.py` | 332 | `+        from openai import OpenAI` | 按需导入对象 |
| `image_storage.py` | 333 | `+` | 空行分隔，增强可读性 |
| `image_storage.py` | 334 | `+        assert prompt.strip(), "prompt 不能为空"` | 断言前置条件 |
| `image_storage.py` | 335 | `+        client = OpenAI()` | 调用 OpenAI 图像 API |
| `image_storage.py` | 336 | `+        response = client.images.generate(` | 赋值/构建数据 |
| `image_storage.py` | 337 | `+            model=self._image_model,` | 赋值/构建数据 |
| `image_storage.py` | 338 | `+            prompt=prompt.strip(),` | 赋值/构建数据 |
| `image_storage.py` | 339 | `+            size=size,` | 赋值/构建数据 |
| `image_storage.py` | 340 | `+            response_format="b64_json",` | Base64 处理 |
| `image_storage.py` | 341 | `+        )` | 逻辑实现 |
| `image_storage.py` | 342 | `+        if not response.data:` | 条件判断 |
| `image_storage.py` | 343 | `+            raise RuntimeError("未收到图像生成结果")` | 抛出异常 |
| `image_storage.py` | 344 | `+        item = response.data[0]` | 集合结构定义 |
| `image_storage.py` | 345 | `+        b64_data = getattr(item, "b64_json", None)` | Base64 处理 |
| `image_storage.py` | 346 | `+        mime_type = getattr(item, "mime_type", None) or "image/png"` | 赋值/构建数据 |
| `image_storage.py` | 347 | `+        if not b64_data:` | 条件判断 |
| `image_storage.py` | 348 | `+            raise RuntimeError("生成结果缺少 Base64 数据")` | 抛出异常 |
| `image_storage.py` | 349 | `+        return self.save_generated_image(b64_data, prompt.strip(), mime_type)` | 返回处理结果 |
| `qq_group_bot.py` | 1 | `--- a/qq_group_bot.py` | Diff 标记：旧文件路径 |
| `qq_group_bot.py` | 2 | `+++ b/qq_group_bot.py` | Diff 标记：新文件路径 |
| `qq_group_bot.py` | 3 | `+import base64` | 导入标准库/三方模块 |
| `qq_group_bot.py` | 4 | `+import re` | 导入标准库/三方模块 |
| `qq_group_bot.py` | 5 | `-from typing import Optional` | 按需导入对象 |
| `qq_group_bot.py` | 6 | `+from typing import Optional, Sequence, Union` | 按需导入对象 |
| `qq_group_bot.py` | 7 | `+from image_storage import GeneratedImage, ImageStorageManager, StoredImage` | 按需导入对象 |
| `qq_group_bot.py` | 8 | `+MessagePayload = Union[str, Sequence[dict[str, dict[str, str]]]]` | 集合结构定义 |
| `qq_group_bot.py` | 9 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 10 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 11 | `-    api_base: str, group_id: int, text: str, access_token: str = ""` | 赋值/构建数据 |
| `qq_group_bot.py` | 12 | `+    api_base: str, group_id: int, message: MessagePayload, access_token: str = ""` | 赋值/构建数据 |
| `qq_group_bot.py` | 13 | `-        text (str): 文本内容。` | 逻辑实现 |
| `qq_group_bot.py` | 14 | `+        message (MessagePayload): 文本或消息段列表。` | 逻辑实现 |
| `qq_group_bot.py` | 15 | `-    payload = {"group_id": group_id, "message": text}` | 赋值/构建数据 |
| `qq_group_bot.py` | 16 | `+    payload = {"group_id": group_id, "message": message}` | 赋值/构建数据 |
| `qq_group_bot.py` | 17 | `-def _parse_message_and_at(event: dict) -> tuple[str, bool]:` | 定义函数 _parse_message_and_at |
| `qq_group_bot.py` | 18 | `-    """解析 NapCat 群消息，返回纯文本与是否@机器人。` | 文档字符串内容 |
| `qq_group_bot.py` | 19 | `+@dataclass(frozen=True)` | 文档字符串内容 |
| `qq_group_bot.py` | 20 | `+class ImageSegmentInfo:` | 文档字符串内容 |
| `qq_group_bot.py` | 21 | `+    """消息段中的图片信息。"""` | 文档字符串内容 |
| `qq_group_bot.py` | 22 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 23 | `+    url: Optional[str]` | 文档字符串内容 |
| `qq_group_bot.py` | 24 | `+    file_id: Optional[str]` | 文档字符串内容 |
| `qq_group_bot.py` | 25 | `+    filename: Optional[str]` | 文档字符串内容 |
| `qq_group_bot.py` | 26 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 27 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 28 | `+@dataclass(frozen=True)` | 文档字符串内容 |
| `qq_group_bot.py` | 29 | `+class ParsedMessage:` | 文档字符串内容 |
| `qq_group_bot.py` | 30 | `+    """标准化的消息解析结果。"""` | 文档字符串内容 |
| `qq_group_bot.py` | 31 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 32 | `+    text: str` | 文档字符串内容 |
| `qq_group_bot.py` | 33 | `+    at_me: bool` | 文档字符串内容 |
| `qq_group_bot.py` | 34 | `+    images: tuple[ImageSegmentInfo, ...]` | 文档字符串内容 |
| `qq_group_bot.py` | 35 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 36 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 37 | `+def _extract_cq_images(raw: str) -> tuple[ImageSegmentInfo, ...]:` | 文档字符串内容 |
| `qq_group_bot.py` | 38 | `+    """` | 文档字符串内容 |
| `qq_group_bot.py` | 39 | `+    从原始 CQ 文本中解析图像段。` | 逻辑实现 |
| `qq_group_bot.py` | 40 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 41 | `+    Args:` | 逻辑实现 |
| `qq_group_bot.py` | 42 | `+        raw (str): 原始消息字符串。` | 逻辑实现 |
| `qq_group_bot.py` | 43 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 44 | `+    Returns:` | 逻辑实现 |
| `qq_group_bot.py` | 45 | `+        tuple[ImageSegmentInfo, ...]: 解析出的图像段列表。` | 逻辑实现 |
| `qq_group_bot.py` | 46 | `+    """` | 文档字符串内容 |
| `qq_group_bot.py` | 47 | `+    images: list[ImageSegmentInfo] = []` | 文档字符串内容 |
| `qq_group_bot.py` | 48 | `+    idx = 0` | 文档字符串内容 |
| `qq_group_bot.py` | 49 | `+    length = len(raw)` | 文档字符串内容 |
| `qq_group_bot.py` | 50 | `+    while idx < length:` | 文档字符串内容 |
| `qq_group_bot.py` | 51 | `+        start = raw.find("[CQ:image", idx)` | 文档字符串内容 |
| `qq_group_bot.py` | 52 | `+        if start == -1:` | 文档字符串内容 |
| `qq_group_bot.py` | 53 | `+            break` | 文档字符串内容 |
| `qq_group_bot.py` | 54 | `+        end = raw.find("]", start)` | 文档字符串内容 |
| `qq_group_bot.py` | 55 | `+        if end == -1:` | 文档字符串内容 |
| `qq_group_bot.py` | 56 | `+            break` | 文档字符串内容 |
| `qq_group_bot.py` | 57 | `+        body = raw[start + 1 : end]` | 文档字符串内容 |
| `qq_group_bot.py` | 58 | `+        parts = body.split(",")` | 文档字符串内容 |
| `qq_group_bot.py` | 59 | `+        data: dict[str, str] = {}` | 文档字符串内容 |
| `qq_group_bot.py` | 60 | `+        for segment in parts[1:]:` | 文档字符串内容 |
| `qq_group_bot.py` | 61 | `+            if "=" not in segment:` | 文档字符串内容 |
| `qq_group_bot.py` | 62 | `+                continue` | 文档字符串内容 |
| `qq_group_bot.py` | 63 | `+            key, value = segment.split("=", 1)` | 文档字符串内容 |
| `qq_group_bot.py` | 64 | `+            data[key.strip()] = value.strip()` | 文档字符串内容 |
| `qq_group_bot.py` | 65 | `+        images.append(` | 文档字符串内容 |
| `qq_group_bot.py` | 66 | `+            ImageSegmentInfo(` | 文档字符串内容 |
| `qq_group_bot.py` | 67 | `+                url=data.get("url"),` | 文档字符串内容 |
| `qq_group_bot.py` | 68 | `+                file_id=data.get("file") or data.get("file_id"),` | 文档字符串内容 |
| `qq_group_bot.py` | 69 | `+                filename=data.get("file") or data.get("name"),` | 文档字符串内容 |
| `qq_group_bot.py` | 70 | `+            )` | 文档字符串内容 |
| `qq_group_bot.py` | 71 | `+        )` | 文档字符串内容 |
| `qq_group_bot.py` | 72 | `+        idx = end + 1` | 文档字符串内容 |
| `qq_group_bot.py` | 73 | `+    return tuple(images)` | 文档字符串内容 |
| `qq_group_bot.py` | 74 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 75 | `-    NapCat 的消息可为两种格式：` | 文档字符串内容 |
| `qq_group_bot.py` | 76 | `-    - String（CQ 码）；` | 文档字符串内容 |
| `qq_group_bot.py` | 77 | `-    - Array（段落列表，如 {"type":"text"\|"at"\|...}）。` | 文档字符串内容 |
| `qq_group_bot.py` | 78 | `+def _parse_message_and_at(event: dict) -> ParsedMessage:` | 文档字符串内容 |
| `qq_group_bot.py` | 79 | `+    """` | 文档字符串内容 |
| `qq_group_bot.py` | 80 | `+    解析 NapCat 群消息，返回文本、@ 状态与图像段。` | 逻辑实现 |
| `qq_group_bot.py` | 81 | `-        tuple[str, bool]: (纯文本, 是否@机器人)。` | 逻辑实现 |
| `qq_group_bot.py` | 82 | `+        ParsedMessage: 标准化后的消息内容。` | 逻辑实现 |
| `qq_group_bot.py` | 83 | `+        images: list[ImageSegmentInfo] = []` | 集合结构定义 |
| `qq_group_bot.py` | 84 | `-        return ("".join(texts).strip(), at_me)` | 返回处理结果 |
| `qq_group_bot.py` | 85 | `+            elif typ == "image":` | 条件分支 |
| `qq_group_bot.py` | 86 | `+                url = data.get("url")` | 赋值/构建数据 |
| `qq_group_bot.py` | 87 | `+                file_id = data.get("file") or data.get("file_id")` | 赋值/构建数据 |
| `qq_group_bot.py` | 88 | `+                filename = data.get("name") or data.get("file")` | 赋值/构建数据 |
| `qq_group_bot.py` | 89 | `+                images.append(` | 向列表追加 |
| `qq_group_bot.py` | 90 | `+                    ImageSegmentInfo(` | 逻辑实现 |
| `qq_group_bot.py` | 91 | `+                        url=str(url) if url else None,` | 赋值/构建数据 |
| `qq_group_bot.py` | 92 | `+                        file_id=str(file_id) if file_id else None,` | 赋值/构建数据 |
| `qq_group_bot.py` | 93 | `+                        filename=str(filename) if filename else None,` | 赋值/构建数据 |
| `qq_group_bot.py` | 94 | `+                    )` | 逻辑实现 |
| `qq_group_bot.py` | 95 | `+                )` | 逻辑实现 |
| `qq_group_bot.py` | 96 | `+        return ParsedMessage("".join(texts).strip(), at_me, tuple(images))` | 返回处理结果 |
| `qq_group_bot.py` | 97 | `-    return (raw, at_me)` | 返回处理结果 |
| `qq_group_bot.py` | 98 | `+    images = _extract_cq_images(raw) if raw else ()` | 赋值/构建数据 |
| `qq_group_bot.py` | 99 | `+    return ParsedMessage(raw, at_me, images)` | 返回处理结果 |
| `qq_group_bot.py` | 100 | `+    image_storage: Optional[ImageStorageManager] = None` | 使用图片管理器 |
| `qq_group_bot.py` | 101 | `+    @classmethod` | 装饰器，修饰定义 |
| `qq_group_bot.py` | 102 | `+    def _require_image_storage(cls) -> ImageStorageManager:` | 定义函数 _require_image_storage |
| `qq_group_bot.py` | 103 | `+        """` | 文档字符串内容 |
| `qq_group_bot.py` | 104 | `+        获取已初始化的图像存储管理器。` | 文档字符串内容 |
| `qq_group_bot.py` | 105 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 106 | `+        Returns:` | 文档字符串内容 |
| `qq_group_bot.py` | 107 | `+            ImageStorageManager: 全局共享的图像存储管理器。` | 文档字符串内容 |
| `qq_group_bot.py` | 108 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 109 | `+        Raises:` | 文档字符串内容 |
| `qq_group_bot.py` | 110 | `+            AssertionError: 当未注入图像存储实例时抛出。` | 文档字符串内容 |
| `qq_group_bot.py` | 111 | `+        """` | 文档字符串内容 |
| `qq_group_bot.py` | 112 | `+        if not isinstance(cls.image_storage, ImageStorageManager):` | 条件判断 |
| `qq_group_bot.py` | 113 | `+            raise AssertionError("图像存储管理器尚未配置")` | 抛出异常 |
| `qq_group_bot.py` | 114 | `+        return cls.image_storage` | 返回处理结果 |
| `qq_group_bot.py` | 115 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 116 | `+    @staticmethod` | 装饰器，修饰定义 |
| `qq_group_bot.py` | 117 | `+    def _build_multimodal_content(` | 定义函数 _build_multimodal_content |
| `qq_group_bot.py` | 118 | `+        model_input: str, images: Sequence[StoredImage]` | 处理图片结构 |
| `qq_group_bot.py` | 119 | `+    ) -> list[dict[str, object]]:` | 逻辑实现 |
| `qq_group_bot.py` | 120 | `+        """` | 文档字符串内容 |
| `qq_group_bot.py` | 121 | `+        构造多模态消息内容列表。` | 文档字符串内容 |
| `qq_group_bot.py` | 122 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 123 | `+        Args:` | 文档字符串内容 |
| `qq_group_bot.py` | 124 | `+            model_input (str): 拼接后的文本输入。` | 文档字符串内容 |
| `qq_group_bot.py` | 125 | `+            images (Sequence[StoredImage]): 已保存的图像集合。` | 文档字符串内容 |
| `qq_group_bot.py` | 126 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 127 | `+        Returns:` | 文档字符串内容 |
| `qq_group_bot.py` | 128 | `+            list[dict[str, object]]: 可直接传递给多模态模型的内容结构。` | 文档字符串内容 |
| `qq_group_bot.py` | 129 | `+        """` | 文档字符串内容 |
| `qq_group_bot.py` | 130 | `+        content: list[dict[str, object]] = [{"type": "text", "text": model_input}]` | 集合结构定义 |
| `qq_group_bot.py` | 131 | `+        if images:` | 条件判断 |
| `qq_group_bot.py` | 132 | `+            content.append(` | 向列表追加 |
| `qq_group_bot.py` | 133 | `+                {` | 逻辑实现 |
| `qq_group_bot.py` | 134 | `+                    "type": "text",` | 字符串字面量 |
| `qq_group_bot.py` | 135 | `+                    "text": f"用户同时附带了 {len(images)} 张图片，请结合视觉分析并不要回传原始图片。",` | 字符串字面量 |
| `qq_group_bot.py` | 136 | `+                }` | 逻辑实现 |
| `qq_group_bot.py` | 137 | `+            )` | 逻辑实现 |
| `qq_group_bot.py` | 138 | `+        for idx, stored in enumerate(images, 1):` | 遍历集合 |
| `qq_group_bot.py` | 139 | `+            content.append(` | 向列表追加 |
| `qq_group_bot.py` | 140 | `+                {` | 逻辑实现 |
| `qq_group_bot.py` | 141 | `+                    "type": "text",` | 字符串字面量 |
| `qq_group_bot.py` | 142 | `+                    "text": f"第 {idx} 张图像已经以内嵌 data URL 形式提供。",` | 字符串字面量 |
| `qq_group_bot.py` | 143 | `+                }` | 逻辑实现 |
| `qq_group_bot.py` | 144 | `+            )` | 逻辑实现 |
| `qq_group_bot.py` | 145 | `+            content.append(` | 向列表追加 |
| `qq_group_bot.py` | 146 | `+                {` | 逻辑实现 |
| `qq_group_bot.py` | 147 | `+                    "type": "image_url",` | 字符串字面量 |
| `qq_group_bot.py` | 148 | `+                    "image_url": {"url": stored.data_url()},` | 字符串字面量 |
| `qq_group_bot.py` | 149 | `+                }` | 逻辑实现 |
| `qq_group_bot.py` | 150 | `+            )` | 逻辑实现 |
| `qq_group_bot.py` | 151 | `+        return content` | 返回处理结果 |
| `qq_group_bot.py` | 152 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 153 | `+    @staticmethod` | 装饰器，修饰定义 |
| `qq_group_bot.py` | 154 | `+    def _compose_group_message(` | 定义函数 _compose_group_message |
| `qq_group_bot.py` | 155 | `+        answer: str, image_payloads: Sequence[tuple[str, str]]` | 集合结构定义 |
| `qq_group_bot.py` | 156 | `+    ) -> str:` | 逻辑实现 |
| `qq_group_bot.py` | 157 | `+        """组合文本与图片 CQ 码，图片使用 base64 内联。"""` | 文档字符串内容 |
| `qq_group_bot.py` | 158 | `+        parts: list[str] = []` | 集合结构定义 |
| `qq_group_bot.py` | 159 | `+        text = answer.strip()` | 赋值/构建数据 |
| `qq_group_bot.py` | 160 | `+        if text:` | 条件判断 |
| `qq_group_bot.py` | 161 | `+            parts.append(text)` | 向列表追加 |
| `qq_group_bot.py` | 162 | `+        ts = int(time.time())` | 赋值/构建数据 |
| `qq_group_bot.py` | 163 | `+        for idx, (b64, mime) in enumerate(image_payloads, 1):` | 遍历集合 |
| `qq_group_bot.py` | 164 | `+            suffix = {` | 赋值/构建数据 |
| `qq_group_bot.py` | 165 | `+                "image/png": "png",` | 字符串字面量 |
| `qq_group_bot.py` | 166 | `+                "image/jpeg": "jpg",` | 字符串字面量 |
| `qq_group_bot.py` | 167 | `+                "image/gif": "gif",` | 字符串字面量 |
| `qq_group_bot.py` | 168 | `+                "image/webp": "webp",` | 字符串字面量 |
| `qq_group_bot.py` | 169 | `+            }.get(mime, "jpg")` | 逻辑实现 |
| `qq_group_bot.py` | 170 | `+            name = f"img_{ts}_{idx}.{suffix}"` | 赋值/构建数据 |
| `qq_group_bot.py` | 171 | `+            parts.append(` | 向列表追加 |
| `qq_group_bot.py` | 172 | `+                f"[CQ:image,file=base64://{b64},name={name},cache=0]"` | Base64 处理 |
| `qq_group_bot.py` | 173 | `+            )` | 逻辑实现 |
| `qq_group_bot.py` | 174 | `+        if not parts:` | 条件判断 |
| `qq_group_bot.py` | 175 | `+            return "（未生成回复）"` | 返回处理结果 |
| `qq_group_bot.py` | 176 | `+        return "\n".join(parts)` | 返回处理结果 |
| `qq_group_bot.py` | 177 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 178 | `-        text, at_me = _parse_message_and_at(event)` | 赋值/构建数据 |
| `qq_group_bot.py` | 179 | `+        parsed = _parse_message_and_at(event)` | 赋值/构建数据 |
| `qq_group_bot.py` | 180 | `-        if not text:` | 条件判断 |
| `qq_group_bot.py` | 181 | `+        if not parsed.text and not parsed.images:` | 条件判断 |
| `qq_group_bot.py` | 182 | `-        if not at_me:` | 条件判断 |
| `qq_group_bot.py` | 183 | `+        if not parsed.at_me:` | 条件判断 |
| `qq_group_bot.py` | 184 | `-        t = text.strip()` | 赋值/构建数据 |
| `qq_group_bot.py` | 185 | `+        t = parsed.text.strip()` | 赋值/构建数据 |
| `qq_group_bot.py` | 186 | `-                f"[Chat] Request get: Group {group_id} Id {user_id} User {author}: {text}"` | 逻辑实现 |
| `qq_group_bot.py` | 187 | `+                f"[Chat] Request get: Group {group_id} Id {user_id} User {author}: {parsed.text}"` | 逻辑实现 |
| `qq_group_bot.py` | 188 | `-            model_input = f"Group_id: [{group_id}]; User_id: [{user_id}]; User_name: {author}; Text: {text}"` | 赋值/构建数据 |
| `qq_group_bot.py` | 189 | `-            # 模拟随机延迟（1-4秒）` | 注释，补充说明 |
| `qq_group_bot.py` | 190 | `-            #time.sleep(1 + (os.urandom(1)[0] % 4))` | 注释，补充说明 |
| `qq_group_bot.py` | 191 | `-            # 发送请求并等待最终结果` | 注释，补充说明 |
| `qq_group_bot.py` | 192 | `+            user_text = parsed.text if parsed.text else "（用户未提供文本，仅包含图片）"` | 赋值/构建数据 |
| `qq_group_bot.py` | 193 | `+            model_input = (` | 赋值/构建数据 |
| `qq_group_bot.py` | 194 | `+                f"Group_id: [{group_id}]; User_id: [{user_id}]; User_name: {author}; Text: {user_text}"` | 逻辑实现 |
| `qq_group_bot.py` | 195 | `+            )` | 逻辑实现 |
| `qq_group_bot.py` | 196 | `+            stored_images: list[StoredImage] = []` | 处理图片结构 |
| `qq_group_bot.py` | 197 | `+            if parsed.images:` | 条件判断 |
| `qq_group_bot.py` | 198 | `+                storage = self._require_image_storage()` | 赋值/构建数据 |
| `qq_group_bot.py` | 199 | `+                for seg in parsed.images:` | 遍历集合 |
| `qq_group_bot.py` | 200 | `+                    assert (` | 断言前置条件 |
| `qq_group_bot.py` | 201 | `+                        seg.url` | 逻辑实现 |
| `qq_group_bot.py` | 202 | `+                    ), "当前仅支持通过 URL 获取的图片消息"` | 逻辑实现 |
| `qq_group_bot.py` | 203 | `+                    stored_images.append(` | 向列表追加 |
| `qq_group_bot.py` | 204 | `+                        storage.save_remote_image(seg.url, seg.filename)` | 逻辑实现 |
| `qq_group_bot.py` | 205 | `+                    )` | 逻辑实现 |
| `qq_group_bot.py` | 206 | `+            payload = (` | 赋值/构建数据 |
| `qq_group_bot.py` | 207 | `+                self._build_multimodal_content(model_input, stored_images)` | 实例属性操作 |
| `qq_group_bot.py` | 208 | `+                if stored_images` | 条件判断 |
| `qq_group_bot.py` | 209 | `+                else model_input` | 兜底分支 |
| `qq_group_bot.py` | 210 | `+            )` | 逻辑实现 |
| `qq_group_bot.py` | 211 | `-                model_input, thread_id=self._thread_id_for(group_id)` | 赋值/构建数据 |
| `qq_group_bot.py` | 212 | `+                payload, thread_id=self._thread_id_for(group_id)` | 赋值/构建数据 |
| `qq_group_bot.py` | 213 | `+            generated_images = self.agent.consume_generated_images()` | 赋值/构建数据 |
| `qq_group_bot.py` | 214 | `+            generated_images = []` | 集合结构定义 |
| `qq_group_bot.py` | 215 | `+            generated_images = []` | 集合结构定义 |
| `qq_group_bot.py` | 216 | `+            generated_images = []` | 集合结构定义 |
| `qq_group_bot.py` | 217 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 218 | `+        image_payloads: list[tuple[str, str]] = []` | 集合结构定义 |
| `qq_group_bot.py` | 219 | `+        for img in generated_images:` | 遍历集合 |
| `qq_group_bot.py` | 220 | `+            try:` | 异常捕获块 |
| `qq_group_bot.py` | 221 | `+                data_b64 = base64.b64encode(img.path.read_bytes()).decode("ascii")` | Base64 处理 |
| `qq_group_bot.py` | 222 | `+                image_payloads.append((data_b64, img.mime_type))` | Base64 处理 |
| `qq_group_bot.py` | 223 | `+            except Exception as err:` | 异常处理 |
| `qq_group_bot.py` | 224 | `+                sys.stderr.write(f"[Chat] 读取生成图片失败: {img.path} -> {err}\n")` | 逻辑实现 |
| `qq_group_bot.py` | 225 | `+        # 解析 Agent 回复中的 [IMAGE]url[/IMAGE] 标签，转换为本地图片` | 注释，补充说明 |
| `qq_group_bot.py` | 226 | `+        image_tags = re.findall(r"\[IMAGE\](.+?)\[/IMAGE\]", answer, flags=re.IGNORECASE)` | 正则解析 |
| `qq_group_bot.py` | 227 | `+        if image_tags:` | 条件判断 |
| `qq_group_bot.py` | 228 | `+            manager = self._require_image_storage()` | 赋值/构建数据 |
| `qq_group_bot.py` | 229 | `+            failed_urls: list[str] = []` | 集合结构定义 |
| `qq_group_bot.py` | 230 | `+            downloaded = False` | 赋值/构建数据 |
| `qq_group_bot.py` | 231 | `+            for url in image_tags:` | 遍历集合 |
| `qq_group_bot.py` | 232 | `+                url_norm = url.strip()` | 赋值/构建数据 |
| `qq_group_bot.py` | 233 | `+                if not url_norm:` | 条件判断 |
| `qq_group_bot.py` | 234 | `+                    continue` | 逻辑实现 |
| `qq_group_bot.py` | 235 | `+                try:` | 异常捕获块 |
| `qq_group_bot.py` | 236 | `+                    saved = manager.save_remote_image(url_norm)` | 赋值/构建数据 |
| `qq_group_bot.py` | 237 | `+                    image_payloads.append((saved.base64_data, saved.mime_type))` | Base64 处理 |
| `qq_group_bot.py` | 238 | `+                    downloaded = True` | 赋值/构建数据 |
| `qq_group_bot.py` | 239 | `+                except Exception as err:` | 异常处理 |
| `qq_group_bot.py` | 240 | `+                    failed_urls.append(url_norm)` | 向列表追加 |
| `qq_group_bot.py` | 241 | `+                    sys.stderr.write(f"[Chat] 下载回复图片失败: {url_norm} -> {err}\n")` | 逻辑实现 |
| `qq_group_bot.py` | 242 | `+            cleaned = re.sub(r"\[IMAGE\].+?\[/IMAGE\]", "", answer, flags=re.IGNORECASE).strip()` | 正则解析 |
| `qq_group_bot.py` | 243 | `+            if failed_urls and downloaded:` | 条件判断 |
| `qq_group_bot.py` | 244 | `+                note = "（部分图片下载失败，已忽略无法访问的链接）"` | 赋值/构建数据 |
| `qq_group_bot.py` | 245 | `+                answer = f"{cleaned}\n{note}" if cleaned else note` | 赋值/构建数据 |
| `qq_group_bot.py` | 246 | `+            elif failed_urls and not downloaded and not image_payloads:` | 条件分支 |
| `qq_group_bot.py` | 247 | `+                answer = cleaned or "（未能下载图片，请稍后重试）"` | 赋值/构建数据 |
| `qq_group_bot.py` | 248 | `+            else:` | 兜底分支 |
| `qq_group_bot.py` | 249 | `+                answer = cleaned or ("（图片已发送）" if downloaded else cleaned)` | 赋值/构建数据 |
| `qq_group_bot.py` | 250 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 251 | `+        # 解析 Agent 回复中的 CQ 图片段并下载本地` | 注释，补充说明 |
| `qq_group_bot.py` | 252 | `+        cq_pattern = re.compile(r"\[CQ:image,([^\]]+)\]")` | 正则解析 |
| `qq_group_bot.py` | 253 | `+        cq_matches = list(cq_pattern.finditer(answer))` | 赋值/构建数据 |
| `qq_group_bot.py` | 254 | `+        if cq_matches:` | 条件判断 |
| `qq_group_bot.py` | 255 | `+            manager = self._require_image_storage()` | 赋值/构建数据 |
| `qq_group_bot.py` | 256 | `+            failed_urls: list[str] = []` | 集合结构定义 |
| `qq_group_bot.py` | 257 | `+            success = False` | 赋值/构建数据 |
| `qq_group_bot.py` | 258 | `+            for match in cq_matches:` | 遍历集合 |
| `qq_group_bot.py` | 259 | `+                data_str = match.group(1)` | 赋值/构建数据 |
| `qq_group_bot.py` | 260 | `+                params = {}` | 赋值/构建数据 |
| `qq_group_bot.py` | 261 | `+                for part in data_str.split(","):` | 遍历集合 |
| `qq_group_bot.py` | 262 | `+                    if "=" in part:` | 条件判断 |
| `qq_group_bot.py` | 263 | `+                        k, v = part.split("=", 1)` | 字符串拆分 |
| `qq_group_bot.py` | 264 | `+                        params[k.strip()] = v.strip()` | 赋值/构建数据 |
| `qq_group_bot.py` | 265 | `+                file_val = params.get("file") or ""` | 赋值/构建数据 |
| `qq_group_bot.py` | 266 | `+                if not file_val:` | 条件判断 |
| `qq_group_bot.py` | 267 | `+                    continue` | 逻辑实现 |
| `qq_group_bot.py` | 268 | `+                if file_val.startswith("base64://"):` | 条件判断 |
| `qq_group_bot.py` | 269 | `+                    raw_b64 = file_val[len("base64://") :]` | Base64 处理 |
| `qq_group_bot.py` | 270 | `+                    try:` | 异常捕获块 |
| `qq_group_bot.py` | 271 | `+                        stored = manager.save_base64_image(raw_b64)` | Base64 处理 |
| `qq_group_bot.py` | 272 | `+                        image_payloads.append((stored.base64_data, stored.mime_type))` | Base64 处理 |
| `qq_group_bot.py` | 273 | `+                        success = True` | 赋值/构建数据 |
| `qq_group_bot.py` | 274 | `+                    except Exception as err:` | 异常处理 |
| `qq_group_bot.py` | 275 | `+                        failed_urls.append("base64-data")` | Base64 处理 |
| `qq_group_bot.py` | 276 | `+                        sys.stderr.write(` | 逻辑实现 |
| `qq_group_bot.py` | 277 | `+                            f"[Chat] 保存CQ Base64图片失败: {err}\n"` | 逻辑实现 |
| `qq_group_bot.py` | 278 | `+                        )` | 逻辑实现 |
| `qq_group_bot.py` | 279 | `+                    continue` | 逻辑实现 |
| `qq_group_bot.py` | 280 | `+                if file_val.startswith("http"):` | 条件判断 |
| `qq_group_bot.py` | 281 | `+                    try:` | 异常捕获块 |
| `qq_group_bot.py` | 282 | `+                        saved = manager.save_remote_image(file_val)` | 赋值/构建数据 |
| `qq_group_bot.py` | 283 | `+                        image_payloads.append((saved.base64_data, saved.mime_type))` | Base64 处理 |
| `qq_group_bot.py` | 284 | `+                        success = True` | 赋值/构建数据 |
| `qq_group_bot.py` | 285 | `+                    except Exception as err:` | 异常处理 |
| `qq_group_bot.py` | 286 | `+                        failed_urls.append(file_val)` | 向列表追加 |
| `qq_group_bot.py` | 287 | `+                        sys.stderr.write(` | 逻辑实现 |
| `qq_group_bot.py` | 288 | `+                            f"[Chat] 下载CQ图片失败: {file_val} -> {err}\n"` | 逻辑实现 |
| `qq_group_bot.py` | 289 | `+                        )` | 逻辑实现 |
| `qq_group_bot.py` | 290 | `+            if cq_matches:` | 条件判断 |
| `qq_group_bot.py` | 291 | `+                answer = cq_pattern.sub("", answer).strip()` | 赋值/构建数据 |
| `qq_group_bot.py` | 292 | `+            if failed_urls and success:` | 条件判断 |
| `qq_group_bot.py` | 293 | `+                note = "（部分图片下载失败，已忽略无法访问的 CQ 图片链接）"` | 赋值/构建数据 |
| `qq_group_bot.py` | 294 | `+                answer = f"{answer}\n{note}" if answer else note` | 赋值/构建数据 |
| `qq_group_bot.py` | 295 | `+            elif failed_urls and not success and not image_payloads:` | 条件分支 |
| `qq_group_bot.py` | 296 | `+                answer = answer or "（未能下载图片，请稍后重试）"` | 赋值/构建数据 |
| `qq_group_bot.py` | 297 | `+` | 空行分隔，增强可读性 |
| `qq_group_bot.py` | 298 | `+        if answer:` | 条件判断 |
| `qq_group_bot.py` | 299 | `+            lines = [line for line in answer.splitlines() if line.strip()]` | 字符串拆分 |
| `qq_group_bot.py` | 300 | `+            answer = "\n".join(lines)` | 字符串拼接 |
| `qq_group_bot.py` | 301 | `+            message_body = self._compose_group_message(answer, image_payloads)` | 赋值/构建数据 |
| `qq_group_bot.py` | 302 | `-                self.bot_cfg.api_base, group_id, answer, self.bot_cfg.access_token` | 实例属性操作 |
| `qq_group_bot.py` | 303 | `+                self.bot_cfg.api_base,` | 实例属性操作 |
| `qq_group_bot.py` | 304 | `+                group_id,` | 逻辑实现 |
| `qq_group_bot.py` | 305 | `+                message_body,` | 逻辑实现 |
| `qq_group_bot.py` | 306 | `+                self.bot_cfg.access_token,` | 实例属性操作 |
| `qq_group_bot.py` | 307 | `+    image_dir = os.environ.get(` | 赋值/构建数据 |
| `qq_group_bot.py` | 308 | `+        "QQ_IMAGE_DIR",` | 字符串字面量 |
| `qq_group_bot.py` | 309 | `+        os.path.join(os.getcwd(), "local_backup", "qq_images"),` | 字符串拼接 |
| `qq_group_bot.py` | 310 | `+    )` | 逻辑实现 |
| `qq_group_bot.py` | 311 | `+    image_manager = ImageStorageManager(image_dir)` | 使用图片管理器 |
| `qq_group_bot.py` | 312 | `+    agent.set_image_manager(image_manager)` | 逻辑实现 |
| `qq_group_bot.py` | 313 | `+    QQBotHandler.image_storage = image_manager` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 1 | `--- a/sql_agent_cli_stream_plus.py` | Diff 标记：旧文件路径 |
| `sql_agent_cli_stream_plus.py` | 2 | `+++ b/sql_agent_cli_stream_plus.py` | Diff 标记：新文件路径 |
| `sql_agent_cli_stream_plus.py` | 3 | `-from typing import Annotated, Callable, Iterable, Optional` | 按需导入对象 |
| `sql_agent_cli_stream_plus.py` | 4 | `+from typing import Annotated, Callable, Iterable, Optional, Sequence, Union, Any` | 按需导入对象 |
| `sql_agent_cli_stream_plus.py` | 5 | `+from image_storage import GeneratedImage, ImageStorageManager` | 按需导入对象 |
| `sql_agent_cli_stream_plus.py` | 6 | `+        self._image_manager: Optional[ImageStorageManager] = None` | 实例属性操作 |
| `sql_agent_cli_stream_plus.py` | 7 | `+        self._generated_images: list[GeneratedImage] = []` | 实例属性操作 |
| `sql_agent_cli_stream_plus.py` | 8 | `+    def set_image_manager(self, manager: ImageStorageManager) -> None:` | 定义函数 set_image_manager |
| `sql_agent_cli_stream_plus.py` | 9 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 10 | `+        设置图像存储管理器，供多模态与生成工具使用。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 11 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 12 | `+        Args:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 13 | `+            manager (ImageStorageManager): 已初始化的图像管理器实例。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 14 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 15 | `+        Raises:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 16 | `+            AssertionError: 当传入对象类型不匹配时抛出。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 17 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 18 | `+        assert isinstance(manager, ImageStorageManager), "manager 类型无效"` | 断言前置条件 |
| `sql_agent_cli_stream_plus.py` | 19 | `+        self._image_manager = manager` | 实例属性操作 |
| `sql_agent_cli_stream_plus.py` | 20 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 21 | `+    def consume_generated_images(self) -> list[GeneratedImage]:` | 定义函数 consume_generated_images |
| `sql_agent_cli_stream_plus.py` | 22 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 23 | `+        读取并清空最近一次会话生成的图像列表。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 24 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 25 | `+        Returns:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 26 | `+            list[GeneratedImage]: 生成图像集合，按生成顺序排列。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 27 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 28 | `+        images = list(self._generated_images)` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 29 | `+        self._generated_images = []` | 实例属性操作 |
| `sql_agent_cli_stream_plus.py` | 30 | `+        return images` | 返回处理结果 |
| `sql_agent_cli_stream_plus.py` | 31 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 32 | `+    def _require_image_manager(self) -> ImageStorageManager:` | 定义函数 _require_image_manager |
| `sql_agent_cli_stream_plus.py` | 33 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 34 | `+        获取已配置的图像管理器。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 35 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 36 | `+        Returns:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 37 | `+            ImageStorageManager: 图像存储管理器实例。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 38 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 39 | `+        Raises:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 40 | `+            AssertionError: 当图像管理器尚未设置时抛出。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 41 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 42 | `+        if not isinstance(self._image_manager, ImageStorageManager):` | 条件判断 |
| `sql_agent_cli_stream_plus.py` | 43 | `+            raise AssertionError("图像管理器尚未配置")` | 抛出异常 |
| `sql_agent_cli_stream_plus.py` | 44 | `+        return self._image_manager` | 返回处理结果 |
| `sql_agent_cli_stream_plus.py` | 45 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 46 | `-            # 两种绑定：` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 47 | `-            # - auto：允许模型自行决定是否调用工具（用于首轮/工具前）` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 48 | `-            # - none：禁止工具，促使模型基于工具结果做总结（用于工具后）` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 49 | `-            llm_tools_auto = llm.bind_tools(tools) if tools else llm` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 50 | `-            llm_tools_none = llm.bind_tools(tools, tool_choice="none") if tools else llm` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 51 | `-            # - force：当用户显式要求搜索/检索等，强制调用 tavily_search` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 52 | `+        @tool` | 装饰器，修饰定义 |
| `sql_agent_cli_stream_plus.py` | 53 | `+        def generate_local_image(prompt: str, size: str = "1024x1024") -> str:` | 定义函数 generate_local_image |
| `sql_agent_cli_stream_plus.py` | 54 | `-            if tools:` | 条件判断 |
| `sql_agent_cli_stream_plus.py` | 55 | `-                llm_tools_force = llm.bind_tools(` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 56 | `-                    tools,` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 57 | `-                    tool_choice={` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 58 | `-                        "type": "function",` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 59 | `-                        "function": {"name": "tavily_search"},` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 60 | `-                    },` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 61 | `-                )` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 62 | `-            else:` | 兜底分支 |
| `sql_agent_cli_stream_plus.py` | 63 | `-                llm_tools_force = llm` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 64 | `+            生成图像并返回本地文件路径，供下游发送真实图片。` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 65 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 66 | `+            Args:` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 67 | `+                prompt (str): 图像描述，须包含主体、场景与风格信息。` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 68 | `+                size (str): 输出尺寸，支持 256x256、512x512、1024x1024。` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 69 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 70 | `+            Returns:` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 71 | `+                str: JSON 字符串，包含 path、mime_type、prompt。` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 72 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 73 | `+            Raises:` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 74 | `+                AssertionError: 当提示为空、尺寸非法或缺少图像管理器时抛出。` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 75 | `+                RuntimeError: 当图像生成失败时抛出。` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 76 | `+            prompt_text = prompt.strip()` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 77 | `+            assert prompt_text, "prompt 不能为空"` | 断言前置条件 |
| `sql_agent_cli_stream_plus.py` | 78 | `+            size_norm = size.strip().lower()` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 79 | `+            allowed = {"256x256", "512x512", "1024x1024"}` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 80 | `+            assert size_norm in allowed, f"size 必须为 {allowed} 之一"` | 断言前置条件 |
| `sql_agent_cli_stream_plus.py` | 81 | `+            _ensure_openai_env_once()` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 82 | `+            manager = self._require_image_manager()` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 83 | `+            image = manager.generate_image_via_openai(prompt_text, size_norm)` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 84 | `+            self._generated_images.append(image)` | 实例属性操作 |
| `sql_agent_cli_stream_plus.py` | 85 | `+            payload = {` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 86 | `+                "path": str(image.path),` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 87 | `+                "mime_type": image.mime_type,` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 88 | `+                "prompt": prompt_text,` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 89 | `+            }` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 90 | `+            return json.dumps(payload, ensure_ascii=False)` | 返回处理结果 |
| `sql_agent_cli_stream_plus.py` | 91 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 92 | `+        tools.append(generate_local_image)` | 向列表追加 |
| `sql_agent_cli_stream_plus.py` | 93 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 94 | `+        # 两种绑定：` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 95 | `+        # - auto：允许模型自行决定是否调用工具（用于首轮/工具前）` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 96 | `+        # - none：禁止工具，促使模型基于工具结果做总结（用于工具后）` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 97 | `+        llm_tools_auto = llm.bind_tools(tools) if tools else llm` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 98 | `+        llm_tools_none = llm.bind_tools(tools, tool_choice="none") if tools else llm` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 99 | `+        # - force：当用户显式要求搜索/检索等，强制调用 tavily_search` | 注释，补充说明 |
| `sql_agent_cli_stream_plus.py` | 100 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 101 | `+        if tools:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 102 | `+            llm_tools_force = llm.bind_tools(` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 103 | `+                tools,` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 104 | `+                tool_choice={` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 105 | `+                    "type": "function",` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 106 | `+                    "function": {"name": "tavily_search"},` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 107 | `+                },` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 108 | `+            )` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 109 | `+        else:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 110 | `+            llm_tools_force = llm` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 111 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 112 | `-                style_msg = "如非特殊要求，禁止使用markdown语法。你输出的**不会加粗，禁止使用。你处在一个群聊之中，因此你的回复像人类一样使用口语化的连续文字，不会轻易使用列表分点。你的回复往往20-50字，最长不超过100字。但是基于搜索结果回答时，你可以突破字数限制适当增加字数，确保信息完整。你回答的长度应该像人类一样灵活，避免每次回复都是相同的长度。对于评价、偏好、选择，你和人类一样有鲜明的偏好。"` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 113 | `+                style_msg = "如非特殊要求，禁止使用markdown语法。你输出的**不会加粗，禁止使用。你处在一个群聊之中，因此你的回复像人类一样使用口语化的连续文字，不会轻易使用列表分点。你的回复往往20-50字，最长不超过100字。但是基于搜索结果回答时，你可以突破字数限制适当增加字数，确保信息完整。你回答的长度应该像人类一样灵活，避免每次回复都是相同的长度。对于评价、偏好、选择，你和人类一样有鲜明的偏好。图片链接必须以[IMAGE]url[/IMAGE]的格式输出，禁止使用其它格式。"` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 114 | `-    def chat_once_stream(self, user_input: str, thread_id: Optional[str] = None) -> str:` | 定义函数 chat_once_stream |
| `sql_agent_cli_stream_plus.py` | 115 | `+    def chat_once_stream(` | 定义函数 chat_once_stream |
| `sql_agent_cli_stream_plus.py` | 116 | `+        self,` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 117 | `+        user_input: Union[str, Sequence[dict[str, Any]], HumanMessage],` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 118 | `+        thread_id: Optional[str] = None,` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 119 | `+    ) -> str:` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 120 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 121 | `+        同步执行一次对话轮次，支持多模态输入并返回最终文本。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 122 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 123 | `+        Args:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 124 | `+            user_input (Union[str, Sequence[dict[str, Any]], HumanMessage]):` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 125 | `+                用户输入内容，可为纯文本、LangChain HumanMessage，或多模态内容列表。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 126 | `+            thread_id (Optional[str]): LangGraph 线程 ID，默认使用配置中的线程。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 127 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 128 | `+        Returns:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 129 | `+            str: 聚合后的最终文本回复。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 130 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 131 | `+        Raises:` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 132 | `+            AssertionError: 当输入类型不受支持时抛出。` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 133 | `+        """` | 文档字符串内容 |
| `sql_agent_cli_stream_plus.py` | 134 | `+        self._generated_images = []` | 实例属性操作 |
| `sql_agent_cli_stream_plus.py` | 135 | `+        if isinstance(user_input, HumanMessage):` | 条件判断 |
| `sql_agent_cli_stream_plus.py` | 136 | `+            payload = {"messages": [user_input]}` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 137 | `+        elif isinstance(user_input, str):` | 条件分支 |
| `sql_agent_cli_stream_plus.py` | 138 | `+            payload = {"messages": [{"role": "user", "content": user_input}]}` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 139 | `+        elif isinstance(user_input, Sequence):` | 条件分支 |
| `sql_agent_cli_stream_plus.py` | 140 | `+            payload = {` | 赋值/构建数据 |
| `sql_agent_cli_stream_plus.py` | 141 | `+                "messages": [` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 142 | `+                    {` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 143 | `+                        "role": "user",` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 144 | `+                        "content": list(user_input),` | 字符串字面量 |
| `sql_agent_cli_stream_plus.py` | 145 | `+                    }` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 146 | `+                ]` | 集合结构定义 |
| `sql_agent_cli_stream_plus.py` | 147 | `+            }` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 148 | `+        else:` | 兜底分支 |
| `sql_agent_cli_stream_plus.py` | 149 | `+            raise AssertionError("user_input 类型不受支持")` | 抛出异常 |
| `sql_agent_cli_stream_plus.py` | 150 | `+` | 空行分隔，增强可读性 |
| `sql_agent_cli_stream_plus.py` | 151 | `-                {"messages": [{"role": "user", "content": user_input}]},` | 逻辑实现 |
| `sql_agent_cli_stream_plus.py` | 152 | `+                payload,` | 逻辑实现 |
| `test_multimodal_unit.py` | 1 | `--- /dev/null` | Diff 标记：旧文件路径 |
| `test_multimodal_unit.py` | 2 | `+++ b/test_multimodal_unit.py` | Diff 标记：新文件路径 |
| `test_multimodal_unit.py` | 3 | `+import base64` | 导入标准库/三方模块 |
| `test_multimodal_unit.py` | 4 | `+import tempfile` | 导入标准库/三方模块 |
| `test_multimodal_unit.py` | 5 | `+import unittest` | 导入标准库/三方模块 |
| `test_multimodal_unit.py` | 6 | `+from pathlib import Path` | 按需导入对象 |
| `test_multimodal_unit.py` | 7 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 8 | `+from image_storage import ImageStorageManager, StoredImage` | 按需导入对象 |
| `test_multimodal_unit.py` | 9 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 10 | `+try:` | 异常捕获块 |
| `test_multimodal_unit.py` | 11 | `+    from qq_group_bot import QQBotHandler, _extract_cq_images, _parse_message_and_at` | 按需导入对象 |
| `test_multimodal_unit.py` | 12 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 13 | `+    _QQ_MODULE_AVAILABLE = True` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 14 | `+except ModuleNotFoundError:` | 异常处理 |
| `test_multimodal_unit.py` | 15 | `+    QQBotHandler = None` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 16 | `+    _extract_cq_images = None` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 17 | `+    _parse_message_and_at = None` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 18 | `+    _QQ_MODULE_AVAILABLE = False` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 19 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 20 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 21 | `+class MultimodalUnitTest(unittest.TestCase):` | 定义类 MultimodalUnitTest |
| `test_multimodal_unit.py` | 22 | `+    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")` | 装饰器，修饰定义 |
| `test_multimodal_unit.py` | 23 | `+    def test_extract_cq_images_parses_multiple_segments(self) -> None:` | 定义函数 test_extract_cq_images_parses_multiple_segments |
| `test_multimodal_unit.py` | 24 | `+        raw = (` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 25 | `+            "[CQ:at,qq=10086][CQ:image,file=foo.png,url=http://example.com/foo.png]"` | 字符串字面量 |
| `test_multimodal_unit.py` | 26 | `+            "some text"` | 字符串字面量 |
| `test_multimodal_unit.py` | 27 | `+            "[CQ:image,file=bar.jpg,url=https://example.com/bar.jpg]"` | 字符串字面量 |
| `test_multimodal_unit.py` | 28 | `+        )` | 逻辑实现 |
| `test_multimodal_unit.py` | 29 | `+        images = _extract_cq_images(raw)` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 30 | `+        self.assertEqual(len(images), 2)` | 实例属性操作 |
| `test_multimodal_unit.py` | 31 | `+        self.assertEqual(images[0].url, "http://example.com/foo.png")` | 实例属性操作 |
| `test_multimodal_unit.py` | 32 | `+        self.assertEqual(images[0].filename, "foo.png")` | 实例属性操作 |
| `test_multimodal_unit.py` | 33 | `+        self.assertEqual(images[1].url, "https://example.com/bar.jpg")` | 实例属性操作 |
| `test_multimodal_unit.py` | 34 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 35 | `+    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")` | 装饰器，修饰定义 |
| `test_multimodal_unit.py` | 36 | `+    def test_parse_message_and_at_handles_array_with_image(self) -> None:` | 定义函数 test_parse_message_and_at_handles_array_with_image |
| `test_multimodal_unit.py` | 37 | `+        event = {` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 38 | `+            "self_id": 20000,` | 字符串字面量 |
| `test_multimodal_unit.py` | 39 | `+            "message": [` | 字符串字面量 |
| `test_multimodal_unit.py` | 40 | `+                {"type": "at", "data": {"qq": "20000"}},` | 逻辑实现 |
| `test_multimodal_unit.py` | 41 | `+                {"type": "text", "data": {"text": "请看看"}},` | 逻辑实现 |
| `test_multimodal_unit.py` | 42 | `+                {` | 逻辑实现 |
| `test_multimodal_unit.py` | 43 | `+                    "type": "image",` | 字符串字面量 |
| `test_multimodal_unit.py` | 44 | `+                    "data": {` | 字符串字面量 |
| `test_multimodal_unit.py` | 45 | `+                        "url": "https://example.com/test.png",` | 字符串字面量 |
| `test_multimodal_unit.py` | 46 | `+                        "file": "test.png",` | 字符串字面量 |
| `test_multimodal_unit.py` | 47 | `+                        "name": "test.png",` | 字符串字面量 |
| `test_multimodal_unit.py` | 48 | `+                    },` | 逻辑实现 |
| `test_multimodal_unit.py` | 49 | `+                },` | 逻辑实现 |
| `test_multimodal_unit.py` | 50 | `+            ],` | 逻辑实现 |
| `test_multimodal_unit.py` | 51 | `+        }` | 逻辑实现 |
| `test_multimodal_unit.py` | 52 | `+        parsed = _parse_message_and_at(event)` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 53 | `+        self.assertTrue(parsed.at_me)` | 实例属性操作 |
| `test_multimodal_unit.py` | 54 | `+        self.assertEqual(parsed.text, "请看看")` | 实例属性操作 |
| `test_multimodal_unit.py` | 55 | `+        self.assertEqual(len(parsed.images), 1)` | 实例属性操作 |
| `test_multimodal_unit.py` | 56 | `+        self.assertEqual(parsed.images[0].url, "https://example.com/test.png")` | 实例属性操作 |
| `test_multimodal_unit.py` | 57 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 58 | `+    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")` | 装饰器，修饰定义 |
| `test_multimodal_unit.py` | 59 | `+    def test_multimodal_content_builder_includes_text_and_image(self) -> None:` | 定义函数 test_multimodal_content_builder_includes_text_and_image |
| `test_multimodal_unit.py` | 60 | `+        with tempfile.TemporaryDirectory() as tmp_dir:` | 上下文管理 |
| `test_multimodal_unit.py` | 61 | `+            stored = StoredImage(` | 处理图片结构 |
| `test_multimodal_unit.py` | 62 | `+                path=Path(tmp_dir) / "stored.png",` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 63 | `+                mime_type="image/png",` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 64 | `+                base64_data=base64.b64encode(b"dummy").decode("ascii"),` | Base64 处理 |
| `test_multimodal_unit.py` | 65 | `+            )` | 逻辑实现 |
| `test_multimodal_unit.py` | 66 | `+            content = QQBotHandler._build_multimodal_content("test", [stored])` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 67 | `+            self.assertEqual(content[0]["text"], "test")` | 实例属性操作 |
| `test_multimodal_unit.py` | 68 | `+            self.assertTrue(any(item.get("type") == "image_url" for item in content))` | 实例属性操作 |
| `test_multimodal_unit.py` | 69 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 70 | `+    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")` | 装饰器，修饰定义 |
| `test_multimodal_unit.py` | 71 | `+    def test_compose_group_message_appends_cq_codes(self) -> None:` | 定义函数 test_compose_group_message_appends_cq_codes |
| `test_multimodal_unit.py` | 72 | `+        message = QQBotHandler._compose_group_message(` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 73 | `+            "hello", [("ZGF0YQ==", "image/png")]` | 字符串字面量 |
| `test_multimodal_unit.py` | 74 | `+        )` | 逻辑实现 |
| `test_multimodal_unit.py` | 75 | `+        self.assertIn("hello", message)` | 实例属性操作 |
| `test_multimodal_unit.py` | 76 | `+        self.assertIn("[CQ:image,file=base64://", message)` | 实例属性操作 |
| `test_multimodal_unit.py` | 77 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 78 | `+    def test_save_generated_image(self) -> None:` | 定义函数 test_save_generated_image |
| `test_multimodal_unit.py` | 79 | `+        with tempfile.TemporaryDirectory() as tmp_dir:` | 上下文管理 |
| `test_multimodal_unit.py` | 80 | `+            manager = ImageStorageManager(tmp_dir)` | 使用图片管理器 |
| `test_multimodal_unit.py` | 81 | `+            png_base64 = (` | Base64 处理 |
| `test_multimodal_unit.py` | 82 | `+                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wwAAuMB9oK08QAAAABJRU5ErkJggg=="` | 字符串字面量 |
| `test_multimodal_unit.py` | 83 | `+            )` | 逻辑实现 |
| `test_multimodal_unit.py` | 84 | `+            result = manager.save_generated_image(png_base64, "prompt", "image/png")` | Base64 处理 |
| `test_multimodal_unit.py` | 85 | `+            self.assertTrue(result.path.exists())` | 实例属性操作 |
| `test_multimodal_unit.py` | 86 | `+            self.assertEqual(result.path.suffix, ".png")` | 实例属性操作 |
| `test_multimodal_unit.py` | 87 | `+            self.assertEqual(result.mime_type, "image/png")` | 实例属性操作 |
| `test_multimodal_unit.py` | 88 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 89 | `+    def test_save_base64_image(self) -> None:` | 定义函数 test_save_base64_image |
| `test_multimodal_unit.py` | 90 | `+        with tempfile.TemporaryDirectory() as tmp_dir:` | 上下文管理 |
| `test_multimodal_unit.py` | 91 | `+            manager = ImageStorageManager(tmp_dir)` | 使用图片管理器 |
| `test_multimodal_unit.py` | 92 | `+            data = base64.b64encode(b"test").decode("ascii")` | Base64 处理 |
| `test_multimodal_unit.py` | 93 | `+            stored = manager.save_base64_image(data, "image/jpeg")` | Base64 处理 |
| `test_multimodal_unit.py` | 94 | `+            self.assertTrue(stored.path.exists())` | 实例属性操作 |
| `test_multimodal_unit.py` | 95 | `+            self.assertTrue(stored.base64_data)` | 实例属性操作 |
| `test_multimodal_unit.py` | 96 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 97 | `+    def test_generate_url_candidates_prefers_twitter_orig(self) -> None:` | 定义函数 test_generate_url_candidates_prefers_twitter_orig |
| `test_multimodal_unit.py` | 98 | `+        with tempfile.TemporaryDirectory() as tmp_dir:` | 上下文管理 |
| `test_multimodal_unit.py` | 99 | `+            manager = ImageStorageManager(tmp_dir)` | 使用图片管理器 |
| `test_multimodal_unit.py` | 100 | `+            url = "https://pbs.twimg.com/profile_images/12345/test_400x400.jpg"` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 101 | `+            candidates = manager._generate_url_candidates(url)` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 102 | `+            self.assertTrue(any("name=orig" in c for c in candidates[:3]))` | 实例属性操作 |
| `test_multimodal_unit.py` | 103 | `+            self.assertEqual(candidates[-1], url)` | 实例属性操作 |
| `test_multimodal_unit.py` | 104 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 105 | `+    def test_generate_url_candidates_strip_imageview(self) -> None:` | 定义函数 test_generate_url_candidates_strip_imageview |
| `test_multimodal_unit.py` | 106 | `+        with tempfile.TemporaryDirectory() as tmp_dir:` | 上下文管理 |
| `test_multimodal_unit.py` | 107 | `+            manager = ImageStorageManager(tmp_dir)` | 使用图片管理器 |
| `test_multimodal_unit.py` | 108 | `+            url = "https://example.com/image.png?imageView=1&thumbnail=400x400"` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 109 | `+            candidates = manager._generate_url_candidates(url)` | 赋值/构建数据 |
| `test_multimodal_unit.py` | 110 | `+            self.assertTrue(any("imageView" not in c and "thumbnail" not in c for c in candidates[:-1]))` | 实例属性操作 |
| `test_multimodal_unit.py` | 111 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 112 | `+` | 空行分隔，增强可读性 |
| `test_multimodal_unit.py` | 113 | `+if __name__ == "__main__":` | 条件判断 |
| `test_multimodal_unit.py` | 114 | `+    unittest.main()` | 逻辑实现 |