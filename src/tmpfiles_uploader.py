"""
TmpFiles 上传工具，负责将本地文件上传并生成可公开访问的下载链接。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests


@dataclass
class TmpFilesUploader:
    """
    调用 tmpfiles.org API 上传文件并返回直接下载地址的工具类。

    Args:
        endpoint (str): 上传服务接口地址，默认 ``https://tmpfiles.org/api/v1/upload``。
        user_agent (str): 请求头中的 User-Agent。
        timeout_seconds (int): 请求超时时间（秒）。
    """

    endpoint: str = "https://tmpfiles.org/api/v1/upload"
    user_agent: str = "curl/8.1.2"
    timeout_seconds: int = 30

    def upload(self, path: Path) -> str:
        """
        上传本地文件并返回 tmpfiles 的直接下载链接。

        Args:
            path (Path): 待上传的文件路径。

        Returns:
            str: 上传成功后生成的 ``http://tmpfiles.org/dl/...`` 下载地址。

        Raises:
            AssertionError: 当文件不存在时抛出。
            ValueError: 当上传失败或响应内容异常时抛出。
        """

        assert path.is_file(), f"本地文件不存在：{path}"
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        try:
            with path.open("rb") as file_handle:
                response = requests.post(
                    self.endpoint,
                    files={"file": (path.name, file_handle)},
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
        except OSError as exc:
            raise ValueError(f"读取本地文件失败：{path}") from exc
        except requests.RequestException as exc:
            raise ValueError("上传文件到 tmpfiles 失败，请检查网络。") from exc

        if response.status_code >= 400:
            raise ValueError(f"上传服务返回异常状态：HTTP {response.status_code}。")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("上传服务返回内容无法解析为 JSON。") from exc

        if not isinstance(payload, dict):
            raise ValueError("上传服务返回内容异常，缺少字典结构。")
        if payload.get("status") != "success":
            error_message = payload.get("error") or "上传服务提示失败。"
            raise ValueError(f"上传服务提示失败：{error_message}")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("上传服务返回内容异常，缺少 data 字段。")

        link = data.get("url")
        if not isinstance(link, str):
            raise ValueError("上传服务返回内容异常，缺少链接地址。")

        parsed = urlparse(link.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("上传服务返回内容异常，未提供有效的 URL。")

        download_path = parsed.path
        if not download_path.startswith("/dl/"):
            if download_path.startswith("/"):
                download_path = "/dl" + download_path
            else:
                download_path = f"/dl/{download_path}"

        return urlunparse(parsed._replace(path=download_path, params="", query="", fragment=""))
