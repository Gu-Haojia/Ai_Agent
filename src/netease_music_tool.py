"""
网易云音乐搜索与 OneBot 音乐卡片发送核心模块。

本模块只负责外部接口调用和响应结构校验，不包含 LangChain 工具包装。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin

import requests


HttpGet = Callable[..., requests.Response]
HttpPost = Callable[..., requests.Response]
DEFAULT_NETEASE_MUSIC_API_BASE = "https://nce.gqzsldy.com"
DEFAULT_ONEBOT_API_BASE = "http://127.0.0.1:3000"
NETEASE_SONG_DETAIL_API_URL = "https://music.163.com/api/song/detail/"
XIANYUW_MUSIC_ARK_API_URL = "https://apii.xianyuw.cn/api/v1/qq-musicArk"
USE_SIGNED_MUSIC_CARD = True


def _parse_json_response(
    response: requests.Response,
    service_name: str,
) -> dict[str, object]:
    """
    解析并校验外部服务的 JSON 对象响应。

    Args:
        response (requests.Response): 外部服务 HTTP 响应。
        service_name (str): 用于错误信息的服务名称。

    Returns:
        dict[str, object]: 已解析的 JSON 对象。

    Raises:
        AssertionError: 当服务名为空、响应不是 JSON 或顶层不是对象时抛出。
    """
    assert service_name.strip(), "service_name 不能为空"
    try:
        payload = response.json()
    except ValueError as exc:
        raise AssertionError(f"{service_name}返回内容不是有效 JSON") from exc
    assert isinstance(payload, dict), f"{service_name}响应顶层必须是对象"
    return payload


class NeteaseMusicToolError(RuntimeError):
    """表示网易云音乐工具可向 Agent 暴露的外部调用错误。"""

    def __init__(self, error_code: str, message: str) -> None:
        """
        初始化带稳定错误码的工具异常。

        Args:
            error_code (str): 供 Agent 判断错误类型的稳定错误码。
            message (str): 面向 Agent 的简短错误说明。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当错误码或错误说明为空时抛出。
        """
        assert error_code.strip(), "error_code 不能为空"
        assert message.strip(), "message 不能为空"
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class NeteaseSong:
    """保存一条标准化后的网易云歌曲搜索结果。"""

    song_id: str
    title: str
    artists: tuple[str, ...]
    album: str
    duration_seconds: int

    def to_dict(self, rank: int) -> dict[str, object]:
        """
        转换为适合 Agent 消费的歌曲字典。

        Args:
            rank (int): 歌曲在搜索结果中的一基排名。

        Returns:
            dict[str, object]: 包含歌曲 ID、名称、歌手和页面链接的字典。

        Raises:
            AssertionError: 当排名不是正整数时抛出。
        """
        assert rank > 0, "rank 必须为正整数"
        return {
            "rank": rank,
            "song_id": self.song_id,
            "title": self.title,
            "artists": list(self.artists),
            "album": self.album,
            "duration_seconds": self.duration_seconds,
            "page_url": f"https://music.163.com/song?id={self.song_id}",
        }


class NeteaseMusicClient:
    """调用单一网易云音乐 API 上游并标准化搜索结果。"""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        http_get: HttpGet | None = None,
    ) -> None:
        """
        初始化网易云音乐搜索客户端。

        Args:
            base_url (str): 网易云音乐 API 基地址。
            timeout_seconds (float): HTTP 请求超时秒数。
            http_get (HttpGet | None): 可注入的 HTTP GET 调用函数。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当基地址为空或超时秒数不大于零时抛出。
        """
        normalized_base_url = base_url.strip()
        assert normalized_base_url, "网易云音乐 API 基地址不能为空"
        assert timeout_seconds > 0, "timeout_seconds 必须大于 0"
        self._base_url = normalized_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_get = http_get or requests.get

    def search(self, keyword: str, limit: int = 5) -> list[NeteaseSong]:
        """
        按关键词搜索网易云单曲。

        Args:
            keyword (str): 歌名或“歌名 + 歌手”搜索词。
            limit (int): 返回候选歌曲数量，范围为 1 至 5，默认 5。

        Returns:
            list[NeteaseSong]: 按上游相关度排序的歌曲候选列表。

        Raises:
            AssertionError: 当参数非法或上游响应结构不符合约定时抛出。
            NeteaseMusicToolError: 当上游请求超时或 HTTP 请求失败时抛出。
        """
        normalized_keyword = keyword.strip()
        assert normalized_keyword, "keyword 不能为空"
        assert len(normalized_keyword) <= 100, "keyword 长度不能超过 100"
        assert 1 <= limit <= 5, "limit 必须在 1 到 5 之间"

        url = urljoin(self._base_url + "/", "search")
        try:
            response = self._http_get(
                url,
                params={
                    "keywords": normalized_keyword,
                    "limit": limit,
                    "offset": 0,
                    "type": 1,
                },
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise NeteaseMusicToolError(
                "upstream_timeout",
                "网易云音乐搜索服务请求超时。",
            ) from exc
        except requests.RequestException as exc:
            raise NeteaseMusicToolError(
                "upstream_http_error",
                f"网易云音乐搜索服务请求失败：{exc}",
            ) from exc

        if response.status_code != 200:
            raise NeteaseMusicToolError(
                "upstream_http_error",
                f"网易云音乐搜索服务返回 HTTP {response.status_code}。",
            )

        payload = _parse_json_response(response, "网易云音乐搜索服务")
        assert payload.get("code") == 200, "网易云音乐搜索响应 code 不是 200"
        result = payload.get("result")
        assert isinstance(result, dict), "网易云音乐搜索响应缺少 result 对象"
        raw_songs = result.get("songs")
        assert isinstance(raw_songs, list), "网易云音乐搜索响应缺少 songs 列表"

        songs: list[NeteaseSong] = []
        for raw_song in raw_songs[:limit]:
            songs.append(self._parse_song(raw_song))
        return songs

    @staticmethod
    def _parse_song(raw_song: object) -> NeteaseSong:
        """
        将上游单曲对象转换为稳定的数据类型。

        Args:
            raw_song (object): 网易云音乐 API 返回的单曲对象。

        Returns:
            NeteaseSong: 标准化后的歌曲结果。

        Raises:
            AssertionError: 当歌曲字段类型或内容不符合约定时抛出。
        """
        assert isinstance(raw_song, dict), "歌曲搜索结果必须是对象"
        song_id = str(raw_song.get("id", "")).strip()
        title = raw_song.get("name")
        raw_artists = raw_song.get("artists")
        raw_album = raw_song.get("album")
        duration_ms = raw_song.get("duration")

        assert song_id.isdigit(), "歌曲搜索结果缺少有效 id"
        assert isinstance(title, str) and title.strip(), "歌曲搜索结果缺少 name"
        assert isinstance(raw_artists, list), "歌曲搜索结果缺少 artists 列表"
        artists: list[str] = []
        for raw_artist in raw_artists:
            assert isinstance(raw_artist, dict), "歌曲歌手信息必须是对象"
            artist_name = raw_artist.get("name")
            assert isinstance(artist_name, str) and artist_name.strip(), (
                "歌曲歌手信息缺少 name"
            )
            artists.append(artist_name.strip())
        assert artists, "歌曲搜索结果至少需要一名歌手"
        assert isinstance(raw_album, dict), "歌曲搜索结果缺少 album 对象"
        album_name = raw_album.get("name")
        assert isinstance(album_name, str), "歌曲专辑信息缺少 name"
        assert isinstance(duration_ms, int) and duration_ms >= 0, (
            "歌曲搜索结果缺少有效 duration"
        )

        return NeteaseSong(
            song_id=song_id,
            title=title.strip(),
            artists=tuple(artists),
            album=album_name.strip(),
            duration_seconds=round(duration_ms / 1000),
        )


class OneBotMusicCardSender:
    """通过 OneBot HTTP API 向指定 QQ 群发送网易云音乐卡片。"""

    def __init__(
        self,
        api_base: str,
        access_token: str = "",
        timeout_seconds: float = 60.0,
        http_get: HttpGet | None = None,
        http_post: HttpPost | None = None,
        signed_api_key: str | None = None,
    ) -> None:
        """
        初始化 OneBot 音乐卡片发送器。

        Args:
            api_base (str): OneBot HTTP API 基地址。
            access_token (str): OneBot Bearer Token，可为空。
            timeout_seconds (float): HTTP 请求超时秒数。
            http_get (HttpGet | None): 可注入的 HTTP GET 调用函数。
            http_post (HttpPost | None): 可注入的 HTTP POST 调用函数。
            signed_api_key (str | None): 签名接口密钥，默认读取现有环境变量。

        Returns:
            None: 构造函数无返回值。

        Raises:
            AssertionError: 当 API 基地址为空或超时秒数不大于零时抛出。
        """
        normalized_api_base = api_base.strip()
        assert normalized_api_base, "OneBot API 基地址不能为空"
        assert timeout_seconds > 0, "timeout_seconds 必须大于 0"
        self._api_base = normalized_api_base.rstrip("/")
        self._access_token = access_token.strip()
        self._timeout_seconds = timeout_seconds
        self._http_get = http_get or requests.get
        self._http_post = http_post or requests.post
        self._signed_api_key = (
            os.environ.get("XIANYUW_API_KEY", "").strip()
            if signed_api_key is None
            else signed_api_key.strip()
        )

    def send(self, song_id: str, group_id: int) -> str:
        """
        向指定 QQ 群发送网易云音乐卡片。

        Args:
            song_id (str): 网易云单曲 ID，必须来自搜索结果。
            group_id (int): 目标 QQ 群号，由 Agent 调用时提供。

        Returns:
            str: OneBot 返回的消息 ID。

        Raises:
            AssertionError: 当参数非法或 OneBot 响应结构不符合约定时抛出。
            NeteaseMusicToolError: 当 OneBot 请求或业务执行失败时抛出。
        """
        normalized_song_id = song_id.strip()
        assert normalized_song_id.isdigit(), "song_id 必须是数字字符串"
        assert group_id > 0, "group_id 必须为正整数"

        if USE_SIGNED_MUSIC_CARD:
            message = self._build_signed_message(normalized_song_id)
        else:
            message = [
                {
                    "type": "music",
                    "data": {"type": "163", "id": normalized_song_id},
                }
            ]
        return self._send_message(message, group_id)

    def _build_signed_message(self, song_id: str) -> list[dict[str, object]]:
        """
        获取歌曲信息和签名 Ark，并构造 OneBot JSON 消息段。

        Args:
            song_id (str): 已校验的网易云数字歌曲 ID。

        Returns:
            list[dict[str, object]]: 可直接交给 OneBot 的 JSON 消息段。

        Raises:
            AssertionError: 当密钥为空或外部响应结构不符合约定时抛出。
            NeteaseMusicToolError: 当歌曲详情或签名接口请求失败时抛出。
        """
        assert self._signed_api_key, "XIANYUW_API_KEY 不能为空"
        detail_payload = self._get_json(
            NETEASE_SONG_DETAIL_API_URL,
            {"id": song_id, "ids": f"[{song_id}]"},
            "网易云歌曲详情接口",
        )
        assert detail_payload.get("code") == 200, "网易云歌曲详情响应 code 不是 200"
        songs = detail_payload.get("songs")
        assert isinstance(songs, list) and songs, "网易云歌曲详情响应缺少 songs"
        song = songs[0]
        assert isinstance(song, dict), "网易云歌曲详情必须是对象"
        title = song.get("name")
        artists = song.get("artists")
        album = song.get("album")
        assert isinstance(title, str) and title.strip(), "网易云歌曲详情缺少 name"
        assert isinstance(artists, list) and artists, "网易云歌曲详情缺少 artists"
        artist_names: list[str] = []
        for artist in artists:
            assert isinstance(artist, dict), "网易云歌曲歌手信息必须是对象"
            artist_name = artist.get("name")
            assert isinstance(artist_name, str) and artist_name.strip(), (
                "网易云歌曲歌手信息缺少 name"
            )
            artist_names.append(artist_name.strip())
        assert isinstance(album, dict), "网易云歌曲详情缺少 album"
        cover = album.get("picUrl")
        assert isinstance(cover, str) and cover.strip(), "网易云歌曲详情缺少封面"

        signed_payload = self._get_json(
            XIANYUW_MUSIC_ARK_API_URL,
            {
                "key": self._signed_api_key,
                "url": f"http://music.163.com/song/media/outer/url?id={song_id}",
                "song": title.strip(),
                "singer": "/".join(artist_names),
                "cover": cover.strip(),
                "jump": f"https://y.music.163.com/m/song/{song_id}",
                "format": "netease",
            },
            "音乐卡片签名接口",
        )
        assert signed_payload.get("code") == 200, "音乐卡片签名响应 code 不是 200"
        ark = signed_payload.get("data")
        assert isinstance(ark, dict), "音乐卡片签名响应缺少 data"
        return [
            {
                "type": "json",
                "data": {
                    "data": json.dumps(
                        ark,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                },
            }
        ]

    def _get_json(
        self,
        url: str,
        params: dict[str, str],
        service_name: str,
    ) -> dict[str, object]:
        """
        请求音乐卡片依赖的 GET 接口并解析 JSON。

        Args:
            url (str): 请求地址。
            params (dict[str, str]): 查询参数。
            service_name (str): 用于错误信息的服务名称。

        Returns:
            dict[str, object]: 已解析的 JSON 对象。

        Raises:
            NeteaseMusicToolError: 当请求超时、失败或返回非 200 时抛出。
            AssertionError: 当响应不是有效 JSON 对象时抛出。
        """
        try:
            response = self._http_get(
                url,
                params=params,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise NeteaseMusicToolError(
                "music_card_request_failed",
                f"{service_name}请求超时。",
            ) from exc
        except requests.RequestException as exc:
            raise NeteaseMusicToolError(
                "music_card_request_failed",
                f"{service_name}请求失败：{exc}",
            ) from exc
        if response.status_code != 200:
            raise NeteaseMusicToolError(
                "music_card_request_failed",
                f"{service_name}返回 HTTP {response.status_code}。",
            )
        return _parse_json_response(response, service_name)

    def _send_message(
        self,
        message: list[dict[str, object]],
        group_id: int,
    ) -> str:
        """
        通过 OneBot 发送已构造的群消息。

        Args:
            message (list[dict[str, object]]): OneBot 消息段列表。
            group_id (int): 目标 QQ 群号。

        Returns:
            str: OneBot 返回的消息 ID。

        Raises:
            AssertionError: 当 OneBot 响应结构不符合约定时抛出。
            NeteaseMusicToolError: 当 OneBot 请求或业务执行失败时抛出。
        """

        headers = {"Content-Type": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        payload = {
            "group_id": group_id,
            "message": message,
        }
        url = urljoin(self._api_base + "/", "send_group_msg")
        try:
            response = self._http_post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise NeteaseMusicToolError(
                "onebot_request_failed",
                "OneBot 发送音乐卡片请求超时。",
            ) from exc
        except requests.RequestException as exc:
            raise NeteaseMusicToolError(
                "onebot_request_failed",
                f"OneBot 发送音乐卡片请求失败：{exc}",
            ) from exc

        if response.status_code != 200:
            raise NeteaseMusicToolError(
                "onebot_request_failed",
                f"OneBot 发送音乐卡片返回 HTTP {response.status_code}。",
            )

        response_payload = _parse_json_response(
            response,
            "OneBot 音乐卡片接口",
        )
        status = response_payload.get("status")
        retcode = response_payload.get("retcode")
        assert isinstance(status, str), "OneBot 响应缺少 status"
        assert isinstance(retcode, int), "OneBot 响应缺少 retcode"
        if status != "ok" or retcode != 0:
            error_message = response_payload.get("message")
            assert isinstance(error_message, str), "OneBot 失败响应缺少 message"
            raise NeteaseMusicToolError(
                "onebot_business_failed",
                f"NapCat 未能生成或发送音乐卡片：{error_message}",
            )

        data = response_payload.get("data")
        assert isinstance(data, dict), "OneBot 成功响应缺少 data 对象"
        message_id = str(data.get("message_id", "")).strip()
        assert message_id, "OneBot 成功响应缺少 message_id"
        return message_id
