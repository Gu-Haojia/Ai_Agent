"""
网易云音乐 Agent 工具核心与包装层单元测试。
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import requests

import sql_agent_cli_stream_plus as agent_module
from src.netease_music_tool import (
    NeteaseMusicClient,
    NeteaseMusicToolError,
    NeteaseSong,
    OneBotMusicCardSender,
)


def _response(payload: object, status_code: int = 200) -> mock.Mock:
    """
    构造固定 JSON 内容的 requests.Response 替身。

    Args:
        payload (object): ``response.json()`` 返回的对象。
        status_code (int): HTTP 状态码。

    Returns:
        mock.Mock: 可用于 HTTP 客户端测试的响应替身。

    Raises:
        AssertionError: 当状态码不是正整数时抛出。
    """
    assert status_code > 0, "status_code 必须为正整数"
    response = mock.Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _search_payload() -> dict[str, object]:
    """
    构造固定的网易云搜索响应。

    Returns:
        dict[str, object]: 包含两条歌曲结果的响应。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return {
        "code": 200,
        "result": {
            "songCount": 2,
            "songs": [
                {
                    "id": 1357375695,
                    "name": "海阔天空",
                    "artists": [{"name": "Beyond"}],
                    "album": {"name": "精选集"},
                    "duration": 239560,
                },
                {
                    "id": 347230,
                    "name": "海阔天空",
                    "artists": [{"name": "Beyond"}],
                    "album": {"name": "海阔天空"},
                    "duration": 326000,
                },
            ],
        },
    }


class NeteaseMusicClientTests(unittest.TestCase):
    """验证网易云音乐搜索客户端的参数和响应校验。"""

    def test_search_returns_normalized_songs(self) -> None:
        """搜索成功时应返回标准化歌曲列表并限制候选数量。"""
        http_get = mock.Mock(return_value=_response(_search_payload()))
        client = NeteaseMusicClient(
            base_url="https://music-api.example/",
            http_get=http_get,
        )

        songs = client.search(" 海阔天空 Beyond ", limit=1)

        self.assertEqual(
            songs,
            [
                NeteaseSong(
                    song_id="1357375695",
                    title="海阔天空",
                    artists=("Beyond",),
                    album="精选集",
                    duration_seconds=240,
                )
            ],
        )
        http_get.assert_called_once_with(
            "https://music-api.example/search",
            params={
                "keywords": "海阔天空 Beyond",
                "limit": 1,
                "offset": 0,
                "type": 1,
            },
            timeout=10.0,
        )

    def test_search_returns_empty_list_when_no_song_exists(self) -> None:
        """合法空结果应返回空列表而不是伪造候选。"""
        http_get = mock.Mock(
            return_value=_response(
                {"code": 200, "result": {"songCount": 0, "songs": []}}
            )
        )
        client = NeteaseMusicClient(
            base_url="https://music-api.example",
            http_get=http_get,
        )

        self.assertEqual(client.search("不存在的歌曲"), [])

    def test_search_exposes_timeout_with_stable_error_code(self) -> None:
        """搜索超时应转换为模型可判断的稳定错误码。"""
        client = NeteaseMusicClient(
            base_url="https://music-api.example",
            http_get=mock.Mock(side_effect=requests.Timeout("timeout")),
        )

        with self.assertRaises(NeteaseMusicToolError) as caught:
            client.search("海阔天空")

        self.assertEqual(caught.exception.error_code, "upstream_timeout")

    def test_search_rejects_unexpected_response_shape(self) -> None:
        """搜索响应字段变化时应显式抛出断言错误。"""
        client = NeteaseMusicClient(
            base_url="https://music-api.example",
            http_get=mock.Mock(return_value=_response({"code": 200})),
        )

        with self.assertRaisesRegex(AssertionError, "result"):
            client.search("海阔天空")


class OneBotMusicCardSenderTests(unittest.TestCase):
    """验证 OneBot 音乐卡片发送参数和业务响应处理。"""

    def test_send_builds_music_segment_and_returns_message_id(self) -> None:
        """发送成功时应构造 163 音乐段并返回消息 ID。"""
        http_post = mock.Mock(
            return_value=_response(
                {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"message_id": 987654},
                    "message": "",
                }
            )
        )
        sender = OneBotMusicCardSender(
            api_base="http://onebot/",
            access_token="token",
            http_post=http_post,
        )

        message_id = sender.send("1357375695", 123456)

        self.assertEqual(message_id, "987654")
        http_post.assert_called_once_with(
            "http://onebot/send_group_msg",
            json={
                "group_id": 123456,
                "message": [
                    {
                        "type": "music",
                        "data": {"type": "163", "id": "1357375695"},
                    }
                ],
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer token",
            },
            timeout=60.0,
        )

    def test_send_exposes_onebot_business_failure(self) -> None:
        """OneBot 业务失败时应返回稳定错误类型而不是伪装成功。"""
        sender = OneBotMusicCardSender(
            api_base="http://onebot",
            http_post=mock.Mock(
                return_value=_response(
                    {
                        "status": "failed",
                        "retcode": 1400,
                        "data": None,
                        "message": "消息体无法解析",
                    }
                )
            ),
        )

        with self.assertRaises(NeteaseMusicToolError) as caught:
            sender.send("1357375695", 123456)

        self.assertEqual(caught.exception.error_code, "onebot_business_failed")
        self.assertIn("消息体无法解析", str(caught.exception))


class NeteaseMusicToolRegistrationTests(unittest.TestCase):
    """验证 Agent 文件中的网易云工具参数包装和 JSON 返回值。"""

    def test_search_wrapper_returns_ranked_candidates(self) -> None:
        """搜索包装层应返回带排名的模型可读 JSON。"""
        song = NeteaseSong(
            song_id="1357375695",
            title="海阔天空",
            artists=("Beyond",),
            album="精选集",
            duration_seconds=240,
        )
        with mock.patch.object(
            agent_module.NeteaseMusicClient,
            "search",
            return_value=[song],
        ) as search:
            output = agent_module.netease_music_search.invoke(  # type: ignore[attr-defined]
                {"keyword": "海阔天空 Beyond"}
            )

        payload = json.loads(output)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["songs"][0]["song_id"], "1357375695")
        self.assertEqual(payload["songs"][0]["rank"], 1)
        search.assert_called_once_with("海阔天空 Beyond", limit=3)

    def test_search_wrapper_returns_structured_failure(self) -> None:
        """搜索外部错误应在工具边界转换成结构化失败。"""
        with mock.patch.object(
            agent_module.NeteaseMusicClient,
            "search",
            side_effect=NeteaseMusicToolError(
                "upstream_timeout",
                "网易云音乐搜索服务请求超时。",
            ),
        ):
            output = agent_module.netease_music_search.invoke(  # type: ignore[attr-defined]
                {"keyword": "海阔天空"}
            )

        payload = json.loads(output)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "upstream_timeout")

    def test_send_wrapper_uses_only_song_and_group_ids(self) -> None:
        """发送包装层应只要求歌曲 ID 与模型提供的群号。"""
        with mock.patch.object(
            agent_module.OneBotMusicCardSender,
            "send",
            return_value="987654",
        ) as send:
            output = agent_module.send_netease_music_card.invoke(  # type: ignore[attr-defined]
                {"song_id": "1357375695", "group_id": 123456}
            )

        payload = json.loads(output)
        self.assertEqual(payload["status"], "sent")
        self.assertEqual(payload["message_id"], "987654")
        send.assert_called_once_with("1357375695", 123456)

    def test_send_wrapper_rejects_invalid_group_id(self) -> None:
        """发送包装层应拒绝非法群号且不执行底层发送。"""
        with mock.patch.object(
            agent_module.OneBotMusicCardSender,
            "send",
        ) as send, self.assertRaises(AssertionError):
            agent_module.send_netease_music_card.invoke(  # type: ignore[attr-defined]
                {"song_id": "1357375695", "group_id": 0}
            )

        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
