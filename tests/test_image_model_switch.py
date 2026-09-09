"""验证生图命令切换与 OpenAI 实际请求的模型一致。"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from image_storage import ImageStorageManager
from qq_group_bot import QQBotHandler


class ImageModelSwitchTest(unittest.TestCase):
    """
    覆盖生图服务商切换、模型切换及生成和编辑请求。

    Returns:
        None: 测试类不返回业务结果。

    Raises:
        AssertionError: 当命令状态与实际请求不一致时抛出。
    """

    def setUp(self) -> None:
        """
        创建仅处理命令的机器人实例。

        Returns:
            None: 初始化测试实例，无返回值。

        Raises:
            None: 无主动抛出的异常。
        """
        self.handler = object.__new__(QQBotHandler)
        self.handler.bot_cfg = SimpleNamespace(
            api_base="http://127.0.0.1:3000",
            access_token="",
            cmd_allowed_users=(),
        )

    def test_commands_change_generation_and_edit_models_without_recreation(self) -> None:
        """
        同一管理器应立即使用切换后的模型，并在重新切入 OpenAI 时重置。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当模型、服务商、回复或请求参数错误时抛出。
        """
        response = SimpleNamespace(data=[SimpleNamespace(b64_json="ZmFrZQ==")])
        client = mock.Mock()
        client.images.generate.return_value = response
        client.images.edit.return_value = response
        transitions = (
            ("/imageprovider", "openai", "gpt-image-2.5-flare"),
            ("/image", "openai", "gpt-image-2.5-sunburst"),
            ("/image", "openai", "gpt-image-2.5-flare"),
            ("/image", "openai", "gpt-image-2.5-sunburst"),
            ("/imageprovider", "gemini", "gpt-image-2.5-sunburst"),
            ("/imageprovider", "openai", "gpt-image-2.5-flare"),
        )

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.dict(
                os.environ,
                {
                    "IMAGE_PROVIDER": "gemini",
                    "NEXT_IMAGE_PROVIDER": " OpenAI ",
                    "IMAGE_MODEL_NAME": "gpt-image-2",
                    "GEMINI_IMAGE_MODEL": "gemini-3.1-flash-image",
                },
            ),
            mock.patch("qq_group_bot._send_group_msg") as send_mock,
            mock.patch("openai.OpenAI", return_value=client),
        ):
            manager = ImageStorageManager(tmp_dir)
            reference_path = Path(tmp_dir) / "reference.png"
            reference_path.write_bytes(b"reference")

            for command, provider, model in transitions:
                with self.subTest(command=command, provider=provider, model=model):
                    self.assertTrue(
                        self.handler._handle_commands(10001, 20002, command)
                    )
                    self.assertEqual(os.environ["IMAGE_PROVIDER"], provider)
                    self.assertEqual(os.environ["IMAGE_MODEL_NAME"], model)
                    self.assertEqual(
                        os.environ["GEMINI_IMAGE_MODEL"], "gemini-3.1-flash-image"
                    )
                    self.assertIn(provider, send_mock.call_args.args[2])
                    if provider == "openai":
                        self.assertIn(model, send_mock.call_args.args[2])
                        self.assertIs(
                            manager.generate_image_via_openai(" draw a cat "), response
                        )
                        client.images.generate.assert_called_with(
                            model=model, prompt="draw a cat"
                        )
                        self.assertIs(
                            manager.generate_image_via_openai(
                                " edit the cat ", reference_path
                            ),
                            response,
                        )
                        client.images.edit.assert_called_with(
                            model=model,
                            image=reference_path.resolve(),
                            prompt="edit the cat",
                        )

    def test_image_uses_flare_as_unconfigured_openai_model(self) -> None:
        """
        OpenAI 未配置模型时应从默认 Flare 开始循环。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当默认模型或切换顺序错误时抛出。
        """
        with (
            mock.patch.dict(os.environ, {"IMAGE_PROVIDER": " OpenAI "}),
            mock.patch("qq_group_bot._send_group_msg"),
        ):
            os.environ.pop("IMAGE_MODEL_NAME", None)
            for expected in ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"):
                self.assertTrue(self.handler._handle_commands(10001, 20002, "/image"))
                self.assertEqual(os.environ["IMAGE_MODEL_NAME"], expected)

    def test_image_preserves_gemini_cycle_and_openai_selection(self) -> None:
        """
        Gemini 应保留原有默认值和循环顺序，且不改变 OpenAI 的模型选择。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当服务商隔离或 Gemini 切换行为错误时抛出。
        """
        cases = (
            (None, "gemini-3-pro-image"),
            ("gemini-3.1-flash-image", "gemini-3-pro-image"),
            ("gemini-3-pro-image", "gemini-3.1-flash-image"),
        )
        for current, expected in cases:
            with (
                self.subTest(current=current),
                mock.patch.dict(
                    os.environ, {"IMAGE_MODEL_NAME": "gpt-image-2.5-sunburst"}
                ),
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
            ):
                os.environ.pop("IMAGE_PROVIDER", None)
                if current is None:
                    os.environ.pop("GEMINI_IMAGE_MODEL", None)
                else:
                    os.environ["GEMINI_IMAGE_MODEL"] = current
                self.assertTrue(self.handler._handle_commands(10001, 20002, "/image"))
                self.assertEqual(os.environ["GEMINI_IMAGE_MODEL"], expected)
                self.assertEqual(os.environ["IMAGE_MODEL_NAME"], "gpt-image-2.5-sunburst")
                self.assertIn("gemini", send_mock.call_args.args[2])

    def test_image_rejects_unsupported_provider_without_changing_models(self) -> None:
        """
        不支持切换的服务商应返回明确错误并保留模型配置。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当非法切换改变配置或未返回错误时抛出。
        """
        for provider in ("xai", "unknown"):
            with (
                self.subTest(provider=provider),
                mock.patch.dict(os.environ, {"IMAGE_PROVIDER": provider}),
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
            ):
                before = dict(os.environ)
                self.assertTrue(self.handler._handle_commands(10001, 20002, "/image"))
                self.assertEqual(dict(os.environ), before)
                self.assertIn("不支持切换生图模型", send_mock.call_args.args[2])

    def test_image_rejects_invalid_current_model_without_changing_configuration(self) -> None:
        """
        当前模型不在候选列表时应明确报错，不自动改用其他模型。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当非法模型被替换或未返回错误时抛出。
        """
        cases = (
            ("openai", "IMAGE_MODEL_NAME", "gpt-image-2"),
            ("openai", "IMAGE_MODEL_NAME", ""),
            ("gemini", "GEMINI_IMAGE_MODEL", "unknown"),
        )
        for provider, variable, model in cases:
            with (
                self.subTest(provider=provider, model=model),
                mock.patch.dict(
                    os.environ, {"IMAGE_PROVIDER": provider, variable: model}
                ),
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
            ):
                before = dict(os.environ)
                self.assertTrue(self.handler._handle_commands(10001, 20002, "/image"))
                self.assertEqual(dict(os.environ), before)
                self.assertIn("可切换列表", send_mock.call_args.args[2])

    def test_explicit_instance_model_takes_precedence_over_environment(self) -> None:
        """
        显式指定实例模型的调用方应保留固定模型语义。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当环境变量覆盖显式模型时抛出。
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.dict(os.environ, {"IMAGE_MODEL_NAME": "gpt-image-2.5-flare"}),
            mock.patch("openai.OpenAI") as factory,
        ):
            manager = ImageStorageManager(tmp_dir, image_model="gpt-image-2.5-sunburst")
            manager.generate_image_via_openai("draw a cat")
            factory.return_value.images.generate.assert_called_once_with(
                model="gpt-image-2.5-sunburst", prompt="draw a cat"
            )

    def test_blank_model_is_rejected_before_creating_api_client(self) -> None:
        """
        空白模型配置应直接报错，避免发起无效请求或自动使用默认模型。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当空白模型未被拒绝或创建了客户端时抛出。
        """
        for explicit_model in (None, "", " "):
            with (
                self.subTest(explicit_model=explicit_model),
                tempfile.TemporaryDirectory() as tmp_dir,
                mock.patch.dict(os.environ, {"IMAGE_MODEL_NAME": " "}),
                mock.patch("openai.OpenAI") as factory,
            ):
                manager = ImageStorageManager(tmp_dir, image_model=explicit_model)
                with self.assertRaisesRegex(AssertionError, "模型名称不能为空"):
                    manager.generate_image_via_openai("draw a cat")
                factory.assert_not_called()
