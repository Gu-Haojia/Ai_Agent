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
        创建共用图像管理器的命令实例，并隔离测试状态。

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
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = ImageStorageManager(self.directory.name)
        storage_patch = mock.patch.object(QQBotHandler, "image_storage", self.manager)
        storage_patch.start()
        self.addCleanup(storage_patch.stop)

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
            manager = self.manager
            reference_path = Path(self.directory.name) / "reference.png"
            reference_path.write_bytes(b"reference")

            for command, provider, model in transitions:
                with self.subTest(command=command, provider=provider, model=model):
                    self.assertTrue(
                        self.handler._handle_commands(10001, 20002, command)
                    )
                    self.assertEqual(os.environ["IMAGE_PROVIDER"], provider)
                    self.assertEqual(manager.openai_image_model, model)
                    self.assertEqual(os.environ["IMAGE_MODEL_NAME"], "gpt-image-2")
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
                    else:
                        self.assertIn(
                            "当前生图模型：gemini-3.1-flash-image。",
                            send_mock.call_args.args[2],
                        )

    def test_image_uses_internal_state_without_creating_environment_variable(self) -> None:
        """
        OpenAI 应从默认 Flare 开始循环，且不创建模型环境变量。

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
                self.assertEqual(self.manager.openai_image_model, expected)
                self.assertNotIn("IMAGE_MODEL_NAME", os.environ)

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
        self.manager.set_openai_image_model("gpt-image-2.5-sunburst")
        for current, expected in cases:
            with (
                self.subTest(current=current),
                mock.patch.dict(os.environ),
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
            ):
                os.environ.pop("IMAGE_PROVIDER", None)
                if current is None:
                    os.environ.pop("GEMINI_IMAGE_MODEL", None)
                else:
                    os.environ["GEMINI_IMAGE_MODEL"] = current
                self.assertTrue(self.handler._handle_commands(10001, 20002, "/image"))
                self.assertEqual(os.environ["GEMINI_IMAGE_MODEL"], expected)
                self.assertEqual(self.manager.openai_image_model, "gpt-image-2.5-sunburst")
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
                self.assertEqual(self.manager.openai_image_model, "gpt-image-2.5-flare")
                self.assertIn("不支持切换生图模型", send_mock.call_args.args[2])

    def test_image_rejects_invalid_gemini_model_without_changing_configuration(self) -> None:
        """
        当前模型不在候选列表时应明确报错，不自动改用其他模型。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当非法模型被替换或未返回错误时抛出。
        """
        with (
            mock.patch.dict(
                os.environ,
                {"IMAGE_PROVIDER": "gemini", "GEMINI_IMAGE_MODEL": "unknown"},
            ),
            mock.patch("qq_group_bot._send_group_msg") as send_mock,
        ):
            before = dict(os.environ)
            self.assertTrue(self.handler._handle_commands(10001, 20002, "/image"))
            self.assertEqual(dict(os.environ), before)
            self.assertIn("可切换列表", send_mock.call_args.args[2])

    def test_instances_keep_independent_model_selections(self) -> None:
        """
        不同管理器应独立保存模型选择，避免依赖共享的环境状态。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当一个实例的切换影响另一个实例时抛出。
        """
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch("openai.OpenAI") as factory,
        ):
            manager = ImageStorageManager(tmp_dir, image_model="gpt-image-2.5-sunburst")
            self.assertEqual(self.manager.openai_image_model, "gpt-image-2.5-flare")
            manager.generate_image_via_openai("draw a cat")
            factory.return_value.images.generate.assert_called_once_with(
                model="gpt-image-2.5-sunburst", prompt="draw a cat"
            )
            manager.set_openai_image_model("gpt-image-2.5-flare")
            self.manager.set_openai_image_model("gpt-image-2.5-sunburst")
            self.assertEqual(manager.openai_image_model, "gpt-image-2.5-flare")

    def test_invalid_model_is_rejected_before_updating_instance(self) -> None:
        """
        空白或不支持的模型应直接报错，并保留原有模型选择。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当非法模型未被拒绝或改变了实例状态时抛出。
        """
        for model in ("", " ", "gpt-image-2", "unknown"):
            with self.subTest(model=model):
                with self.assertRaises(AssertionError):
                    self.manager.set_openai_image_model(model)
                self.assertEqual(self.manager.openai_image_model, "gpt-image-2.5-flare")

    def test_legacy_model_environment_does_not_affect_requests(self) -> None:
        """
        遗留环境变量应被忽略，新实例始终从 Flare 开始生成和编辑。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当环境变量改变默认模型或被写入时抛出。
        """
        for legacy_model in ("", "gpt-image-2", "gpt-image-2.5-sunburst"):
            with (
                self.subTest(legacy_model=legacy_model),
                tempfile.TemporaryDirectory() as tmp_dir,
                mock.patch.dict(os.environ, {"IMAGE_MODEL_NAME": legacy_model}),
                mock.patch("openai.OpenAI") as factory,
            ):
                manager = ImageStorageManager(tmp_dir)
                reference_path = Path(tmp_dir) / "reference.png"
                reference_path.write_bytes(b"reference")
                manager.generate_image_via_openai("draw a cat")
                factory.return_value.images.generate.assert_called_once_with(
                    model="gpt-image-2.5-flare", prompt="draw a cat"
                )
                manager.generate_image_via_openai("edit the cat", reference_path)
                factory.return_value.images.edit.assert_called_once_with(
                    model="gpt-image-2.5-flare",
                    image=reference_path.resolve(),
                    prompt="edit the cat",
                )
                self.assertEqual(os.environ["IMAGE_MODEL_NAME"], legacy_model)

    def test_provider_switch_requires_initialized_storage(self) -> None:
        """
        缺少图像管理器时应明确报错，并保持当前服务商不变。

        Returns:
            None: 测试无返回值。

        Raises:
            AssertionError: 当初始化失败仍改变服务商时抛出。
        """
        with (
            mock.patch.object(QQBotHandler, "image_storage", None),
            mock.patch.dict(
                os.environ,
                {"IMAGE_PROVIDER": "gemini", "NEXT_IMAGE_PROVIDER": "openai"},
            ),
            mock.patch("qq_group_bot._send_group_msg") as send_mock,
        ):
            self.assertTrue(self.handler._handle_commands(10001, 20002, "/imageprovider"))
            self.assertEqual(os.environ["IMAGE_PROVIDER"], "gemini")
            self.assertIn("图像存储管理器尚未配置", send_mock.call_args.args[2])
