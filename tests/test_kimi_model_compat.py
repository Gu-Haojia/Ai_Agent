import os

from sql_agent_cli_stream_plus import _memory_index_config, _prepare_model_env


def _clear_model_env(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "KIMI_API_KEY",
        "KIMI_OPENAI_BASE_URL",
        "KIMI_PROTOCOL",
        "MEM_EMBED_DIMS",
        "MEM_EMBED_MODEL",
        "MOONSHOT_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_kimi_code_alias_defaults_to_anthropic(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("KIMI_API_KEY", "dummy-kimi-key")

    resolved = _prepare_model_env("kimi-code:kimi-for-coding")

    assert resolved == "anthropic:kimi-for-coding"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.kimi.com/coding/"
    assert os.environ["ANTHROPIC_API_KEY"] == "dummy-kimi-key"


def test_kimi_code_alias_can_use_openai_protocol(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("KIMI_PROTOCOL", "openai")
    monkeypatch.setenv("KIMI_API_KEY", "dummy-kimi-key")

    resolved = _prepare_model_env("kimi-code:kimi-for-coding")

    assert resolved == "openai:kimi-for-coding"
    assert os.environ["OPENAI_BASE_URL"] == "https://api.kimi.com/coding/v1"
    assert os.environ["OPENAI_API_KEY"] == "dummy-kimi-key"


def test_moonshot_alias_uses_openai_compatible_platform(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MOONSHOT_API_KEY", "dummy-moonshot-key")

    resolved = _prepare_model_env("moonshot:kimi-k2")

    assert resolved == "openai:kimi-k2"
    assert os.environ["OPENAI_BASE_URL"] == "https://api.moonshot.cn/v1"
    assert os.environ["OPENAI_API_KEY"] == "dummy-moonshot-key"


def test_kimi_only_deployment_does_not_default_to_openai_embedding(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-kimi-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.kimi.com/coding/v1")

    assert _memory_index_config() is None


def test_memory_index_can_be_enabled_explicitly(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MEM_EMBED_MODEL", "openai:text-embedding-3-small")
    monkeypatch.setenv("MEM_EMBED_DIMS", "1536")

    assert _memory_index_config() == {
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
