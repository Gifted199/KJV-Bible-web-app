"""Settings tests: no network, no real .env (every test passes its own environment)."""

import pytest

from app import config

OPENAI = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "openai-secret-key-123",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.4-mini",
    "AZURE_OPENAI_EMBED_DEPLOYMENT": "text-embedding-3-small",
}


def test_require_returns_the_groups_settings():
    settings = config.require("openai", OPENAI)
    assert settings["AZURE_OPENAI_CHAT_DEPLOYMENT"] == "gpt-5.4-mini"
    assert settings["AZURE_OPENAI_EMBED_DEPLOYMENT"] == "text-embedding-3-small"


def test_missing_settings_are_named_but_no_value_is_shown():
    env = {**OPENAI, "AZURE_OPENAI_EMBED_DEPLOYMENT": "  "}  # blank counts as missing
    with pytest.raises(config.MissingSettings) as error:
        config.require("openai", env)
    message = str(error.value)
    assert "AZURE_OPENAI_EMBED_DEPLOYMENT" in message
    assert "openai-secret-key-123" not in message


def test_redact_masks_every_secret_in_an_error_message():
    env = {"AZURE_OPENAI_API_KEY": "openai-secret-key-123", "AZURE_SEARCH_API_KEY": "search-secret-456"}
    text = "401 for key openai-secret-key-123 and search-secret-456"
    assert config.redact(text, env) == "401 for key *** and ***"


@pytest.mark.parametrize("url", [
    "https://example.openai.azure.com",
    "https://example.openai.azure.com/",
    "https://example.openai.azure.com/openai/responses?api-version=2025-04-01-preview",
    "https://example.openai.azure.com/openai/deployments/x/embeddings?api-version=2023-05-15",
])
def test_base_endpoint_accepts_the_portals_target_uri(url):
    assert config.base_endpoint(url) == "https://example.openai.azure.com"


def test_chat_options_use_max_completion_tokens():
    assert config.chat_options(100) == {"max_completion_tokens": 100, "reasoning_effort": "none"}
    assert config.chat_options(100, temperature=0)["temperature"] == 0


def test_chat_options_refuse_what_the_model_rejects():
    with pytest.raises(ValueError):
        config.chat_options(100, reasoning_effort="low", temperature=0)  # temperature needs reasoning off
    with pytest.raises(ValueError):
        config.chat_options(100, reasoning_effort="maximum")
    assert "temperature" not in config.chat_options(100, reasoning_effort="low")
