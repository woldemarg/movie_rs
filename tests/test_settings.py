from backend.settings import ROOT, Settings


def test_local_profile_needs_no_key_and_openrouter_needs_one():
    settings = Settings(_env_file=None, llm_api_key="")

    local = settings.llm_profile("local")
    assert (local.provider, local.api_key, local.extra_body) == ("local", "", {})
    assert settings.llm_profile("openrouter") is None


def test_openrouter_profile_carries_the_key_and_the_provider_order():
    settings = Settings(
        _env_file=None, llm_api_key="k", llm_provider_order=" deepinfra, ,parasail "
    )

    profile = settings.llm_profile("openrouter")

    assert (profile.api_key, profile.model) == ("k", "google/gemma-4-26b-a4b-it")
    assert profile.extra_body == {"provider": {"order": ["deepinfra", "parasail"]}}


def test_env_sample_lists_only_real_settings():
    lines = (ROOT / ".env.sample").read_text(encoding="utf-8").splitlines()
    keys = {line.split("=", 1)[0].lower() for line in lines if line and not line.startswith("#")}

    assert keys <= set(Settings.model_fields)
    assert {"local_llm_model", "llm_model", "llm_api_key", "tmdb_api_key"} <= keys
