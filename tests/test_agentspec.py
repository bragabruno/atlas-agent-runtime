"""AGT-1 AgentSpec tests.

Covers:
- regdoc-qa.yaml loads and validates correctly (all fields, correct values).
- Missing required field raises ValidationError with an explicit message.
- Extra / unknown field raises ValidationError (extra='forbid').
- Non-semver agent_version rejected.
- Non-positive integers for caps rejected.
- Empty tool name in whitelist rejected.
- Non-dict YAML raises ValueError.
- load_spec raises FileNotFoundError for missing files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agentspec import AgentSpec, load_spec

# Path to the real atlas-prompts agent YAML — resolved relative to this file.
_REPO_ROOT = Path(__file__).parent.parent.parent  # Developer-local/Atlas
_REGDOC_YAML = _REPO_ROOT / "atlas-prompts" / "agents" / "regdoc-qa.yaml"


class TestLoadRegdocYaml:
    """regdoc-qa.yaml must load and validate without error."""

    def test_file_exists(self) -> None:
        assert _REGDOC_YAML.exists(), f"regdoc-qa.yaml not found at {_REGDOC_YAML}"

    def test_loads_as_agentspec(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert isinstance(spec, AgentSpec)

    def test_agent_name(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.agent_name == "regdoc-qa"

    def test_agent_version(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.agent_version == "1.0.0"

    def test_model_alias(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.model_alias == "claude-sonnet-4-6"

    def test_tool_whitelist(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.tool_whitelist == ["doc_search", "verify_citation"]

    def test_max_iterations(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.max_iterations == 8

    def test_token_budget(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.token_budget == 4096

    def test_timeout_s(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.timeout_s == 60

    def test_system_prompt_ref(self) -> None:
        spec = load_spec(_REGDOC_YAML)
        assert spec.system_prompt_ref == "prompts/regdoc-qa/1.0.0/template.jinja"


class TestInvalidSpecs:
    """Invalid YAML data must fail fast with explicit ValidationError."""

    def _valid_data(self) -> dict[str, object]:
        return {
            "agent_name": "test-agent",
            "agent_version": "1.0.0",
            "system_prompt_ref": "prompts/test.txt",
            "model_alias": "atlas-default",
            "tool_whitelist": ["doc_search"],
            "max_iterations": 5,
            "token_budget": 8000,
            "timeout_s": 30,
        }

    def test_missing_agent_name_raises(self) -> None:
        data = self._valid_data()
        del data["agent_name"]
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_missing_max_iterations_raises(self) -> None:
        data = self._valid_data()
        del data["max_iterations"]
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_missing_token_budget_raises(self) -> None:
        data = self._valid_data()
        del data["token_budget"]
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_missing_timeout_s_raises(self) -> None:
        data = self._valid_data()
        del data["timeout_s"]
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_extra_field_raises(self) -> None:
        data = self._valid_data()
        data["unknown_field"] = "oops"
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_invalid_semver_raises(self) -> None:
        data = self._valid_data()
        data["agent_version"] = "not-semver"
        with pytest.raises(ValidationError, match="semver"):
            AgentSpec.model_validate(data)

    def test_zero_max_iterations_raises(self) -> None:
        data = self._valid_data()
        data["max_iterations"] = 0
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_negative_token_budget_raises(self) -> None:
        data = self._valid_data()
        data["token_budget"] = -1
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)

    def test_empty_tool_name_raises(self) -> None:
        data = self._valid_data()
        data["tool_whitelist"] = ["doc_search", ""]
        with pytest.raises(ValidationError):
            AgentSpec.model_validate(data)


class TestLoadSpecHelpers:
    """load_spec file-level error handling."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_spec(tmp_path / "does_not_exist.yaml")

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises((ValueError, ValidationError)):
            load_spec(p)

    def test_accepts_path_as_string(self) -> None:
        spec = load_spec(str(_REGDOC_YAML))
        assert spec.agent_name == "regdoc-qa"
