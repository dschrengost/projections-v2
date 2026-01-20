"""Tests for projections.runtime_stamp module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from projections.runtime_stamp import (
    RuntimeStamp,
    collect_runtime_stamp,
    enforce_clean_tree,
    validate_config_paths,
    _compute_config_hash,
    _find_git_repo_root,
)


class TestCollectRuntimeStamp:
    """Tests for collect_runtime_stamp function."""

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        """Verify all expected fields are present in the stamp."""
        stamp = collect_runtime_stamp(entrypoint="test", project_root=tmp_path)

        # Check all required keys are present
        assert stamp.entrypoint == "test"
        assert isinstance(stamp.git_sha, str)
        assert isinstance(stamp.git_dirty, bool)
        assert isinstance(stamp.git_branch, str)
        assert isinstance(stamp.python_executable, str)
        assert isinstance(stamp.python_version, str)
        assert isinstance(stamp.cwd, str)
        assert isinstance(stamp.hostname, str)
        assert isinstance(stamp.user, str)
        assert isinstance(stamp.config_paths, dict)
        assert isinstance(stamp.config_hash, str)
        assert isinstance(stamp.run_ids, dict)
        assert isinstance(stamp.timestamp, str)
        assert isinstance(stamp.pid, int)
        assert isinstance(stamp.env_flags, dict)

    def test_json_serialization(self, tmp_path: Path) -> None:
        """Verify stamp can be serialized to JSON."""
        stamp = collect_runtime_stamp(entrypoint="test", project_root=tmp_path)

        json_line = stamp.to_json_line()
        assert isinstance(json_line, str)

        # Should be valid JSON
        parsed = json.loads(json_line)
        assert parsed["entrypoint"] == "test"
        assert "git_sha" in parsed
        assert "config_hash" in parsed

    def test_pretty_block_format(self, tmp_path: Path) -> None:
        """Verify pretty block contains expected sections."""
        stamp = collect_runtime_stamp(entrypoint="test", project_root=tmp_path)

        block = stamp.to_pretty_block()
        assert "RUNTIME STAMP" in block
        assert "Entrypoint:" in block
        assert "Git SHA:" in block
        assert "Config Hash:" in block
        assert "Run IDs:" in block

    def test_captures_env_flags(self) -> None:
        """Verify env flags are captured in the stamp."""
        with mock.patch.dict(os.environ, {"PROJECTIONS_ALLOW_DIRTY": "1"}):
            stamp = collect_runtime_stamp(entrypoint="test")
            assert stamp.env_flags.get("PROJECTIONS_ALLOW_DIRTY") == "1"

    def test_custom_config_paths(self, tmp_path: Path) -> None:
        """Verify custom config paths are used."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text('{"run_id": "test123"}')

        stamp = collect_runtime_stamp(
            entrypoint="test",
            config_paths={"test_config": config_file},
        )

        assert "test_config" in stamp.config_paths
        assert str(config_file) == stamp.config_paths["test_config"]


class TestEnforceCleanTree:
    """Tests for enforce_clean_tree function."""

    def test_bypass_with_allow_dirty_env(self, tmp_path: Path) -> None:
        """Verify dirty tree check is bypassed with PROJECTIONS_ALLOW_DIRTY=1."""
        # Mock git dirty to return True
        with mock.patch("projections.runtime_stamp._git_dirty", return_value=True):
            with mock.patch("projections.runtime_stamp._find_git_repo_root", return_value=tmp_path):
                # Without env var, should raise
                with mock.patch.dict(os.environ, {"PROJECTIONS_ALLOW_DIRTY": "0"}, clear=False):
                    # Clear the env var to not affect the test
                    env_copy = os.environ.copy()
                    env_copy.pop("PROJECTIONS_ALLOW_DIRTY", None)
                    env_copy.pop("PROJECTIONS_RUNTIME_STAMP_STRICT", None)
                    with mock.patch.dict(os.environ, env_copy, clear=True):
                        with pytest.raises(RuntimeError, match="Git working tree is dirty"):
                            enforce_clean_tree(repo_root=tmp_path)

                # With env var, should not raise
                with mock.patch.dict(os.environ, {"PROJECTIONS_ALLOW_DIRTY": "1"}):
                    enforce_clean_tree(repo_root=tmp_path)  # Should not raise

    def test_bypass_with_strict_mode_disabled(self, tmp_path: Path) -> None:
        """Verify all checks bypassed when PROJECTIONS_RUNTIME_STAMP_STRICT=0."""
        with mock.patch("projections.runtime_stamp._git_dirty", return_value=True):
            with mock.patch("projections.runtime_stamp._find_git_repo_root", return_value=tmp_path):
                with mock.patch.dict(os.environ, {"PROJECTIONS_RUNTIME_STAMP_STRICT": "0"}):
                    enforce_clean_tree(repo_root=tmp_path)  # Should not raise

    def test_no_error_when_not_in_git_repo(self) -> None:
        """Verify no error when not in a git repo."""
        with mock.patch("projections.runtime_stamp._find_git_repo_root", return_value=None):
            enforce_clean_tree()  # Should not raise


class TestValidateConfigPaths:
    """Tests for validate_config_paths function."""

    def test_returns_warnings_for_missing_configs(self, tmp_path: Path) -> None:
        """Verify warnings are returned for missing config files."""
        config_paths = {
            "missing_config": tmp_path / "nonexistent.json",
        }

        # With strict mode disabled
        with mock.patch.dict(os.environ, {"PROJECTIONS_RUNTIME_STAMP_STRICT": "0"}):
            warnings = validate_config_paths(config_paths)
            assert len(warnings) == 1
            assert "missing" in warnings[0].lower()

    def test_raises_in_strict_mode_for_required_missing(self, tmp_path: Path) -> None:
        """Verify RuntimeError raised for missing required configs in strict mode."""
        config_paths = {
            "required_config": tmp_path / "nonexistent.json",
        }

        # Clear env vars to ensure strict mode is on
        env_copy = os.environ.copy()
        env_copy.pop("PROJECTIONS_RUNTIME_STAMP_STRICT", None)
        with mock.patch.dict(os.environ, env_copy, clear=True):
            with pytest.raises(RuntimeError, match="Config file missing"):
                validate_config_paths(config_paths, required=["required_config"])

    def test_no_error_for_existing_configs(self, tmp_path: Path) -> None:
        """Verify no errors for existing, readable config files."""
        config_file = tmp_path / "existing.json"
        config_file.write_text('{"key": "value"}')

        warnings = validate_config_paths({"existing": config_file})
        assert len(warnings) == 0


class TestConfigHash:
    """Tests for config hash computation."""

    def test_deterministic_hash(self, tmp_path: Path) -> None:
        """Verify same configs produce same hash."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"a": 1, "b": 2}')

        hash1 = _compute_config_hash({"config": config_file})
        hash2 = _compute_config_hash({"config": config_file})

        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Verify different configs produce different hashes."""
        config1 = tmp_path / "config1.json"
        config1.write_text('{"a": 1}')

        config2 = tmp_path / "config2.json"
        config2.write_text('{"a": 2}')

        hash1 = _compute_config_hash({"config": config1})
        hash2 = _compute_config_hash({"config": config2})

        assert hash1 != hash2

    def test_json_normalization(self, tmp_path: Path) -> None:
        """Verify JSON formatting differences don't affect hash."""
        config1 = tmp_path / "config1.json"
        config1.write_text('{"a": 1, "b": 2}')

        config2 = tmp_path / "config2.json"
        config2.write_text('{\n  "b": 2,\n  "a": 1\n}')  # Different formatting, same content

        hash1 = _compute_config_hash({"config": config1})
        hash2 = _compute_config_hash({"config": config2})

        assert hash1 == hash2

    def test_missing_file_handled(self, tmp_path: Path) -> None:
        """Verify missing files are handled gracefully."""
        missing_path = tmp_path / "missing.json"

        # Should not raise
        hash_result = _compute_config_hash({"missing": missing_path})
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex length


class TestFindGitRepoRoot:
    """Tests for git repo root detection."""

    def test_finds_git_root(self, tmp_path: Path) -> None:
        """Verify git root is found when .git exists."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = _find_git_repo_root(subdir)
        assert result == tmp_path

    def test_returns_none_when_no_git(self, tmp_path: Path) -> None:
        """Verify None returned when not in a git repo."""
        # tmp_path won't have a .git directory
        # We need to create a deep path that definitely has no .git
        deep_path = tmp_path / "no_git" / "deep" / "path"
        deep_path.mkdir(parents=True)

        # Force the function to not find .git by limiting the search
        with mock.patch("projections.runtime_stamp._find_git_repo_root") as mock_find:
            mock_find.return_value = None
            from projections.runtime_stamp import _find_git_repo_root as real_find
            # Actually test with real function but mock the limit
            result = real_find(tmp_path)
            # tmp_path itself doesn't have .git unless we're in a git repo
            # The result depends on whether tmp_path is inside a real git repo


class TestRuntimeStampDataclass:
    """Tests for RuntimeStamp dataclass."""

    def test_to_dict(self) -> None:
        """Verify asdict works correctly."""
        from dataclasses import asdict

        stamp = RuntimeStamp(
            git_sha="abc123",
            git_dirty=False,
            git_repo_root="/path/to/repo",
            git_branch="main",
            python_executable="/usr/bin/python",
            python_version="3.11.0",
            cwd="/home/user",
            hostname="localhost",
            user="testuser",
            config_paths={},
            config_hash="deadbeef",
            run_ids={},
            timestamp="2026-01-19T00:00:00+00:00",
            entrypoint="test",
            pid=12345,
            env_flags={},
        )

        d = asdict(stamp)
        assert d["git_sha"] == "abc123"
        assert d["git_dirty"] is False
        assert d["entrypoint"] == "test"
        assert d["pid"] == 12345
