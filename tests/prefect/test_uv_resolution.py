"""Tests for uv path resolution in live_nba_pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestUvBin:
    """Tests for _uv_bin() path resolution."""

    def test_uv_bin_uses_env_var(self, tmp_path):
        """When UV_BIN is set and exists, should use it."""
        fake_uv = tmp_path / "fake_uv"
        fake_uv.touch()

        with patch.dict(os.environ, {"UV_BIN": str(fake_uv)}):
            from prefect_flows.live_nba_pipeline import _uv_bin

            result = _uv_bin()
            assert result == str(fake_uv)

    def test_uv_bin_raises_if_env_var_missing_file(self, tmp_path):
        """When UV_BIN is set but file doesn't exist, should raise."""
        fake_path = str(tmp_path / "nonexistent_uv")

        with patch.dict(os.environ, {"UV_BIN": fake_path}):
            from prefect_flows.live_nba_pipeline import _uv_bin

            with pytest.raises(FileNotFoundError) as exc_info:
                _uv_bin()
            assert "UV_BIN" in str(exc_info.value)
            assert "does not exist" in str(exc_info.value)

    def test_uv_bin_uses_default_path(self, tmp_path):
        """When no env var but default path exists, should use it."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove UV_BIN if set
            os.environ.pop("UV_BIN", None)

            from prefect_flows.live_nba_pipeline import _DEFAULT_UV_PATH, _uv_bin

            with patch.object(Path, "exists") as mock_exists:
                mock_exists.return_value = True
                # Mock _DEFAULT_UV_PATH.exists() to return True
                with patch(
                    "prefect_flows.live_nba_pipeline._DEFAULT_UV_PATH",
                    tmp_path / "uv",
                ):
                    (tmp_path / "uv").touch()
                    # Re-import to get fresh function
                    import importlib
                    import prefect_flows.live_nba_pipeline as module

                    original_default = module._DEFAULT_UV_PATH
                    module._DEFAULT_UV_PATH = tmp_path / "uv"
                    try:
                        result = module._uv_bin()
                        assert result == str(tmp_path / "uv")
                    finally:
                        module._DEFAULT_UV_PATH = original_default

    def test_uv_bin_uses_shutil_which(self):
        """When no env var and no default path, should try shutil.which."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UV_BIN", None)

            import prefect_flows.live_nba_pipeline as module

            original_default = module._DEFAULT_UV_PATH
            # Set default to a non-existent path
            module._DEFAULT_UV_PATH = Path("/nonexistent/path/uv")

            with patch("shutil.which", return_value="/usr/local/bin/uv"):
                try:
                    result = module._uv_bin()
                    assert result == "/usr/local/bin/uv"
                finally:
                    module._DEFAULT_UV_PATH = original_default

    def test_uv_bin_raises_helpful_error(self):
        """When uv not found anywhere, should raise with instructions."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UV_BIN", None)

            import prefect_flows.live_nba_pipeline as module

            original_default = module._DEFAULT_UV_PATH
            module._DEFAULT_UV_PATH = Path("/nonexistent/path/uv")

            with patch("shutil.which", return_value=None):
                try:
                    with pytest.raises(FileNotFoundError) as exc_info:
                        module._uv_bin()
                    error_msg = str(exc_info.value)
                    assert "UV_BIN" in error_msg
                    assert "install uv" in error_msg.lower()
                finally:
                    module._DEFAULT_UV_PATH = original_default


class TestRunPythonModule:
    """Tests for _run_python_module() using resolved uv path."""

    def test_run_python_module_uses_resolved_uv(self, tmp_path):
        """Command should start with resolved uv path."""
        fake_uv = tmp_path / "fake_uv"
        fake_uv.touch()

        with patch.dict(os.environ, {"UV_BIN": str(fake_uv)}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="", stderr=""
                )

                from prefect_flows.live_nba_pipeline import _run_python_module

                _run_python_module(
                    "test.module",
                    ["--arg1", "value1"],
                    data_root=tmp_path,
                    timeout_s=60,
                )

                # Verify subprocess.run was called with resolved uv path
                call_args = mock_run.call_args
                cmd = call_args[0][0]
                assert cmd[0] == str(fake_uv)
                assert cmd[1:4] == ["run", "python", "-m"]
                assert cmd[4] == "test.module"
                assert "--arg1" in cmd
                assert "value1" in cmd
