"""Pytest compatibility patches for Windows temp directory handling.

On this environment, Python 3.12 + Windows can produce inaccessible
directories when `Path.mkdir(mode=0o700)` is used. Pytest's tmpdir
implementation uses mode 0o700 by default, which breaks `tmp_path`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import _pytest.tmpdir as _tmpdir
from _pytest.pathlib import rm_rf


if os.name == "nt":
    _WINDOWS_TMP_MODE = 0o755

    def _getbasetemp_windows_safe(self: _tmpdir.TempPathFactory) -> Path:
        if self._basetemp is not None:
            return self._basetemp

        if self._given_basetemp is not None:
            basetemp = self._given_basetemp
            if basetemp.exists():
                rm_rf(basetemp)
            basetemp.mkdir(mode=_WINDOWS_TMP_MODE)
            basetemp = basetemp.resolve()
        else:
            # Avoid ACL-poisoned `%TEMP%\pytest-of-*` trees by using repo-local temp dirs.
            root = Path.cwd().joinpath("temp")
            root.mkdir(parents=True, exist_ok=True)
            basetemp = root.joinpath(f"pytest-{uuid.uuid4().hex[:10]}")
            basetemp.mkdir(mode=_WINDOWS_TMP_MODE)
            basetemp = basetemp.resolve()

        self._basetemp = basetemp
        self._trace("new basetemp", basetemp)
        return basetemp

    def _mktemp_windows_safe(
        self: _tmpdir.TempPathFactory, basename: str, numbered: bool = True
    ) -> Path:
        basename = self._ensure_relative_to_basetemp(basename)
        if not numbered:
            path = self.getbasetemp().joinpath(basename)
            path.mkdir(mode=_WINDOWS_TMP_MODE)
        else:
            path = _tmpdir.make_numbered_dir(
                root=self.getbasetemp(),
                prefix=basename,
                mode=_WINDOWS_TMP_MODE,
            )
            self._trace("mktemp", path)
        return path

    _tmpdir.TempPathFactory.getbasetemp = _getbasetemp_windows_safe
    _tmpdir.TempPathFactory.mktemp = _mktemp_windows_safe
