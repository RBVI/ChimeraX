"""Only Windows needs to discover bundle directories for DLL search paths."""
import os
import sys
from types import SimpleNamespace

import pytest

from chimerax.core.toolshed.info import BundleInfo


@pytest.mark.parametrize('platform', ['darwin', 'linux'])
def test_other_platforms_do_not_resolve_unused_library_paths(monkeypatch, platform):
    monkeypatch.setattr(sys, 'platform', platform)
    def unexpected():
        pytest.fail('Resolved a DLL directory on a platform that does not use it')
    BundleInfo.update_library_path(SimpleNamespace(library_dir=unexpected))


@pytest.mark.parametrize('directory', [None, '/example/bundle/lib'])
def test_windows_still_adds_existing_bundle_library_directories(monkeypatch, directory):
    monkeypatch.setattr(sys, 'platform', 'win32')
    paths = []
    monkeypatch.setattr(os, 'add_dll_directory', paths.append, raising=False)
    BundleInfo.update_library_path(SimpleNamespace(library_dir=lambda: directory))
    assert paths == ([directory] if directory else [])
