import os
import tempfile

import pytest


# Minimal pyproject.toml content for testing
MINIMAL_PYPROJECT_TOML = """
[build-system]
requires = ["setuptools", "ChimeraX-BundleBuilder"]
build-backend = "chimerax.bundle_builder.cx_pep517"

[project]
name = "ChimeraX-TestBundle"
license = { text = "Free for non-commercial use" }
authors = [{name = "Test", email = "test@test.com"}]
description = "Test bundle"
dependencies = ["ChimeraX-Core ~=1.0"]
dynamic = ["classifiers", "requires-python", "version"]

[project.urls]
Home = "https://example.com"

[tool.chimerax]
min-session-version = 1
max-session-version = 1
categories = ["General"]
classifiers = ["Development Status :: 2 - Pre-Alpha"]
"""

MINIMAL_INIT_PY = '''
__version__ = "1.0.0"
'''


class MockLogger:
    """Mock logger for testing."""
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


@pytest.fixture
def temp_bundle():
    """Create a temporary bundle directory with minimal pyproject.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = os.path.join(tmpdir, "test_bundle")
        src_dir = os.path.join(bundle_dir, "src")
        os.makedirs(src_dir)

        # Write pyproject.toml
        toml_path = os.path.join(bundle_dir, "pyproject.toml")
        with open(toml_path, "w") as f:
            f.write(MINIMAL_PYPROJECT_TOML)

        # Write src/__init__.py with version
        init_path = os.path.join(src_dir, "__init__.py")
        with open(init_path, "w") as f:
            f.write(MINIMAL_INIT_PY)

        yield bundle_dir, toml_path


def test_from_path_uses_bundle_path_not_cwd(temp_bundle):
    """Test that Bundle.from_path uses the specified bundle path, not os.getcwd().

    This is a regression test for GitHub issue #212 where editable installs
    would use the wrong path because Bundle.__init__ was using os.getcwd()
    instead of the actual bundle path passed to from_path.
    """
    from chimerax.bundle_builder import BundleBuilderTOML

    bundle_dir, _ = temp_bundle
    logger = MockLogger()

    # Save original cwd
    original_cwd = os.getcwd()

    try:
        # Change to a different directory (temp directory root)
        different_dir = os.path.dirname(bundle_dir)
        os.chdir(different_dir)

        # Create bundle from path - this should use bundle_dir, not cwd
        bundle = BundleBuilderTOML.from_path(logger, bundle_dir)

        # The bundle path should be the bundle_dir, not the current working directory
        # Use realpath to handle macOS /var -> /private/var symlink
        assert os.path.realpath(bundle.path) == os.path.realpath(bundle_dir), (
            f"Bundle.path should be '{bundle_dir}' but got '{bundle.path}'. "
            f"Current working directory was '{os.getcwd()}'"
        )
    finally:
        os.chdir(original_cwd)


def test_from_toml_file_uses_bundle_path_not_cwd(temp_bundle):
    """Test that Bundle.from_toml_file uses the toml file's directory, not os.getcwd()."""
    from chimerax.bundle_builder import BundleBuilderTOML

    bundle_dir, toml_path = temp_bundle
    logger = MockLogger()

    # Save original cwd
    original_cwd = os.getcwd()

    try:
        # Change to a different directory
        different_dir = os.path.dirname(bundle_dir)
        os.chdir(different_dir)

        # Create bundle from toml file path
        bundle = BundleBuilderTOML.from_toml_file(logger, toml_path)

        # The bundle path should be the directory containing pyproject.toml
        # Use realpath to handle macOS /var -> /private/var symlink
        assert os.path.realpath(bundle.path) == os.path.realpath(bundle_dir), (
            f"Bundle.path should be '{bundle_dir}' but got '{bundle.path}'. "
            f"Current working directory was '{os.getcwd()}'"
        )
    finally:
        os.chdir(original_cwd)


def test_bundle_path_is_absolute(temp_bundle):
    """Test that Bundle.path is always an absolute path."""
    from chimerax.bundle_builder import BundleBuilderTOML

    bundle_dir, _ = temp_bundle
    logger = MockLogger()

    # Save original cwd
    original_cwd = os.getcwd()

    try:
        # Change to parent of bundle dir
        parent_dir = os.path.dirname(bundle_dir)
        os.chdir(parent_dir)

        # Use relative path to create bundle
        rel_path = os.path.basename(bundle_dir)
        bundle = BundleBuilderTOML.from_path(logger, rel_path)

        # The bundle path should be absolute
        assert os.path.isabs(bundle.path), (
            f"Bundle.path should be absolute but got '{bundle.path}'"
        )
        # Use realpath to handle macOS /var -> /private/var symlink
        assert os.path.realpath(bundle.path) == os.path.realpath(bundle_dir)
    finally:
        os.chdir(original_cwd)
