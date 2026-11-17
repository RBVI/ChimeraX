"""Bundle version bump lint, exercised against throwaway git repositories."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

XML = '<BundleInfo name="ChimeraX-Xml"\n    version="{}" package="chimerax.xml">\n</BundleInfo>\n'
DYNAMIC_TOML = (
    '[project]\nname = "ChimeraX-Dyn"\ndynamic = ["version"]\n'
    '[tool.setuptools.dynamic]\nversion = { attr = "src.__version__" }\n'
)
STATIC_TOML = '[project]\nname = "ChimeraX-Static"\nversion = "{}"\n'


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_module("utils/ci/check_bundle_versions.py")


class Repo:
    def __init__(self, path):
        self.path = path
        self.git("init", "-q", "-b", "develop")

    def git(self, *args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
             "-c", "commit.gpgsign=false", *args],
            cwd=self.path, check=True, capture_output=True,
        )

    def commit(self, files):
        for name, content in files.items():
            path = self.path / "src/bundles" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "change")

    def failures(self):
        return check.unbumped_bundles("base", "HEAD", cwd=self.path)


@pytest.fixture
def repo(tmp_path):
    repo = Repo(tmp_path)
    repo.commit({
        "xml/bundle_info.xml": XML.format("1.0"),
        "xml/src/__init__.py": "",
        "dyn/pyproject.toml": DYNAMIC_TOML,
        "dyn/src/__init__.py": '__version__ = "2.0"\n',
        "static/pyproject.toml": STATIC_TOML.format("3.0"),
        "static/src/__init__.py": "",
        "core/pyproject.toml": DYNAMIC_TOML,
        "core/src/__init__.py": '__version__ = "1.0"\n',
        "nested/Lib/bundle_info.xml": XML.format("1.0"),
        "nested/Lib/src/__init__.py": "",
        "xml/tests/bundles/fixture/bundle_info.xml": XML.format("0.1"),
    })
    repo.git("tag", "base")
    return repo


def test_no_changes_passes(repo):
    assert repo.failures() == {}


@pytest.mark.parametrize("bundle, version", [
    ("xml", "1.0"), ("dyn", "2.0"), ("static", "3.0"), ("nested/Lib", "1.0"),
])
def test_change_without_bump_fails(repo, bundle, version):
    repo.commit({f"{bundle}/src/cmd.py": "x = 1\n"})
    assert repo.failures() == {f"src/bundles/{bundle}": version}


@pytest.mark.parametrize("bundle, bump", [
    ("xml", {"xml/bundle_info.xml": XML.format("1.1")}),
    ("dyn", {"dyn/src/__init__.py": '__version__ = "2.1"\n'}),
    ("static", {"static/pyproject.toml": STATIC_TOML.format("3.1")}),
])
def test_one_bump_covers_several_commits(repo, bundle, bump):
    repo.commit({f"{bundle}/src/cmd.py": "x = 1\n"})
    repo.commit({f"{bundle}/src/cmd.py": "x = 2\n"})
    repo.commit(bump)
    assert repo.failures() == {}


def test_deleted_file_counts_as_change(repo):
    repo.git("rm", "-q", "src/bundles/xml/src/__init__.py")
    repo.git("commit", "-q", "-m", "delete")
    assert repo.failures() == {"src/bundles/xml": "1.0"}


def test_tests_and_fixture_bundles_are_ignored(repo):
    repo.commit({
        "xml/tests/test_cmd.py": "",
        "xml/tests/bundles/fixture/src/__init__.py": "",
    })
    assert repo.failures() == {}


def test_core_and_new_bundles_are_exempt(repo):
    repo.commit({
        "core/src/cmd.py": "",
        "new/bundle_info.xml": XML.format("0.1"),
        "new/src/__init__.py": "",
    })
    assert repo.failures() == {}


def test_base_is_the_merge_base(repo):
    repo.git("checkout", "-q", "-b", "topic")
    repo.commit({"dyn/src/__init__.py": '__version__ = "2.1"\n', "dyn/src/cmd.py": ""})
    repo.git("checkout", "-q", "develop")
    repo.commit({"xml/src/cmd.py": ""})
    assert check.unbumped_bundles("develop", "topic", cwd=repo.path) == {}


def test_unreadable_version_fails(repo):
    repo.commit({"static/pyproject.toml": '[project]\nname = "ChimeraX-Static"\n'})
    assert repo.failures() == {"src/bundles/static": None}


def test_main_exits_nonzero_on_missing_bump(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo.path)
    assert check.main(["--base", "base"]) == 0
    repo.commit({"xml/src/cmd.py": ""})
    assert check.main(["--base", "base"]) == 1
    assert "src/bundles/xml: changed since base" in capsys.readouterr().out


def test_every_real_bundle_version_is_readable():
    roots = check.bundle_configs("HEAD", cwd=ROOT)
    assert len(roots) > 150
    unreadable = [
        root for root, names in roots.items()
        if root not in check.EXEMPT_BUNDLES
        and check.read_version("HEAD", root, names, cwd=ROOT) is None
    ]
    assert unreadable == []
