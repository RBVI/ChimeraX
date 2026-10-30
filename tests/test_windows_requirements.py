"""Windows dependency filtering keeps all explicit application requirements."""

from pathlib import Path
import re
import shlex
import subprocess
import tomllib


def test_windows_only_skips_cuda_plugin_transitive_dependencies():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = [dep for group in project["project"]["optional-dependencies"].values()
                    for dep in group]
    source = "\n".join(requirements) + "\n"
    makefile = (root / "prereqs/pips/Makefile").read_text()
    expressions = re.findall(r"\tsed (.*?) app_requirements.txt > app_requirements-(?:with|no)-deps.txt", makefile)
    assert len(expressions) == 2
    with_deps, no_deps = [subprocess.check_output(
        ["sed", *shlex.split(expression)], input=source, text=True).splitlines()
        for expression in expressions]
    assert len(no_deps) == 1
    assert no_deps[0].startswith("openmm-cuda-12==")
    assert any(dep.startswith("openmm==") for dep in with_deps)
    assert sorted(with_deps + no_deps) == sorted(requirements)
    assert "$(APP_PIP_INSTALL) --no-deps -r app_requirements-no-deps.txt" in makefile
    assert "PATCHES += openmm_metadata.patch" in makefile
    clean = makefile.split("\nclean:\n", 1)[1].split("\noutdated:", 1)[0]
    ignored = (root / ".gitignore").read_text().splitlines()
    for name in ("app_requirements-with-deps.txt", "app_requirements-no-deps.txt"):
        assert name in clean
        assert "prereqs/pips/" + name in ignored
