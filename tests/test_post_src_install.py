"""Post-install configuration must not extend an earlier target's recipe."""

from pathlib import Path


def test_post_install_configuration_has_no_recipe_commands():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    configuration = makefile.split("POST_SRC_INSTALL =", 1)[1].split(
        ".PHONY: post-src-install", 1)[0]
    continued = False
    for line in configuration.splitlines():
        # Tab-indented variable continuations are valid; recipe commands here
        # would attach to the preceding target and run before bundles finish.
        assert continued or not line.startswith("\t"), line
        continued = line.rstrip().endswith("\\")


def test_pip_check_runs_only_in_its_post_install_target():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    command = "\t$(APP_PYTHON_EXE) -m pip check"
    assert makefile.splitlines().count(command) == 1
    assert "pip-check.post-install:\n" + command in makefile
