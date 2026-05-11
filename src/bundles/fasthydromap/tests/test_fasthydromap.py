import importlib.util
import sys
import types
from pathlib import Path

import pytest


BUNDLE_SRC = Path(__file__).resolve().parents[1] / "src"


def _stub_chimerax_modules(monkeypatch):
    for name in list(sys.modules):
        if name == "chimerax" or name.startswith("chimerax."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    chimerax = types.ModuleType("chimerax")
    chimerax.__path__ = []

    core = types.ModuleType("chimerax.core")
    core.__path__ = []

    errors = types.ModuleType("chimerax.core.errors")

    class UserError(Exception):
        pass

    errors.UserError = UserError

    settings = types.ModuleType("chimerax.core.settings")

    class Settings:
        def __init__(self, *args, **kwargs):
            pass

        def save(self):
            pass

    settings.Settings = Settings

    toolshed = types.ModuleType("chimerax.core.toolshed")

    class BundleAPI:
        pass

    toolshed.BundleAPI = BundleAPI

    monkeypatch.setitem(sys.modules, "chimerax", chimerax)
    monkeypatch.setitem(sys.modules, "chimerax.core", core)
    monkeypatch.setitem(sys.modules, "chimerax.core.errors", errors)
    monkeypatch.setitem(sys.modules, "chimerax.core.settings", settings)
    monkeypatch.setitem(sys.modules, "chimerax.core.toolshed", toolshed)
    return UserError


def _load_bundle_package(monkeypatch):
    _stub_chimerax_modules(monkeypatch)
    for name in ["fhm_bundle", "fhm_bundle.cmd", "fhm_bundle.install", "fhm_bundle.settings"]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    spec = importlib.util.spec_from_file_location(
        "fhm_bundle",
        BUNDLE_SRC / "__init__.py",
        submodule_search_locations=[str(BUNDLE_SRC)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fhm_bundle"] = module
    spec.loader.exec_module(module)
    return module


def _load_submodule(monkeypatch, submodule_name):
    _load_bundle_package(monkeypatch)
    fullname = f"fhm_bundle.{submodule_name}"
    path = BUNDLE_SRC / f"{submodule_name}.py"
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def test_register_command_dispatches(monkeypatch):
    module = _load_bundle_package(monkeypatch)
    calls = []

    cmd_module = types.ModuleType("fhm_bundle.cmd")
    cmd_module.register_fasthydromap_command = lambda logger: calls.append(("fasthydromap", logger))
    install_module = types.ModuleType("fhm_bundle.install")
    install_module.register_fasthydromap_install_command = lambda logger: calls.append(
        ("fasthydromap install", logger)
    )
    monkeypatch.setitem(sys.modules, "fhm_bundle.cmd", cmd_module)
    monkeypatch.setitem(sys.modules, "fhm_bundle.install", install_module)

    module.bundle_api.register_command("fasthydromap", "log1")
    module.bundle_api.register_command("fasthydromap install", "log2")

    assert calls == [("fasthydromap", "log1"), ("fasthydromap install", "log2")]


def test_settings_only_persist_install_location(monkeypatch):
    settings_module = _load_submodule(monkeypatch, "settings")
    assert settings_module._FastHydroMapSettings.EXPLICIT_SAVE == {
        "fasthydromap_install_location": "",
    }


def test_managed_executable_uses_saved_install_location(monkeypatch):
    install_module = _load_submodule(monkeypatch, "install")

    saved = types.SimpleNamespace(fasthydromap_install_location="/tmp/fhm", save=lambda: None)
    monkeypatch.setattr(install_module, "_fasthydromap_settings", lambda session: saved)
    monkeypatch.setattr(install_module, "exists", lambda path: path == "/tmp/fhm/bin/fasthydromap")

    exe = install_module.managed_fasthydromap_executable(object())
    assert exe == "/tmp/fhm/bin/fasthydromap"


def test_fasthydromap_command_prefers_env_override(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    monkeypatch.setenv("FASTHYDROMAP_EXE", "/custom/fasthydromap")

    command = cmd_module._fasthydromap_command(object(), Path("model.pdb"), Path("out"), install_location=None)
    assert command == ["/custom/fasthydromap", "predict", "model.pdb", "-o", "out"]


def test_fasthydromap_command_uses_managed_install(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    monkeypatch.delenv("FASTHYDROMAP_EXE", raising=False)
    monkeypatch.setattr(cmd_module, "managed_fasthydromap_executable", lambda session, install_location=None: "/venv/bin/fasthydromap")

    command = cmd_module._fasthydromap_command(object(), Path("model.pdb"), Path("out"), install_location="/venv")
    assert command == ["/venv/bin/fasthydromap", "predict", "model.pdb", "-o", "out"]


def test_fasthydromap_command_requires_install(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    monkeypatch.delenv("FASTHYDROMAP_EXE", raising=False)
    monkeypatch.setattr(cmd_module, "managed_fasthydromap_executable", lambda session, install_location=None: None)

    with pytest.raises(cmd_module.UserError, match="Run 'fasthydromap install' first"):
        cmd_module._fasthydromap_command(object(), Path("model.pdb"), Path("out"), install_location=None)


def test_read_single_structure_scores(monkeypatch, tmp_path):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("frame,A:1,_:2,3A\n0,4.1,5.2,6.3\n", encoding="utf-8")

    scores = cmd_module._read_single_structure_scores(csv_path)
    assert scores == {"A:1": 4.1, "_:2": 5.2, "3A": 6.3}


def test_read_single_structure_scores_current_long_format(monkeypatch, tmp_path):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    csv_path = tmp_path / "scores_long.csv"
    csv_path.write_text("residue,Fdewet,Fdewet_intrinsic\nA:1,4.1,2.0\n_:2,5.2,2.5\n3A,6.3,3.1\n", encoding="utf-8")

    scores = cmd_module._read_single_structure_scores(csv_path)
    assert scores == {"A:1": 4.1, "_:2": 5.2, "3A": 6.3}


@pytest.mark.parametrize(
    "label,expected",
    [
        ("A:42", ("A", 42, "")),
        ("_:7B", (" ", 7, "B")),
        ("12A", (None, 12, "A")),
    ],
)
def test_parse_residue_label(monkeypatch, label, expected):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    assert cmd_module._parse_residue_label(label) == expected


def test_color_structure_uses_fixed_palette_and_range(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    calls = []

    commands = types.ModuleType("chimerax.core.commands")
    commands.run = lambda session, command: calls.append(command)
    monkeypatch.setitem(sys.modules, "chimerax.core.commands", commands)

    structure = types.SimpleNamespace(atomspec="#1")
    cmd_module._color_structure(object(), structure, target="acs", show_atoms=False)

    assert calls[-1] == (
        "color byattribute r:fasthydromap_score #1 target acs "
        "palette ^lipophilicity range 4,6.5 novalue gray"
    )
    assert "hide #1 bonds" not in calls


def test_find_structure_residue_handles_chainless_and_insertion_code(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")

    direct_residue = object()
    chainless_residue = object()
    find_calls = []

    structure = types.SimpleNamespace(
        find_residue=lambda chain_id, resid, insert=" ": find_calls.append((chain_id, resid, insert)) or direct_residue
    )
    no_chain_lookup = {"12A": chainless_residue}

    assert cmd_module._find_structure_residue(structure, "A:7B", no_chain_lookup) is direct_residue
    assert find_calls == [("A", 7, "B")]
    assert cmd_module._find_structure_residue(structure, "12A", no_chain_lookup) is chainless_residue


def test_install_command_uses_pypi_and_cli_torch(monkeypatch):
    install_module = _load_submodule(monkeypatch, "install")
    recorded = []

    monkeypatch.setattr(
        install_module,
        "log_subprocess_output",
        lambda session, command, finished_callback, wait=False: recorded.append(command),
    )

    session = types.SimpleNamespace(
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
        ui=types.SimpleNamespace(is_gui=False),
    )
    installer = install_module.InstallFastHydroMap.__new__(install_module.InstallFastHydroMap)
    installer._session = session
    installer._directory = "/tmp/fhm"
    installer._package_spec = "fasthydromap"
    installer._torch_variant = "cpu"
    installer._wait = True

    installer._upgrade_packaging_tools()
    installer._install_fasthydromap_package()
    installer._install_torch()

    assert recorded == [
        ["/tmp/fhm/bin/python", "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"],
        ["/tmp/fhm/bin/python", "-m", "pip", "install", "fasthydromap"],
        ["/tmp/fhm/bin/fasthydromap", "install-torch", "--variant", "cpu"],
    ]
