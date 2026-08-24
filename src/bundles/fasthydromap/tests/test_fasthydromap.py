import importlib.util
import sys
import tomllib
import types
from pathlib import Path

import pytest


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SRC = BUNDLE_ROOT / "src"


def test_bundle_metadata_release_constraints():
    with open(BUNDLE_ROOT / "pyproject.toml", "rb") as metadata_file:
        metadata = tomllib.load(metadata_file)

    assert metadata["project"]["license"] == "MIT"
    assert "ChimeraX-Core ~=1.12" in metadata["project"]["dependencies"]
    assert not any(
        classifier.startswith("License ::")
        for classifier in metadata["tool"]["chimerax"]["classifiers"]
    )

    init_text = (BUNDLE_SRC / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "0.1.1"' in init_text


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

    command = cmd_module._fasthydromap_command(
        object(), Path("model.pdb"), Path("out"), quantity="pc1", install_location=None
    )
    assert command == ["/custom/fasthydromap", "predict", "model.pdb", "-o", "out", "--quantity", "pc1"]


def test_fasthydromap_command_uses_managed_install(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    monkeypatch.delenv("FASTHYDROMAP_EXE", raising=False)
    monkeypatch.setattr(cmd_module, "managed_fasthydromap_executable", lambda session, install_location=None: "/venv/bin/fasthydromap")

    command = cmd_module._fasthydromap_command(
        object(), Path("model.pdb"), Path("out"), quantity="fdewet", install_location="/venv"
    )
    assert command == ["/venv/bin/fasthydromap", "predict", "model.pdb", "-o", "out", "--quantity", "fdewet"]


def test_fasthydromap_command_requires_install(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    monkeypatch.delenv("FASTHYDROMAP_EXE", raising=False)
    monkeypatch.setattr(cmd_module, "managed_fasthydromap_executable", lambda session, install_location=None: None)

    with pytest.raises(cmd_module.UserError, match="Run 'fasthydromap install' first"):
        cmd_module._fasthydromap_command(
            object(), Path("model.pdb"), Path("out"), quantity="fdewet", install_location=None
        )


def test_fasthydromap_without_open_structure_shows_quick_start(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    atomic = types.ModuleType("chimerax.atomic")
    atomic.Residue = object
    atomic.all_atomic_structures = lambda session: []
    monkeypatch.setitem(sys.modules, "chimerax.atomic", atomic)

    with pytest.raises(cmd_module.UserError, match="open 1a1u") as error:
        cmd_module.fasthydromap(object())

    assert "fasthydromap #1" in str(error.value)
    assert "help fasthydromap" in str(error.value)


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


def test_read_single_structure_scores_pc_quantity(monkeypatch, tmp_path):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    csv_path = tmp_path / "pc_scores.csv"
    csv_path.write_text("residue,PC1\nA:1,7.1\n_:2,-1.2\n", encoding="utf-8")

    scores = cmd_module._read_single_structure_scores(csv_path, quantity="pc1")
    assert scores == {"A:1": 7.1, "_:2": -1.2}


def test_save_protein_only_pdb_excludes_nucleic_acids_and_ligands(
    monkeypatch, tmp_path
):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    deleted_atoms = []
    saved_models = []

    class FakeResidue:
        PT_AMINO = 1

        def __init__(self, polymer_type, atoms, name, number, chain_id="A"):
            self.polymer_type = polymer_type
            self.atoms = atoms
            self.name = name
            self.number = number
            self.chain_id = chain_id
            self.insertion_code = " "

    class FakeAtoms(list):
        def delete(self):
            deleted_atoms.extend(self)

    copy_model = types.SimpleNamespace(
        residues=[
            FakeResidue(
                FakeResidue.PT_AMINO,
                ["protein-N", "protein-CA"],
                "ALA",
                1,
            ),
            FakeResidue(2, ["dna-P", "dna-C1'"], "DC", 3, chain_id="P"),
            FakeResidue(0, ["zinc"], "ZN", 1301),
        ],
        delete=lambda: setattr(copy_model, "was_deleted", True),
        was_deleted=False,
    )
    structure = types.SimpleNamespace(
        name="mixed complex",
        residues=[
            FakeResidue(FakeResidue.PT_AMINO, ["original-protein"], "ALA", 1),
            FakeResidue(2, ["original-dna"], "DC", 3, chain_id="P"),
            FakeResidue(0, ["original-zinc"], "ZN", 1301),
        ],
        copy=lambda name: copy_model,
    )
    warnings = []
    session = types.SimpleNamespace(
        logger=types.SimpleNamespace(warning=warnings.append)
    )

    atomic = types.ModuleType("chimerax.atomic")
    atomic.Atoms = FakeAtoms
    pdb = types.ModuleType("chimerax.pdb")
    pdb.save_pdb = lambda session, path, models: saved_models.extend(models)
    monkeypatch.setitem(sys.modules, "chimerax.atomic", atomic)
    monkeypatch.setitem(sys.modules, "chimerax.pdb", pdb)

    cmd_module._save_protein_only_pdb(session, structure, tmp_path / "model.pdb")

    assert deleted_atoms == ["dna-P", "dna-C1'", "zinc"]
    assert saved_models == [copy_model]
    assert copy_model.was_deleted
    assert len(warnings) == 1
    assert "ignoring 2 non-protein residues" in warnings[0]
    assert "P:3 DC" in warnings[0]
    assert "A:1301 ZN" in warnings[0]


def test_prediction_warning_describes_histidine_charge_limitations(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")

    def residue(name, number):
        return types.SimpleNamespace(
            name=name,
            number=number,
            chain_id="A",
            insertion_code=" ",
        )

    warnings = []
    session = types.SimpleNamespace(
        logger=types.SimpleNamespace(warning=warnings.append)
    )
    cmd_module._warn_prediction_limitations(
        session,
        "model #1",
        protein_residues=[residue("HIS", 5), residue("HID", 9), residue("ALA", 10)],
        nonprotein_residues=[],
    )

    assert len(warnings) == 1
    assert "assumes neutral histidine chemistry" in warnings[0]
    assert "HID/HIE" in warnings[0]
    assert "HIP/HSP" in warnings[0]
    assert "A:5 HIS" in warnings[0]
    assert "A:9 HID" in warnings[0]


def test_save_protein_only_pdb_rejects_nonprotein_model(monkeypatch, tmp_path):
    cmd_module = _load_submodule(monkeypatch, "cmd")

    class FakeResidue:
        PT_AMINO = 1
        polymer_type = 2

    structure = types.SimpleNamespace(residues=[FakeResidue()])

    with pytest.raises(cmd_module.UserError, match="contains no protein residues"):
        cmd_module._save_protein_only_pdb(object(), structure, tmp_path / "model.pdb")


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
    colors = types.ModuleType("chimerax.core.colors")
    colors.BuiltinColors = {"gray": "gray-color"}
    monkeypatch.setitem(sys.modules, "chimerax.core.colors", colors)
    std_color = types.ModuleType("chimerax.std_commands.color")
    std_color.color_by_attr = lambda session, attr_name, **kwargs: calls.append(
        ("color_by_attr", attr_name, kwargs)
    )
    monkeypatch.setitem(sys.modules, "chimerax.std_commands.color", std_color)
    monkeypatch.setattr(cmd_module, "_resolve_palette", lambda palette: f"RESOLVED({palette})")

    structure = types.SimpleNamespace(atomspec="#1", atoms="atoms-object")
    cmd_module._color_structure(
        object(),
        structure,
        target="acs",
        show_atoms=False,
        attr_name="fasthydromap_fdewet",
        palette="^lipophilicity",
        color_range=(4.0, 6.5),
    )

    assert calls[-1] == (
        "color_by_attr",
        "r:fasthydromap_fdewet",
        {
            "atoms": "atoms-object",
            "target": "acs",
            "palette": "RESOLVED(^lipophilicity)",
            "range": (4.0, 6.5),
            "no_value_color": "gray-color",
            "log_info": False,
        },
    )
    assert "hide #1 bonds" not in calls


def test_color_structure_uses_palette_and_range_overrides(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    calls = []

    commands = types.ModuleType("chimerax.core.commands")
    commands.run = lambda session, command: calls.append(command)
    monkeypatch.setitem(sys.modules, "chimerax.core.commands", commands)
    colors = types.ModuleType("chimerax.core.colors")
    colors.BuiltinColors = {"gray": "gray-color"}
    monkeypatch.setitem(sys.modules, "chimerax.core.colors", colors)
    std_color = types.ModuleType("chimerax.std_commands.color")
    std_color.color_by_attr = lambda session, attr_name, **kwargs: calls.append(
        ("color_by_attr", attr_name, kwargs)
    )
    monkeypatch.setitem(sys.modules, "chimerax.std_commands.color", std_color)
    palette_object = object()
    monkeypatch.setattr(cmd_module, "_resolve_palette", lambda palette: f"RESOLVED({palette is palette_object})")

    structure = types.SimpleNamespace(atomspec="#1", atoms="atoms-object")
    cmd_module._color_structure(
        object(),
        structure,
        target="acs",
        show_atoms=True,
        attr_name="fasthydromap_pc1",
        palette=palette_object,
        color_range=(3.5, 7),
    )

    assert calls[-1] == (
        "color_by_attr",
        "r:fasthydromap_pc1",
        {
            "atoms": "atoms-object",
            "target": "acs",
            "palette": "RESOLVED(True)",
            "range": (3.5, 7),
            "no_value_color": "gray-color",
            "log_info": False,
        },
    )


def test_pc_color_specs_use_requested_palettes_and_ranges(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")
    assert cmd_module.QUANTITY_SPECS["pc1"]["palette"] == "red-white-blue"
    assert cmd_module.QUANTITY_SPECS["pc1"]["range"] == (-8.0, 8.0)
    assert cmd_module.QUANTITY_SPECS["pc2"]["palette"] == "cyanmaroon"
    assert cmd_module.QUANTITY_SPECS["pc2"]["range"] == (-2.0, 8.0)
    assert cmd_module.QUANTITY_SPECS["pc3"]["palette"] == "^lipophilicity"
    assert cmd_module.QUANTITY_SPECS["pc3"]["range"] == (-2.0, 2.0)


def test_resolve_palette_supports_builtin_and_reversed(monkeypatch):
    cmd_module = _load_submodule(monkeypatch, "cmd")

    class FakeMap:
        def __init__(self, name):
            self.name = name

        def reversed(self):
            return f"reversed-{self.name}"

    colors = types.ModuleType("chimerax.core.colors")
    colors.BuiltinColormaps = {
        "lipophilicity": FakeMap("lipophilicity"),
        "red-white-blue": FakeMap("red-white-blue"),
    }
    monkeypatch.setitem(sys.modules, "chimerax.core.colors", colors)

    assert cmd_module._resolve_palette("^lipophilicity") == "reversed-lipophilicity"
    assert cmd_module._resolve_palette("red-white-blue").name == "red-white-blue"


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

    assert install_module.DEFAULT_PACKAGE_SPEC == "fasthydromap>=0.1.4,<0.2"

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
    installer._package_spec = "fasthydromap>=0.1.3,<0.2"
    installer._torch_variant = "cpu"
    installer._wait = True

    installer._upgrade_packaging_tools()
    installer._install_fasthydromap_package()
    installer._install_torch()

    assert recorded == [
        ["/tmp/fhm/bin/python", "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"],
        [
            "/tmp/fhm/bin/python",
            "-m",
            "pip",
            "install",
            "fasthydromap>=0.1.3,<0.2",
        ],
        ["/tmp/fhm/bin/fasthydromap", "install-torch", "--variant", "cpu"],
    ]


def test_macos_app_translocation_detection(monkeypatch):
    install_module = _load_submodule(monkeypatch, "install")

    monkeypatch.setattr(sys, "platform", "darwin")
    assert install_module._is_macos_app_translocation(
        "/private/var/folders/example/AppTranslocation/UUID/d/ChimeraX.app/Contents/bin/python3.11"
    )
    assert not install_module._is_macos_app_translocation(
        "/Applications/ChimeraX-1.12.app/Contents/bin/python3.11"
    )

    monkeypatch.setattr(sys, "platform", "linux")
    assert not install_module._is_macos_app_translocation(
        "/private/var/folders/example/AppTranslocation/UUID/d/ChimeraX.app/Contents/bin/python3.11"
    )


def test_install_success_logs_clickable_quick_start(monkeypatch):
    install_module = _load_submodule(monkeypatch, "install")
    calls = []
    session = types.SimpleNamespace(
        logger=types.SimpleNamespace(
            info=lambda message, **kwargs: calls.append((message, kwargs))
        )
    )

    install_module._log_ready_message(session)

    message, kwargs = calls[0]
    assert "FastHydroMap is ready" in message
    assert 'cxcmd:open 1a1u' in message
    assert 'cxcmd:fasthydromap #1' in message
    assert 'cxcmd:help fasthydromap' in message
    assert kwargs == {"is_html": True}
