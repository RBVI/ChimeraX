"""SNFG symbols are drawn inside their structure (#20888) and track it (#20437)."""

import math

import numpy
import pytest


def _depiction(structure):
    return getattr(structure, "_snfg_depiction", None)


def _flush_changes(session):
    from chimerax.atomic import check_for_changes
    check_for_changes(session)


def _add_sugar(structure, number, name="NAG", offset=0.0):
    residue = structure.new_residue(name, "A", number)
    ring = []
    for i, atom_name in enumerate(("C1", "C2", "C3", "C4", "C5", "O5")):
        atom = structure.new_atom(atom_name, atom_name[0])
        residue.add_atom(atom)
        angle = i * math.pi / 3
        atom.coord = (offset + 1.5 * math.cos(angle), 1.5 * math.sin(angle), 0)
        ring.append(atom)
    for a1, a2 in zip(ring, ring[1:] + ring[:1]):
        structure.new_bond(a1, a2)
    return residue


def _structure(session, names=("NAG", "NAG")):
    from chimerax.atomic import AtomicStructure

    structure = AtomicStructure(session, auto_style=False, log_info=False)
    residues = [_add_sugar(structure, i + 1, name, 6.0 * i) for i, name in enumerate(names)]
    for r1, r2 in zip(residues, residues[1:]):
        structure.new_bond(r1.find_atom("C4"), r2.find_atom("C1"))
    session.models.add([structure])
    # Later changes are only reported per structure once its creation has been.
    _flush_changes(session)
    return structure


def _alpha(depiction):
    return set(depiction.shapes.vertex_colors[:, 3].tolist())


def test_symbols_are_drawings_of_the_structure(test_production_session):
    from chimerax.core.models import ADD_MODELS
    from chimerax.atomic import AtomicStructure
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    structure = AtomicStructure(session, auto_style=False, log_info=False)
    _add_sugar(structure, 1)
    # Reproduce the camera initialization still pending while a model opens.
    with session.triggers.block_trigger(ADD_MODELS):
        session.models.add([structure])
    assert session.models._initialize_camera

    assert snfg_command(session, structures=[structure], atoms=False) == 1

    assert session.models.list() == [structure]
    assert session.models._initialize_camera
    depiction = _depiction(structure)
    assert depiction.shapes in structure.child_drawings()
    assert depiction.connections in structure.child_drawings()
    assert not structure.atoms.displays.any()


def test_no_sugars_adds_nothing(test_production_session):
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    structure = _structure(session, names=("ALA",))
    drawings = list(structure.child_drawings())

    assert snfg_command(session, structures=[structure]) == 0
    assert _depiction(structure) is None
    assert structure.child_drawings() == drawings
    assert snfg_command(session, "hide", structures=[structure]) == 0


def test_hide_restores_atoms_and_removes_drawings(test_production_session):
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    structure = _structure(session)
    structure.atoms.displays = True
    drawings = list(structure.child_drawings())

    assert snfg_command(session, structures=[structure], atoms=False) == 2
    assert _depiction(structure).connections.triangles is not None
    assert snfg_command(session, "hide", structures=[structure]) == 2

    assert structure.atoms.displays.all()
    assert _depiction(structure) is None
    assert structure.child_drawings() == drawings


def test_deleted_residues_lose_their_symbols(test_production_session):
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    structure = _structure(session, names=("NAG", "NAG", "ALA"))
    snfg_command(session, structures=[structure], atoms=False)
    depiction = _depiction(structure)
    first, second, _ = structure.residues

    second.atoms.delete()
    _flush_changes(session)
    assert list(depiction.shapes.residues()) == [first]
    assert len(depiction.shapes._shapes) == 1
    assert depiction.connections.triangles is None

    first.atoms.delete()
    _flush_changes(session)
    assert _depiction(structure) is None
    assert depiction.shapes not in structure.child_drawings()
    assert depiction.connections not in structure.child_drawings()


def test_symbols_follow_moved_atoms(test_production_session):
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    structure = _structure(session, names=("NAG",))
    snfg_command(session, structures=[structure])
    depiction = _depiction(structure)
    residue = structure.residues[0]

    structure.atoms.coords = structure.atoms.coords + (0, 0, 10)
    _flush_changes(session)

    assert depiction.shapes.get_centroid(residue) == pytest.approx((0, 0, 10), abs=1e-5)
    assert depiction.shapes.vertices[:, 2].mean() == pytest.approx(10, abs=1e-3)


def test_glycan_selector(test_production_session):
    from chimerax.core.commands import run

    session = test_production_session
    structure = _structure(session, names=("NAG", "ALA"))
    run(session, "select glycan", log=False)
    assert set(structure.atoms[structure.atoms.selected].residues.names) == {"NAG"}
    assert structure.atoms.selected.sum() == 6


def test_scene_restore_and_fade(test_production_session):
    from chimerax.core.state import State
    from chimerax.snfg.snfg import snfg_command, _manager

    session = test_production_session
    structure = _structure(session)
    mgr = _manager(session)
    without = mgr.take_snapshot(session, State.SCENE)
    snfg_command(session, structures=[structure], atoms=False, size=2.0)
    small = mgr.take_snapshot(session, State.SCENE)
    snfg_command(session, structures=[structure], atoms=False, size=4.0)
    large = mgr.take_snapshot(session, State.SCENE)

    mgr.restore_scene(without)
    assert _depiction(structure) is None
    mgr.restore_scene(None)
    assert _depiction(structure) is None

    mgr.restore_scene(small)
    depiction = _depiction(structure)
    assert set(depiction.sizes.values()) == {2.0}
    assert depiction.connections.display

    mgr.interpolate_scene(small, large, 0.25)
    assert _depiction(structure) is depiction
    assert set(depiction.sizes.values()) == {2.5}
    assert depiction.shapes._shape_sizes == [2.5, 2.5]

    mgr.interpolate_scene(small, without, 0.25)
    assert _alpha(depiction) == {191}
    assert depiction.connections.color[3] == 191
    mgr.interpolate_scene(None, small, 0.5, switchover=True)
    assert _alpha(depiction) == {128}
    mgr.restore_scene(small)
    assert _alpha(depiction) == {255}


def test_scenes_include_snfg(test_production_session, monkeypatch):
    from chimerax.scenes.scene import Scene
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    monkeypatch.setattr(Scene, "take_thumbnail", lambda self: "")
    structure = _structure(session)
    structure.atoms.displays = True
    session.scenes.save_scene("plain")
    snfg_command(session, structures=[structure], atoms=False)
    session.scenes.save_scene("symbols")

    session.scenes.restore_scene("plain")
    assert _depiction(structure) is None
    assert structure.atoms.displays.all()

    session.scenes.restore_scene("symbols")
    assert len(_depiction(structure).sizes) == 2
    assert not structure.atoms.displays.any()

    session.scenes.interpolate_scenes("plain", "symbols", 0.25)
    assert _alpha(_depiction(structure)) == {64}


def test_session_round_trip(test_production_session, tmp_path):
    from chimerax.core.commands import run
    from chimerax.atomic import all_atomic_structures
    from chimerax.snfg.snfg import snfg_command

    session = test_production_session
    structure = _structure(session)
    structure.atoms.displays = True
    snfg_command(session, structures=[structure], atoms=False, size=3.0)
    path = tmp_path / "snfg.cxs"
    run(session, f"save {path}", log=False)
    run(session, f"open {path}", log=False)

    restored, = all_atomic_structures(session)
    assert restored is not structure
    assert session.models.list() == [restored]
    depiction = _depiction(restored)
    assert set(depiction.sizes.values()) == {3.0}
    assert depiction.connections.triangles is not None

    assert snfg_command(session, "hide") == 2
    assert restored.atoms.displays.all()


def test_old_session_model_becomes_drawings(test_production_session):
    from chimerax.core.models import Model
    from chimerax.core.state import State
    from chimerax.snfg import bundle_api

    session = test_production_session
    structure = _structure(session)
    residues = list(structure.residues)
    placeholder = Model("SNFG symbols", session)
    structure.add([placeholder])
    data = placeholder.take_snapshot(session, State.SESSION)
    session.models.close([placeholder])
    data.update({
        'structure': structure,
        'hidden_residues': residues,
        'shapes_drawing': {'shape_residues': residues, 'shape_sizes': [2.0, 3.0]},
    })

    model = bundle_api.get_class('SNFGModel').restore_snapshot(session, data)
    assert model in structure.child_models()
    session.triggers.activate_trigger("end restore session", session)

    assert session.models.list() == [structure]
    depiction = _depiction(structure)
    assert depiction.sizes == dict(zip(residues, (2.0, 3.0)))
    assert depiction.hidden_residues == set(residues)
    assert not depiction.show_atoms
    assert numpy.ptp(depiction.shapes.vertices[:, 0]) > 6
