import os

import pytest

from chimerax.bundle_builder import xml_to_toml, __version__
import importlib.metadata

bb_toml = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
with open(bb_toml, "r") as f:
    for line in f:
        if "numpy" in line:
            break
    numpy_version = line.split("==")[1][:-3].strip().rstrip()


@pytest.mark.parametrize(
    "xml, toml",
    [
        ("alignment_algs.xml", "alignment_algs.toml"),
        ("arrays.xml", "arrays.toml"),
        ("atom_search.xml", "atom_search.toml"),
        ("atomic.xml", "atomic.toml"),
        ("atomic_lib.xml", "atomic_lib.toml"),
        ("chem_group.xml", "chem_group.toml"),
        ("connect_structure.xml", "connect_structure.toml"),
        ("coulombic.xml", "coulombic.toml"),
        ("dssp.xml", "dssp.toml"),
        ("geometry.xml", "geometry.toml"),
        ("graphics.xml", "graphics.toml"),
        ("map.xml", "map.toml"),
        ("mask.xml", "mask.toml"),
        ("mlp.xml", "mlp.toml"),
        ("mmcif.xml", "mmcif.toml"),
        ("mmtf.xml", "mmtf.toml"),
        ("morph.xml", "morph.toml"),
        ("nucleotides.xml", "nucleotides.toml"),
        ("pdb.xml", "pdb.toml"),
        ("pdb_lib.xml", "pdb_lib.toml"),
        ("preset_mgr.xml", "preset_mgr.toml"),
        ("realsense.xml", "realsense.toml"),
        ("segment.xml", "segment.toml"),
        ("shortcuts.xml", "shortcuts.toml"),
        ("struct_measure.xml", "struct_measure.toml"),
        ("stl.xml", "stl.toml"),
        ("surface.xml", "surface.toml"),
        ("webcam.xml", "webcam.toml"),
    ],
)
def test_xml_to_toml(xml, toml):
    xml_file = os.path.join(os.path.dirname(__file__), "xml", xml)
    toml_file = os.path.join(os.path.dirname(__file__), "toml", toml)
    print(numpy_version)
    with open(toml_file, "r") as f:
        expected = f.read().strip()
        expected = expected.replace("CURRENT_NUMPY_VERSION", numpy_version)
        expected = expected.replace("CURRENT_BUNDLE_BUILDER_VERSION", __version__)
        assert xml_to_toml(xml_file).strip() == str(expected)
