import os

import pytest

from chimerax.bundle_builder import xml_to_toml


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
        ("pdb.xml", "pdb.toml"),
        ("pdb_lib.xml", "pdb_lib.toml"),
        ("realsense.xml", "realsense.toml"),
        ("segment.xml", "segment.toml"),
        ("stl.xml", "stl.toml"),
        ("surface.xml", "surface.toml"),
        ("webcam.xml", "webcam.toml"),
    ],
)
def test_xml_to_toml(xml, toml):
    xml_file = os.path.join(os.path.dirname(__file__), "xml", xml)
    toml_file = os.path.join(os.path.dirname(__file__), "toml", toml)
    with open(toml_file, "r") as f:
        expected = f.read().strip()
        assert xml_to_toml(xml_file).strip() == str(expected)
