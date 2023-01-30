#!/bin/python3
# vi:set expandtab shiftwidth=4:
#
# Try to import every ChimeraX package and module
# except for the blacklisted ones.
#
import pkgutil
import importlib
import chimerax
import pytest

blacklist = set([
    "chimerax.add_charge.process_lib",  # creates data.py
    "chimerax.alignment_algs.libalign_algs",  # non-importable dynamic lib
    "chimerax.alphafold.alphafold_predict_colab",    # IPython notebook with syntax magic
    "chimerax.alphafold.alphafold21_predict_colab",  # IPython notebook with syntax magic
    "chimerax.alphafold.alphafold22_predict_colab",  # IPython notebook with syntax magic
    "chimerax.alphafold.colabfold_predict",          # IPython notebook with syntax magic
    "chimerax.alphafold.fix_seq_titles",	# Alphafold database processing script.
    "chimerax.atomic.libmolc",  # non-importable dynamic lib
    "chimerax.atomic.md_crds.dcd.MDToolsMarch97.md_tests",  # test code
    "chimerax.build_structure.process",  # processes Chimera fragment files
    "chimerax.coulombic.create_data",  # creates data.py
    "chimerax.dicom.scan_dicoms",  # development script
    "chimerax.map.data.memoryuse", # unported code
    "chimerax.map.filter.square",  # unported code
    "chimerax.map.series.align",   # unported code
    "chimerax.modeller.script_head",  # fragment of a Modeller script
    "chimerax.modeller.script_tail",  # fragment of a Modeller script
    "chimerax.remote_control.run",    # imports Python2 xmlrpclib
    "chimerax.segger.ChimeraExtension",  # unported segger features
    "chimerax.segger.Mesh",
    "chimerax.segger.extract_region_dialog",
    "chimerax.segger.imageviewer",
    "chimerax.segger.iseg_dialog",
    "chimerax.segger.modelz",
    "chimerax.segger.promod_dialog",
    "chimerax.segger.rseg_dialog",
    "chimerax.segger.segloop_dialog",
    "chimerax.structcomp",  # ChimeraX command script
    "chimerax.surface.geodesic", # development script
    "chimerax.webcam.camera",		# Uses QVideoSink only in Qt6
    "chimerax.webcam.camera_qt5",	# Uses QAbstractVideoSurface only in Qt5
    "chimerax.addh.tool",
    "chimerax.alignment_algs.options",
    "chimerax.alignment_headers.conservation",
    "chimerax.alignment_headers.rmsd",
    "chimerax.alphafold.pae",
])

@pytest.mark.skip(reason="Still need to decide what to do about e.g app_dirs, tools")
def test_chimerax_imports():
    for info in pkgutil.walk_packages(chimerax.__path__, prefix=chimerax.__name__ + '.'):
        module_finder, name, is_pkg = info
        if name.endswith("tool") or name in blacklist:
            continue
        m = importlib.import_module(name)
