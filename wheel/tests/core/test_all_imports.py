#!/bin/python3
# vi:set expandtab shiftwidth=4:
#
# Try to import every ChimeraX package and module
# except for the blacklisted ones.
#
import pkgutil
import importlib
import chimerax

module_blacklist = set([
    "chimerax.amber_info" # needs app_bin_dir
    , "chimerax.webservices" # needs cxservices
    , "chimerax.blastprotein" # needs webservices
    , "chimerax.build_structure" # needs Qt
    , "chimerax.dicom" # tries to import its .ui submodule in __init__
    , "chimerax.structcomp"  # ChimeraX command script
    # Not going in the library, but part of test suite for GUI ChimeraX
    , "chimerax.ui" # tries to import Qt
])

fine_blacklist = set([
    "chimerax.amber_info" # needs app_bin_dir
    , "chimerax.alignment_algs.options" # imports chimerax.ui
    , "chimerax.alignment_headers.conservation" # imports chimerax.ui
    , "chimerax.alignment_headers.rmsd" # imports chimerax.ui
    , "chimerax.alphafold.pae" # imports chimerax.ui
    , "chimerax.atomic.inspectors" # imports chimerax.ui
    , "chimerax.atomic.options" # imports chimerax.ui
    , "chimerax.atomic.widgets" # imports chimerax.ui
    , "chimerax.clashes.gui" # imports Qt 
    , "chimerax.core_formats.gui" # imports Qt 
    , "chimerax.esmfold.pae" # imports chimerax.alphafold.pae, which imports chimerax.ui
    , "chimerax.hbonds.gui" # imports Qt 
    , "chimerax.image_formats.gui" # imports Qt 
    , "chimerax.looking_glass.lookingglass" # imports Qt 
    , "chimerax.map_filter.morph_gui" # imports chimerax.ui
    , "chimerax.map_series.slider" # imports chimerax.ui
    , "chimerax.md_crds.gui" # imports Qt 
    , "chimerax.md_crds.dcd.MDToolsMarch97.md_tests" # prints a bunch of stuff to the screen, but works
    , "chimerax.mmcif.build_ui" # imports Qt 
    , "chimerax.mmcif.gui" # imports Qt 
    , "chimerax.mmcif.mmcif_write" # imports app_dirs
    , "chimerax.model_series.mseries" # imports chimerax.ui
    , "chimerax.modeller.common" # imports webservices
    , "chimerax.pdb.gui" # imports Qt
    , "chimerax.pubchem.build_ui" # imports Qt
    , "chimerax.registration.gui" # imports chimerax.ui
    , "chimerax.save_command.widgets" # imports Qt
    , "chimerax.seq_view.feature_browser" # imports Qt
    , "chimerax.seq_view.settings" # imports chimerax.ui
    , "chimerax.seq_view.seq_canvas" # imports chimerax.seq_view.settings
    , "chimerax.seqalign.widgets" # imports chimerax.ui
    , "chimerax.sim_matrices.options" # imports chimerax.ui
    , "chimerax.smiles.build_ui" # imports Qt
    , "chimerax.std_commands.coordset_gui" # imports chimerax.ui
    , "chimerax.std_commands.defattr_gui" # imports Qt
    # Not going in the library, but part of test suite for GUI ChimeraX
    , "chimerax.ui.core_settings_ui" # imports settings from core_settings before initialized if imported standalone
    # Held over from cxtestimports.py
    , "chimerax.add_charge.process_lib"  # creates data.py
    , "chimerax.alignment_algs.libalign_algs"  # non-importable dynamic lib
    , "chimerax.alphafold.alphafold_predict_colab"    # IPython notebook with syntax magic
    , "chimerax.alphafold.alphafold21_predict_colab"  # IPython notebook with syntax magic
    , "chimerax.alphafold.alphafold22_predict_colab"  # IPython notebook with syntax magic
    , "chimerax.alphafold.colabfold_predict"          # IPython notebook with syntax magic
    , "chimerax.alphafold.fix_seq_titles"	# Alphafold database processing script.
    , "chimerax.atomic.libmolc"  # non-importable dynamic lib
    , "chimerax.atomic.md_crds.dcd.MDToolsMarch97.md_tests"  # test code
    , "chimerax.build_structure.process"  # processes Chimera fragment files
    , "chimerax.coulombic.create_data"  # creates data.py
    , "chimerax.dicom.scan_dicoms"  # development script
    , "chimerax.map.data.memoryuse" # unported code
    , "chimerax.map.filter.square"  # unported code
    , "chimerax.map.series.align"   # unported code
    , "chimerax.modeller.script_head"  # fragment of a Modeller script
    , "chimerax.modeller.script_tail"  # fragment of a Modeller script
    , "chimerax.remote_control.run"    # imports Python2 xmlrpclib
    , "chimerax.segger.ChimeraExtension"  # unported segger features
    , "chimerax.segger.Mesh"
    , "chimerax.segger.extract_region_dialog"
    , "chimerax.segger.imageviewer"
    , "chimerax.segger.iseg_dialog"
    , "chimerax.segger.modelz"
    , "chimerax.segger.promod_dialog"
    , "chimerax.segger.rseg_dialog"
    , "chimerax.segger.segloop_dialog"
    , "chimerax.structcomp"  # ChimeraX command script
    , "chimerax.surface.geodesic" # development script
    , "chimerax.webcam.camera"		# Uses QVideoSink only in Qt6
    , "chimerax.webcam.camera_qt5"	# Uses QAbstractVideoSurface only in Qt5
])

def test_chimerax_top_level_imports():
    for info in pkgutil.iter_modules(chimerax.__path__, prefix=chimerax.__name__ + '.'):
        module_finder, name, is_pkg = info
        if name in module_blacklist:
            continue
        m = importlib.import_module(name)

def test_all_imports():
    def check_if_true_error(pkg):
        if pkg not in fine_blacklist and pkg not in module_blacklist:
            raise
    for info in pkgutil.walk_packages(chimerax.__path__, prefix=chimerax.__name__ + '.', onerror=check_if_true_error):
        module_finder, name, is_pkg = info
        if name.endswith(".tool") or (name in fine_blacklist) or (name in module_blacklist):
            continue
        m = importlib.import_module(name)
