# vim: set expandtab shiftwidth=4 softtabstop=4:
#
# Shared exclusion lists for import tests and the wheel build.
#

# Modules that cannot be imported (scripts, non-importable dynamic libs, etc.).
IMPORT_EXCLUDES = {
    "chimerax.add_charge.process_lib",  # creates data.py
    "chimerax.alignment_algs.libalign_algs",  # non-importable dynamic lib
    "chimerax.alphafold.alphafold_predict_colab",  # IPython notebook with syntax magic
    "chimerax.alphafold.alphafold21_predict_colab",  # IPython notebook with syntax magic
    "chimerax.alphafold.alphafold22_predict_colab",  # IPython notebook with syntax magic
    "chimerax.alphafold.colabfold_predict",  # IPython notebook with syntax magic
    "chimerax.alphafold.colabfold_predict_test",  # IPython notebook with syntax magic
    "chimerax.alphafold.fix_seq_titles",  # Alphafold database processing script
    "chimerax.atomic.libmolc",  # non-importable dynamic lib
    "chimerax.atomic.md_crds.dcd.MDToolsMarch97.md_tests",  # test code
    "chimerax.boltz.download_weights_and_ccd",  # Boltz install script
    "chimerax.boltz.make_ccd_atom_counts_file",  # Boltz install script
    "chimerax.build_structure.process",  # processes Chimera fragment files
    "chimerax.coulombic.create_data",  # creates data.py
    "chimerax.kvfinder.cmd",  # top-level import of optional dependency
    "chimerax.map.data.memoryuse",  # unported code
    "chimerax.map.filter.square",  # unported code
    "chimerax.map.series.align",  # unported code
    "chimerax.modeller.script_head",  # fragment of a Modeller script
    "chimerax.modeller.script_tail",  # fragment of a Modeller script
    "chimerax.openfold.download_weights", # OpenFold install script
    "chimerax.remote_control.run",  # imports Python2 xmlrpclib
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
    "chimerax.surface.geodesic",  # development script
    "chimerax.webcam.camera",  # Uses QVideoSink only in Qt6
    "chimerax.webcam.camera_qt5",  # Uses QAbstractVideoSurface only in Qt5
}

# Entire packages excluded from the wheel build.
WHEEL_MODULE_EXCLUDES = {
    "chimerax.amber_info",  # needs app_bin_dir
    "chimerax.animations",  # GUI tool, imports tool.py which gets filtered out
    "chimerax.blastprotein",  # needs webservices
    "chimerax.bug_reporter",  # imports a GUI / not needed
    "chimerax.build_structure",  # needs Qt
    "chimerax.kvfinder",  # tries to import an optional dependency at the top level
    "chimerax.lighting_gui",  # GUI tool, needs Qt
    "chimerax.structcomp",  # ChimeraX command script
    "chimerax.ui",  # tries to import Qt
    "chimerax.vive",  # GUI only bundle
    "chimerax.webservices",  # needs cxservices
}

# Additional per-module exclusions for the wheel build (Qt/UI dependencies).
_WHEEL_FINE_EXTRA = {
    "chimerax.alignment_algs.options",  # imports chimerax.ui
    "chimerax.alignment_headers.conservation",  # imports chimerax.ui
    "chimerax.alignment_headers.rmsd",  # imports chimerax.ui
    "chimerax.alphafold.pae",  # imports chimerax.ui
    "chimerax.amber_info",  # needs app_bin_dir
    "chimerax.atomic.inspectors",  # imports chimerax.ui
    "chimerax.atomic.options",  # imports chimerax.ui
    "chimerax.atomic.widgets",  # imports chimerax.ui
    "chimerax.clashes.gui",  # imports Qt
    "chimerax.core_formats.gui",  # imports Qt
    "chimerax.esmfold.pae",  # imports chimerax.alphafold.pae -> chimerax.ui
    "chimerax.hbonds.gui",  # imports Qt
    "chimerax.image_formats.gui",  # imports Qt
    "chimerax.looking_glass.lookingglass",  # imports Qt
    "chimerax.map_filter.morph_gui",  # imports chimerax.ui
    "chimerax.map_series.slider",  # imports chimerax.ui
    "chimerax.md_crds.gui",  # imports Qt
    "chimerax.mmcif.build_ui",  # imports Qt
    "chimerax.mmcif.gui",  # imports Qt
    "chimerax.mmcif.mmcif_write",  # imports app_dirs
    "chimerax.model_series.mseries",  # imports chimerax.ui
    "chimerax.modeller.common",  # imports webservices
    "chimerax.pdb.gui",  # imports Qt
    "chimerax.pubchem.build_ui",  # imports Qt
    "chimerax.registration.gui",  # imports chimerax.ui
    "chimerax.save_command.widgets",  # imports Qt
    "chimerax.segmentations.job",  # imports core_settings, runtime-generated attr
    "chimerax.segmentations.ui",  # imports Qt
    "chimerax.segmentations.view",  # imports UI elements
    "chimerax.seq_view.feature_browser",  # imports Qt
    "chimerax.seq_view.seq_canvas",  # imports chimerax.seq_view.settings
    "chimerax.seq_view.settings",  # imports chimerax.ui
    "chimerax.seqalign.widgets",  # imports chimerax.ui
    "chimerax.sim_matrices.options",  # imports chimerax.ui
    "chimerax.smiles.build_ui",  # imports Qt
    "chimerax.std_commands.coordset_gui",  # imports chimerax.ui
    "chimerax.std_commands.defattr_gui",  # imports Qt
    "chimerax.ui.core_settings_ui",  # imports settings from core_settings before initialized
}

WHEEL_FINE_EXCLUDES = IMPORT_EXCLUDES | _WHEEL_FINE_EXTRA
