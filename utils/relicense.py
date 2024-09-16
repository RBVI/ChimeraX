#!/usr/bin/env python3
# This is NOT THIS FILE'S LICENSE. It's here to make it easy to copy and paste
# for any files the script may have missed.
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

import os

new_license_text = """{comment} Copyright 2022 Regents of the University of California. All rights reserved.
{comment} The ChimeraX application is provided pursuant to the ChimeraX license
{comment} agreement, which covers academic and commercial uses. For more details, see
{comment} <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
{comment}
{comment} This particular file is part of the ChimeraX library. You can also
{comment} redistribute and/or modify it under the terms of the GNU Lesser General
{comment} Public License version 2.1 as published by the Free Software Foundation.
{comment} For more details, see
{comment} <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
{comment}
{comment} THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
{comment} EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
{comment} OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
{comment} LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
{comment} VERSION 2.1
{comment}
{comment} This notice must be embedded in or attached to all copies, including partial
{comment} copies, of the software or any revisions or derivations thereof."""


new_license_text_python = new_license_text.format(comment = "#")
new_license_text_cpp = new_license_text.format(comment = " *")

# TODO: In the future, it would be good to look for the presence or absence of  the
# string "This file is a part of the ChimeraX library" and decide what to do based on
# that instead of a specific filename or path, now that that information will be
# embedded in the file.

# Copied directly from wheel/filter_modules.py:
bundle_blacklist = set([
    "amber_info" # needs app_bin_dir
    , "webservices" # needs cxservices
    , "blastprotein" # needs webservices
    , "build_structure" # needs Qt
    , "dicom" # tries to import its .ui submodule in __init__
    , "nifti" # tries to import dicom
    , "nrrd" # tries to import dicom
    , "structcomp"  # ChimeraX command script
    # Not going in the library, but part of test suite for GUI ChimeraX
    , "ui" # tries to import Qt
    , "vive" # GUI only bundle
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

folders_to_skip = [
    "dist"
    , "build"
    , "__pycache__"
    , "tests"
    , "test"
    , "test_data"
    , "ui"
    , "doc"
    , "docs"
    , "gromacs" # submodule of md_crds
]

skipped_files = [
    "bundle_info.xml"
    , "pyproject.toml"
    , "Makefile"
    , "tool.py"
    , "README"
    , "al2co"
    , "cxtest.py"
]

skipped_extensions = [
    "obj"
    , "xml"
    , "o"
    , "exe"
    , "js"
    , "json"
    , "ipynb"
    , "a"
    , "dll"
    , "exp"
    , "lib"
    , "gz"
    , "png"
    , "pyd"
    , "zip"
    , "dylib"
    , "so"
    , "maegz"
    , "jpg"
    , "gif"
    , "webarchive"
    , "pdf"
    , "dcd"
    , "xtc"
    , "ico"
    , "cxs"
]

c_extensions = [
    "cpp"
    , "c"
    , "h"
    , "hpp"
]

chimerax_repo_root = os.path.dirname(os.path.dirname(__file__))
chimerax_bundle_root = os.path.join(chimerax_repo_root, "src", "bundles")
chimerax_main = os.path.join(chimerax_repo_root, "src", "bundles", "core", "src", "__main__.py")

problematic_files = [
    chimerax_main
    , os.path.join(chimerax_repo_root, "src", "bundles", "map_data", "src", "ims", "ims_format.py")
    ,
]

modified_files = []
for dirpath, dirnames, filenames in os.walk(chimerax_bundle_root, topdown = True):
    # I did this with a really nice list comprehension at first, I promise.
    # Python proceeded to take the result of that comprehension, completely
    # throw it away and ignore it, and recurse into unwanted directories anyway.
    # If you can mutate something you might as well be able to reassign it.
    # One day language authors will learn this basic truth!
    if dirpath == chimerax_bundle_root:
        for dir in bundle_blacklist:
            if dir in dirnames:
                dirnames.remove(dir)
        continue
    else:
        for dir in folders_to_skip:
            if dir in dirnames:
                dirnames.remove(dir)
        for dir in dirnames:
            if dir.endswith("egg-info"):
                dirnames.remove(dir)
    # OK, now we can do the actual work.
    for file in filenames:
        if file.startswith('.'):
            continue
        if file in skipped_files:
            continue
        if file.split('.')[0].endswith('gui') or file.split('.')[0].endswith('cgi'):
            continue
        if file.split('.')[-1] in skipped_extensions:
            continue
        path_to_thing = os.path.join(dirpath, file)
        if path_to_thing in problematic_files:
            continue
        modified_files.append(path_to_thing)
        with open(path_to_thing, "r+") as f:
            lines = f.read().split('\n')
            license_start = None
            license_end = None
            for lineno, line in enumerate(lines):
                if "UCSF ChimeraX Copyright" in line:
                    if not license_start:
                        license_start = lineno
                    else:
                        license_end = lineno
            # I guess don't relicense files that aren't ours!
            if license_start and license_end:
                del lines[license_start + 1:license_end]
                if any(file.endswith(ext) for ext in c_extensions):
                    lines.insert(license_start + 1, new_license_text_cpp)
                else:
                    lines.insert(license_start + 1, new_license_text_python)
                f.seek(0)
                f.write('\n'.join(lines))
print('\n'.join(modified_files))
