import pytest
import sys

import import_cycle_finder

@pytest.mark.parametrize("path,bundle_name", [
    ("./segger/seggerx/Segger/src/fit_dialog.py", "segger")
    , ("./rotamer_libs/Dunbrack/src/lib.py", "dunbrack_rotamer_lib")
    , ("./rotamer_libs/Dynameomics/src/lib.py", "dynameomics_rotamer_lib")
    , ("./rotamer_libs/Richardson/src/lib.py", "richardson_rotamer_lib")
    , ("./blastprotein/src/cmd.py", "blastprotein")
])
def test_get_bundle_name(path, bundle_name):
    assert import_cycle_finder.get_bundle_name(path) == bundle_name

@pytest.mark.parametrize("package,bundles", [
    ("from chimerax.core import foo", ["core"])
    , ("from chimerax.core.toolshed import foo as bar", ["core"])
    , ("import chimerax.core.toolshed", ["core"])
    , ("import chimerax.core.toolshed as ts, chimerax.blastprotein", sorted(["core", "blastprotein"]))
    , ("import chimerax.core.toolshed as ts, chimerax.blastprotein, chimerax.alphafold", sorted(["core", "blastprotein", "alphafold"]))
    , ("import chimerax.app_dirs", [import_cycle_finder.top_level_package_placeholder])
    , ("import chimerax", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import atomic", ["atomic"])
    , ("from chimerax import map", ["map"])
    , ("from chimerax import mmcif", ["mmcif"])
    , ("from chimerax import atomic, map, mmcif", ["atomic", "map", "mmcif"])
    , ("import chimerax.map_data as VD", ["map_data"])
    , ("import chimerax.app_dirs_unversioned", [import_cycle_finder.top_level_package_placeholder])
    , ("import chimerax", [import_cycle_finder.top_level_package_placeholder])
    , ("import chimerax.map_fit", ["map_fit"])
    , ("import os, chimerax", [import_cycle_finder.top_level_package_placeholder])
    , ("import os, sys, chimerax", [import_cycle_finder.top_level_package_placeholder])
    , ("import sys, chimerax, os", [import_cycle_finder.top_level_package_placeholder])
    , ("import chimerax, os", [import_cycle_finder.top_level_package_placeholder])
    , ("import chimerax, os, sys", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_lib_dir", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs as ad", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs as ad, app_data_dir", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs_unversioned", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs_unversioned as app_dirs", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs, app_data_dir", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_bin_dir", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_data_dir", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_data_dir as directory", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_data_dir, app_dirs", [import_cycle_finder.top_level_package_placeholder])
    , ("from chimerax import app_dirs, app_dirs_unversioned", [import_cycle_finder.top_level_package_placeholder])
    , ('_debug("chimerax.registration import failed")', [])
    , ('step_mod = importlib.import_module("chimerax." + mod_name)', [])
    , ('from . import chimerax_uuid', [])
    , ('from chimerax import arrays ; arrays.load_libarrays()', ["arrays"])
    , ('from chimerax import atomic_lib, pdb_lib', ["atomic_lib", "pdb_lib"])
    , ('from chimerax import atomic_lib, pdb_lib # ensure libs we need are linked', sorted(["atomic_lib", "pdb_lib"]))
    , ('from chimerax import graphics as g', ['graphics'])
    , ('from chimerax import pdb_matrices as pm, crystal', sorted(['pdb_matrices', 'crystal']))
    , ('from chimerax import pdb_matrices as m', ['pdb_matrices'])
])
def test_get_imported_bundle(package, bundles):
    assert import_cycle_finder.get_imported_bundle(package) == bundles
