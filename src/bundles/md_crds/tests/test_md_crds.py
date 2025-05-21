import os

import pytest

test_data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test-data")
test_pdb_1 = os.path.join(test_data_folder, "chimera_test.pdb")
test_pdb_2 = os.path.join(test_data_folder, "start.pdb")
test_xtc = os.path.join(test_data_folder, "chimera_test.xtc")
test_dcd = os.path.join(test_data_folder, "test.dcd")
test_lammps_psf = os.path.join(test_data_folder, "gly.psf")
test_lammps_xtc = os.path.join(test_data_folder, "gly.xtc")
test_lammps_data = os.path.join(test_data_folder, "gly.data")
test_lammps_dump = os.path.join(test_data_folder, "gly.dump")

@pytest.mark.parametrize("test_pdb,test_crd_file,expected_coordsets", [(test_pdb_1, test_xtc, 21), (test_pdb_2, test_dcd, 2), (test_lammps_psf, test_lammps_xtc, 4), (test_lammps_data, test_lammps_dump, 4)])
def test_md_crds(test_production_session, test_pdb, test_crd_file, expected_coordsets):
    session = test_production_session
    from chimerax.core.commands import run
    if test_pdb.endswith(".pdb"):
        run(session, "open %s" % test_pdb)
        run(session, "open %s structureModel #1" % test_crd_file)
    else:
        run(session, "open %s coords %s" % (test_pdb, test_crd_file))
    assert(session.models[0].num_coordsets == expected_coordsets), "Expected %i coordinate sets; actually produced %s" % (expected_coordsets, session.models[0].num_coordsets)
