# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Routines to extract PDB transformation matrices from PDB file header records
# and produce crystal unit cell symmetry matrices.
#

from .structure_matrices import (unit_cell_parameters,
    				 unit_cell_matrices,
                                 crystal_symmetries,
                                 noncrystal_symmetries,
                                 space_group_symmetries,
                                 biological_unit_matrices)

from .parsepdb import (pdb_biomt_matrices,
                       pdb_biomolecules,
                       pdb_smtry_matrices,
                       pdb_crystal_origin,
                       pdb_mtrix_matrices,
                       pdb_unit_cell_matrices,
                       pdb_3x3x3_unit_cell_matrices,
                       pdb_crystal_symmetry_matrices,
                       pdb_space_group_matrices,
                       pdb_crystal_parameters,
                       pdb_pack_matrices,
                       set_pdb_biomt_remarks,
                       transform_pdb_biomt_remarks,
                       restore_pdb_biomt_remarks)

from .parsemmcif import (mmcif_unit_cell_matrices,
                         mmcif_unit_cell_parameters,
                         mmcif_crystal_symmetry_matrices,
                         mmcif_ncs_matrices,
                         mmcif_biounit_matrices)

from .parsecif import (cif_unit_cell_matrices,
                       cif_unit_cell_parameters,
                       cif_crystal_symmetry_matrices,
                       cif_ncs_matrices)

