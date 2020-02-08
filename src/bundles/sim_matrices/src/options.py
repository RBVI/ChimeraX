# vim: set expandtab ts=4 sw=4:

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

from chimerax.ui.options import SymbolicEnumOption, EnumOption
from . import matrices, matrix_name_key_func, protein_matrix

mats = matrices()

class SimilarityMatrixOption(SymbolicEnumOption):
    labels = sorted(mats.keys(), key=matrix_name_key_func)
    values = [mats[name] for name in labels]

class ProteinSimilarityMatrixOption(SymbolicEnumOption):
    labels = sorted([name for name in mats.keys() if protein_matrix(name)], key=matrix_name_key_func)
    values = [mats[name] for name in labels]

class NucleicSimilarityMatrixOption(SymbolicEnumOption):
    labels = sorted([name for name in mats.keys() if not protein_matrix(name)], key=matrix_name_key_func)
    values = [mats[name] for name in labels]

class SimilarityMatrixNameOption(EnumOption):
    values = sorted(mats.keys(), key=matrix_name_key_func)

class ProteinSimilarityMatrixNameOption(EnumOption):
    values = sorted([name for name in mats.keys() if protein_matrix(name)], key=matrix_name_key_func)

class NucleicSimilarityMatrixNameOption(EnumOption):
    values = sorted([name for name in mats.keys() if not protein_matrix(name)], key=matrix_name_key_func)
