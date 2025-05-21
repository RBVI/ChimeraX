# vim: set expandtab ts=4 sw=4:

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
