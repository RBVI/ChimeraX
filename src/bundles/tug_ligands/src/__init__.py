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

#
# The file moriarty_and_case_ligands.zip contains OpenMM Amber parameters for ligands and was
# copied from the ISOLDE github repository
#
#   https://github.com/tristanic/isolde/tree/master/isolde/src/openmm/amberff
#
# These ligand parameters come originally from Nigel Moriarty, calculated as described in article
#
# Improved chemistry restraints for crystallographic refinement by integrating the Amber force field into Phenix
# Nigel W Moriarty, Pawel A Janowski, Jason M Swails, Hai Nguyen, Jane S Richardson, David A Case, Paul D Adams
# Acta Crystallogr D Struct Biol 2020 Jan 1;76(Pt 1):51-62.  doi: 10.1107/S2059798319015134. Epub 2020 Jan 1.
#
from os.path import dirname
parameters_directory = dirname(__file__)

from chimerax.core.toolshed import BundleAPI
class _TugLigandsAPI(BundleAPI):
    pass
bundle_api = _TugLigandsAPI()
