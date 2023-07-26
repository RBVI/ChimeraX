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
