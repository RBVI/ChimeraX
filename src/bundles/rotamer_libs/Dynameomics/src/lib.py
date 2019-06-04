# vim: set expandtab shiftwidth=4 softtabstop=4:

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

_cache = {}

from chimerax.atomic.rotamers.manager import RotamerLibrary, RotamerParams, \
    UnsupportedResNameError, NoResidueRotamersError

class DynameomicsRotamerLibrary(RotamerLibrary):

    @property
    def display_name(self):
        return "Dynameomics"

    @property
    def description(self):
        return "Dynameomics backbone-independent rotamer library -- March '12"

    @property
    def citation(self):
        return """A.D. Scouras and V. Daggett (2011)
The dynameomics rotamer library:
  Amino acid side chain conformations and dynamics from
  comprehensive molecular dynamics simulations in water
Protein Science 20, 341-352."""

    @property
    def cite_pubmed_id(self):
        return 21280126

    @property
    def residue_names(self):
        return ["ARG", "ASN", "ASP", "CYS", "CYH", "GLN", "GLU", "HID", "HIE", "HIP", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

    @property
    def res_name_mapping(self):
        return { "CYH": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS" }

    def rotamer_params(self, res_name, phi, psi):
        return self._get_params(res_name, res_name, _cache, "rotamerData.zip")
