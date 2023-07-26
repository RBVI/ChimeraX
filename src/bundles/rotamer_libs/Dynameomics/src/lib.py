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

from chimerax.rotamers import RotamerLibrary, RotamerParams, UnsupportedResTypeError, NoResidueRotamersError

class DynameomicsRotamerLibrary(RotamerLibrary):

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

    _rotamer_res_names = set(["ARG", "ASN", "ASP", "CYS", "CYH", "GLN", "GLU", "HID", "HIE", "HIP", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"])
    @property
    def residue_names(self):
        return self._rotamer_res_names

    @property
    def res_name_descriptions(self):
        mapping = super().res_name_descriptions
        mapping.update({
            "CYH": "non-disulfide-bonded cysteine",
            "CYS": "disulfide-bonded cysteine",
            "HID": "\N{GREEK SMALL LETTER DELTA}-protonated histidine",
            "HIE": "\N{GREEK SMALL LETTER EPSILON}-protonated histidine",
            "HIP": "doubly protonated histidine",
            "HIS": "singly protonated histidine (HID+HIE)"
        })
        return mapping

    @property
    def res_name_mapping(self):
        return { "CYH": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS" }

    def rotamer_params(self, res_name, phi, psi, *, cis=False):
        return self._get_params(res_name, res_name, _cache, "rotamerData.zip")
