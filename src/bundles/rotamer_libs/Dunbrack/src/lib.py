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

_dependent_cache = {}
_independent_cache = {}

from chimerax.rotamers import RotamerLibrary, RotamerParams, UnsupportedResTypeError, NoResidueRotamersError

class DunbrackRotamerLibrary(RotamerLibrary):

    @property
    def citation(self):
        return """Shapovalov, M.S., and Dunbrack, R.L., Jr. (2011)
A Smoothed Backbone-Dependent Rotamer Library for Proteins
    Derived from Adaptive Kernel Density Estimates and Regressions
Structure, 19, 844-858."""

    @property
    def cite_pubmed_id(self):
        return 21645855

    _rotamer_res_names = set(["ARG", "ASN", "ASP", "CPR", "CYD", "CYH", "CYS", "GLN", "GLU", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TPR", "TRP", "TYR", "VAL"])
    @property
    def residue_names(self):
        return self._rotamer_res_names

    @property
    def res_name_descriptions(self):
        mapping = super().res_name_descriptions
        mapping.update({
            "CPR": "cis proline",
            "CYD": "disulfide-bonded cysteine",
            "CYH": "non-disulfide-bonded cysteine",
            "CYS": "cysteine (CYD+CYH)",
            "PRO": "proline (CPR+TPR)",
            "TPR": "trans proline"
        })
        return mapping

    @property
    def res_name_mapping(self):
        return { "CPR": "PRO", "CYD": "CYS", "CYH": "CYS", "TPR": "PRO" }

    def rotamer_params(self, res_name, phi, psi, *, cis=False):
        if phi is None or psi is None:
            file_name = res_name
            archive = "independentRotamerData2002.zip"
            cache = _independent_cache
        else:
            from math import floor
            phi = floor((phi + 5) / 10.0) * 10
            psi = floor((psi + 5) / 10.0) * 10
            file_name = "%s%d%d" % (res_name, phi, psi)
            archive = "dependentRotamerData.zip"
            cache = _dependent_cache
        return self._get_params(res_name, file_name, cache, archive)
