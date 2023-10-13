# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from abc import abstractmethod

class NoResidueRotamersError(ValueError):
    pass

class UnsupportedResTypeError(NoResidueRotamersError):
    pass

class RotamerParams:
    def __init__(self, p, chis):
        """ 'p' is the probability of this rotamer.  'chis' is a list of the chi angles. """
        self.p = p
        self.chis = chis

class RotamerLibrary:
    """Provide all the information needed to display/list and use a rotamer library.

       Needed methods are described in their doc strings, and methods that must be
       implemented by subclasses are marked with the 'abstractmethod' decorator.

       If possible, reading the actual rotamer library itself should be done in a
       "lazy" (i.e. on demand) manner if possible to minimize startup time.
    """

    def __init__(self, name, ui_name):
        # name as given in Provider tag
        self.name = name
        self.ui_name = ui_name

    @property
    def citation(self):
        """If your library has a reference to cite, the text of the citation.  Used as the 'cite'
           argument to the chimerax.ui.widgets.Citation constructor.  Example:

           Shapovaeov, M.S., and Dunbrack, R.L., Jr. (2011)
           A Smoothed Backbone-Dependent Rotamer Library for Proteins
               Derived from Adaptive Kernel Density Estimates and Regressions
           Structure, 19, 844-858.
        """
        return None

    @property
    def cite_pubmed_id(self):
        """The (integer) PubMed ID corresponding to your citation"""
        return None

    # display_name and description is now a Provider attribute (display_name changed to ui_name),
    # since they are needed even if the library is not yet installed

    def map_res_name(self, res_name, exemplar=None):
        """Take a residue name and map it to a name that this library supports.  For, example if
           the library supports HIE, HID, and HID but not HIS per se, then map "HIS" to one of the
           three supported names.  'exemplar', if provided, is an example residue whose name needs
           mapping.  The default implementation handles some common HIS, PRO and CYS variants.

           This routine should return None if no mapping can be determined, though "ALA" and "GLY"
           should simply return themselves.
        """
        if res_name == "ALA" or res_name == "GLY":
            return res_name

        supported_names = set(self.residue_names)
        if res_name == "HIS":
            if "HID" in supported_names and "HIE" in supported_names:
                if exemplar:
                    if exemplar.find_atom("HD1"):
                        if "HIP" in supported_names and exemplar.find_atom("HE2"):
                            return "HIP"
                        return "HID"
                    return "HIE"
                return "HID"
        elif res_name in ["HID", "HIE", "HIP"]:
            if res_name in supported_names:
                return res_name
            if "HIS" in  supported_names:
                return "HIS"
        elif res_name == "CYS":
            if "CYH" in supported_names or "CYD" in supported_names:
                if exemplar:
                    sg = exemplar.find_atom("SG")
                    if sg:
                        for nb in sg.neighbors:
                            if nb.residue != sg.residue:
                                if "CYD" in supported_names:
                                    return "CYD"
                                break
                        else:
                            if "CYH" in supported_names:
                                return "CYH"
        elif res_name == "CYH" or res_name == "CYD":
            if res_name in supported_names:
                return res_name
            if "CYS" in supported_names:
                return "CYS"
        elif res_name == "PRO":
            if "CPR" in supported_names or "TPR" in supported_names:
                if exemplar:
                    omega = exemplar.omega
                    if omega is not None and abs(omega) < 90:
                        if "CPR" in supported_names:
                            return "CPR"
                    elif "TPR" in supported_names:
                        return "TPR"
        if res_name in supported_names:
            return res_name
        return None

    std_rotamer_res_names = frozenset(["ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU",
            "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"])
    @property
    def residue_names(self):
        """A set of the residue names that this rotamer library provides rotamers for.  Typically just
           the 18 standard amino acids that actually have side chains, but some libraries provide
           rotamers for certain protonation states (e.g. CYH for non-disulphide cysteine) or conformers
           (e.g. CPR for cis-proline).
        """
        return self.std_rotamer_res_names.copy()

    std_rotamer_res_descriptions = { "ALA": "alanine", "ASN": "asparagine", "ASP": "aspartic acid",
            "CYS": "cysteine", "GLN": "glutamine", "GLU": "glutamic acid", "GLY": "glycine",
            "HIS": "histidine", "ILE": "isoleucine", "LEU": "leucine", "LYS": "lysine",
            "MET": "methionine", "PHE": "phenylalinine", "PRO": "proline", "SER": "serine",
            "THR": "threonine", "TRP": "tryptophan", "TYR": "tyrosine", "VAL": "valine",
        }
    @property
    def res_name_descriptions(self):
        """A dictionary mapping the 3-letter residue name to a full text description of the residue,
           e.g. "leucine" for LEU or "doubly protonated histidine" for HIP.  All normal amino acids
           are included in the default implementation.  All residues provided by the library should
           be in the dictionary that this property returns.
        """
        return self.std_rotamer_res_descriptions.copy()

    @property
    def res_name_mapping(self):
        """For libraries that have non-standard residue names that correspond to certain states of
           standard residues (see the residue_names method), this dictionary maps the non-standard
           name to the corresponding standard name.
        """
        return {}

    @property
    def res_template_func(self):
        """If a rotamer library supports non-standard residues, this should return a function that
           when given the residue name as its argument, returns a TmplResidue that can be used
           to build out the rotamer (and returns None for residues not in the library).
        """
        from chimerax.atomic import TmplResidue
        return TmplResidue.get_template

    @abstractmethod
    def rotamer_params(self, res_name, phi, psi, *, cis=False):
        """Return a list of RotamerParams instances corresponding to the residue name 'res_name' and
           the backbone angle 'phi' and 'psi'.  Backbone-independent libraries will ignore phi and psi.
           Note that phi or psi can be None for chain-terminal residues.  Backbone-dependent libraries
           will have to use some fallback procedure for generating parameters in those cases, or throw
           NoResidueRotamersError.  If 'res_name' does not correspond to a name supported by the library,
           throw UnsupportedResTypeError.

           For rotamer libraries that support cis vs. trans rotamers, the cis keyword can be used
           to decide which rotamers to return.
        """
        pass

    def _get_params(self, res_name, file_name, cache, archive):
        """Possibly useful utility routine for fetching parameters stored in zip archives"""
        try:
            return cache[file_name]
        except KeyError:
            pass
        base_name = self._non_cistrans_res_name(res_name)
        if base_name not in self.residue_names:
            raise UnsupportedResTypeError(
                "%s library does not support residue type '%s'" % (self.ui_name, base_name))
        import os.path, inspect
        my_dir = os.path.split(inspect.getfile(self.__class__))[0]
        from zipfile import ZipFile
        zf = ZipFile(os.path.join(my_dir, archive), "r")
        try:
            data = zf.read(file_name)
        except KeyError:
            raise NoResidueRotamersError(
                "'%s' library has no rotamers for '%s'" % (self.ui_name, file_name))
        from struct import unpack, calcsize
        sz1 = calcsize("!ii")
        num_rotamers, num_params, = unpack("!ii", data[:sz1])
        sz2 = calcsize("!%df" % num_params)
        rotamers = []
        for i in range(num_rotamers):
            params = unpack("!%df" % num_params, data[sz1 + i * sz2 : sz1 + (i+1) * sz2])
            p = params[0]
            chis = params[1:]
            rotamers.append(RotamerParams(p, chis))
        cache[file_name] = rotamers
        return rotamers

    def _non_cistrans_res_name(self, res_name):
        if res_name.endswith('cis'):
            return res_name[:-4]
        if res_name.endswith('trans'):
            return res_name[:-6]
        return res_name
