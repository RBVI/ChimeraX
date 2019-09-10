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
    def cite_name(self):
        """In the preamble to the citation, cite_name replaces XXX in "Publications using XXX rotamers
           should cite:"
        """
        return self.display_name

    @property
    def cite_pubmed_id(self):
        """The (integer) PubMed ID corresponding to your citation"""
        return None

    @property
    @abstractmethod
    def display_name(self):
        """Short name to display in a list of libraries, e.g. "Dunbrack 2010" or "Dynameomics"
           Should be the same as the 'name' attribute used in your Provider tag in bundle_info.xml.
           Also used as argument to swapaa command.
        """
        pass

    @property
    def description(self):
        """A somewhat longer decription than 'display_name' (though typically still one line)
        to show in interfaces when the library has been selected, e.g.:

        Dunbrack 2010 backbone-dependent rotamer library -- 5% stepdown
        """
        return self.display_name

    std_rotamer_res_names = set(["ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET",
                "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"])
    @property
    def residue_names(self):
        """The residue names that this rotamer library provides rotamers for.  Typically just the
           18 standard amino acids that actually have side chains, but some libraries provide rotamers
           for certain protonation states (e.g. CYH for non-disulphide cysteine) or conformers
           (e.g. CPR for cis-proline).
        """
        return self.std_rotamer_res_names

    @property
    def res_name_description(self):
        """A dictionary mapping the 3-letter residue name to a full text description of the residue,
           e.g. "leucine" for LEU or "doubly protonated histidine" for HIP.  All normal amino acids
           are included in the default implementation.  All residues provided by the library should
           be in the dictionary that this property returns.
        """
        return {
            "ALA": "alanine",
            "ASN": "asparagine",
            "ASP": "aspartic acid",
            "CYS": "cysteine",
            "GLN": "glutamine",
            "GLU": "glutamic acid",
            "GLY": "glycine",
            "HIS": "histidine",
            "ILE": "isoleucine",
            "LEU": "leucine",
            "LYS": "lysine",
            "MET": "methionine",
            "PHE": "phenylalinine",
            "PRO": "proline",
            "SER": "serine",
            "THR": "threonine",
            "TRP": "tryptophan",
            "TYR": "tyrosine",
            "VAL": "valine",
        }

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
                "%s library does not support residue type '%s'" % (self.display_name, base_name))
        import os.path, inspect
        my_dir = os.path.split(inspect.getfile(self.__class__))[0]
        from zipfile import ZipFile
        zf = ZipFile(os.path.join(my_dir, archive), "r")
        try:
            data = zf.read(file_name)
        except KeyError:
            raise NoResidueRotamersError(
                "'%s' library has no rotamers for '%s'" % (self.display_name, file_name))
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
