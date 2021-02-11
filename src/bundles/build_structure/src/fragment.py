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

from chimerax.core.toolshed import ProviderManager

class Fragment:
    def __init__(self, name, atoms, bonds):
        """
        *name* is the fragment name (e.g. "benzene")

        *atoms* is a list of tuples: (element name, xyz)

        *bonds* is a list of tuples: (indices, depict)
        where *indices* is a two-tuple into the atom list and *depict* is either None [single bond]
        or an xyz [center of ring] for a double bond.  Depiction of non-ring double bonds not supported yet.
        File a bug report if you need them.
        """
        self.name = name
        self.atoms = atoms
        self.bonds = bonds
