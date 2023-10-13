# vim: set expandtab ts=4 sw=4:

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

PreferredAtoms = [
        "CA",
        "P",
        "N",
        "C",
        "O5'",
        "O3'",
]

def segment_alignment_atoms(rList):
        """Construct a list of CA/P atoms from residue list (ignoring
        residues that do not have either atom)"""
        aList = []
        if len(rList) > 2:
                atomsPerResidue = 1
        elif len(rList) > 1:
                atomsPerResidue = 2
        else:
                atomsPerResidue = 3
        for r in rList:
                atoms = set()
                failed = False
                while len(atoms) < atomsPerResidue:
                        for aname in PreferredAtoms:
                                a = r.find_atom(aname)
                                if a is None or a in atoms:
                                        continue
                                atoms.add(a)
                                break
                        else:
                                for a in r.atoms:
                                        if a not in atoms:
                                                atoms.add(a)
                                                break
                                else:
                                        failed = True
                        if failed:
                                break
                aList.extend(atoms)
        #print "%d residues -> %d atoms" % (len(rList), len(aList))
        return aList
                        
def copyMolecule(m):
        """
        Copy molecule and return both copy and map of corresponding atoms.
        Delete alt locs and all but the active coordinate set in the copy,
        and make the active coordset have id 1.
        """
        c = m.copy()
        c.delete_alt_locs()
        if c.num_coordsets != 1 or c.active_coordset_id != 1:
                xyz = c.atoms.coords
                c.remove_coordsets()
                c.add_coordset(1, xyz)
                c.active_coordset_id = 1
        atomMap = {a:ca for a,ca in zip(m.atoms, c.atoms)}
        residueMap = {r:cr for r,cr in zip(m.residues, c.residues)}
        return c, atomMap, residueMap

def timestamp(s):
        import time
        print ("%s: %s" % (time.ctime(time.time()), s))
