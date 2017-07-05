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
def write_coords(session, file_name, format, models):
    from .dcd.MDToolsMarch97.md_DCD import DCDWrite
    dcd = DCDWrite(file_name, DCDAtoms(models[0].atoms))
    for m in models:
        matoms = m.atoms
        ocid = m.active_coordset_id
        for cid in m.coordset_ids:
            m.active_coordset_id = cid
            dcd.append(matoms.coords)
        m.active_coordset_id = ocid
    dcd.file.close()

class DCDAtoms:
    def __init__(self, atoms):
        self.atoms = atoms        # Writing DCD file only uses len(atoms)

