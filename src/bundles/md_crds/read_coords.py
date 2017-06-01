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

def read_coords(session, file_name, model, format_name, replace=True):
    from numpy import array, float64
    from chimerax.core.errors import UserError
    if format_name == "Gromacs compressed coordinates":
        from ._gromacs import read_xtc_file
        session.logger.status("Reading Gromacs xtc coordinates", blank_after=0)
        num_atoms, coords_list = read_xtc_file(file_name)
        coords = array(coords_list, dtype=float64)
        coords *= 10.0
        session.logger.status("Finished reading Gromacs xtc coordinates")
    elif format_name == "Gromacs full-precision coordinates":
        from ._gromacs import read_trr_file
        session.logger.status("Reading Gromacs trr coordinates", blank_after=0)
        num_atoms, coords_list = read_trr_file(file_name)
        coords = array(coords_list, dtype=float64)
        coords *= 10.0
        session.logger.status("Finished reading Gromacs trr coordinates")
    elif format_name == "DCD coordinates":
        from .dcd.MDToolsMarch97.md_DCD import DCD
        session.logger.status("Reading DCD coordinates", blank_after=0)
        dcd = DCD(file_name)
        num_atoms = dcd.numatoms
        coords_list = [dcd[i] for i in range(dcd.numframes)]
        coords = array(coords_list, dtype=float64)
        session.logger.status("Finished reading DCD coordinates")
    else:
        raise ValueError("Unknown MD coordinate format: %s" % format_name)
    if model.num_atoms != num_atoms:
        raise UserError("Specified structure has %d atoms"
            " whereas the coordinates are for %d atoms" % (model.num_atoms, num_atoms))
    model.add_coordsets(coords, replace=replace)
    return len(coords)
