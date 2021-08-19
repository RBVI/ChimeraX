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
    if format_name == "xtc":
        from ._gromacs import read_xtc_file
        session.logger.status("Reading Gromacs xtc coordinates", blank_after=0)
        num_atoms, coords_list = read_xtc_file(file_name)
        coords = array(coords_list, dtype=float64)
        coords *= 10.0
        session.logger.status("Finished reading Gromacs xtc coordinates")
    elif format_name == "trr":
        from ._gromacs import read_trr_file
        session.logger.status("Reading Gromacs trr coordinates", blank_after=0)
        num_atoms, coords_list = read_trr_file(file_name)
        coords = array(coords_list, dtype=float64)
        coords *= 10.0
        session.logger.status("Finished reading Gromacs trr coordinates")
    elif format_name == "dcd":
        from .dcd.MDToolsMarch97.md_DCD import DCD
        session.logger.status("Reading DCD coordinates", blank_after=0)
        dcd = DCD(file_name)
        num_frames = _set_model_dcd_coordinates(model, dcd, replace)
        session.logger.status("Finished reading DCD coordinates")
        return num_frames
    elif format_name == "amber":
        from netCDF4 import Dataset
        ds = Dataset(file_name, "r")
        try:
            # netCDF4 has a builtin __array__ that doesn't allow a second argumeht...
            coords = array(array(ds.variables['coordinates']), dtype=float64)
        except KeyError:
            raise UserError("File is not an Amber netCDF coordinates file (no coordinates found)")
        num_atoms = len(coords[0])
    else:
        raise ValueError("Unknown MD coordinate format: %s" % format_name)
    if model.num_atoms != num_atoms:
        raise UserError("Specified structure has %d atoms"
            " whereas the coordinates are for %d atoms" % (model.num_atoms, num_atoms))
    model.add_coordsets(coords, replace=replace)
    return len(coords)

def _set_model_dcd_coordinates(model, dcd, replace = True):
    '''Read DCD coordinates and add to model efficiently when there are thousands of frames.'''
    num_atoms = dcd.numatoms
    if model.num_atoms != num_atoms:
        from chimerax.core.errors import UserError
        raise UserError("Specified structure has %d atoms"
            " whereas the coordinates are for %d atoms" % (model.num_atoms, num_atoms))
    if replace:
        model.remove_coordsets()
        base = 1
    else:
        base = max(model.coordset_ids) + 1
    from numpy import asarray, float64
    for i in range(dcd.numframes):
        model.add_coordset(base+i, asarray(dcd[i], float64, order = 'C'))
    model.active_coordset_id = base
    return dcd.numframes
