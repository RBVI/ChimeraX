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

from chimerax.core.errors import UserError

def read_coords(session, file_name, model, format_name, *, replace=True, start=1, step=1, end=None):
    from numpy import array, float64
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
        num_frames = _set_model_dcd_coordinates(session, model, dcd, replace, start, step, end)
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
    start, step, end = process_limit_args(session, start, step, end, len(coords))
    if start > 0 or step > 1 or end < len(coords):
        coords = coords[start:end:step]

    model.add_coordsets(coords, replace=replace)
    return len(coords)

def process_limit_args(session, start, step, end, num_coords):
    if end is None:
        end = num_coords
    if end < start:
        raise UserError("'start' (%d) must be less than or equal to 'end' (%d)" % (start, end))
    start -= 1
    if start > 0 or step > 1 or end < num_coords:
        session.logger.info("start: %d, step: %d, end %d" % (start+1, step, end))
    return start, step, end

def _set_model_dcd_coordinates(session, model, dcd, replace, start, step, end):
    '''Read DCD coordinates and add to model efficiently when there are thousands of frames.'''
    num_atoms = dcd.numatoms
    if model.num_atoms != num_atoms:
        from chimerax.core.errors import UserError
        raise UserError("Specified structure has %d atoms"
            " whereas the coordinates are for %d atoms" % (model.num_atoms, num_atoms))
    start, step, end = process_limit_args(session, start, step, end, dcd.numframes)
    if replace:
        model.remove_coordsets()
        base = 1
    else:
        base = max(model.coordset_ids) + 1
    from numpy import asarray, float64
    num_frames = 0
    for i in range(start, end, step):
        model.add_coordset(base+num_frames, asarray(dcd[i], float64, order = 'C'))
        num_frames += 1
    model.active_coordset_id = base
    return num_frames
