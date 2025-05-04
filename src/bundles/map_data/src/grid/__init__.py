# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
LAMMPS grid format support for ChimeraX.

This module provides support for reading LAMMPS grid files produced by the
'dump grid' command in LAMMPS.
"""

# Register the grid format
def register_grid_format():
    from chimerax.map_data import data_formats, FileFormatInfo
    data_formats['grid'] = FileFormatInfo('grid',
                                         'LAMMPS grid',
                                         ['grid'],
                                         'LAMMPS grid format',
                                         module_name = 'chimerax.map_data.grid',
                                         available = True,
                                         reference_url = 'https://docs.lammps.org/Howto_grid.html')

def open(stream, session, name=None, array_mode=None):
    """
    Read grid data from a LAMMPS grid format file and return a list of GridData objects.
    
    Parameters
    ----------
    stream : io.IOBase
        File stream containing LAMMPS grid data.
    session : Session
        ChimeraX session.
    name : str, optional
        Name to assign to the grid data.
    array_mode : str, optional
        Specifies whether to return full data arrays.
        Can be None (return data arrays), 'mmap' (memory map file), or 'header-only' (don't read arrays).
    
    Returns
    -------
    list of GridData objects
    """
    from chimerax.core.errors import UserError
    from .grid import read_lammps_grid

    try:
        grids = read_lammps_grid(stream, session, name, array_mode)
        return grids
    except Exception as e:
        raise UserError(f"Error reading LAMMPS grid file: {str(e)}")

def save(grid_data, path, session=None):
    """
    Write grid data to a LAMMPS grid format file.
    
    Parameters
    ----------
    grid_data : GridData or sequence of GridData
        Grid data to save.
    path : str
        Path to the output file.
    session : Session, optional
        ChimeraX session.
        
    Returns
    -------
    None
    """
    from chimerax.core.errors import UserError
    from .grid import write_lammps_grid

    try:
        write_lammps_grid(grid_data, path)
    except Exception as e:
        raise UserError(f"Error writing LAMMPS grid file: {str(e)}")

# Register the format when module is imported
register_grid_format()
