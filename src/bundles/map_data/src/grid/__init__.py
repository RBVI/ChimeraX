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

from chimerax.core.errors import UserError

def open(path, session, name = None, array_mode = None):
    """
    Read LAMMPS grid format files.
    
    Parameters
    ----------
    path : str
        Path to file to read.
    session : Session instance
        For reporting progress.
    name : str
        Name to assign to the grid data.
    array_mode : str
        Specifies whether to return full data arrays.
        Can be None (return data arrays), 'mmap' (memory map file),
        or 'header-only' (don't read arrays).
    
    Returns
    -------
    list of GridData objects
    """
    from .grid_format import read_lammps_grid
    
    if path.endswith('.gz'):
        import gzip
        f = gzip.open(path, 'rt')
    else:
        f = open(path, 'r')
    
    try:
        data = read_lammps_grid(f, session, name)
    except Exception as e:
        raise UserError(f"Error reading LAMMPS grid file: {str(e)}")
    finally:
        f.close()
    
    return data

def save(grid_data, path, session = None, options = {}):
    """
    Write LAMMPS grid format files.
    
    Parameters
    ----------
    grid_data : GridData or sequence of GridData
        Grid data to save.
    path : str
        Path to write file to.
    session : Session instance
        For reporting progress.
    options : dict
        Save options. Not used.
    """
    from .grid_format import write_lammps_grid
    
    try:
        write_lammps_grid(grid_data, path)
    except Exception as e:
        raise UserError(f"Error writing LAMMPS grid file: {str(e)}")
