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
LAMMPS grid format parsing and writing.
"""

import os

def read_lammps_grid(stream, session = None, name = None):
    """
    Read a LAMMPS grid format file and return GridData objects.
    
    Parameters
    ----------
    stream : file-like object
        Stream containing the grid data.
    session : Session instance, optional
        For reporting progress.
    name : str, optional
        Name to assign to the grid data.
    
    Returns
    -------
    list of GridData objects
    """
    from .grid_grid import parse_lammps_grid
    
    # Get base name if name is not specified
    if name is None and hasattr(stream, 'name'):
        name = os.path.basename(stream.name)
    
    # Read and parse the grid data
    return parse_lammps_grid(stream, name=name)

def write_lammps_grid(grid_data, path):
    """
    Write grid data to a LAMMPS grid format file.
    
    Parameters
    ----------
    grid_data : GridData or sequence of GridData
        Grid data to save.
    path : str
        Path to write the file to.
    
    Returns
    -------
    None
    """
    from .grid_grid import write_lammps_grid_file
    
    # Use the grid_grid implementation
    write_lammps_grid_file(grid_data, path)
