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
LAMMPS grid3d format parsing.
"""

print("*** ok 2")

import os
import numpy as np
from chimerax.map_data import ArrayGridData

def read_lammps_grid3d(stream, session = None, name = None):
    """
    Read a LAMMPS grid3d format file and return GridData objects.

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

    print("*** ok 3")

    # Get base name if name is not specified
    if name is None and hasattr(stream, 'name'):
        name = os.path.basename(stream.name)
    
    grids = []
    
    # Get base name from name parameter
    base_name = name if name else 'lammps_grid'
    
    # Process the file line by line
    lines = stream.readlines()
    
    # Variables to store during parsing
    timesteps = []
    grid_size = None
    origin = None
    cell_size = None
    columns = None
    current_timestep_data = None
    
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        
        # Parse ITEM: TIMESTEP
        if line.startswith("ITEM: TIMESTEP"):
            # Save previous timestep data if exists
            if current_timestep_data is not None and grid_size is not None:
                grids.extend(_create_grids_from_data(current_timestep_data, grid_size, 
                                                    origin, cell_size, columns, 
                                                    timesteps[-1], base_name))
            
            line_index += 1
            timestep = int(lines[line_index].strip())
            timesteps.append(timestep)
            current_timestep_data = None
        
        # Parse ITEM: GRID SIZE
        elif line.startswith("ITEM: GRID SIZE"):
            line_index += 1
            parts = lines[line_index].strip().split()
            if len(parts) >= 3:
                # In LAMMPS, grid dimensions are typically (nx, ny, nz)
                grid_size = (int(parts[0]), int(parts[1]), int(parts[2]))
        
        # Parse ITEM: GRID ORIGIN
        elif line.startswith("ITEM: GRID ORIGIN"):
            line_index += 1
            parts = lines[line_index].strip().split()
            if len(parts) >= 3:
                origin = (float(parts[0]), float(parts[1]), float(parts[2]))
        
        # Parse ITEM: GRID SPACING
        elif line.startswith("ITEM: GRID SPACING"):
            line_index += 1
            parts = lines[line_index].strip().split()
            if len(parts) >= 3:
                cell_size = (float(parts[0]), float(parts[1]), float(parts[2]))
        
        # Parse ITEM: GRID CELLS
        elif line.startswith("ITEM: GRID CELLS"):
            header = line.split()
            # Skip "ITEM: GRID CELLS" and get column names
            columns = header[3:]
            
            if grid_size is None:
                raise ValueError("Grid size not defined before GRID CELLS section")
            
            nx, ny, nz = grid_size
            
            # Initialize data array - LAMMPS usually writes in C-order (row-major)
            # We need to prepare for potentially non-consecutive indices
            current_timestep_data = {}
            for idx, col in enumerate(columns):
                current_timestep_data[col] = np.zeros((nx, ny, nz), dtype=np.float32)
            
            # Read grid data
            cell_count = 0
            while cell_count < nx * ny * nz:
                line_index += 1
                if line_index >= len(lines) or lines[line_index].strip().startswith("ITEM:"):
                    break
                
                parts = lines[line_index].strip().split()
                if len(parts) < 3 + len(columns):
                    continue  # Skip malformed lines
                
                # LAMMPS output is: ix iy iz value1 value2 ...
                ix, iy, iz = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Verify indices are within grid bounds
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    for col_idx, col_name in enumerate(columns):
                        value = float(parts[3 + col_idx])
                        current_timestep_data[col_name][ix, iy, iz] = value
                
                cell_count += 1
            
            # Don't advance the line counter here, as we might have hit a new ITEM:
            continue
        
        line_index += 1
    
    # Process the last timestep data if it exists
    if current_timestep_data is not None and grid_size is not None:
        grids.extend(_create_grids_from_data(current_timestep_data, grid_size, 
                                           origin, cell_size, columns, 
                                           timesteps[-1] if timesteps else 0, 
                                           base_name))
    
    if not grids:
        raise ValueError("No grid data found in file")
    
    return grids

