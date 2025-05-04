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
LAMMPS grid format support.

This module provides functions to read and write LAMMPS grid format files
created by the LAMMPS 'dump grid' command.
"""

import numpy as np
import os
from chimerax.map_data import GridData, ArrayGridData

def read_lammps_grid(stream, session=None, name=None, array_mode=None):
    """
    Read a LAMMPS grid format file and return GridData objects.
    
    Parameters
    ----------
    stream : io.IOBase
        File stream containing LAMMPS grid data.
    session : Session, optional
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
    grids = []
    
    # Get base name from stream if available
    if hasattr(stream, 'name'):
        base_name = name if name else os.path.basename(stream.name)
    else:
        base_name = name if name else 'lammps_grid'
    
    # Handle both text and binary streams
    if hasattr(stream, 'mode') and 'b' in stream.mode:
        # If binary stream, wrap in TextIOWrapper
        import io
        stream = io.TextIOWrapper(stream, encoding='utf-8')
    
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

def _create_grids_from_data(data_dict, grid_size, origin, cell_size, columns, timestep, base_name):
    """
    Helper function to create GridData objects from parsed data.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping column names to data arrays.
    grid_size : tuple
        (nx, ny, nz) grid dimensions.
    origin : tuple
        (x0, y0, z0) grid origin.
    cell_size : tuple
        (dx, dy, dz) grid spacing.
    columns : list
        List of column names.
    timestep : int
        Timestep for the data.
    base_name : str
        Base name for the grid data.
        
    Returns
    -------
    list of GridData objects
    """
    grids = []
    
    for col_name in columns:
        if col_name not in data_dict:
            continue
            
        # Extract data for this column
        data = data_dict[col_name]
        
        # Create a name for this grid
        grid_name = f"{base_name}_{col_name}"
        if timestep is not None:
            grid_name += f"_timestep_{timestep}"
            
        # Create GridData object
        # Note: ChimeraX expects data in Fortran order (column-major)
        # but LAMMPS outputs in C order (row-major), so we may need to transpose
        # Here we assume data is already properly ordered as nx, ny, nz
        grid = ArrayGridData(data,
                            origin=origin if origin else (0, 0, 0),
                            step=cell_size if cell_size else (1, 1, 1),
                            cell_angles=(90, 90, 90),
                            name=grid_name)
                            
        grids.append(grid)
    
    return grids

def write_lammps_grid(grid_data, path):
    """
    Write grid data to a LAMMPS grid format file.
    
    Parameters
    ----------
    grid_data : GridData or sequence of GridData
        Grid data to save.
    path : str
        Path to the output file.
        
    Returns
    -------
    None
    """
    # Handle single or multiple grid data objects
    if isinstance(grid_data, GridData):
        grids = [grid_data]
    else:
        grids = list(grid_data)
    
    if not grids:
        raise ValueError("No grid data to save")
    
    # All grids should have the same dimensions and origin for LAMMPS format
    ref_grid = grids[0]
    grid_size = ref_grid.size
    origin = ref_grid.origin
    step = ref_grid.step
    
    # Check if all grids have the same dimensions
    for grid in grids[1:]:
        if grid.size != grid_size or not np.allclose(grid.origin, origin) or not np.allclose(grid.step, step):
            raise ValueError("All grids must have the same dimensions, origin, and step size for LAMMPS grid format")
    
    # Write to file
    with open(path, 'w') as f:
        # Only one timestep for now (could be extended to multiple)
        timestep = 0
        
        # Write header
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        
        # Grid size
        nx, ny, nz = grid_size
        f.write("ITEM: GRID SIZE\n")
        f.write(f"{nx} {ny} {nz}\n")
        
        # Grid origin
        f.write("ITEM: GRID ORIGIN\n")
        f.write(f"{origin[0]} {origin[1]} {origin[2]}\n")
        
        # Grid spacing
        f.write("ITEM: GRID SPACING\n")
        f.write(f"{step[0]} {step[1]} {step[2]}\n")
        
        # Prepare column names from grid names
        col_names = [grid.name.split('_')[0] if '_' in grid.name else f"grid{i+1}" 
                    for i, grid in enumerate(grids)]
        
        # Write data header
        f.write(f"ITEM: GRID CELLS ix iy iz {' '.join(col_names)}\n")
        
        # Write grid data
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Collect values from all grids for this cell
                    values = [grid.matrix()[ix, iy, iz] for grid in grids]
                    values_str = ' '.join(f"{val}" for val in values)
                    
                    # Write cell data
                    f.write(f"{ix} {iy} {iz} {values_str}\n")
    
    return None
