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

# -----------------------------------------------------------------------------
# Read LAMMPS grid3d format map data.
#
import os
import numpy

class Lammps_Grid_Data:
    """
    Parses LAMMPS grid3d format files produced by the 'dump grid' command.
    """
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        
        # Parse file to get grid metadata
        if path.endswith('.gz'):
            import gzip
            f = gzip.open(path, 'rt')
        else:
            f = open(path, 'r')

        try:
            self.parse_grid3d(f)
        finally:
            f.close()
            
    def parse_grid3d(self, file):
        """Parse the LAMMPS grid3d file and extract metadata."""
        # LAMMPS grid data variables
        self.timesteps = []
        self.matrix_size = None  # (nx, ny, nz)
        self.origin = None       # (x0, y0, z0)
        self.step = None         # (dx, dy, dz)
        self.columns = None      # List of data column names
        self.data_sections = []  # List of (timestep, start_pos, num_entries)
        self.box_bounds = None   # [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        self.dimension = 3       # Default to 3D
        
        # Read through file to find all sections
        lines = file.readlines()
        line_index = 0
        
        while line_index < len(lines):
            line = lines[line_index].strip()
            
            # Parse ITEM: TIMESTEP
            if line.startswith("ITEM: TIMESTEP"):
                line_index += 1
                timestep = int(lines[line_index].strip())
                self.timesteps.append(timestep)
            
            # Parse ITEM: BOX BOUNDS
            elif line.startswith("ITEM: BOX BOUNDS"):
                self.box_bounds = []
                for dim in range(3):  # Assume 3D box
                    line_index += 1
                    if line_index < len(lines):
                        parts = lines[line_index].strip().split()
                        if len(parts) >= 2:
                            self.box_bounds.append((float(parts[0]), float(parts[1])))
                
                # Calculate origin from box bounds
                if len(self.box_bounds) == 3:
                    self.origin = (self.box_bounds[0][0], self.box_bounds[1][0], self.box_bounds[2][0])
            
            # Parse ITEM: DIMENSION
            elif line.startswith("ITEM: DIMENSION"):
                line_index += 1
                if line_index < len(lines):
                    self.dimension = int(lines[line_index].strip())
            
            # Parse ITEM: GRID SIZE (with nx ny nz tokens)
            elif line.startswith("ITEM: GRID SIZE"):
                line_index += 1
                parts = lines[line_index].strip().split()
                if len(parts) >= 3:
                    # LAMMPS grid dimensions are (nx, ny, nz)
                    self.matrix_size = (int(parts[0]), int(parts[1]), int(parts[2]))
                    
                    # If we have box_bounds and matrix_size but no step, calculate it
                    if self.box_bounds and self.matrix_size and not self.step:
                        nx, ny, nz = self.matrix_size
                        dx = (self.box_bounds[0][1] - self.box_bounds[0][0]) / nx if nx > 1 else 1.0
                        dy = (self.box_bounds[1][1] - self.box_bounds[1][0]) / ny if ny > 1 else 1.0
                        dz = (self.box_bounds[2][1] - self.box_bounds[2][0]) / nz if nz > 1 else 1.0
                        self.step = (dx, dy, dz)
            
            # Parse ITEM: GRID CELLS
            elif line.startswith("ITEM: GRID CELLS"):
                parts = line.split()
                # Skip "ITEM: GRID CELLS" and get column names
                self.columns = parts[3:] if len(parts) > 3 else ["data"]
                
                if self.matrix_size is None:
                    raise ValueError("Grid size not defined before GRID CELLS section")
                    
                # Record the position in the file for later data reading
                start_pos = line_index + 1
                
                # Calculate expected number of data entries
                nx, ny, nz = self.matrix_size
                num_entries = nx * ny * nz
                
                # Record data section info
                current_timestep = self.timesteps[-1] if self.timesteps else 0
                self.data_sections.append((current_timestep, start_pos, num_entries))
                
                # Skip the data section - we'll read it on demand
                line_index += num_entries
                continue
                
            line_index += 1
        
        # Use first column for the initial grid
        if self.columns and len(self.columns) > 0:
            self.current_column = self.columns[0]
        else:
            raise ValueError("No grid data columns found")
        
        # Set element type
        from numpy import float32
        self.element_type = float32
        
        # Validate that we have all required metadata
        if self.matrix_size is None:
            raise ValueError("Grid size not found in file")
        if self.origin is None:
            raise ValueError("Grid origin not found in file")
        if self.step is None:
            raise ValueError("Grid spacing not found in file")
        if not self.data_sections:
            raise ValueError("No grid data sections found in file")
    
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        """
        Read and return the specified submatrix from the file.
        For LAMMPS grid3d format, we need to read and parse the entire grid.
        """
        # Currently just read the first data section and first column
        timestep, start_pos, num_entries = self.data_sections[0]
        
        # Read the whole file again
        if self.path.endswith('.gz'):
            import gzip
            f = gzip.open(self.path, 'rt')
        else:
            f = open(self.path, 'r')
        
        try:
            lines = f.readlines()
            
            # Create numpy array for the data
            nx, ny, nz = self.matrix_size
            data = numpy.zeros((nz, ny, nx), dtype=self.element_type)
            
            # Read the data section
            grid_index = 0
            for i in range(num_entries):
                if progress and i % 1000 == 0:
                    progress.percent = 100.0 * i / num_entries
                
                line_idx = start_pos + i
                if line_idx >= len(lines):
                    break
                    
                line = lines[line_idx].strip()
                if not line:  # Skip empty lines
                    continue
                
                # Parse value - each line just contains a single value
                try:
                    value = float(line)
                except ValueError:
                    continue
                
                # Calculate grid coordinates from linear index
                # Assuming data is ordered with x changing fastest, then y, then z
                # As specified in the LAMMPS documentation
                ix = grid_index % nx
                iy = (grid_index // nx) % ny
                iz = grid_index // (nx * ny)
                
                # Store in data array (in ZYX index order)
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    data[iz, iy, ix] = value
                
                grid_index += 1
            
            # Extract the requested submatrix
            submatrix = self._submatrix(data, ijk_origin, ijk_size, ijk_step)
            return submatrix
            
        finally:
            f.close()
    
    def _submatrix(self, data, ijk_origin, ijk_size, ijk_step):
        """Extract a submatrix from the data array."""
        ox, oy, oz = ijk_origin
        sx, sy, sz = ijk_size
        stx, sty, stz = ijk_step
        
        # Adjust for our ZYX index order
        i0, i1, i2 = oz, oy, ox
        s0, s1, s2 = sz, sy, sx
        st0, st1, st2 = stz, sty, stx
        
        # Extract the submatrix
        if st0 == 1 and st1 == 1 and st2 == 1:
            submatrix = data[i0:i0+s0, i1:i1+s1, i2:i2+s2].copy()
        else:
            submatrix = data[
                i0:i0+s0*st0:st0,
                i1:i1+s1*st1:st1,
                i2:i2+s2*st2:st2].copy()
            
        return submatrix
