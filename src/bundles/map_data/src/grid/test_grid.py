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

# Place this file in src/bundles/map_data/src/grid/test_grid.py

"""Unit tests for LAMMPS grid format reader/writer."""

import unittest
import numpy as np
import os
import tempfile
from chimerax.core.errors import UserError
from chimerax.map_data import GridData, ArrayGridData
from chimerax.map_data.grid import read_lammps_grid, write_lammps_grid

class LAMMPSGridTest(unittest.TestCase):
    """Test LAMMPS grid format reader and writer."""
    
    def setUp(self):
        """Set up test case."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, 'test.grid')
        
        # Create a simple test grid
        size = (10, 15, 20)
        origin = (0.0, 0.0, 0.0)
        step = (1.0, 1.0, 1.0)
        
        # Create two test grids with different data
        data1 = np.zeros(size, dtype=np.float32)
        data2 = np.zeros(size, dtype=np.float32)
        
        # Fill with test data
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    data1[i, j, k] = i + j + k
                    data2[i, j, k] = i * j * k
        
        self.grid1 = ArrayGridData(data1, origin=origin, step=step, name="value1")
        self.grid2 = ArrayGridData(data2, origin=origin, step=step, name="value2")
        
        # Write a test grid file
        with open(self.test_file, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: GRID SIZE\n")
            f.write("10 15 20\n")
            f.write("ITEM: GRID ORIGIN\n")
            f.write("0.0 0.0 0.0\n")
            f.write("ITEM: GRID SPACING\n")
            f.write("1.0 1.0 1.0\n")
            f.write("ITEM: GRID CELLS ix iy iz value1 value2\n")
            
            for i in range(10):
                for j in range(15):
                    for k in range(20):
                        val1 = i + j + k
                        val2 = i * j * k
                        f.write(f"{i} {j} {k} {val1} {val2}\n")
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_read_grid(self):
        """Test reading a LAMMPS grid file."""
        # Test with file stream
        with open(self.test_file, 'r') as stream:
            grids = read_lammps_grid(stream)
        
        # Should have two grids for the two columns
        self.assertEqual(len(grids), 2)
        
        # Check grid properties
        for grid in grids:
            self.assertEqual(grid.size, (10, 15, 20))
            self.assertTrue(np.allclose(grid.origin, (0.0, 0.0, 0.0)))
            self.assertTrue(np.allclose(grid.step, (1.0, 1.0, 1.0)))
        
        # Check data values
        value1_grid = next((g for g in grids if "value1" in g.name), None)
        value2_grid = next((g for g in grids if "value2" in g.name), None)
        
        self.assertIsNotNone(value1_grid)
        self.assertIsNotNone(value2_grid)
        
        for i in range(10):
            for j in range(15):
                for k in range(20):
                    self.assertAlmostEqual(value1_grid.matrix()[i, j, k], i + j + k)
                    self.assertAlmostEqual(value2_grid.matrix()[i, j, k], i * j * k)
    
    def test_write_read_grid(self):
        """Test writing and then reading back a LAMMPS grid file."""
        output_file = os.path.join(self.temp_dir.name, 'output.grid')
        
        # Write grids to file
        write_lammps_grid([self.grid1, self.grid2], output_file)
        
        # Read back the file
        with open(output_file, 'r') as stream:
            grids = read_lammps_grid(stream)
        
        # Should have two grids
        self.assertEqual(len(grids), 2)
        
        # Check data values
        value1_grid = next((g for g in grids if "value1" in g.name), None)
        value2_grid = next((g for g in grids if "value2" in g.name), None)
        
        self.assertIsNotNone(value1_grid)
        self.assertIsNotNone(value2_grid)
        
        # Compare with original grids
        data1 = self.grid1.matrix()
        data2 = self.grid2.matrix()
        
        # Check a few sample points
        for i, j, k in [(0, 0, 0), (5, 7, 9), (9, 14, 19)]:
            self.assertAlmostEqual(value1_grid.matrix()[i, j, k], data1[i, j, k])
            self.assertAlmostEqual(value2_grid.matrix()[i, j, k], data2[i, j, k])
    
    def test_error_handling(self):
        """Test error handling for malformed files."""
        # Create a malformed file
        bad_file = os.path.join(self.temp_dir.name, 'bad.grid')
        with open(bad_file, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            # Missing grid size information
            f.write("ITEM: GRID CELLS ix iy iz value\n")
            f.write("0 0 0 1.0\n")
        
        # Should raise an error
        with self.assertRaises(ValueError):
            with open(bad_file, 'r') as stream:
                read_lammps_grid(stream)
    
    def test_compressed_grid(self):
        """Test reading a compressed grid file."""
        import gzip
        import io
        
        # Create a gzipped test file
        gz_file = os.path.join(self.temp_dir.name, 'test.grid.gz')
        
        # Write compressed file
        with gzip.open(gz_file, 'wt') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: GRID SIZE\n")
            f.write("3 3 3\n")
            f.write("ITEM: GRID ORIGIN\n")
            f.write("0.0 0.0 0.0\n")
            f.write("ITEM: GRID SPACING\n")
            f.write("1.0 1.0 1.0\n")
            f.write("ITEM: GRID CELLS ix iy iz value\n")
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = i + j + k
                        f.write(f"{i} {j} {k} {val}\n")
        
        # Read compressed file
        with gzip.open(gz_file, 'rt') as stream:
            grids = read_lammps_grid(stream, name='test.grid.gz')
        
        # Check that it was read correctly
        self.assertEqual(len(grids), 1)
        grid = grids[0]
        self.assertEqual(grid.size, (3, 3, 3))
        
        # Check a few data values
        self.assertAlmostEqual(grid.matrix()[0, 0, 0], 0.0)
        self.assertAlmostEqual(grid.matrix()[1, 1, 1], 3.0)
        self.assertAlmostEqual(grid.matrix()[2, 2, 2], 6.0)

if __name__ == '__main__':
    unittest.main()
