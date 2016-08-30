// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Extend a map using crystal symmetry.
//
#ifndef EXTENDMAP_HEADER_INCLUDED
#define EXTENDMAP_HEADER_INCLUDED

#include <Python.h>			// use PyObject

namespace Map_Cpp
{

extern "C" {
  
// ----------------------------------------------------------------------------
// Compute map values by interpolating another map at positions related by
// symmetry operators and periodic unit cell symmetries.
// Returns the number of grid point that were not covered (assigned 0) and
// the maximum discrepancy for points that were multiply covered.  Multiply
// covered points receive the average value of all symmetric positions.
// Output array must have 32-bit float values.
//
//  extend_crystal_map(in_array, ijk_cell_size, ijk_symmetries, out_array, out_ijk_to_in_ijk_transform)
//
PyObject *extend_crystal_map(PyObject *, PyObject *args, PyObject *keywds);

}	// end extern C

}	// end of namespace Map_Cpp

#endif
