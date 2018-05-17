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

#ifndef OCCUPANCY_HEADER_INCLUDED
#define OCCUPANCY_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

// Each xyz point contributes to the nearest 8 grid bins.
// Occupancy grid indices are ordered z, y, x and must be a NumPy float32 array.
PyObject *fill_occupancy_map(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
