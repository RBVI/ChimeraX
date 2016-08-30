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
// Compute linear combination of matrices.  5x faster than numpy.
//
#ifndef COMBINE_HEADER_INCLUDED
#define COMBINE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// m = f1*m1 + f2*m2.  Requires 3-d contiguous arrays of same type.
// void linear_combination(float f1, PyObject *m1, float f2, PyObject *m2, PyObject *m);
PyObject *linear_combination(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
