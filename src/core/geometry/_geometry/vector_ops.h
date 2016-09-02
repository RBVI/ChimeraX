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
// Compute linear combination of vectors.  5x faster than numpy.
//
#ifndef VECTOR_OPS_HEADER_INCLUDED
#define VECTOR_OPS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// Computes sum in 64-bit, 1-d arrays only.
PyObject *inner_product_64(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *inner_product_64_doc;

}

#endif
