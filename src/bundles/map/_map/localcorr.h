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
#ifndef LOCALCORR_HEADER_INCLUDED
#define LOCALCORR_HEADER_INCLUDED

#include <Python.h>			// use PyObject

namespace Map_Cpp
{
//
// Compute local correlation between two maps over a moving window of size
// N by N by N voxels.  Maps must be contiguous, input maps must have the
// same value type, and output map must be float32.  The input 3-d numpy
// matrices // must be equal in size and the resulting map is smaller by N-1
// in each dimension.
//
// local_correlation(map1, map2, window_size, subtract_mean, mapc)
extern "C" PyObject *local_correlation(PyObject *, PyObject *args, PyObject *keywds);

}	// end of namespace Map_Cpp

#endif
