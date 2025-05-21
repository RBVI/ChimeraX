/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
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
