// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Routines to compute bounds.
//
#ifndef BOUNDS_HEADER_INCLUDED
#define BOUNDS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// Return value bounds is a 2 by 3 float32 numpy array xyz_min and xyz_max vectors.
  
// point_bounds(points) -> bounds
PyObject *point_bounds(PyObject *, PyObject *args, PyObject *keywds);
// point_copies_bounds(points, positions) -> bounds.  Positions is N by 3 by 4 float32 array.
PyObject *point_copies_bounds(PyObject *, PyObject *args, PyObject *keywds);
// sphere_xyz_bounds(centers, radii) -> bounds
PyObject *sphere_bounds(PyObject *, PyObject *args, PyObject *keywds);
// axes_sphere_bounds(centers, radii, axes) -> axes_bounds
PyObject *sphere_axes_bounds(PyObject *, PyObject *args, PyObject *keywds);
// spheres_in_bounds(centers, radii, axes, axes_bounds, padding) -> indices
PyObject *spheres_in_bounds(PyObject *, PyObject *args, PyObject *keywds);
// bounds_overlap(bounds1, bounds2, padding) -> true/false
PyObject *bounds_overlap(PyObject *, PyObject *args, PyObject *keywds);
// points_within_planes(points, planes) -> point mask
PyObject *points_within_planes(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
