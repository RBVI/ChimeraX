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
// Routines to compute rotation of standard cylinder aligned along z axis
// to have specified end points.  For drawing molecule bonds.
//
#ifndef CYLINDERROT_HEADER_INCLUDED
#define CYLINDERROT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// cylinder_rotations(xyz0, xyz1, radii, rot44)
PyObject *cylinder_rotations(PyObject *, PyObject *args, PyObject *keywds);
// half_cylinder_rotations(xyz0, xyz1, radii, rot44)
PyObject *half_cylinder_rotations(PyObject *, PyObject *args, PyObject *keywds);
// cylinder_rotations_x3d(xyz0, xyz1, radii, float [n * 9])
PyObject *cylinder_rotations_x3d(PyObject *, PyObject *args, PyObject *keywds);
}

#endif
