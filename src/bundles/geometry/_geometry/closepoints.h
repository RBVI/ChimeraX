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
// Find pairs of close points given two sets of points and a distance.
//

#ifndef CLOSEPOINTS_HEADER_INCLUDED
#define CLOSEPOINTS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// find_close_points(xyz1, xyz2, max_dist) -> (indices1, indices2)
extern "C" PyObject *find_close_points(PyObject *, PyObject *args, PyObject *keywds);
extern const char *find_close_points_doc;

// find_closest_points(xyz1, xyz2, max_dist) -> (indices1, indices2, nearest1)
extern "C" PyObject *find_closest_points(PyObject *, PyObject *args, PyObject *keywds);
extern const char *find_closest_points_doc;

// find_close_points_sets(tp1, tp2, max_dist) -> (indices1, indices2) with tp1 = [(transform1, xyz1), ...]
extern "C" PyObject *find_close_points_sets(PyObject *, PyObject *args, PyObject *keywds);
extern const char *find_close_points_sets_doc;
}

#endif
