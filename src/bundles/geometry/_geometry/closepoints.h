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
