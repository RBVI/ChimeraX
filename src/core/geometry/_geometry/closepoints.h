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
