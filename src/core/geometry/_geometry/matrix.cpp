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
// Compute matrices
//
#include <Python.h>			// use PyObject
#include <cmath>            // sqrt

#include <arrays/pythonarray.h>		// use array_from_python()
#include "matrix.h"

inline void normalize(double *xyz)
{
    double len = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
    if (len == 0.0) return;
    xyz[0] /= len;
    xyz[1] /= len;
    xyz[2] /= len;
}

inline void cross(const double *v1, const double *v2, double *out)
{
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

static PyObject *look_at(double* from_pt, double* to_pt, double* up)
{
    double x[3], y[3], z[3];
    double diff[3], trans[3];
    for (int i = 0; i < 3; ++i) {
        diff[i] = to_pt[i] - from_pt[i];
    }
    normalize(diff);
    cross(up, diff, x);
    normalize(x);
    cross(diff, x, y);
    normalize(y);
    z[0] = -diff[0]; z[1] = -diff[1]; z[2] = -diff[2];
    auto fx = from_pt[0];
    auto fy = from_pt[1];
    auto fz = from_pt[2];
    trans[0] = 0.0 - (x[0]*fx + x[1]*fy + x[2]*fz);
    trans[1] = 0.0 - (y[0]*fx + y[1]*fy + y[2]*fz);
    trans[2] = 0.0 - (z[0]*fx + z[1]*fy + z[2]*fz);
    double matrix[12];
    double* m = &matrix[0];
    *m++ = x[0]; *m++ = x[1], *m++ = x[2]; *m++ = trans[0];
    *m++ = y[0]; *m++ = y[1], *m++ = y[2]; *m++ = trans[1];
    *m++ = z[0]; *m++ = z[1], *m++ = z[2]; *m = trans[2];
    return c_array_to_python(matrix, 3, 4);
}

extern "C" PyObject *look_at(PyObject *, PyObject *args)
{
  double from_pt[3], to_pt[3], up[3];
  if (PyArg_ParseTuple(args, const_cast<char *>("O&O&O&"),
		       parse_double_3_array, from_pt,
		       parse_double_3_array, to_pt,
		       parse_double_3_array, up))
    {
      return look_at(from_pt, to_pt, up);
    }
  return NULL;
}
