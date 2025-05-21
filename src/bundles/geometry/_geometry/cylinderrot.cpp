// vi: set expandtab shiftwidth=4 softtabstop=4:

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
// Transform points with shift, scale and linear operations.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use sqrtf()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use call_template_function()

#ifndef M_PI
# define M_PI		3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
//
static void cylinder_rotations(float *axyz0, float *axyz1, int n, float *radii,
			       float *rot44)
{
  for (int i = 0 ; i < n ; ++i, axyz0 += 3, axyz1 += 3)
    {
      float vx = axyz1[0] - axyz0[0];
      float vy = axyz1[1] - axyz0[1];
      float vz = axyz1[2] - axyz0[2];
      float h = sqrtf(vx*vx + vy*vy + vz*vz);
      if (h == 0)
	{ vx = vy = 0 ; vz = 1; }
      else
	{ vx /= h; vy /= h; vz /= h; }

      float r = *radii++;
      float sx = r, sy = r, sz = h;
      
      // Avoid degenerate vz = -1 case.
      if (vz < 0)
	{ vx = -vx; vy = -vy; vz = -vz; sx = -r; sz = -h; }

      float c1 = 1.0/(1+vz);
      float vxx = c1*vx*vx, vyy = c1*vy*vy, vxy = c1*vx*vy;

      *rot44++ = sx*(vyy + vz);
      *rot44++ = -sx*vxy;
      *rot44++ = -sx*vx;
      *rot44++ = 0;

      *rot44++ = -sy*vxy;
      *rot44++ = sy*(vxx + vz);
      *rot44++ = -sy*vy;
      *rot44++ = 0;

      *rot44++ = sz*vx;
      *rot44++ = sz*vy;
      *rot44++ = sz*vz;
      *rot44++ = 0;

      *rot44++ = 0;
      *rot44++ = 0;
      *rot44++ = 0;
      *rot44++ = 1;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *cylinder_rotations(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz0, xyz1, radii, rot44;
  const char *kwlist[] = {"xyz0", "xyz1", "radii", "rot44", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz0,
				   parse_float_n3_array, &xyz1,
				   parse_float_n_array, &radii,
				   parse_writable_float_3d_array, &rot44))
    return NULL;

  int64_t n = xyz0.size(0);
  if (xyz1.size(0) != n || radii.size(0) != n)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end-point and radii arrays must have same size, got sizes %s %s %s",
			xyz0.size_string(0).c_str(), xyz1.size_string(0).c_str(),
			radii.size_string(0).c_str());
  if (rot44.size(0) != n || rot44.size(1) != 4 || rot44.size(2) != 4)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder rotations wrong size, got (%s), expected (%s,4,4)",
			rot44.size_string().c_str(), xyz0.size_string(0).c_str());
  if (!xyz0.is_contiguous() || !xyz1.is_contiguous() || !radii.is_contiguous()
      || !rot44.is_contiguous())
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end point, radii or rotation array not contiguous.");

  cylinder_rotations(xyz0.values(), xyz1.values(), n, radii.values(), rot44.values());

  return python_none();
}

// -----------------------------------------------------------------------------
//
static void half_cylinder_rotations(float *axyz0, float *axyz1, int64_t n, float *radii,
				    float *rot44)
{
  float *rot44b = rot44 + 16*n;
  for (int64_t i = 0 ; i < n ; ++i, axyz0 += 3, axyz1 += 3)
    {
      float x0 = axyz0[0], x1 = axyz1[0];
      float y0 = axyz0[1], y1 = axyz1[1];
      float z0 = axyz0[2], z1 = axyz1[2];
      float vx = x1-x0, vy = y1-y0, vz = z1-z0;
      float h = sqrtf(vx*vx + vy*vy + vz*vz);
      if (h == 0)
	{ vx = vy = 0 ; vz = 1; }
      else
	{ vx /= h; vy /= h; vz /= h; }

      float r = *radii++;
      float sx = r, sy = r, sz = h;

      // Avoid degenerate vz = -1 case.
      if (vz < 0)
	{ vx = -vx; vy = -vy; vz = -vz; sx = -r; sz = -h; }

      float c1 = 1.0/(1+vz);
      float vxx = c1*vx*vx, vyy = c1*vy*vy, vxy = c1*vx*vy;

      *rot44++ = *rot44b++ = sx*(vyy + vz);
      *rot44++ = *rot44b++ = -sx*vxy;
      *rot44++ = *rot44b++ = -sx*vx;
      *rot44++ = *rot44b++ = 0;

      *rot44++ = *rot44b++ = -sy*vxy;
      *rot44++ = *rot44b++ = sy*(vxx + vz);
      *rot44++ = *rot44b++ = -sy*vy;
      *rot44++ = *rot44b++ = 0;

      *rot44++ = *rot44b++ = sz*vx;
      *rot44++ = *rot44b++ = sz*vy;
      *rot44++ = *rot44b++ = sz*vz;
      *rot44++ = *rot44b++ = 0;

      *rot44++ = .75*x0 + .25*x1;
      *rot44++ = .75*y0 + .25*y1;
      *rot44++ = .75*z0 + .25*z1;
      *rot44++ = 1;

      *rot44b++ = .25*x0 + .75*x1;
      *rot44b++ = .25*y0 + .75*y1;
      *rot44b++ = .25*z0 + .75*z1;
      *rot44b++ = 1;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *half_cylinder_rotations(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz0, xyz1, radii, rot44;
  const char *kwlist[] = {"xyz0", "xyz1", "radii", "rot44", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz0,
				   parse_float_n3_array, &xyz1,
				   parse_float_n_array, &radii,
				   parse_writable_float_3d_array, &rot44))
    return NULL;

  int64_t n = xyz0.size(0);
  if (xyz1.size(0) != n || radii.size(0) != n)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end-point and radii arrays must have same size, got sizes %s %s %s",
			xyz0.size_string(0).c_str(), xyz1.size_string(0).c_str(),
			radii.size_string(0).c_str());
  if (rot44.size(0) != 2*n || rot44.size(1) != 4 || rot44.size(2) != 4)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder rotations wrong size, got (%s), expected (2*%s,4,4)",
			rot44.size_string().c_str(), xyz0.size_string(0).c_str());
  if (!xyz0.is_contiguous() || !xyz1.is_contiguous() || !radii.is_contiguous()
      || !rot44.is_contiguous())
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end point, radii or rotation array not contiguous.");

  half_cylinder_rotations(xyz0.values(), xyz1.values(), n, radii.values(), rot44.values());

  return python_none();
}

// -----------------------------------------------------------------------------
//
static void cylinder_rotations_x3d(float *axyz0, float *axyz1, int64_t n,
                                   float *radii, float *info)
{
# define OFF_HEIGHT 0
# define OFF_RADIUS 1
# define OFF_ROT_AXIS_X 2
# define OFF_ROT_AXIS_Y 3
# define OFF_ROT_AXIS_Z 4
# define OFF_ROT_ANGLE 5
# define OFF_TRANS_X 6
# define OFF_TRANS_Y 7
# define OFF_TRANS_Z 8
# define STRIDE 9
    // Object axis, in this case the cylinder axis
    // X3D cylinders are along y axis
# define OBJ_AXIS_X 0.0f
# define OBJ_AXIS_Y 1.0f
# define OBJ_AXIS_Z 0.0f
    float rx, ry, rz, angle;
    for (int64_t i = 0 ; i < n * STRIDE ; i += STRIDE) {
        float vx = *axyz1++ - *axyz0++;
        float vy = *axyz1++ - *axyz0++;
        float vz = *axyz1++ - *axyz0++;
        float d = sqrtf(vx * vx + vy * vy + vz * vz);
        if (d <= 0) {
            rx = ry = 0 ; rz = 1; angle = 0;
        } else {
            vx /= d; vy /= d; vz /= d;
            // rotation axis is cross product of v and cylinder axis
            rx = OBJ_AXIS_Y * vz - OBJ_AXIS_Z * vy;
            ry = OBJ_AXIS_Z * vx - OBJ_AXIS_X * vz;
            rz = OBJ_AXIS_X * vy - OBJ_AXIS_Y * vx;
            float cosine = OBJ_AXIS_X * vx + OBJ_AXIS_Y * vy + OBJ_AXIS_Z * vz;
            angle = acosf(cosine);
            float r_sq_len = rx * rx + ry * ry + rz * rz;

            if (r_sq_len > 0) {
                float r_len = sqrtf(r_sq_len);
                rx /= r_len;
                ry /= r_len;
                rz /= r_len;
            } else if (cosine < 0) { // Only happens if axis = -cyl_axis
                if (OBJ_AXIS_Z == 0) {
                    rx = ry = 0; rz = 1;
                } else {
                    rx = -OBJ_AXIS_Y; ry = OBJ_AXIS_X; rz = 0;
                }
                angle = float(M_PI);
            }
        }

        info[i + OFF_HEIGHT] = d * 0.5;
        info[i + OFF_RADIUS] = *radii++;
        info[i + OFF_ROT_AXIS_X] = rx;
        info[i + OFF_ROT_AXIS_Y] = ry;
        info[i + OFF_ROT_AXIS_Z] = rz;
        info[i + OFF_ROT_ANGLE] = angle;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *cylinder_rotations_x3d(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz0, xyz1, radii, info;
  const char *kwlist[] = {"xyz0", "xyz1", "radii", "info", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz0,
				   parse_float_n3_array, &xyz1,
				   parse_float_n_array, &radii,
				   parse_writable_float_n9_array, &info))
    return NULL;

  int64_t n = xyz0.size(0);
  if (xyz1.size(0) != n || radii.size(0) != n)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end-point and radii arrays must have same size, got sizes %s %s %s",
			xyz0.size_string(0).c_str(), xyz1.size_string(0).c_str(),
			radii.size_string(0).c_str());
  if (info.size(0) != n || info.size(1) != 9)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder rotations wrong size, got (%s), expected (%s, 9)",
			info.size_string().c_str(), xyz0.size_string(0).c_str());
  if (!xyz0.is_contiguous() || !xyz1.is_contiguous() || !radii.is_contiguous()
      || !info.is_contiguous())
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end point, radii or rotation array not contiguous.");

  cylinder_rotations_x3d(xyz0.values(), xyz1.values(), n, radii.values(), info.values());

  return python_none();
}
