// vi: set expandtab shiftwidth=4 softtabstop=4:
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
  for (int i = 0 ; i < n ; ++i)
    {
      float vx = *axyz1++ - *axyz0++;
      float vy = *axyz1++ - *axyz0++;
      float vz = *axyz1++ - *axyz0++;
      float d = sqrtf(vx*vx + vy*vy + vz*vz);
      if (d == 0)
	{ vx = vy = 0 ; vz = 1; }
      else
	{ vx /= d; vy /= d; vz /= d; }

      float c = vz, c1;
      if (c <= -1)
	c1 = 0;       // Degenerate -z axis case.
      else
	c1 = 1.0/(1+c);

      float wx = -vy, wy = vx;
      float cx = c1*wx, cy = c1*wy;
      float r = *radii++;
      float h = d;

      *rot44++ = r*(cx*wx + c);
      *rot44++ = r*cy*wx;
      *rot44++ = -r*wy;
      *rot44++ = 0;

      *rot44++ = r*cx*wy;
      *rot44++ = r*(cy*wy + c);
      *rot44++ = r*wx;
      *rot44++ = 0;

      *rot44++ = h*wy;
      *rot44++ = -h*wx;
      *rot44++ = h*c;
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

  int n = xyz0.size(0);
  if (xyz1.size(0) != n || radii.size(0) != n)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end-point and radii arrays must have same size, got %d and %d",
			n, xyz1.size(0), radii.size(0));
  if (rot44.size(0) != n || rot44.size(1) != 4 || rot44.size(2) != 4)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder rotations wrong size, got %d %d %d, expected %d 4 4",
			rot44.size(0), rot44.size(1), rot44.size(2), n);
  if (!xyz0.is_contiguous() || !xyz1.is_contiguous() || !radii.is_contiguous()
      || !rot44.is_contiguous())
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end point, radii or rotation array not contiguous.");

  cylinder_rotations(xyz0.values(), xyz1.values(), n, radii.values(), rot44.values());

  return python_none();
}

// -----------------------------------------------------------------------------
//
static void cylinder_rotations_x3d(float *axyz0, float *axyz1, int n,
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
    for (int i = 0 ; i < n * STRIDE ; i += STRIDE) {
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

  int n = xyz0.size(0);
  if (xyz1.size(0) != n || radii.size(0) != n)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end-point and radii arrays must have same size, got %d and %d",
			n, xyz1.size(0), radii.size(0));
  if (info.size(0) != n || info.size(1) != 9)
    return PyErr_Format(PyExc_ValueError,
			"Cylinder rotations wrong size, got %d %d, expected %d 9",
			info.size(0), info.size(1), n);
  if (!xyz0.is_contiguous() || !xyz1.is_contiguous() || !radii.is_contiguous()
      || !info.is_contiguous())
    return PyErr_Format(PyExc_ValueError,
			"Cylinder end point, radii or rotation array not contiguous.");

  cylinder_rotations_x3d(xyz0.values(), xyz1.values(), n, radii.values(), info.values());

  return python_none();
}
