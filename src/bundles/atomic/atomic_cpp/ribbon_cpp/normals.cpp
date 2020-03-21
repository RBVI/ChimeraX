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

// Need _USE_MATH_DEFINES on Windows to get M_PI from cmath
#define _USE_MATH_DEFINES
#include <cmath>			// use std:isnan()

#include <arrays/pythonarray.h>		// use parse_uint8_n_array()
#include <arrays/rcarray.h>		// use CArray

#define DEBUG_CONSTRAINED_NORMALS   0

#define FLIP_MINIMIZE   0
#define FLIP_PREVENT    1
#define FLIP_FORCE      2

inline float inner(const float* u, const float* v)
{
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

inline float* cross(const float* u, const float* v, float* result)
{
    result[0] = u[1]*v[2] - u[2]*v[1];
    result[1] = u[2]*v[0] - u[0]*v[2];
    result[2] = u[0]*v[1] - u[1]*v[0];
    return result;
}

// -------------------------------------------------------------------------
// ribbon functions

static void _rotate_around(float* n, float c, float s, float* v)
{
    float c1 = 1 - c;
    float m00 = c + n[0] * n[0] * c1;
    float m01 = n[0] * n[1] * c1 - s * n[2];
    float m02 = n[2] * n[0] * c1 + s * n[1];
    float m10 = n[0] * n[1] * c1 + s * n[2];
    float m11 = c + n[1] * n[1] * c1;
    float m12 = n[2] * n[1] * c1 - s * n[0];
    float m20 = n[0] * n[2] * c1 - s * n[1];
    float m21 = n[1] * n[2] * c1 + s * n[0];
    float m22 = c + n[2] * n[2] * c1;
    // Use temporary so that v[0] does not get set too soon
    float x = m00 * v[0] + m01 * v[1] + m02 * v[2];
    float y = m10 * v[0] + m11 * v[1] + m12 * v[2];
    float z = m20 * v[0] + m21 * v[1] + m22 * v[2];
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

static void _parallel_transport_normals(int num_pts, float* tangents, float* n0, float* normals)
{
    // First normal is same as given normal
    normals[0] = n0[0];
    normals[1] = n0[1];
    normals[2] = n0[2];
    // n: normal updated at each step
    // b: binormal defined by cross product of two consecutive tangents
    // b_hat: normalized b
    float n[3] = { n0[0], n0[1], n0[2] };
    float b[3];
    float b_hat[3];
    for (int i = 1; i != num_pts; ++i) {
        float *ti1 = tangents + (i - 1) * 3;
        float *ti = ti1 + 3;
        cross(ti1, ti, b);
        float b_len = sqrtf(inner(b, b));
        if (!std::isnan(b_len) && b_len > 0) {
            b_hat[0] = b[0] / b_len;
            b_hat[1] = b[1] / b_len;
            b_hat[2] = b[2] / b_len;
            float c = inner(ti1, ti);
            if (!std::isnan(c)) {
                float s = sqrtf(1 - c*c);
                if (!std::isnan(s))
                    _rotate_around(b_hat, c, s, n);
            }
        }
        float *ni = normals + i * 3;
        ni[0] = n[0];
        ni[1] = n[1];
        ni[2] = n[2];
    }
}

inline float delta_to_angle(float twist, float f)
{
    // twist is total twist
    // f is between 0 and 1
    // linear interpolation - show cusp artifact
    // return twist * f;
    // cosine interpolation - second degree continuity
    // return (1 - cos(f * M_PI)) / 2 * twist;
    // sigmoidal interpolation - second degree continuity
    return (1.0 / (1 + exp(-8.0 * (f - 0.5)))) * twist;
}

#if DEBUG_CONSTRAINED_NORMALS > 0
inline float rad2deg(float r)
{
    return 180.0 / M_PI * r;
}
#endif

static void constrained_normals(float *tangents, int num_pts, float *n_start, float *n_end,
				int flip_mode, bool start_flipped, bool end_flipped, bool no_twist,
				float *normals, bool *return_need_flip)
{
#if DEBUG_CONSTRAINED_NORMALS > 0
    std::cerr << "constrained_normals\n";
#endif
    // Convert Python objects to arrays and pointers
#if DEBUG_CONSTRAINED_NORMALS > 0
    std::cerr << "n_start" << ' ' << n_start[0] << ' ' << n_start[1] << ' ' << n_start[2] << '\n';
    std::cerr << "n_end" << ' ' << n_end[0] << ' ' << n_end[1] << ' ' << n_end[2] << '\n';
    std::cerr << "start inner: " << inner(n_start, tangents)
        << " end inner: " << inner(n_end, tangents + num_pts * 3) << '\n';
#if DEBUG_CONSTRAINED_NORMALS > 1
    std::cerr << "tangents\n";
    for (int i = 0; i != num_pts; ++i) {
        float *tp = tangents + i * 3;
        std::cerr << "  " << i << ' ' << tp[0] << ' ' << tp[1] << ' ' << tp[2] << '\n';
    }
#endif
#endif
    _parallel_transport_normals(num_pts, tangents, n_start, normals);
#if DEBUG_CONSTRAINED_NORMALS > 1
    std::cerr << "returned from _parallel_transport_normals\n";
    for (int i = 0; i != num_pts; ++i) {
        float *np = normals + i * 3;
        std::cerr << "  " << i << ' ' << np[0] << ' ' << np[1] << ' ' << np[2] << '\n';
    }
#endif
    // Then figure out what twist is needed to make the
    // ribbon end up with the desired ending normal
    float* n = normals + (num_pts - 1) * 3;
    float other_end[3] = { n_end[0], n_end[1], n_end[2] };
    float twist = 0;
    bool need_flip = false;
    if (!no_twist) {
        twist = acos(inner(n, n_end));
        if (std::isnan(twist))
            twist = 0;
#if DEBUG_CONSTRAINED_NORMALS > 0
        std::cerr << "initial twist " << rad2deg(twist) << " degrees, sqlen(n): "
            << inner(n, n) << " sqlen(other_end): " << inner(other_end, other_end) << "\n";
#endif
        // Now we figure out whether to flip the ribbon or not
        if (flip_mode == FLIP_MINIMIZE) {
            // If twist is greater than 90 degrees, turn the opposite
            // direction.  (Assumes that ribbons are symmetric.)
            if (twist > 0.6 * M_PI)
            // if (twist > M_PI / 2)
                need_flip = true;
        } else if (flip_mode == FLIP_PREVENT) {
            // Make end_flip the same as start_flip
            if (end_flipped != start_flipped)
                need_flip = true;
        } else if (flip_mode == FLIP_FORCE) {
            // Make end_flip the opposite of start_flip
            if (end_flipped == start_flipped)
                need_flip = true;
        }
#if DEBUG_CONSTRAINED_NORMALS > 0
        std::cerr << "flip_mode: " << flip_mode << " start_flipped: " << start_flipped
                  << " end_flipped: " << end_flipped << " need_flip: " << need_flip << '\n';
#endif
        if (need_flip) {
#if DEBUG_CONSTRAINED_NORMALS > 0
            std::cerr << "flipped twist " << rad2deg(twist) << " degrees, sqlen(n): " << inner(n, n)
                      << " sqlen(other_end): " << inner(other_end, other_end) << "\n";
#endif
            for (int i = 0; i != 3; ++i)
                other_end[i] = -n_end[i];
            twist = acos(inner(n, other_end));
        }
        // Figure out direction of twist (right-hand rule)
        float *last_tangent = tangents + (num_pts - 1) * 3;
        float tmp[3];
        if (inner(cross(n, other_end, tmp), last_tangent) < 0)
            twist = -twist;
    }
#if DEBUG_CONSTRAINED_NORMALS > 0
    std::cerr << "final twist " << rad2deg(twist) << " degrees, need_flip " << need_flip << "\n";
#endif
    // Compute fraction per step
    float delta = 1.0 / (num_pts - 1);
#if DEBUG_CONSTRAINED_NORMALS > 0
    std::cerr << "per step delta " << delta << "\n";
#endif
    // Apply twist to each normal along path
    for (int i = 1; i != num_pts; ++i) {
        int offset = i * 3;
        float angle = delta_to_angle(twist, i * delta);
        float c = cos(angle);
        float s = sin(angle);
#if DEBUG_CONSTRAINED_NORMALS > 1
        float before = inner(tangents + offset, normals + offset);
        std::cerr << "twist " << i << " angle " << angle << " -> ";
#endif
        _rotate_around(tangents + offset, c, s, normals + offset);
#if DEBUG_CONSTRAINED_NORMALS > 1
        float after = inner(tangents + offset, normals + offset);
        float* n = normals + offset;
        std::cerr << n[0] << ' ' << n[1] << ' ' << n[2]
            << " before/after: " << before << ' ' << after << '\n';
#endif
    }
#if DEBUG_CONSTRAINED_NORMALS > 1
    float *last_n = normals + (num_pts - 1) * 3;
    std::cerr << "check: last n: " << last_n[0] << ' ' << last_n[1] << ' ' << last_n[2]
            << " other_end: " << other_end[0] << ' ' << other_end[1] << ' ' << other_end[2]
            << " dot: " << inner(last_n, other_end) << '\n';
#endif
#if DEBUG_CONSTRAINED_NORMALS > 0
    if (fabs(inner(normals + (num_pts - 1) * 3, other_end)) < (1 - 1e-2))
        std::cerr << "***** WRONG ROTATION *****\n";
#endif

    *return_need_flip = need_flip;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
ribbon_constrained_normals(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray tangents;
  float start_normal[3], end_normal[3];
  int flip_mode;
  int start_flipped, end_flipped, no_twist; // bool values
  const char *kwlist[] = {"tangents", "start_normal", "end_normal",
			  "flip_mode", "start_flipped", "end_flipped", "no_twist", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&ippp"),
				   (char **)kwlist,
				   parse_float_n3_array, &tangents,
				   parse_float_3_array, &start_normal[0],
				   parse_float_3_array, &end_normal[0],
				   &flip_mode, &start_flipped, &end_flipped, &no_twist))
    return NULL;

  FArray tang = tangents.contiguous_array();
  bool need_flip = false;
  float *normals = NULL;
  int num_pts = tang.size(0);
  PyObject *py_normals = python_float_array(num_pts, 3, &normals);
  constrained_normals(tang.values(), num_pts, &start_normal[0], &end_normal[0],
		      flip_mode, start_flipped, end_flipped, no_twist,
		      normals, &need_flip);

  return python_tuple(py_normals, python_bool(need_flip));
}
