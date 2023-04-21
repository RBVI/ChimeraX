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
#include <iostream>
#include <map>				// use std::map
#include <vector>			// use std::vector

#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use python_float_array
#include <atomstruct/Atom.h>		// use Atom
using atomstruct::Atom;
#include <atomstruct/Coord.h>		// use Coord
using atomstruct::Coord;
#include <atomstruct/Residue.h>		// use Residue
using atomstruct::Residue;
using atomstruct::BBE_RIBBON;
#include <atomstruct/Structure.h>	// use Structure
using atomstruct::Structure;


#include "normals.h"			// use parallel_transport_normals, dihedral_angle
#include "parse.h"			// use parse_residues(), parse_string_float_map()

// -----------------------------------------------------------------------------
//
static void cubic_path(const double *c, double tmin, double tmax, int n, float *coords, float *tangents)
{
  double step = (n > 1 ? (tmax - tmin) / (n-1) : 0);
  double x0 = c[0], x1 = c[1], x2 = c[2], x3 = c[3];
  double y0 = c[4], y1 = c[5], y2 = c[6], y3 = c[7];
  double z0 = c[8], z1 = c[9], z2 = c[10], z3 = c[11];
  for (int i = 0 ; i < n ; ++i)
    {
      double t = tmin + i*step;
      double t_2 = 2*t;
      double t2 = t*t;
      double t2_3 = 3*t2;
      double t3 = t*t2;
      *coords = x0 + t*x1 + t2*x2 + t3*x3; ++coords;
      *coords = y0 + t*y1 + t2*y2 + t3*y3; ++coords;
      *coords = z0 + t*z1 + t2*z2 + t3*z3; ++coords;
      float tx = x1 + t_2*x2 + t2_3*x3;
      float ty = y1 + t_2*y2 + t2_3*y3;
      float tz = z1 + t_2*z2 + t2_3*z3;
      float tn = sqrtf(tx*tx + ty*ty + tz*tz);
      if (tn != 0)
	{
	  tx /= tn; ty /= tn; tz /= tn;
	}
      *tangents = tx; ++tangents;
      *tangents = ty; ++tangents;
      *tangents = tz; ++tangents;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
const char *cubic_path_doc =
  "cubic_path(coeffs, tmin, tmax, num_points) -> coords, tangents\n"
  "\n"
  "Compute a path in 3D using x,y,z cubic polynomials.\n"
  "Polynomial coefficients are given in 3x4 matrix coeffs, 64-bit float.\n"
  "The path is computed from t = tmin to tmax with num_points points.\n"
  "Points on the path and normalized tangent vectors are returned.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "coeffs : 3 by 4 float64 array\n"
  "  x,y,z cubic polynomial coefficients c0 + c1*t + c2*t*t + c3*t*t*t.\n"
  "tmin : float64\n"
  "  minimum t value.\n"
  "tmax : float64\n"
  "  maximum t value.\n"
  "num_points : int\n"
  "  number of points.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "coords : n by 3 float array\n"
  "  points on cubic path.\n"
  "tangents : n by 3 float array\n"
  "  normalized tangent vectors at each point of path.\n";

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *cubic_path(PyObject *, PyObject *args, PyObject *keywds)
{
  double coeffs[12];
  double tmin, tmax;
  int num_points;
  const char *kwlist[] = {"coeffs", "tmin", "tmax", "num_points", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ddi"),
				   (char **)kwlist,
				   parse_double_3x4_array, &coeffs[0],
				   &tmin, &tmax, &num_points))
    return NULL;

  float *coords, *tangents;
  PyObject *coords_py = python_float_array(num_points, 3, &coords);
  PyObject *tangents_py = python_float_array(num_points, 3, &tangents);

  cubic_path(&coeffs[0], tmin, tmax, num_points, coords, tangents);

  PyObject *ct = python_tuple(coords_py, tangents_py);
  return ct;
}

// -----------------------------------------------------------------------------
//
static void spline_path(const double *coeffs, int nseg, const float *normals,
			const unsigned char *flip, const unsigned char *twist, int ndiv,
			float *ca, float *ta, float *na)
{
  int np = ndiv/2;
  cubic_path(coeffs, -0.3, 0, np+1, ca, ta);
  bool backwards = true;  // Parallel transport normal backwards
  parallel_transport(np+1, ta, normals, na, backwards);
  ca += 3*np; ta += 3*np ; na += 3*np;
  
  const float *end_normal = NULL;
  float flipped_normal[3];
  for (int seg = 0 ; seg < nseg ; ++seg)
    {
      np = ndiv+1;
      cubic_path(coeffs+12*seg, 0, 1, np, ca, ta);
      const float *start_normal = (seg == 0 ? normals : end_normal);
      parallel_transport(np, ta, start_normal, na);
      end_normal = normals + 3*(seg + 1);
      
      if (twist[seg])
	{
	  if (flip[seg])
	    {
	      // Decide whether to flip the spline segment start normal so that it aligns
	      // better with the preceding segment parallel transported normal.
	      float a = dihedral_angle(na + 3*ndiv, end_normal, ta + 3*ndiv);
	      bool flip= (fabs(a) > 0.6 * M_PI);	// Not sure why this is not 0.5 * M_PI
	      if (flip)
		{
		  for (int i = 0 ; i < 3 ; ++i)
		    flipped_normal[i] = -end_normal[i];
		  end_normal = flipped_normal;
		}
	    }
	  smooth_twist(ta, np, na, end_normal);
	}
      ca += 3*ndiv; ta += 3*ndiv ; na += 3*ndiv;
    }

  np = (ndiv + 1)/2;
  cubic_path(coeffs + 12*(nseg-1), 1, 1.3, np, ca, ta);
  parallel_transport(np, ta, end_normal, na);
}

// -----------------------------------------------------------------------------
//
extern "C" const char *spline_path_doc =
  "spline_path(coeffs, normals, flip_normals, twist, ndiv) -> coords, tangents, normals\n"
  "\n"
  "Compute a path in 3D from segments (x(t),y(t),z(t)) cubic polynomials in t.\n"
  "The path is also extrapolated before the first segment and after the last segment.\n"
  "Polynomial coefficients are given by N 3x4 matrix coeffs, 64-bit float.\n"
  "Normal vectors a the start of the N segments are specified and are.\n"
  "parallel transported along the path.  If flip_normals[i] is true and\n"
  "the parallel transported normal for segment i does is more than 90 degrees\n"
  "away from the specified normal for segment i+1 then the i+1 segment\n"
  "starts with a flipped normal.  If twist[i] then the segment normals\n"
  "are rotated so that the end of segment normal is in the same plane\n"
  "as the starting normal of the next segment, with each normal in the\n"
  "segment having a twist about its tangent vector applied in linearly\n"
  "increasing amounts.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "coeffs : N by 3 by 4 float64 array\n"
  "  x,y,z cubic polynomial coefficients c0 + c1*t + c2*t*t + c3*t*t*t for N segments.\n"
  "normals : N+1 by 3 float array\n"
  "  normal vectors for segment end points.\n"
  "flip_normals : unsigned char\n"
  "  boolean value for each segment whether to allow flipping normals.\n"
  "twist : unsigned char\n"
  "  boolean value for each segment whether to twist normals.\n"
  "ndiv : int\n"
  "  number of points per segment.  Left end is include, right end excluded.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "coords : M by 3 float array\n"
  "  points on cubic path. M = (N+1) * ndiv\n"
  "tangents : M by 3 float array\n"
  "  normalized tangent vectors at each point of path.\n"
  "normals : M by 3 float array\n"
  "  normal vectors at each point of path.\n";

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *spline_path(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray coeffs;
  FArray normals;
  Reference_Counted_Array::Array<unsigned char> flip, twist; // boolean
  int ndiv;
  const char *kwlist[] = {"coeffs", "normals", "flip_normals", "twist", "ndiv", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&i"),
				   (char **)kwlist,
				   parse_contiguous_double_n34_array, &coeffs,
				   parse_float_n3_array, &normals,
				   parse_uint8_n_array, &flip,
				   parse_uint8_n_array, &twist,
				   &ndiv))
    return NULL;

  if (!normals.is_contiguous() || !flip.is_contiguous() || !twist.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError,
		   "spline_path(): normals, flip and twist arrays must be contiguous");
      return NULL;
    }
  if (coeffs.size(0)+1 != normals.size(0))
    {
      PyErr_Format(PyExc_TypeError,
		   "spline_path(): Normals array (%s) must be one longer than coefficients array (%s)",
		   normals.size_string().c_str(), coeffs.size_string().c_str());
      return NULL;
    }
  if (flip.size(0) < coeffs.size(0) || twist.size(0) < coeffs.size(0))
    {
      PyErr_Format(PyExc_TypeError,
		   "spline_path(): Flip array (%s) and twist array (%s) must have same size as coefficients array (%s)",
		   flip.size_string().c_str(), twist.size_string().c_str(), coeffs.size_string().c_str());
      return NULL;
    }

  int nseg = coeffs.size(0);
  int num_points = (nseg+1) * ndiv;
  float *ca, *ta, *na;
  PyObject *pcoords = python_float_array(num_points, 3, &ca);
  PyObject *ptangents = python_float_array(num_points, 3, &ta);
  PyObject *pnormals = python_float_array(num_points, 3, &na);

  spline_path(coeffs.values(), nseg, normals.values(), flip.values(), twist.values(), ndiv,
	      ca, ta, na);

  PyObject *ctn = python_tuple(pcoords, ptangents, pnormals);
  return ctn;
}

// -----------------------------------------------------------------------------
//
static void tridiagonal(int n, double *a, double *b, const double *c, double *d)
{
  /*
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    Hacked source from
    http://ofan666.blogspot.com/2012/02/tridiagonal-matrix-algorithm-solver-in.html
  */

  for (int i = 1 ; i < n ; ++i)
    {
      double mc = a[i] / b[i - 1];
      b[i] = b[i] - mc * c[i - 1];
      d[i] = d[i] - mc * d[i - 1];
    }

  a[n-1] = d[n-1] / b[n-1];
  for (int i = n-2 ; i >= 0 ; --i)
    a[i] = (d[i] - c[i] * a[i + 1]) / b[i];
}

// -----------------------------------------------------------------------------
//
static void cubic_spline(const float *coords, int num_pts, double *coef)
{
  // Extend ends
  int ne = num_pts + 2;
  double *temp = new double[ne*7];

  double *x = temp, *y = temp + ne, *z = temp + 2*ne;
  double *xyz[3] = {x, y, z};
  const float *c0 = coords, *c1 = coords+3;
  for (int axis = 0 ; axis < 3 ; ++axis)
    xyz[axis][0] = c0[axis] - (c1[axis] - c0[axis]);
  for (int i = 0 ; i < num_pts ; ++i)
    { x[i+1] = coords[3*i]; y[i+1] = coords[3*i+1]; z[i+1] = coords[3*i+2]; }
  const float *e0 = coords+3*(num_pts-1), *e1 = coords+3*(num_pts-2);
  for (int axis = 0 ; axis < 3 ; ++axis)
    xyz[axis][ne-1] = e0[axis] + (e0[axis] - e1[axis]);

  double *a = temp + 3*ne, *b = temp + 4*ne, *c = temp + 5*ne, *d = temp + 6*ne;
  for (int axis = 0 ; axis < 3 ; ++axis)
    {
      // 1D cubic spline from http://mathworld.wolfram.com/CubicSpline.html
      // Set b[0] and b[-1] to 1 to match TomG code in VolumePath
      double *values = xyz[axis];
      for (int i = 0 ; i < ne ; ++i)
	{ a[i] = 1; b[i] = 4; c[i] = 1; }
      b[0] = b[ne-1] = 2;
      // b[0] = b[ne-1] = 1;
      d[0] = values[1] - values[0];
      for (int i = 1 ; i < ne-1 ; ++i)
	d[i] = 3 * (values[i+1] - values[i-1]);
      d[ne-1] = 3 * (values[ne-1] - values[ne-2]);
      tridiagonal(ne, a, b, c, d); // Result returned in a.
      for (int i = 0 ; i < num_pts-1 ; ++i)
	{
	  double *cf = coef + 12*i + 4*axis;
	  cf[0] = values[i+1];
	  cf[1] = a[i+1];
	  double delta = values[i+2] - values[i+1];
	  cf[2] = 3 * delta - 2 * a[i+1] - a[i+2];
	  cf[3] = 2 * -delta + a[i+1] + a[i+2];
	}
    }
  delete [] temp;
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *cubic_spline(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz;
  const char *kwlist[] = {"xyz", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz))
    return NULL;

  if (!xyz.is_contiguous())
    {
      PyErr_Format(PyExc_TypeError,
		   "cubic_spline(): xyz array must be contiguous");
      return NULL;
    }

  int n = xyz.size(0);
  if (n < 2)
    {
      PyErr_Format(PyExc_TypeError,
		   "cubic_spline(): Must have 2 or more coordinates, got %d", n);
      return NULL;
    }

  double *coefficients;
  PyObject *coef = python_double_array(n-1, 3, 4, &coefficients);
  cubic_spline(xyz.values(), n, coefficients);

  return coef;
}

// -----------------------------------------------------------------------------
//
static void atom_spline_positions(const Residues &residues,
				  const std::map<std::string, float> &atom_offset_map,
				  std::vector<Atom *> &atoms, std::vector<float> &offsets)
{
  int nr = residues.count;
  for (int ri = 0 ; ri < nr ; ++ri)
    {
      /*
      Residue *r = residues.pointers[ri];
      for (auto a = atom_offset_map.begin() ; a != atom_offset_map.end() ; ++a)
	{
	  Atom *atom = r->find_atom(a->first.c_str());
	  if (atom)
	    {
	      atoms.push_back(atom);
	      offsets.push_back(ri + a->second);
	    }
	}
      */

      const Residue::Atoms &ratoms = residues.pointers[ri]->atoms();
      for (auto a = ratoms.begin() ; a != ratoms.end() ; ++a)
	{
	  Atom *atom = *a;
	  if (atom->is_backbone(BBE_RIBBON))
	    {
	      auto ai = atom_offset_map.find(atom->name().c_str());
	      if (ai != atom_offset_map.end())
		{
		  atoms.push_back(atom);
		  offsets.push_back(ri + ai->second);
		}
	    }
	}
    }
}
// -----------------------------------------------------------------------------
//
inline void spline_position(float offset,const double *coef, int num_pts, double *xyz)
{
  int seg = int(offset);
  float t = offset - seg;
  if (seg < 0)
    {
      t += seg;
      seg = 0;
    }
  else if (seg > num_pts-1)
    {
      t += seg - (num_pts-1);
      seg = num_pts-1;
    }
  const double *c = coef + 12*seg;
  xyz[0] = c[0] + t*(c[1] + t*(c[2] + t*c[3]));
  xyz[1] = c[4] + t*(c[5] + t*(c[6] + t*c[7]));
  xyz[2] = c[8] + t*(c[9] + t*(c[10] + t*c[11]));
}

// -----------------------------------------------------------------------------
//
static void spline_positions(const std::vector<float> &offsets,
			     const double *coef, int num_pts,
			     double *positions)
{
  int n = offsets.size();
  for (int i = 0 ; i < n ; ++i, positions +=3)
    spline_position(offsets[i], coef, num_pts, positions);
}

// -----------------------------------------------------------------------------
//
static PyObject *python_atom_pointers(const std::vector<Atom *> &atoms)
{
  void **data;
  size_t n = atoms.size();
  PyObject *a = python_voidp_array(n, &data);
  for (size_t i = 0 ; i < n ; ++i)
    data[i] = atoms[i];
  return a;
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *atom_spline_positions(PyObject *, PyObject *args, PyObject *keywds)
{
  Residues residues;
  std::map<std::string, float> atom_offsets;
  DArray coef;
  const char *kwlist[] = {"residues", "atom_offsets", "spline_coef", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_residues, &residues,
				   parse_string_float_map, &atom_offsets,
				   parse_contiguous_double_n34_array, &coef))
    return NULL;
  
  std::vector<Atom *> atoms;
  std::vector<float> offsets;
  atom_spline_positions(residues, atom_offsets,	atoms, offsets);
  double *positions;
  PyObject *xyz = python_double_array(offsets.size(), 3, &positions);
  spline_positions(offsets,  coef.values(), coef.size(0), positions);
  
  return python_tuple(python_atom_pointers(atoms), xyz);
}

//
// Position of atoms on ribbon in spline parameter units.
// These should correspond to the "minimum" backbone atoms
// listed in atomstruct/Residue.cpp.
// Negative means on the spline between previous residue
// and this one; positive between this and next.
// These are copied from Chimera.  May want to do a survey
// of closest spline parameters across many structures instead.
//
static std::map<std::string, float> _tether_positions = {
   // Amino acid
   {"N", -1/3.},
   {"H", -1/3.},
   {"CA", 0.},
   {"C",    1/3.},
   {"OXT",  1/3.},
   // Nucleotide
   {"P",    -2/6.},
   {"O5'",  -1/6.},
   {"C5'",   0.},
   {"C4'",   1/6.},
   {"C3'",   2/6.},
   {"O3'",   3/6.}
};

static std::map<std::string, float> _non_tether_positions = {
    // Amino acid
    {"O",    1/3.},
    {"OT1",  1/3.},
    {"OT2",  1/3.},
    // Nucleotide
    {"OP1", -2/6.},
    {"O1P", -2/6.},
    {"OP2", -2/6.},
    {"O2P", -2/6.},
    {"OP3", -2/6.},
    {"O3P", -2/6.},
    {"O2'", -1/6.},
    {"C2'",  2/6.},
    {"O4'",  1/6.},
    {"C1'",  1.5/6.},
    {"O3'",  2/6.},
};

// -----------------------------------------------------------------------------
//
static void set_atom_ribbon_positions(const Residues &residues,
				      const std::map<std::string, float> &atom_offset_map,
				      const double *coef, int num_pts,
				      std::vector<Atom *> *atoms = NULL)
{
  int nr = residues.count;
  atomstruct::Coord c;
  double xyz[3];
  for (int ri = 0 ; ri < nr ; ++ri)
    {
      const Residue::Atoms &ratoms = residues.pointers[ri]->atoms();
      for (auto a = ratoms.begin() ; a != ratoms.end() ; ++a)
	{
	  Atom *atom = *a;
	  if (atom->is_backbone(BBE_RIBBON))
	    {
	      auto ai = atom_offset_map.find(atom->name().c_str());
	      if (ai != atom_offset_map.end())
		{
		  if (atoms)
		    atoms->push_back(atom);
		  spline_position(ri + ai->second, coef, num_pts, xyz);
		  c.set_xyz(xyz[0], xyz[1], xyz[2]);
		  atom->set_ribbon_coord(c);
		}
	    }
	}
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *set_atom_tether_positions(PyObject *, PyObject *args, PyObject *keywds)
{
  Residues residues;
  DArray coef;
  const char *kwlist[] = {"residues", "spline_coef", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_residues, &residues,
				   parse_contiguous_double_n34_array, &coef))
    return NULL;
  
  set_atom_ribbon_positions(residues, _non_tether_positions, coef.values(), coef.size(0));
  std::vector<Atom *> atoms;
  set_atom_ribbon_positions(residues, _tether_positions, coef.values(), coef.size(0), &atoms);
  
  return python_atom_pointers(atoms);
}

static void residue_update_hide(Residue *r, Atom *anchor)
{
    if (r->ribbon_display() && r->ribbon_hide_backbone()) {
        // Ribbon is shown and hides backbone, so hide backbone atoms and bonds
        for (auto atom: r->atoms())
            if ((atom->hide() & Atom::HIDE_RIBBON) == 0
                    && atom->is_backbone(BBE_RIBBON) && atom != anchor)
                atom->set_hide_bits(Atom::HIDE_RIBBON);
    }
    else {
        // Ribbon is not shown or does not hide backbone
        // so unhide backbone atoms and bonds
        for (auto atom: r->atoms())
            if ((atom->hide() & Atom::HIDE_RIBBON) != 0
                    && atom->is_backbone(BBE_RIBBON) && atom != anchor)
                atom->clear_hide_bits(Atom::HIDE_RIBBON);
    }
}

// -------------------------------------------------------------------------
// geometry functions
//
inline float* atom_vector(const Atom* f, const Atom* t, float* result)
{
    const Coord &fc = f->coord();
    const Coord &tc = t->coord();
    result[0] = tc[0] - fc[0];
    result[1] = tc[1] - fc[1];
    result[2] = tc[2] - fc[2];
    return result;
}

inline float* cross(const float* u, const float* v, float* result)
{
    result[0] = u[1]*v[2] - u[2]*v[1];
    result[1] = u[2]*v[0] - u[0]*v[2];
    result[2] = u[0]*v[1] - u[1]*v[0];
    return result;
}

inline bool normalize(float *v)
{
    try {
        float length = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (length > 0) {
            v[0] /= length;
            v[1] /= length;
            v[2] /= length;
            return true;
        }
        return false;
    } catch (std::domain_error &) {
        return false;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *get_polymer_spline(PyObject *, PyObject *args, PyObject *keywds)
{
    Residues residues;
    const char *kwlist[] = {"residues", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				     (char **)kwlist,
				     parse_residues, &residues))
        return NULL;

    bool want_peptide = true;
    bool want_guides = true;
    Residue **res_array = residues.pointers;
    size_t n = residues.count;

    // If no ribbon is displayed for any residue, return Nones
    bool any_display = false;
    for (size_t i = 0; i != n; ++i)
        if (res_array[i]->ribbon_display())
            any_display = true;
    if (!any_display) {
	    return python_tuple(python_bool(false), python_none(), python_none(), python_none());
    }

    // Find all the ribbon-relevant atoms
    struct PeptidePlane {
        float normal[3];
    };
    std::vector<Atom *> centers;
    std::vector<Atom *> guides;
    std::unordered_map<int, PeptidePlane> peptide_planes;
    bool has_guides = want_guides;
    Atom *prev_c = NULL;
    Atom *prev_o = NULL;
    for (size_t i = 0; i != n; ++i) {
        Residue* r = res_array[i];
        for (auto atom: r->atoms())
            atom->set_in_ribbon(false);
        Atom *ca = r->find_atom("CA");
        if (ca != NULL) {
            // Case 1: amino acid
            centers.push_back(ca);
            if (!want_guides) {
                residue_update_hide(r, ca);
                continue;
            }
            Atom *o = r->find_atom("O");
            if (want_guides) {
                if (o != NULL)
                    guides.push_back(o);
                else
                    want_peptide = has_guides = false;
            }
            if (want_peptide && o != NULL) {
                Atom *n = r->find_atom("N");
                Atom *c = r->find_atom("C");
                if (n == NULL || c == NULL) {
                    want_peptide = false;
                    residue_update_hide(r, ca);
                    continue;
                }
                if (prev_c != NULL) {
                    float co[3], cn[3];
                    PeptidePlane peptide;
                    atom_vector(prev_c, prev_o, co);
                    atom_vector(prev_c, n, cn);
                    cross(co, cn, peptide.normal);
                    // NB: do not bother normalizing now since we will
                    // use them in a cross product later and will
                    // have to normalize that result
                    peptide_planes[i] = peptide;
                }
                prev_c = c;
                prev_o = o;
            }
            residue_update_hide(r, ca);
        }
        else {
            prev_c = NULL;
            // Look for nucleotide
            Atom *a = r->find_atom("C5'");
            Atom *anchor = a;
            if (a == NULL) {
                a = r->find_atom("P");
                if (a == NULL) {
                    // Case 2: not a nucleotide
                    r->set_ribbon_display(false);
                    residue_update_hide(r, NULL);
                    continue;
                }
            }
            // Case 3: Nucleotide
            centers.push_back(a);
            if (want_guides) {
                Atom *c1p = r->find_atom("C1'");
                if (c1p)
                    guides.push_back(c1p);
                else
                    has_guides = false;
            }
            residue_update_hide(r, anchor);
        }
    }

    // Create Python return value: tuple of (atoms, control points, guide points)
    void **adata;
    PyObject *alist = python_voidp_array(centers.size(), &adata);
    for (auto atom : centers)
        *adata++ = atom;

    float *data;
    float *cdata;
    PyObject *ca = python_float_array(centers.size(), 3, &cdata);
    data = cdata;
    for (auto atom : centers) {
        const Coord &c = atom->coord();
        *data++ = c[0];
        *data++ = c[1];
        *data++ = c[2];
        atom->set_in_ribbon(true);
    }

    PyObject *ga;
    if (!has_guides || centers.size() < 2) {
        ga = python_none();
    }
    else {
        float *gdata;
        ga = python_float_array(centers.size(), 3, &gdata);
        size_t last = centers.size() - 1;
        //
        // For all but the first and last residues, compute the orientation.
        // First and last are different if they use peptide orientation
        //
        for (size_t i = 1; i != last; ++i) {
            Residue* r = res_array[i];
            float* center = cdata + i*3;
            float* guide = gdata + i*3;
            if (want_peptide
		&& r->polymer_type() == atomstruct::PT_AMINO
		&& r->structure()->ribbon_orient(r) == Structure::RIBBON_ORIENT_PEPTIDE) {
                // "peptide_planes" are relative to the previous
                // residue, so the i'th element is the peptide
                // plane between centers[i-1] and centers[i].
                // We want to average (i-1,i) and (i,i+1).
                const float* prev_pp = peptide_planes[i].normal;
                const float* this_pp = peptide_planes[i+1].normal;
                cross(prev_pp, this_pp, guide);
                if (normalize(guide)) {
                    guide[0] += center[0];
                    guide[1] += center[1];
                    guide[2] += center[2];
                    continue;
                }
                std::cerr << "normalization error " << i
                          << ' ' << r->str() << '\n';
                std::cerr << "prev: " << prev_pp[0] << ' '
                          << prev_pp[1] << ' ' << prev_pp[2] << '\n';
                std::cerr << "this: " << this_pp[0] << ' '
                          << this_pp[1] << ' ' << this_pp[2] << '\n';
                std::cerr << "guide: " << guide[0] << ' '
                          << guide[1] << ' ' << guide[2] << '\n';
            }
            // Either peptide calculation failed or we want guides
            const Coord &c = guides[i]->coord();
            guide[0] = c[0];
            guide[1] = c[1];
            guide[2] = c[2];
            guides[i]->set_in_ribbon(true);
        }
        //
        // Handle first residue
        //
        {
            Residue* r = res_array[0];
            float* guide = gdata;
            float* source;
            if (want_peptide
		&& r->polymer_type() == atomstruct::PT_AMINO
		&& r->structure()->ribbon_orient(r) == Structure::RIBBON_ORIENT_PEPTIDE) {
                // Want peptide.  Copy from second residue.
                source = gdata + 3;
                guide[0] = source[0];
                guide[1] = source[1];
                guide[2] = source[2];
            }
            else {
                const Coord &c = guides[0]->coord();
                guide[0] = c[0];
                guide[1] = c[1];
                guide[2] = c[2];
                guides[0]->set_in_ribbon(true);
            }
        }
        //
        // Handle last residue
        //
        {
            Residue* r = res_array[last];
            float* guide = gdata + last*3;
            float* source;
            if (want_peptide
		&& r->polymer_type() == atomstruct::PT_AMINO
		&& r->structure()->ribbon_orient(r) == Structure::RIBBON_ORIENT_PEPTIDE) {
                // Want peptide.  Copy from next to last residue.
                source = gdata + (last-1)*3;
                guide[0] = source[0];
                guide[1] = source[1];
                guide[2] = source[2];
            }
            else {
                const Coord &c = guides[last]->coord();
                guide[0] = c[0];
                guide[1] = c[1];
                guide[2] = c[2];
                guides[last]->set_in_ribbon(true);
            }
        }
    }
    return python_tuple(python_bool(true), alist, ca, ga);
}
