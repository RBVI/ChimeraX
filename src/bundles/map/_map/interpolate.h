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
// Routines to interpolate volume data using trilinear interpolation,
// and to interpolate a colormap.  These are for coloring surfaces using
// volume data values.
//
#ifndef INTERPOLATE_HEADER_INCLUDED
#define INTERPOLATE_HEADER_INCLUDED

#include <vector>			// use std::vector<>
#include <arrays/rcarray.h>		// use Numeric_Array

namespace Interpolate
{
enum  Interpolation_Method {INTERP_LINEAR, INTERP_NEAREST};

void interpolate_volume_data(float vertices[][3], int n,
			     float vtransform[3][4],
			     const Reference_Counted_Array::Numeric_Array &data,
			     Interpolation_Method method,
			     float *values, std::vector<int> &outside);

void interpolate_volume_gradient(float vertices[][3], int n,
				 float vtransform[3][4],
				 const Reference_Counted_Array::Numeric_Array &data,
				 Interpolation_Method method,
				 float gradients[][3],
				 std::vector<int> &outside);

void interpolate_colormap(float values[], int n,
			  float color_data_values[], int m,
			  float rgba_colors[][4],
			  float rgba_above_value_range[4],
			  float rgba_below_value_range[4],
			  float rgba[][4]);

void set_outside_volume_colors(int *outside, int n,
			       float rgba_outside_volume[4],
			       float rgba[][4]);

}  // end of namespace interpolate

#endif
