// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Routines to interpolate volume data using trilinear interpolation,
// and to interpolate a colormap.  These are for coloring surfaces using
// volume data values.
//
#ifndef INTERPOLATE_HEADER_INCLUDED
#define INTERPOLATE_HEADER_INCLUDED

#include <vector>			// use std::vector<>
#include "rcarray.h"			// use Numeric_Array

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
