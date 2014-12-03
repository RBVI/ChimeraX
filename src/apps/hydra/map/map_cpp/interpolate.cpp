// ----------------------------------------------------------------------------
//
#include <math.h>			// use floor()
#include <vector>			// use std::vector

#include "interpolate.h"
#include "rcarray.h"			// use Array<T>, Numeric_Array

namespace Interpolate
{

// ----------------------------------------------------------------------------
//
inline bool data_cell(float xyz[3], float vtransform[3][4], int dsize[3],
		      int bijk[3], float fijk[3], int edge_pad)
{
  for (int a = 0 ; a < 3 ; ++a)
    {
      float ia = (vtransform[a][0]*xyz[0] + vtransform[a][1]*xyz[1] +
		  vtransform[a][2]*xyz[2] + vtransform[a][3]);
      float fia = floor(ia);
      int bia = static_cast<int>(fia);
      //
      // Would be better to test bounds with ia instead of bia because
      // if ia exceeds range of int then bia may wrap back into range.
      // But I observed case on Linux/Pentium4 where ia ~= 80, fia >= int(80),
      // but not ia >= int(80).
      //
      if (bia < edge_pad || bia >= dsize[a]-edge_pad-1)
	return false;
      bijk[a] = bia;
      fijk[a] = ia - fia;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
template <class T>
static void interpolate_volume(float vertices[][3], int n,
			       float vtransform[3][4],
			       const Reference_Counted_Array::Array<T> &data,
			       Interpolation_Method method,
			       float *values, std::vector<int> &outside)
{
  int dsize[3] = {data.size(2), data.size(1), data.size(0)};
  long si = data.stride(2), sj = data.stride(1), sk = data.stride(0);
  T *d = data.values();
  int bijk[3];
  float fijk[3];
  for (int m = 0 ; m < n ; ++m)
    {
      float *xyz = vertices[m];
      if (!data_cell(xyz, vtransform, dsize, bijk, fijk, 0))
	{
	  values[m] = 0;
	  outside.push_back(m);
	  continue;
	}
      long offset = bijk[0]*si + bijk[1]*sj + bijk[2]*sk;
      T *dc = d + offset;
      if (method == INTERP_LINEAR)
	{
	  float fi1 = fijk[0], fj1 = fijk[1], fk1 = fijk[2];
	  float fi0 = 1 - fi1, fj0 = 1 - fj1, fk0 = 1 - fk1;
	  values[m] = (fk0*(fj0*(fi0 * dc[0] + fi1 * dc[si]) +
			    fj1*(fi0 * dc[sj] + fi1 * dc[si+sj])) +
		       fk1*(fj0*(fi0 * dc[sk] + fi1 * dc[si+sk]) +
			    fj1*(fi0 * dc[sj+sk] + fi1 * dc[si+sj+sk])));
	}
      else
	values[m] = dc[(fijk[0]>=0.5 ? si : 0) +
		       (fijk[1]>=0.5 ? sj : 0) +
		       (fijk[2]>=0.5 ? sk : 0)];
    }
}

// ----------------------------------------------------------------------------
//
void interpolate_volume_data(float vertices[][3], int n,
			     float vtransform[3][4],
			     const Reference_Counted_Array::Numeric_Array &data,
			     Interpolation_Method method,
			     float *values, std::vector<int> &outside)
{
  call_template_function(interpolate_volume, data.value_type(),
  			 (vertices, n, vtransform, data, method,
			  values, outside));
}

// ----------------------------------------------------------------------------
//
template <class T>
static void interpolate_gradient(float vertices[][3], int n,
				 float vtransform[3][4],
				 const Reference_Counted_Array::Array<T> &data,
				 Interpolation_Method method,
				 float gradients[][3],
				 std::vector<int> &outside)
{
  int dsize[3] = {data.size(2), data.size(1), data.size(0)};
  long stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  T *d = data.values();
  int bijk[3];
  float fijk[3];
  for (int m = 0 ; m < n ; ++m)
    {
      float *xyz = vertices[m];
      float *grad = gradients[m];
      if (!data_cell(xyz, vtransform, dsize, bijk, fijk, 1))
	{
	  grad[0] = grad[1] = grad[2] = 0;
	  outside.push_back(m);
	  continue;
	}
      long offset = bijk[0]*stride[0] + bijk[1]*stride[1] + bijk[2]*stride[2];
      T *dc = d + offset;
      float gijk[3] = {0, 0, 0};
      if (method == INTERP_LINEAR)    // Average gradients at 8 cell corners.
	for (int oi = 0 ; oi < 2 ; ++oi)
	  {
	    float wi = (oi ? fijk[0] : 1-fijk[0]);
	    for (int oj = 0 ; oj < 2 ; ++oj)
	      {
		float wij = wi * (oj ? fijk[1] : 1-fijk[1]);
		for (int ok = 0 ; ok < 2 ; ++ok)
		  {
		    float wijk = .5 * wij * (ok ? fijk[2] : 1-fijk[2]);
		    long oijk = oi*stride[0] + oj*stride[1] + ok*stride[2];
		    T *dijk = dc + oijk;
		    for (int a = 0 ; a < 3 ; ++a)
		      gijk[a] += wijk * (*(dijk+stride[a]) - *(dijk-stride[a]));
		  }
	      }
	  }
      else	// Nearest grid point
	{
	  T *dijk = &dc[(fijk[0]>=0.5 ? stride[0] : 0) +
			(fijk[1]>=0.5 ? stride[1] : 0) +
			(fijk[2]>=0.5 ? stride[2] : 0)];
	  for (int a = 0 ; a < 3 ; ++a)
	    gijk[a] = *(dijk+stride[a]) - *(dijk-stride[a]);
	}
      // Transform discrete gradients to vertex coordinate system.
      for (int a = 0 ; a < 3 ; ++a)
	grad[a] = (gijk[0]*vtransform[0][a] +
		   gijk[1]*vtransform[1][a] +
		   gijk[2]*vtransform[2][a]);
    }
}

// ----------------------------------------------------------------------------
//
void interpolate_volume_gradient(float vertices[][3], int n,
				 float vtransform[3][4],
				 const Reference_Counted_Array::Numeric_Array &data,
				 Interpolation_Method method,
				 float gradients[][3],
				 std::vector<int> &outside)
{
  call_template_function(interpolate_gradient, data.value_type(),
  			 (vertices, n, vtransform, data, method,
			  gradients, outside));
}
    
// ----------------------------------------------------------------------------
//
void interpolate_colormap(float values[], int n,
			  float color_data_values[], int m,
			  float rgba_colors[][4],
			  float rgba_above_value_range[4],
			  float rgba_below_value_range[4],
			  float rgba[][4])
{
  for (int k = 0 ; k < n ; ++k)
    {
      float v = values[k];
      float *rgbak = rgba[k];
      if (v < color_data_values[0])
	for (int a = 0 ; a < 4 ; ++a)
	  rgbak[a] = rgba_below_value_range[a];
      else if (v > color_data_values[m-1])
	for (int a = 0 ; a < 4 ; ++a)
	  rgbak[a] = rgba_above_value_range[a];
      else
	{
	  int j = 1;
	  while (v > color_data_values[j])
	    j += 1;
	  float v0 = color_data_values[j-1], v1 = color_data_values[j];
	  float f1 = (v - v0) / (v1 - v0);
	  float f0 = 1 - f1;
	  float *c0 = rgba_colors[j-1], *c1 = rgba_colors[j];
	  for (int a = 0 ; a < 4 ; ++a)
            rgbak[a] = f0*c0[a]+f1*c1[a];
	}
    }
}
            
// ----------------------------------------------------------------------------
//
void set_outside_volume_colors(int *outside, int n,
			       float rgba_outside_volume[4],
			       float rgba[][4])
{
  for (int k = 0 ; k < n ; ++k)
    {
      float *rgbak = rgba[outside[k]];
      for (int a = 0 ; a < 4 ; ++a)
	rgbak[a] = rgba_outside_volume[a];
    }
}

}  // end of namespace interpolate
