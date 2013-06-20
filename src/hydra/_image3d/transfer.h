// ----------------------------------------------------------------------------
// Routines to evaluate transfer functions mapping numeric data values to
// RGBA color and opacity.
//
#ifndef TRANSFER_HEADER_INCLUDED
#define TRANSFER_HEADER_INCLUDED

#include <Python.h>			// use PyObject

namespace Image_3d
{

extern "C" {

// ----------------------------------------------------------------------------
// Fill the 2 dimensional (size by 4) colormap array with rgba values
// corresponding to the center of the data bins.  The colormap array is not
// modified for bin centers outside the domain of the transfer function.
// This routine is useful in conjunction with data_to_bin_index().  The two
// routines map data values to bin index, and bin index to rgba.
//
// void transfer_function_colormap(PyObject *transfer_func,
//				float bcfirst, float bclast, PyObject *colormap,
//				int bins = 0, int bin_step = 1,
//				bool blend = false);
//
PyObject *transfer_function_colormap(PyObject *s, PyObject *args,
				     PyObject *keywds);

// ----------------------------------------------------------------------------
// Convert float color values to uint8 or uint16.
//
// void colors_float_to_uint(PyObject *colors_float, PyObject *colors_uint);
//
PyObject *colors_float_to_uint(PyObject *s, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Map data values to colors using a colormap.  Out of range data values
// map to the end colormap values if clamp is true, otherwise they have
// color components set to zero.
//
// void data_to_colors(PyObject *data, float dmin, float dmax,
//		    PyObject *colormap, bool clamp, PyObject *colors);
//
PyObject *data_to_colors(PyObject *s, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Map data values to rgba colors (float) by interpolating a colormap (float).
//
// void data_to_colormap_colors(PyObject *data, float dmin, float dmax,
//			     PyObject *colormap, PyObject *colors, bool blend);
//
PyObject *data_to_colormap_colors(PyObject *s, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// The data value range is divided into equal sized bins with the first bin
// centered at bcfirst and the last centered at bclast.
// Data values below bcfirst and above bclast belong to the end bins.
// The 3 dimensional 8-bit or 16-bit integer index_values array gets
// the bin number of the corresponding data value.  The bins are numbered 0
// to bins-1 scaled by bin_step.  The scaling is to allow compositing
// multiple data components.  If add is false the indices are written to 
// index_values. If add is true, the indices are added to index_values.
//
// void data_to_bin_index(PyObject *data, float bcfirst, float bclast, int bins,
//		       int bin_step, PyObject *index_values, bool add);
//
PyObject *data_to_bin_index(PyObject *s, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Map array of colormap indices (int8, uint8, int16, uint16, int32, uint32
// non-negative index values) to colors from a colormap (uint8, uint16).
// Colormap can have 1 to 4 components.
// No bounds checking is done on the indices.
// If modulate is true then color values are multiplied by colormap values
// scaled to 0-1 range, otherwise color values replace existing values.
//
// void indices_to_colors(PyObject *indices, PyObject *colormap,
//		       PyObject *colors, bool modulate = false);
//
PyObject *indices_to_colors(PyObject *s, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Resample a colormap into a multi-dimensional colormap.
//
// void resample_colormap(float bcfirst1, float bclast1, PyObject *colormap1,
//		       float bcfirst2, float bclast2, PyObject *colormap2,
//		       int bins = 0, int bin_step = 1, bool blend = false);
//
PyObject *resample_colormap(PyObject *s, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Data is a 3 dimensional NumPy array.
// The transfer function is a piecewise linear map represented
// as a sorted list of (data_value, intensity_scale, red, green, blue, alpha).
// The interpolated intensity scale modulates the interpolated rgba value.
// The 4 dimensional NumPy float array rgba is modified.
// If blend is false the rgba values are replaced.  If blend is true
// the rgb values are added, and alpha is set so that 1-alpha gets
// scaled by 1-(transfer alpha).
//
// void data_to_rgba(PyObject *data, PyObject *transfer_func, PyObject *rgba,
//		  bool blend);
//
PyObject *data_to_rgba(PyObject *s, PyObject *args, PyObject *keywds);

}	// end extern C

}	// end of namespace Volume_Display

#endif
