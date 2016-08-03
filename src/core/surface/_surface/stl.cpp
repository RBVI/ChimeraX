// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Pack triangles in Stereo Lithography (STL) file format.
//
#include <Python.h>			// use PyObject

//#include <iostream>			// use std::cerr for debugging
//#include <set>				// use std::set<>

#include <math.h>			// use sqrtf()
#include <string.h>			// use memcpy()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray, IArray

// ----------------------------------------------------------------------------
//
static void pack_triangles(const FArray &va, const IArray &ta, unsigned char *data)
{
  int nt = ta.size(0);
  long t0 = ta.stride(0), t1 = ta.stride(1);
  int *tv = ta.values();
  int *tv0 = tv, *tv1 = tv + t1, *tv2 = tv + 2*t1;
  long vs0 = va.stride(0), vs1 = va.stride(1);
  float *vv = va.values();
  float *vv0 = vv, *vv1 = vv + vs1, *vv2 = vv + 2*vs1;
  float f[12];
  for (int t = 0 ; t < nt ; ++t)
    {
      // Get 3 triangle vertices
      int ti = t*t0;
      int v0 = tv0[ti], v1 = tv1[ti], v2 = tv2[ti];
      int v0i = v0*vs0, v1i = v1*vs0, v2i = v2*vs0;
      float v0x = vv0[v0i], v0y = vv1[v0i], v0z = vv2[v0i];
      float v1x = vv0[v1i], v1y = vv1[v1i], v1z = vv2[v1i];
      float v2x = vv0[v2i], v2y = vv1[v2i], v2z = vv2[v2i];

      // Compute triangle normal.
      float e10x = v1x-v0x, e10y = v1y-v0y, e10z = v1z-v0z;
      float e20x = v2x-v0x, e20y = v2y-v0y, e20z = v2z-v0z;
      float nx = e10y*e20z - e10z*e20y, ny = e10z*e20x - e10x*e20z, nz = e10x*e20y - e10y*e20x;
      float nn = sqrtf(nx*nx + ny*ny + nz*nz);
      if (nn > 0)
	{ nx /= nn; ny /= nn; nz /= nn; }

      // Pack in array.
      f[0] = nx; f[1] = ny; f[2] = nz;
      f[3] = v0x; f[4] = v0y; f[5] = v0z;
      f[6] = v1x; f[7] = v1y; f[8] = v1z;
      f[9] = v2x; f[10] = v2y; f[11] = v2z;
      unsigned char *d = data + t*50;
      memcpy(d, static_cast<void*>(&f[0]), 48);
      d[48] = d[49] = 0;
    }

  // Packing must be little endian.
  unsigned int i = 0x12345678;
  bool little_endian = (*(unsigned char *)&i == 0x78);
  if (!little_endian)
    {
      unsigned short *s = (unsigned short *)(data);
      for (int t = 0 ; t < nt ; ++t)
	{
	  unsigned short *d = s + t*25;
	  for (int i = 0 ; i < 25 ; i += 2)
	    {
	      unsigned short t0 = d[i], t1 = d[i+1];
	      d[i] = ((t1 & 0xff)<<8) + ((t1 & 0xff00)>>8);
	      d[i+1] = ((t0 & 0xff)<<8) + ((t0 & 0xff00)>>8);
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
stl_pack(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray varray;
  IArray tarray;
  const char *kwlist[] = {"vertices", "triangles", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray))
    return NULL;

  int nt = tarray.size(0);
  unsigned char *data;
  PyObject *a = python_uint8_array(nt, 50, &data);	// 50 bytes per triangle.

  pack_triangles(varray, tarray, data);
  
  return a;
}
