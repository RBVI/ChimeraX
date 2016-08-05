// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Pack triangles in Stereo Lithography (STL) file format.
//
#include <Python.h>			// use PyObject

//include <iostream>			// use std::cerr for debugging
#include <map>				// use std::map
#include <vector>			// use std::vector

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

// ----------------------------------------------------------------------------
//
class Vertex
{
public:
  float x,y,z;
  Vertex() {};
  bool operator<(const Vertex &v) const
    { return (x < v.x || (x == v.x && (y < v.y || (y == v.y && z < v.z)))); }
};

// -----------------------------------------------------------------------------
// Read 50 bytes per triangle containing float32 normal vector
// followed three float32 vertices, followed by two "attribute bytes"
// sometimes used to hold color information, but ignored by this reader.
//
static void unpack(const char *geom, int ntri,
		   int *tri, std::vector<float> &v, std::vector<float> &n)
{
  // Find unique vertices and record triangle vertex indices.
  std::map<Vertex,int> vertices; // Map unique vertices to index.
  float tv[12];
  Vertex vxyz;
  int vi;
  for (int t = 0 ; t < ntri ; ++t)
    {
      memcpy(&tv[0], geom + t*50 + 12, 36);
      for (int c = 0 ; c < 3 ; ++c)
	{
	  float *tc = tv + 3*c;
	  vxyz.x = tc[0];
	  vxyz.y = tc[1];
	  vxyz.z = tc[2];
	  auto vit = vertices.find(vxyz);
	  if (vit == vertices.end())
	    {
	      vertices[vxyz] = vi = vertices.size();
	      v.push_back(vxyz.x);
	      v.push_back(vxyz.y);
	      v.push_back(vxyz.z);
	      n.push_back(0);
	      n.push_back(0);
	      n.push_back(0);
	    }
	  else
	    vi = vit->second;
	  tri[3*t+c] = vi;
	}
    }

  // Compute vertex normals as average of triangle normals.
  float nxyz[3];
  for (int t = 0 ; t < ntri ; ++t)
    {
      memcpy(&nxyz[0], geom + t*50, 12);
      for (int c = 0 ; c < 3 ; ++c)
	{
	  vi = tri[3*t+c];
	  for (int a = 0 ; a < 3 ; ++a)
	    n[3*vi+a] += nxyz[a];
	}
    }

  // Normalize normals.
  int nv = v.size() / 3;
  for (vi = 0 ; vi < nv ; ++vi)
    {
      float nx = n[3*vi], ny = n[3*vi+1], nz = n[3*vi+2];
      float nn = sqrtf(nx*nx + ny*ny + nz*nz);
      if (nn > 0)
	{ n[3*vi] = nx/nn; n[3*vi+1] = ny/nn; n[3*vi+2] = nz/nn; }
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
stl_unpack(PyObject *, PyObject *args, PyObject *keywds)
{
  const char *data;
  int nbytes;
  const char *kwlist[] = {"data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("y#"),
				   (char **)kwlist,
				   &data, &nbytes))
    return NULL;

  int ntri = nbytes / 50;
  int *tri;
  PyObject *tpy = python_int_array(ntri, 3, &tri);
  std::vector<float> v, n;
  unpack(data, ntri, tri, v, n);
  int nv = v.size() / 3;
  PyObject *vpy = c_array_to_python(v, nv, 3);
  PyObject *npy = c_array_to_python(n, nv, 3);
  
  return python_tuple(vpy, npy, tpy);
}

// ----------------------------------------------------------------------------
//
static PyMethodDef stl_methods[] = {
  {const_cast<char*>("stl_pack"), (PyCFunction)stl_pack,
   METH_VARARGS|METH_KEYWORDS,
   "stl_pack(vertices, triangles)\n"
   "\n"
   "Compute the STL (Stereo Lithography) file format packing of specified triangles.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("stl_unpack"), (PyCFunction)stl_unpack,
   METH_VARARGS|METH_KEYWORDS,
   "stl_unpack(data)\n"
   "\n"
   "Return vertices, normals and triangle vertex indices by unpacking STL (Stereo Lithography)\n"
   "file format data.\n"
   "Implemented in C++.\n"
  },
};


static struct PyModuleDef stl_def =
{
	PyModuleDef_HEAD_INIT,
	"_stl",
	"STL file read/write",
	-1,
	stl_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__stl()
{
	return PyModule_Create(&stl_def);
}
