// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef MOMENTS_HEADER_INCLUDED
#define MOMENTS_HEADER_INCLUDED

namespace Map_Cpp
{

extern "C"
{
  // moments_py(array_3d) -> m33, m3, m0
  PyObject *moments_py(PyObject *, PyObject *args, PyObject *keywds);
  // affine_scale_py(array_3d, c, u[3], invert)
  //  a -> a * (c+u*(i,j,k)) if invert false
  //  a -> a / (c+u*(i,j,k)) if invert true
  PyObject *affine_scale_py(PyObject *, PyObject *args, PyObject *keywds);  
}
 
} // end of namespace Map_Cpp

#endif
