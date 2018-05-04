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
