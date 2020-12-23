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

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>			// Use CGLSetParameter
#endif

#include <arrays/pythonarray.h>		// use python_bool()

// ----------------------------------------------------------------------------
//
static bool set_mac_swap_interval(int sync)
{
#ifdef __APPLE__
  GLint                       gl_sync = sync;
  CGLContextObj               ctx = CGLGetCurrentContext();

  if (ctx == 0)
    return false;
  CGLSetParameter(ctx, kCGLCPSwapInterval, &gl_sync);
  // CGLSetParameter(ctx, NSOpenGLCPSwapInterval, &gl_sync);
  return true;
#endif

  return false;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
set_mac_swap_interval(PyObject *, PyObject *args, PyObject *keywds)
{
  int sync;
  const char *kwlist[] = {"sync", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("i"),
				   (char **)kwlist,
				   &sync))
    return NULL;

  bool success = set_mac_swap_interval(sync);
  return python_bool(success);
}
