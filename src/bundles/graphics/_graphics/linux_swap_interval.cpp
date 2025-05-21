// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifdef __linux__
#include <GL/glx.h>
#include <dlfcn.h>
#endif

#include <arrays/pythonarray.h>                // use python_bool()

// ----------------------------------------------------------------------------
//
static bool set_linux_swap_interval(int sync)
{
#ifdef GLX_SGI_swap_control
   PFNGLXSWAPINTERVALSGIPROC glx_swap_interval_sgi =
          reinterpret_cast<PFNGLXSWAPINTERVALSGIPROC> (
               dlsym(RTLD_DEFAULT, "glXSwapIntervalSGI"));
  if (glx_swap_interval_sgi && glx_swap_interval_sgi(sync) == 0)
    return true;
#endif

#ifdef GLX_MESA_swap_control
   PFNGLXSWAPINTERVALMESAPROC glx_swap_interval_mesa =
          reinterpret_cast<PFNGLXSWAPINTERVALMESAPROC> (
                  dlsym(RTLD_DEFAULT, "glXSwapIntervalMESA"));
  if (glx_swap_interval_mesa && glx_swap_interval_mesa(sync) == 0)
    return true;
#endif

  return false;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
set_linux_swap_interval(PyObject *, PyObject *args, PyObject *keywds)
{
  int sync;
  const char *kwlist[] = {"sync", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("i"),
				   (char **)kwlist,
				   &sync))
    return NULL;

  bool success = set_linux_swap_interval(sync);
  return python_bool(success);
}
