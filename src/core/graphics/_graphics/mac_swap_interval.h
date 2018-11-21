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
#ifndef MAC_SWAP_INTERVAL_HEADER_INCLUDED
#define MAC_SWAP_INTERVAL_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// bool set_mac_swap_interval(int) -> true success / false failure
PyObject *set_mac_swap_interval(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
