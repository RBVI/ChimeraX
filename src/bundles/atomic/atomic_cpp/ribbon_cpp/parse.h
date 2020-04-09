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
#ifndef RPARSE_HEADER_INCLUDED
#define RPARSE_HEADER_INCLUDED

#include <Python.h>			// use PyObject

#include <atomstruct/Residue.h>		// use Residue
using atomstruct::Residue;

class Residues
{
public:
  int count;
  Residue **pointers;
};

extern "C" int parse_residues(PyObject *arg, void *res);
extern "C" int parse_string_float_map(PyObject *arg, void *sf);


#endif
