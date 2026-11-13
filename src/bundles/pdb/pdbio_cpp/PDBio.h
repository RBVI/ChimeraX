// vi: set expandtab ts=4 sw=4:

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

// Declarations shared by PDBio_read.cpp (reading, and the module) and
// PDBio_write.cpp (writing), split up so that they compile in parallel

#ifndef pdbio_h
#define pdbio_h


#include <algorithm>  // for std::sort, std::find
#include <cctype>
#include <cmath> // abs
#include <fstream>
#include <set>
#include <sstream>
#include <stdio.h>  // fgets
#include <unordered_map>

#include "Python.h"

#include <arrays/pythonarray.h>	// Use python_voidp_array(), array_from_python()
#include <atomstruct/Atom.h>
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Bond.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Sequence.h>
#include <atomstruct/destruct.h>
#include <atomstruct/tmpl/residues.h>
#include <logger/logger.h>
#include "pdb/connect.h"
#include <PDB.h>

using atomstruct::Atom;
using atomstruct::AtomicStructure;
using atomstruct::AtomName;
using atomstruct::Bond;
using atomstruct::ChainID;
using atomstruct::CoordSet;
using element::Element;
using atomstruct::MolResId;
using atomstruct::Real;
using atomstruct::Residue;
using atomstruct::ResName;
using atomstruct::Sequence;
using atomstruct::Structure;
using atomstruct::Coord;

using namespace pdb;
using namespace pdb_connect;

// Defined in PDBio_read.cpp
extern std::string pdb_segment;
extern std::string pdb_charge;
extern std::string pqr_charge;

// Defined in PDBio_write.cpp
extern "C" PyObject *write_pdb_file(PyObject *, PyObject *args);

#endif  // pdbio_h
