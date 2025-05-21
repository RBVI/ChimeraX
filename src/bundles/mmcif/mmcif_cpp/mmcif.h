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

#include <Python.h>
#include <string>
#include <functional>
#include <vector>

#include <atomstruct/string_types.h>
#include <atomstruct/Bond.h>

namespace tmpl {
    class Residue;
}

namespace atomstruct {
    class Bond;
}

namespace mmcif {

using atomstruct::ResName;
using atomstruct::Bond;

PyObject*   parse_mmCIF_file(const char* filename, PyObject* logger,
                             bool coordsets, bool atomic, bool ignore_styling);
PyObject*   parse_mmCIF_file(const char* filename,
                             const std::vector<std::string> &extra_categories,
                             PyObject* logger, bool coordsets, bool atomic, bool ignore_styling);
PyObject*   parse_mmCIF_buffer(const unsigned char* buffer, PyObject* logger,
                               bool coordsets, bool atomic, bool ignore_styling);
PyObject*   parse_mmCIF_buffer(const unsigned char* buffer,
                             const std::vector<std::string> &extra_categories,
                             PyObject* logger, bool coordsets, bool atomic, bool ignore_styling);
void        load_mmCIF_templates(const char* filename);
void        set_Python_locate_function(PyObject* function);

PyObject*   extract_CIF_tables(const char* filename,
                               const std::vector<std::string> &categories,
                               bool all_data_blocks);

PyObject*   quote_value(PyObject* value, int max_len=60);
typedef std::vector<const Bond*> Bonds;
void        non_standard_bonds(const Bond** bonds, size_t num_bonds, bool selected_only, bool displayed_only, Bonds& disulfide, Bonds& covalent);

#ifndef WrapPy
const tmpl::Residue*
            find_template_residue(const ResName& name, bool start = false, bool stop = false);
typedef std::function<std::string (const ResName& residue_type)>
            LocateFunc;
void        set_locate_template_function(LocateFunc func);
#endif

}  // namespace mmcif
