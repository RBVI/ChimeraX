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

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "destruct.h"
#include "Structure.h"
#include "PBGroup.h"

#include <pysupport/convert.h>

#include <Python.h>
#include <arrays/pythonarray.h>   // Uses python_int_array(), python_float_array()

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::BaseManager>;

namespace atomstruct {

BaseManager::~BaseManager()
{
    // assign to var so it lives to end of destructor
    auto du = DestructionUser(this);
    // delete groups while DestructionUser active
    for (auto name_grp: _groups) {
        delete name_grp.second;
    }
}

void
BaseManager::change_category(Proxy_PBGroup* grp, std::string& category)
{
    if (_groups.find(category) != _groups.end())
        throw std::invalid_argument("Another pb group with that category already exists.");
    _groups.erase(grp->category());
    _groups[category] = grp;
}
void
AS_PBManager::change_cs(const CoordSet* cs)
{
    for (auto pbg_info: _groups) pbg_info.second->change_cs(cs);
}

void
BaseManager::clear()
{
    for (auto cat_grp: _groups)
        delete cat_grp.second;
    _groups.clear();
}

void
BaseManager::delete_group(Proxy_PBGroup* group)
{
    auto gmi = _groups.find(group->category());
    if (gmi == _groups.end())
        throw std::invalid_argument("Asking for deletion of group not in manager!");
    delete group;
    _groups.erase(gmi);
}

Proxy_PBGroup*
BaseManager::get_group(const std::string& name) const
{
    auto gmi = _groups.find(name);
    if (gmi != _groups.end()) {
        return (*gmi).second;
    }
    return nullptr;
}

Proxy_PBGroup*
AS_PBManager::get_group(const std::string& name, int create)
{
    Proxy_PBGroup* grp = get_group(name);
    if (grp) {
        if (create != GRP_NONE && grp->group_type() != create)
            throw std::invalid_argument("Group type mismatch");
        return grp;
    }

    if (create == GRP_NONE)
        return nullptr;

    grp = new Proxy_PBGroup(this, name, structure(), create);
    if (name == structure()->PBG_METAL_COORDINATION)
        grp->set_color(147, 112, 219);
    else if (name == structure()->PBG_MISSING_STRUCTURE)
        grp->set_halfbond(true);
    else if (name == structure()->PBG_HYDROGEN_BONDS)
        grp->set_color(0, 191, 255);
    _groups[name] = grp;
    return grp;
}

Proxy_PBGroup*
PBManager::get_group(const std::string& name, int create)
{
    Proxy_PBGroup* grp = get_group(name);
    if (grp) {
        if (create != GRP_NONE && grp->group_type() != create) {
            throw std::invalid_argument("Group type mismatch");
        }
        return grp;
    }

    if (create == GRP_NONE)
        return nullptr;

    if (create != GRP_NORMAL)
        throw std::invalid_argument("Can only create normal pseudobond groups"
            " in global non-structure-associated pseudobond manager");

    grp = new Proxy_PBGroup(this, name, nullptr, create);
    _groups[name] = grp;
    return grp;
}

StructureManager::StructureManager(Structure* as):
    BaseManager(as->change_tracker()), _structure(as) {}

void
AS_PBManager::remove_cs(const CoordSet* cs) {
    for (auto pbg_info: _groups) pbg_info.second->remove_cs(cs);
}

int
BaseManager::session_info(PyObject** ints, PyObject** floats, PyObject** misc) const
{
    *misc = PyList_New(0);
    if (*misc == nullptr) {
        throw std::runtime_error("Can't allocate list for pseudobond misc info");
    }

    int num_ints = group_map().size(); // to remember group types
    int num_floats = 0;
    std::vector<Proxy_PBGroup*> groups;
    PyObject* cat_list = PyList_New(group_map().size());
    if (cat_list == nullptr) {
        throw std::runtime_error("Can't allocate list for pseudobond category names");
    }
    if (PyList_Append(*misc, cat_list) < 0)
        throw std::runtime_error("Can't append pseudobond category name list to misc list");
    using pysupport::cchar_to_pystring;
    int cat_index = 0;
    for (auto cat_grp: group_map()) {
        auto cat = cat_grp.first;
        auto grp = cat_grp.second;
        groups.push_back(grp);
        num_ints += grp->session_num_ints();
        num_floats += grp->session_num_floats();
        PyList_SET_ITEM(cat_list, cat_index++, cchar_to_pystring(cat, "pb group category"));
    }
    int* int_array;
    *ints = python_int_array(num_ints, &int_array);
    float* float_array;
    *floats = python_float_array(num_floats, &float_array);
    for (auto grp: groups) {
        *int_array++ = grp->group_type();
        grp->session_save(&int_array, &float_array);
    }
    return CURRENT_SESSION_VERSION;
}

void
BaseManager::session_save_setup() const
{
    session_save_pbs = new SessionSavePbMap;
    _ses_struct_to_id_map = new SessionStructureToIDMap;
    // since pseudobond session IDs may be asked for before
    // the structure/manager is itself asked to save, need
    // to populate the maps here instead of during session_info
    for (auto& cat_grp: group_map())
        cat_grp.second->session_save_setup();
}

void
BaseManager::session_restore(int version, int** ints, float** floats, PyObject* misc)
{
    if (version > CURRENT_SESSION_VERSION)
        throw std::invalid_argument("Don't know how to restore new session data; update your"
            " version of ChimeraX");

    clear(); // only really relevant for global manager, but oh well

    auto& int_ptr = *ints;
    auto& float_ptr = *floats;

    if (!(PyTuple_Check(misc) || PyList_Check(misc)) || PySequence_Fast_GET_SIZE(misc) != 1) {
        throw std::invalid_argument("PBManager::session_restore: third arg is not a"
            " 1-element tuple");
    }
    using pysupport::pysequence_of_string_to_cvec;
    std::vector<std::string> categories;
    pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(misc, 0), categories, "PB Group category");
    for (auto cat: categories) {
        auto grp = get_group(cat, *int_ptr++);
        grp->session_restore(version, &int_ptr, &float_ptr);
    }
}

}  // namespace atomstruct
