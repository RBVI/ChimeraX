// vi: set expandtab ts=4 sw=4:

#include "AtomicStructure.h"
#include "PBGroup.h"

#include <basegeom/destruct.h>
#include <pysupport/convert.h>

#include <Python.h>
#include <pythonarray.h>

namespace atomstruct {

BaseManager::~BaseManager()
{
    // assign to var so it lives to end of destructor
    auto du = basegeom::DestructionUser(this);
    // delete groups while DestructionUser active
    for (auto name_grp: this->_groups) {
        delete name_grp.second;
    }
}

void
AS_PBManager::delete_group(Proxy_PBGroup* group)
{
    auto gmi = this->_groups.find(group->category());
    if (gmi == this->_groups.end())
        throw std::invalid_argument("Asking for deletion of group not in manager!");
    delete group;
    this->_groups.erase(gmi);
}

void
PBManager::delete_group(Proxy_PBGroup* group)
{
    auto gmi = this->_groups.find(group->category());
    if (gmi == this->_groups.end())
        throw std::invalid_argument("Asking for deletion of group not in manager!");
    delete group;
    this->_groups.erase(gmi);
}

Proxy_PBGroup*
AS_PBManager::get_group(const std::string& name, int create)
{
    Proxy_PBGroup* grp;
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end()) {
        grp = (*gmi).second;
        if (create != GRP_NONE && grp->group_type() != create) {
            throw std::invalid_argument("Group type mismatch");
        }
        return grp;
    }

    if (create == GRP_NONE)
        return nullptr;

    grp = new Proxy_PBGroup(this, name, structure(), create);
    if (name == structure()->PBG_METAL_COORDINATION)
        grp->set_default_color(147, 112, 219);
    else if (name == structure()->PBG_MISSING_STRUCTURE)
        grp->set_default_halfbond(true);
    else if (name == structure()->PBG_HYDROGEN_BONDS)
        grp->set_default_color(0, 204, 230);
    _groups[name] = grp;
    return grp;
}

Proxy_PBGroup*
PBManager::get_group(const std::string& name, int create)
{
    Proxy_PBGroup* grp;
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end()) {
        grp = (*gmi).second;
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

StructureManager::StructureManager(AtomicStructure* as):
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
    std::vector<std::string> categories;
    std::vector<Proxy_PBGroup*> groups;
    PyObject* cat_list = PyList_New(group_map().size());
    if (cat_list == nullptr) {
        throw std::runtime_error("Can't allocate list for pseudobond category names");
    }
    if (PyList_Append(*misc, cat_list) < 0)
        throw std::runtime_error("Can't append pseudobond category name list to misc list");
    int cat_index = 0;
    for (auto cat_grp: group_map()) {
        auto cat = cat_grp.first;
        auto grp = cat_grp.second;
        categories.push_back(cat);
        groups.push_back(grp);
        num_ints += grp->session_num_ints();
        num_floats += grp->session_num_floats();
        PyObject* py_cat = PyUnicode_FromString(cat.c_str());
        if (py_cat == nullptr)
            throw std::runtime_error("Cannot create Python string for pb group category");
        PyList_SET_ITEM(cat_list, cat_index++, py_cat);
    }
    int* int_array;
    *ints = python_int_array(num_ints, &int_array);
    float* float_array;
    *floats = python_float_array(num_floats, &float_array);
    for (auto grp: groups) {
        *int_array++ = grp->group_type();
        grp->session_save(&int_array, &float_array);
    }
    return 1;
}

void
BaseManager::session_restore(int** ints, float** floats, PyObject* misc)
{
    auto& int_ptr = *ints;

    if (!PyList_Check(misc) || PyList_Size(misc) != 1) {
        throw std::invalid_argument("PBManager::session_restore: third arg is not a"
            " 1-element list");
    }
    using pysupport::pylist_of_string_to_cvector;
    std::vector<std::string> categories;
    pylist_of_string_to_cvector(PyList_GET_ITEM(misc, 0), categories, "PB Group category");
    auto grp_type_ints = int_ptr;
    int_ptr += categories.size();
    for (auto cat: categories) {
        auto grp = get_group(cat, *grp_type_ints++);
        grp->session_restore(ints, floats);
    }
}

}  // namespace atomstruct
