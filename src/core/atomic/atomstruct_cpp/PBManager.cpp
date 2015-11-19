// vi: set expandtab ts=4 sw=4:

#include "AtomicStructure.h"
#include "PBGroup.h"

#include <basegeom/destruct.h>

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

//TODO: code should just be directly adding ints/floats, not lists of ints/floats
#if 0
void
AS_PBManager::_grp_session_info(Group::Pseudobonds& pbonds,
    PyObject* ints, PyObject* floats, PyObject* misc) const
{

    size_t n = pbonds.size();
    int* int_array;
    PyObject* npy_array = python_int_array(n, 0, &int_array);
    if (PyList_Append(ints, npy_array) < 0)
        throw std::runtime_error(
            "Can't append numpy int array to pseudobond group list");

    float* float_array;
    npy_array = python_float_array(n, 0, &float_array);
    if (PyList_Append(floats, npy_array) < 0)
        throw std::runtime_error(
            "Can't append numpy float array to pseudobond group list");

    for (auto pb: pbonds) {
        //TODO
    }
}
#endif

StructureManager::StructureManager(AtomicStructure* as):
    BaseManager(as->change_tracker()), _structure(as) {}

void
AS_PBManager::remove_cs(const CoordSet* cs) {
    for (auto pbg_info: _groups) pbg_info.second->remove_cs(cs);
}

int
AS_PBManager::session_info(PyObject* ints, PyObject* floats, PyObject* misc) const
{
    PyObject* int_list = PyList_New(0);
    if (int_list == nullptr) {
        throw std::runtime_error("Can't allocate list for pseudobond group ints");
    }
    if (PyList_Append(ints, int_list) < 0)
        throw std::runtime_error("Can't append pseudobond ints to global list");
    PyObject* float_list = PyList_New(0);
    if (float_list == nullptr) {
        throw std::runtime_error("Can't allocate list for pseudobond group floats");
    }
    if (PyList_Append(floats, float_list) < 0)
        throw std::runtime_error("Can't append pseudobond floats to global list");
    PyObject* misc_list = PyList_New(0);
    if (misc_list == nullptr) {
        throw std::runtime_error("Can't allocate list for pseudobond group misc info");
    }
    if (PyList_Append(misc, misc_list) < 0)
        throw std::runtime_error("Can't append pseudobond misc info to global list");

//TODO: fix code below to keep ints/floats as simple lists, not lists of lists
#if 0
    for (auto cat_proxy: _groups) {
        auto category = cat_proxy.first;
        auto proxy = cat_proxy.second;

        PyObject* attr_list = PyList_New(2);
        if (attr_list == nullptr)
            throw std::runtime_error("Cannot create Python list for pb group misc info");
        if (PyList_Append(misc_list, attr_list) < 0)
            throw std::runtime_error("Can't append attr list to pb group list");
        // category
        PyObject* py_cat = PyUnicode_FromString(category.c_str());
        if (py_cat == nullptr)
            throw std::runtime_error("Cannot create Python string for pb group category");
        PyList_SET_ITEM(attr_list, 0, py_cat);
        PyObject* grp_attrs = PyList_New(0);
        if (grp_attrs == nullptr)
            throw std::runtime_error("Cannot create list for pb group attrs");
        PyList_SET_ITEM(attr_list, 1, grp_attrs);

        if (proxy->group_type() == GRP_NORMAL) {
            _grp_session_info(proxy->pseudobonds(), int_list, float_list, grp_attrs);
        } else {
            // per-coord-set group
            PyObject* cs_int_list = PyList_New(0);
            PyObject* cs_float_list = PyList_New(0);
            PyObject* cs_misc_list = PyList_New(0);
            if (cs_int_list == nullptr || cs_float_list == nullptr
            || cs_misc_list == nullptr)
                throw std::runtime_error("Can't create per-coord-set list");
            if (PyList_Append(int_list, cs_int_list) < 0)
                throw std::runtime_error("Cannot append to per-cs int list");
            if (PyList_Append(float_list, cs_float_list) < 0)
                throw std::runtime_error("Cannot append to per-cs float list");
            if (PyList_Append(misc_list, cs_misc_list) < 0)
                throw std::runtime_error("Cannot append to per-cs misc list");
            for (auto cs: structure()->coord_sets()) {
                _grp_session_info(proxy->pseudobonds(cs), cs_int_list, cs_float_list,
                    cs_misc_list);
            }
        }
    }
#endif
    return 1;
}

}  // namespace atomstruct
