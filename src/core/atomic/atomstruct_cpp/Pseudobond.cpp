// vi: set expandtab ts=4 sw=4:

#include "Atom.h"
#include "AtomicStructure.h"
#include <basegeom/destruct.h>
#include "Pseudobond.h"
#include <pythonarray.h>

#include <Python.h>

namespace atomstruct {

void
Owned_PBGroup_Base::_check_ownership(Atom* a1, Atom* a2)
{
    if (_owner != nullptr && (a1->structure() != _owner || a2->structure() != _owner))
        throw std::invalid_argument("Pseudobond endpoints not in "
            " atomic structure associated with group");
}

static void
_check_destroyed_atoms(PBonds& pbonds, const std::set<void*>& destroyed,
    GraphicsContainer* gc)
{
    PBonds remaining;
    for (auto pb: pbonds) {
        auto& pb_atoms = pb->atoms();
        if (destroyed.find(static_cast<void*>(pb_atoms[0])) != destroyed.end()
        || destroyed.find(static_cast<void*>(pb_atoms[1])) != destroyed.end()) {
            delete pb;
        } else {
            remaining.insert(pb);
        }
    }
    if (remaining.size() == 0) {
        pbonds.clear();
        gc->set_gc_shape();
    } else if (remaining.size() != pbonds.size()) {
        pbonds.swap(remaining);
        gc->set_gc_shape();
    }
}

void
CS_PBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = basegeom::DestructionBatcher(this);
    for (auto& cs_pbs: _pbonds)
        _check_destroyed_atoms(cs_pbs.second, destroyed,
            static_cast<GraphicsContainer*>(this));
}

void
Owned_PBGroup::check_destroyed_atoms(const std::set<void*>& destroyed)
{
    auto db = basegeom::DestructionBatcher(this);
    _check_destroyed_atoms(_pbonds, destroyed,
        static_cast<GraphicsContainer*>(this));
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

    grp = new Proxy_PBGroup(static_cast<Proxy_PBGroup::BaseManager*>(this),
        name, _owner, create);
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

    grp = new Proxy_PBGroup(static_cast<Proxy_PBGroup::BaseManager*>(this),
        name, nullptr, create);
    _groups[name] = grp;
    return grp;
}

void
AS_PBManager::_grp_session_info(const std::set<PBond*>& pbonds, PyObject* ints,
    PyObject* floats, PyObject* misc) const
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

PBond*
CS_PBGroup::new_pseudobond(Atom* a1, Atom* a2)
{
    return new_pseudobond(a1, a2, a1->structure()->active_coord_set());
}

PBond*
CS_PBGroup::new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs)
{
    _check_ownership(a1, a2);
    PBond* pb = new PBond(a1, a2, this);
    auto pbi = _pbonds.find(cs);
    if (pbi == _pbonds.end()) {
        _pbonds[cs].insert(pb);
    } else {
        (*pbi).second.insert(pb);
    }
    return pb;
}

const std::set<PBond*>&
CS_PBGroup::pseudobonds() const
{
    return pseudobonds(_owner->active_coord_set());
}

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
            for (auto cs: _owner->coord_sets()) {
                _grp_session_info(proxy->pseudobonds(cs), cs_int_list, cs_float_list,
                    cs_misc_list);
            }
        }
    }
    return 1;
}

}  // namespace atomstruct
