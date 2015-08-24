// vi: set expandtab shiftwidth=4 softtabstop=4:
#include <Python.h>	// Use PyUnicode_FromString

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>      // use PyArray_*(), NPY_*

#include "atomstruct/Atom.h"
#include "atomstruct/Bond.h"
#include "atomstruct/Chain.h"
#include "atomstruct/Pseudobond.h"
#include "atomstruct/Residue.h"
#include "basegeom/destruct.h"     // Use DestructionObserver
#include "pythonarray.h"           // Use python_voidp_array()

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <functional>
#include <stdint.h>

// Argument delcaration types:
//
// numpy array arguments are sized, so use uint8_t for numpy's uint8,
// float32_t_t for numpys float32_t, etc.  The integer _t types are from
// <stdint.h>.  Special case is for numpy/C/C++ booleans which are
// processed in all cases as bytes:
// 	1 == numpy.bool_().nbytes in Python
// 	1 == sizeof (bool) in C++ and in C from <stdbool.h>
// 	25 == sizeof (bool [25]) in C++ and C
//
// Other arguments are their normal C types and are specified with the
// appropriate ctypes annotations on the Python side.
//
// There should be very few 'int' specifications.  Any int-array should
// have a specific size, eg., int32_t, for its elements.
//
typedef uint8_t npy_bool;
typedef float float32_t;
typedef double float64_t;
typedef void *pyobject_t;

inline PyObject* unicode_from_string(const char *data, size_t size)
{
    return PyUnicode_DecodeUTF8(data, size, "replace");
}

inline PyObject* unicode_from_string(const std::string& str)
{
    return PyUnicode_DecodeUTF8(str.data(), str.size(), "replace");
}

template <int len, char... description_chars>
inline PyObject* unicode_from_string(const chutil::CString<len, description_chars...>& cstr)
{
    return PyUnicode_DecodeUTF8(static_cast<const char*>(cstr), cstr.size(),
                            "replace");
}

static void
molc_error()
{
    // generic exception handler
    if (PyErr_Occurred())
        return;   // nothing to do, already set
    try {
        throw;    // rethrow exception to look at it
    } catch (std::bad_alloc&) {
        PyErr_SetString(PyExc_MemoryError, "not enough memory");
    } catch (std::invalid_argument& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
    } catch (std::length_error& e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
    } catch (std::out_of_range& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (std::overflow_error& e) {
        PyErr_SetString(PyExc_OverflowError, e.what());
    } catch (std::range_error& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (std::underflow_error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (std::logic_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (std::ios_base::failure& e) {
        PyErr_SetString(PyExc_IOError, e.what());
    } catch (std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception");
    }
}

// wrap an arbitrary function
template <typename F, typename... Args> auto
error_wrap(const F& func, Args... args) -> decltype(func(args...))
{
    try {
        return func(args...);
    } catch (...) {
        molc_error();
        return decltype(func(args...))();
    }
}

// wrap a member function
template <typename R, typename T, typename... Args> R
error_wrap(T* inst, R (T::*pm)(Args...), Args... args)
{
    try {
        return (inst->*pm)(args...);
    } catch (...) {
        molc_error();
        return R();
    }
}

// wrap a constant member function
template <typename R, typename T, typename... Args> R
error_wrap(T* inst, R (T::*pm)(Args...) const, Args... args)
{
    try {
        return (inst->*pm)(args...);
    } catch (...) {
        molc_error();
        return R();
    }
}

// wrap getting array elements via const member function
template <typename T, typename Elem, typename Elem2 = Elem> void
error_wrap_array_get(T** instances, size_t n, Elem (T::*pm)() const, Elem2* args)
{
    try {
        for (size_t i = 0; i < n; ++i)
            args[i] = (instances[i]->*pm)();
    } catch (...) {
        molc_error();
    }
}

// wrap setting array elements via member function
template <typename T, typename Elem, typename Elem2 = Elem> void
error_wrap_array_set(T** instances, size_t n, void (T::*pm)(Elem), Elem2* args)
{
    try {
        for (size_t i = 0; i < n; ++i)
            (instances[i]->*pm)(args[i]);
    } catch (...) {
        molc_error();
    }
}


using namespace atomstruct;
using basegeom::Coord;
using basegeom::Real;
using basegeom::DestructionObserver;

extern "C" void atom_bfactor(void *atoms, size_t n, float32_t *bfactors)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::bfactor, bfactors);
}

extern "C" void set_atom_bfactor(void *atoms, size_t n, float32_t *bfactors)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set(a, n, &Atom::set_bfactor, bfactors);
}

extern "C" void atom_bonds(void *atoms, size_t n, pyobject_t *bonds)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Atom::Bonds &b = a[i]->bonds();
            for (size_t j = 0; j != b.size(); ++j)
                *bonds++ = b[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_bonded_atoms(void *atoms, size_t n, pyobject_t *batoms)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Atom::Bonds &b = a[i]->bonds();
            for (size_t j = 0; j != b.size(); ++j)
                *batoms++ = b[j]->other_atom(a[i]);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_chain_id(void *atoms, size_t n, pyobject_t *cids)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(a[i]->residue()->chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_color(void *atoms, size_t n, uint8_t *rgba)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Rgba &c = a[i]->color();
            *rgba++ = c.r;
            *rgba++ = c.g;
            *rgba++ = c.b;
            *rgba++ = c.a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_atom_color(void *atoms, size_t n, uint8_t *rgba)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        Rgba c;
        for (size_t i = 0; i != n; ++i) {
            c.r = *rgba++;
            c.g = *rgba++;
            c.b = *rgba++;
            c.a = *rgba++;
            a[i]->set_color(c);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" bool atom_connects_to(pyobject_t atom1, pyobject_t atom2)
{
    Atom *a1 = static_cast<Atom *>(atom1), *a2 = static_cast<Atom *>(atom2);
#if 1
    return error_wrap([&] () {
                      return a1->connects_to(a2);
                      });
#else
    try {
        return a1->connects_to(a2);
    } catch (...) {
        molc_error();
        return false;
    }
#endif
}

extern "C" void atom_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Coord &c = a[i]->coord();
            *xyz++ = c[0];
            *xyz++ = c[1];
            *xyz++ = c[2];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_atom_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            Real x = *xyz++, y = *xyz++, z = *xyz++;
            a[i]->set_coord(Coord(x,y,z));
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_delete(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        std::map<AtomicStructure *, std::vector<Atom *> > matoms;
        for (size_t i = 0; i != n; ++i)
            matoms[a[i]->structure()].push_back(a[i]);

        for (auto ma: matoms)
            ma.first->delete_atoms(ma.second);
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_display(void *atoms, size_t n, npy_bool *disp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::display, disp);
}

extern "C" void set_atom_display(void *atoms, size_t n, npy_bool *disp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, bool, npy_bool>(a, n, &Atom::set_display, disp);
}

extern "C" void atom_draw_mode(void *atoms, size_t n, int32_t *modes)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, int>(a, n, &Atom::draw_mode, modes);
}

extern "C" void set_atom_draw_mode(void *atoms, size_t n, int32_t *modes)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, int, int>(a, n, &Atom::set_draw_mode, modes);
}

extern "C" void atom_element_name(void *atoms, size_t n, pyobject_t *names)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(a[i]->element().name());
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_element_number(void *atoms, size_t n, uint8_t *nums)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            nums[i] = a[i]->element().number();
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_in_chain(void *atoms, size_t n, npy_bool *in_chain)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            in_chain[i] = (a[i]->residue()->chain() != NULL);
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_is_backbone(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::is_backbone, sel);
}

extern "C" void set_atom_is_backbone(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, bool, npy_bool>(a, n, &Atom::set_is_backbone, sel);
}

extern "C" void atom_structure(void *atoms, size_t n, pyobject_t *molp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::structure, molp);
}

extern "C" void atom_name(void *atoms, size_t n, pyobject_t *names)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(a[i]->name());
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_num_bonds(void *atoms, size_t n, size_t *nbonds)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            nbonds[i] = a[i]->bonds().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_radius(void *atoms, size_t n, float32_t *radii)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::radius, radii);
}

extern "C" void set_atom_radius(void *atoms, size_t n, float32_t *radii)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set(a, n, &Atom::set_radius, radii);
}

extern "C" void atom_residue(void *atoms, size_t n, pyobject_t *resp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::residue, resp);
}

// Apply per-structure transform to atom coordinates.
extern "C" void atom_scene_coords(void *atoms, size_t n, void *mols, size_t m, float64_t *mtf, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    AtomicStructure **ma = static_cast<AtomicStructure **>(mols);

    try {
        std::map<AtomicStructure *, double *> tf;
        for (size_t i = 0; i != m; ++i)
            tf[ma[i]] = mtf + 12*i;

        for (size_t i = 0; i != n; ++i) {
            AtomicStructure *s = a[i]->structure();
            double *t = tf[s];
            const Coord &c = a[i]->coord();
            double x = c[0], y = c[1], z = c[2];
            *xyz++ = t[0]*x + t[1]*y + t[2]*z + t[3];
            *xyz++ = t[4]*x + t[5]*y + t[6]*z + t[7];
            *xyz++ = t[8]*x + t[9]*y + t[10]*z + t[11];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void atom_selected(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::selected, sel);
}

extern "C" void set_atom_selected(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, bool, npy_bool>(a, n, &Atom::set_selected, sel);
}

extern "C" size_t atom_num_selected(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    size_t s = 0;
    try {
        for (size_t i = 0; i != n; ++i)
            if (a[i]->selected())
                s += 1;
        return s;
    } catch (...) {
        molc_error();
        return 0;
    }
}

extern "C" void bond_atoms(void *bonds, size_t n, pyobject_t *atoms)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Bond::Atoms &a = b[i]->atoms();
            *atoms++ = a[0];
            *atoms++ = a[1];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void bond_color(void *bonds, size_t n, uint8_t *rgba)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Rgba &c = b[i]->color();
            *rgba++ = c.r;
            *rgba++ = c.g;
            *rgba++ = c.b;
            *rgba++ = c.a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_bond_color(void *bonds, size_t n, uint8_t *rgba)
{
    Bond **b = static_cast<Bond **>(bonds);
    Rgba c;
    try {
        for (size_t i = 0; i != n; ++i) {
            c.r = *rgba++;
            c.g = *rgba++;
            c.b = *rgba++;
            c.a = *rgba++;
            b[i]->set_color(c);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void bond_display(void *bonds, size_t n, uint8_t *disp)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i)
            disp[i] = static_cast<uint8_t>(b[i]->display());
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_bond_display(void *bonds, size_t n, uint8_t *disp)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i)
            b[i]->set_display(static_cast<Bond::BondDisplay>(disp[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" void bond_halfbond(void *bonds, size_t n, npy_bool *halfb)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, bool, npy_bool>(b, n, &Bond::halfbond, halfb);
}

extern "C" void set_bond_halfbond(void *bonds, size_t n, npy_bool *halfb)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, bool, npy_bool>(b, n, &Bond::set_halfbond, halfb);
}

extern "C" void bond_radius(void *bonds, size_t n, float32_t *radii)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, float>(b, n, &Bond::radius, radii);
}

extern "C" void set_bond_radius(void *bonds, size_t n, float32_t *radii)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, float>(b, n, &Bond::set_radius, radii);
}

extern "C" void pseudobond_atoms(void *pbonds, size_t n, pyobject_t *atoms)
{
    PBond **b = static_cast<PBond **>(pbonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            const PBond::Atoms &a = b[i]->atoms();
            *atoms++ = a[0];
            *atoms++ = a[1];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void pseudobond_color(void *pbonds, size_t n, uint8_t *rgba)
{
    PBond **b = static_cast<PBond **>(pbonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Rgba &c = b[i]->color();
            *rgba++ = c.r;
            *rgba++ = c.g;
            *rgba++ = c.b;
            *rgba++ = c.a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_pseudobond_color(void *pbonds, size_t n, uint8_t *rgba)
{
    PBond **b = static_cast<PBond **>(pbonds);
    Rgba c;
    try {
        for (size_t i = 0; i != n; ++i) {
            c.r = *rgba++;
            c.g = *rgba++;
            c.b = *rgba++;
            c.a = *rgba++;
            b[i]->set_color(c);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void pseudobond_display(void *pbonds, size_t n, uint8_t *disp)
{
    PBond **b = static_cast<PBond **>(pbonds);
    try {
        for (size_t i = 0; i != n; ++i)
            disp[i] = static_cast<uint8_t>(b[i]->display());
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_pseudobond_display(void *pbonds, size_t n, uint8_t *disp)
{
    PBond **b = static_cast<PBond **>(pbonds);
    try {
        for (size_t i = 0; i != n; ++i)
            b[i]->set_display(static_cast<Bond::BondDisplay>(disp[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" void pseudobond_halfbond(void *pbonds, size_t n, npy_bool *halfb)
{
    PBond **b = static_cast<PBond **>(pbonds);
    error_wrap_array_get<PBond, bool, npy_bool>(b, n, &PBond::halfbond, halfb);
}

extern "C" void set_pseudobond_halfbond(void *pbonds, size_t n, npy_bool *halfb)
{
    PBond **b = static_cast<PBond **>(pbonds);
    error_wrap_array_set<PBond, bool, npy_bool>(b, n, &PBond::set_halfbond, halfb);
}

extern "C" void pseudobond_radius(void *pbonds, size_t n, float32_t *radii)
{
    PBond **b = static_cast<PBond **>(pbonds);
    error_wrap_array_get<PBond, float>(b, n, &PBond::radius, radii);
}

extern "C" void set_pseudobond_radius(void *pbonds, size_t n, float32_t *radii)
{
    PBond **b = static_cast<PBond **>(pbonds);
    error_wrap_array_set<PBond, float>(b, n, &PBond::set_radius, radii);
}

extern "C" void pseudobond_group_category(void *pbgroups, int n, void **categories)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (int i = 0 ; i < n ; ++i)
            categories[i] = unicode_from_string(pbg[i]->category());
    } catch (...) {
        molc_error();
    }
}

extern "C" void pseudobond_group_gc_color(void *pbgroups, size_t n, npy_bool *color_changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_get<Proxy_PBGroup, bool, npy_bool>(pbg, n, &Proxy_PBGroup::get_gc_color, color_changed);
}

extern "C" void set_pseudobond_group_gc_color(void *pbgroups, size_t n, npy_bool *color_changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_set<Proxy_PBGroup, bool, npy_bool>(pbg, n, &Proxy_PBGroup::set_gc_color, color_changed);
}

extern "C" void pseudobond_group_gc_select(void *pbgroups, size_t n, npy_bool *select_changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_get<Proxy_PBGroup, bool, npy_bool>(pbg, n, &Proxy_PBGroup::get_gc_select, select_changed);
}

extern "C" void set_pseudobond_group_gc_select(void *pbgroups, size_t n, npy_bool *select_changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_set<Proxy_PBGroup, bool, npy_bool>(pbg, n, &Proxy_PBGroup::set_gc_select, select_changed);
}

extern "C" void pseudobond_group_gc_shape(void *pbgroups, size_t n, npy_bool *shape_changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_get<Proxy_PBGroup, bool, npy_bool>(pbg, n, &Proxy_PBGroup::get_gc_shape, shape_changed);
}

extern "C" void set_pseudobond_group_gc_shape(void *pbgroups, size_t n, npy_bool *shape_changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_set<Proxy_PBGroup, bool, npy_bool>(pbg, n, &Proxy_PBGroup::set_gc_shape, shape_changed);
}

extern "C" void *pseudobond_group_new_pseudobond(void *pbgroup, void *atom1, void *atom2)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        PBond *b = pbg->new_pseudobond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2));
        return b;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void pseudobond_group_num_pseudobonds(void *pbgroups, size_t n, size_t *num_pseudobonds)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (size_t i = 0; i != n; ++i)
            *num_pseudobonds++ = pbg[i]->pseudobonds().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" void pseudobond_group_pseudobonds(void *pbgroups, size_t n, pyobject_t *pseudobonds)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (size_t i = 0 ; i != n ; ++i)
            for (auto pb: pbg[i]->pseudobonds())
                *pseudobonds++ = pb;
    } catch (...) {
        molc_error();
    }
}

extern "C" void *pseudobond_create_global_manager()
{
    try {
        auto pb_manager = new PBManager();
        return pb_manager;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void* pseudobond_global_manager_get_group(void *manager, const char* name, int create)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        return mgr->get_group(name, create);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void residue_atoms(void *residues, size_t n, pyobject_t *atoms)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Residue::Atoms &a = r[i]->atoms();
            for (size_t j = 0; j != a.size(); ++j)
                *atoms++ = a[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_chain_id(void *residues, size_t n, pyobject_t *cids)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(r[i]->chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_is_helix(void *residues, size_t n, npy_bool *is_helix)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::is_helix, is_helix);
}

extern "C" void set_residue_is_helix(void *residues, size_t n, npy_bool *is_helix)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_is_helix, is_helix);
}

extern "C" void residue_is_sheet(void *residues, size_t n, npy_bool *is_sheet)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::is_sheet, is_sheet);
}

extern "C" void set_residue_is_sheet(void *residues, size_t n, npy_bool *is_sheet)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_is_sheet, is_sheet);
}

extern "C" void residue_ss_id(void *residues, size_t n, int32_t *ss_id)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ss_id, ss_id);
}

extern "C" void set_residue_ss_id(void *residues, size_t n, int32_t *ss_id)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ss_id, ss_id);
}

extern "C" void residue_ribbon_display(void *residues, size_t n, npy_bool *ribbon_display)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ribbon_display, ribbon_display);
}

extern "C" void set_residue_ribbon_display(void *residues, size_t n, npy_bool *ribbon_display)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ribbon_display, ribbon_display);
}

extern "C" void residue_structure(void *residues, size_t n, pyobject_t *molp)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::structure, molp);
}

extern "C" void residue_name(void *residues, size_t n, pyobject_t *names)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(r[i]->name());
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_num_atoms(void *residues, size_t n, size_t *natoms)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            natoms[i] = r[i]->atoms().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_number(void *residues, size_t n, int32_t *nums)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::position, nums);
}

extern "C" void residue_str(void *residues, size_t n, pyobject_t *strs)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            strs[i] = unicode_from_string(r[i]->str().c_str());
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_unique_id(void *residues, size_t n, int32_t *rids)
{
    Residue **res = static_cast<Residue **>(residues);
    int32_t rid = -1;
    const Residue *rprev = NULL;
    try {
        for (size_t i = 0; i != n; ++i) {
            const Residue *r = res[i];
            if (rprev == NULL || r->position() != rprev->position() || r->chain_id() != rprev->chain_id()) {
                rid += 1;
                rprev = r;
            }
            rids[i] = rid;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_add_atom(void *res, void *atom)
{
    Residue *r = static_cast<Residue *>(res);
    try {
        r->add_atom(static_cast<Atom *>(atom));
    } catch (...) {
        molc_error();
    }
}

extern "C" void residue_ribbon_color(void *residues, size_t n, uint8_t *rgba)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Rgba &c = r[i]->ribbon_color();
            *rgba++ = c.r;
            *rgba++ = c.g;
            *rgba++ = c.b;
            *rgba++ = c.a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_residue_ribbon_color(void *residues, size_t n, uint8_t *rgba)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        Rgba c;
        for (size_t i = 0; i != n; ++i) {
            c.r = *rgba++;
            c.g = *rgba++;
            c.b = *rgba++;
            c.a = *rgba++;
            r[i]->set_ribbon_color(c);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void chain_chain_id(void *chains, size_t n, pyobject_t *cids)
{
    Chain **c = static_cast<Chain **>(chains);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(c[i]->chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" void chain_structure(void *chains, size_t n, pyobject_t *molp)
{
    Chain **c = static_cast<Chain **>(chains);
    error_wrap_array_get(c, n, &Chain::structure, molp);
}

extern "C" void chain_num_residues(void *chains, size_t n, size_t *nres)
{
    Chain **c = static_cast<Chain **>(chains);
    try {
        for (size_t i = 0; i != n; ++i)
            nres[i] = c[i]->residues().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" void chain_residues(void *chains, size_t n, pyobject_t *res)
{
    Chain **c = static_cast<Chain **>(chains);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Chain::Residues &r = c[i]->residues();
            for (size_t j = 0; j != r.size(); ++j)
                *res++ = r[i];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void *structure_copy(void *mol)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    try {
        return m->copy();
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void structure_gc_color(void *mols, size_t n, npy_bool *color_changed)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get<AtomicStructure, bool, npy_bool>(m, n, &AtomicStructure::get_gc_color, color_changed);
}

extern "C" void set_structure_gc_color(void *mols, size_t n, npy_bool *color_changed)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_set<AtomicStructure, bool, npy_bool>(m, n, &AtomicStructure::set_gc_color, color_changed);
}

extern "C" void structure_gc_select(void *mols, size_t n, npy_bool *select_changed)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get<AtomicStructure, bool, npy_bool>(m, n, &AtomicStructure::get_gc_select, select_changed);
}

extern "C" void set_structure_gc_select(void *mols, size_t n, npy_bool *select_changed)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_set<AtomicStructure, bool, npy_bool>(m, n, &AtomicStructure::set_gc_select, select_changed);
}

extern "C" void structure_gc_shape(void *mols, size_t n, npy_bool *shape_changed)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get<AtomicStructure, bool, npy_bool>(m, n, &AtomicStructure::get_gc_shape, shape_changed);
}

extern "C" void set_structure_gc_shape(void *mols, size_t n, npy_bool *shape_changed)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_set<AtomicStructure, bool, npy_bool>(m, n, &AtomicStructure::set_gc_shape, shape_changed);
}

extern "C" void structure_name(void *mols, size_t n, pyobject_t *names)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(m[i]->name().c_str());
    } catch (...) {
        molc_error();
    }
}

extern "C" void set_structure_name(void *mols, size_t n, pyobject_t *names)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
            m[i]->set_name(PyUnicode_AsUTF8(static_cast<PyObject *>(names[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" void structure_num_atoms(void *mols, size_t n, size_t *natoms)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
            natoms[i] = m[i]->atoms().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" void structure_atoms(void *mols, size_t n, pyobject_t *atoms)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const AtomicStructure::Atoms &a = m[i]->atoms();
            for (size_t j = 0; j != a.size(); ++j)
                *atoms++ = a[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void structure_num_bonds(void *mols, size_t n, size_t *nbonds)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get(m, n, &AtomicStructure::num_bonds, nbonds);
}

extern "C" void structure_bonds(void *mols, size_t n, pyobject_t *bonds)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const AtomicStructure::Bonds &b = m[i]->bonds();
            for (size_t j = 0; j != b.size(); ++j)
                *bonds++ = b[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void structure_num_residues(void *mols, size_t n, size_t *nres)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get(m, n, &AtomicStructure::num_residues, nres);
}

extern "C" void structure_residues(void *mols, size_t n, pyobject_t *res)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const AtomicStructure::Residues &r = m[i]->residues();
            for (size_t j = 0; j != r.size(); ++j)
                *res++ = r[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void structure_num_coord_sets(void *mols, size_t n, size_t *ncoord_sets)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get(m, n, &AtomicStructure::num_coord_sets, ncoord_sets);
}

extern "C" void structure_num_chains(void *mols, size_t n, size_t *nchains)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    error_wrap_array_get(m, n, &AtomicStructure::num_chains, nchains);
}

extern "C" void structure_chains(void *mols, size_t n, pyobject_t *chains)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const AtomicStructure::Chains &c = m[i]->chains();
            for (size_t j = 0; j != c.size(); ++j)
                *chains++ = c[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void structure_pbg_map(void *mols, size_t n, pyobject_t *pbgs)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    PyObject* pbg_map = NULL;
    try {
        for (size_t i = 0; i != n; ++i) {
            pbg_map = PyDict_New();
            for (auto grp_info: m[i]->pb_mgr().group_map()) {
                PyObject* name = unicode_from_string(grp_info.first.c_str());
                PyObject *pbg = PyLong_FromVoidPtr(grp_info.second);
                PyDict_SetItem(pbg_map, name, pbg);
            }
            pbgs[i] = pbg_map;
            pbg_map = NULL;
        }
    } catch (...) {
        Py_XDECREF(pbg_map);
        molc_error();
    }
}

extern "C" Proxy_PBGroup *structure_pseudobond_group(void *mol, const char *name, int create_type)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    try {
        Proxy_PBGroup *pbg = m->pb_mgr().get_group(name, create_type);
        return pbg;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" PyObject *structure_polymers(void *mol, int consider_missing_structure, int consider_chains_ids)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    PyObject *poly = NULL;
    try {
        std::vector<Chain::Residues> polymers = m->polymers(consider_missing_structure, consider_chains_ids);
        poly = PyTuple_New(polymers.size());
        size_t p = 0;
        for (auto resvec: polymers) {
            void **ra;
            PyObject *r_array = python_voidp_array(resvec.size(), &ra);
            size_t i = 0;
            for (auto r: resvec)
                ra[i++] = static_cast<void *>(r);
            PyTuple_SetItem(poly, p++, r_array);
        }
        return poly;
    } catch (...) {
        Py_XDECREF(poly);
        molc_error();
        return nullptr;
    }
}

extern "C" void *structure_new()
{
    try {
        AtomicStructure *m = new AtomicStructure();
        return m;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void structure_delete(void *mol)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    try {
        delete m;
    } catch (...) {
        molc_error();
    }
}

extern "C" void *structure_new_atom(void *mol, const char *atom_name, const char *element_name)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    try {
        Atom *a = m->new_atom(atom_name, Element(element_name));
        return a;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void *structure_new_bond(void *mol, void *atom1, void *atom2)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    try {
        Bond *b = m->new_bond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2));
        return b;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void *structure_new_residue(void *mol, const char *residue_name, const char *chain_id, int pos)
{
    AtomicStructure *m = static_cast<AtomicStructure *>(mol);
    try {
        Residue *r = m->new_residue(residue_name, chain_id, pos, ' ');
        return r;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void *element_new_name(const char *name)
{
    try {
        Element *e = new Element(name);
        return e;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void *element_new_number(size_t number)
{
    try {
        Element *e = new Element(number);
        return e;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" void element_name(void *elements, size_t n, pyobject_t *names)
{
    Element **e = static_cast<Element **>(elements);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = PyUnicode_FromString(e[i]->name());
    } catch (...) {
        molc_error();
    }
}

extern "C" void element_number(void *elements, size_t n, uint8_t *number)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::number, number);
}

extern "C" void element_mass(void *elements, size_t n, float *mass)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::mass, mass);
}

extern "C" void element_is_metal(void *elements, size_t n, npy_bool *metal)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::is_metal, metal);
}

static void *init_numpy()
{
    import_array(); // Initialize use of numpy
    return NULL;
}

// ---------------------------------------------------------------------------
// When a C++ object is deleted eliminate it from numpy arrays of pointers.
//
class Array_Updater : DestructionObserver
{
public:
    Array_Updater()
    {
        init_numpy();
    }
    void add_array(PyObject *numpy_array)
    {
        arrays.insert(reinterpret_cast<PyArrayObject *>(numpy_array));
    }
    void remove_array(void *numpy_array)
    {
        arrays.erase(reinterpret_cast<PyArrayObject *>(numpy_array));
    }
    size_t array_count()
    {
        return arrays.size();
    }
private:
    virtual void  destructors_done(const std::set<void*>& destroyed)
    {
        for (auto a: arrays)
            filter_array(a, destroyed);
    }

    void filter_array(PyArrayObject *a, const std::set<void*>& destroyed)
    {
        // Remove any destroyed pointers from numpy array and shrink the array in place.
        // Numpy array must be contiguous, 1 dimensional array.
        void **ae = static_cast<void **>(PyArray_DATA(a));
        npy_intp s = PyArray_SIZE(a);
        npy_intp j = 0;
        for (npy_intp i = 0; i < s ; ++i)
            if (destroyed.find(ae[i]) == destroyed.end())
                ae[j++] = ae[i];
        if (j < s) {
            //std::cerr << "resizing array " << a << " from " << s << " to " << j << std::endl;
            *PyArray_DIMS(a) = j;        // TODO: This hack may break numpy.
            /*
            // Numpy array can't be resized with weakref made by weakref.finalize().  Not sure why.
            // Won't work anyways because array will reallocate while looping over old array of atoms being deleted.
            PyArray_Dims dims;
            dims.len = 1;
            dims.ptr = &j;
            std::cerr << " base " << PyArray_BASE(a) << " weak " << ((PyArrayObject_fields *)a)->weakreflist << std::endl;
            if (PyArray_Resize(a, &dims, 0, NPY_CORDER) == NULL) {
            std::cerr << "Failed to delete structure object pointers from numpy array." << std::endl;
            PyErr_Print();
            }
            */
        }
    }
    std::set<PyArrayObject *> arrays;
};

class Array_Updater *array_updater = NULL;

extern "C" void remove_deleted_c_pointers(PyObject *numpy_array)
{
    try {
        if (array_updater == NULL)
            array_updater = new Array_Updater();

        array_updater->add_array(numpy_array);
    } catch (...) {
        molc_error();
    }
}

extern "C" void pointer_array_freed(void *numpy_array)
{
    try {
        if (array_updater) {
            array_updater->remove_array(numpy_array);
            if (array_updater->array_count() == 0) {
                delete array_updater;
                array_updater = NULL;
            }
        }
    } catch (...) {
        molc_error();
    }
}

class Object_Map_Deletion_Handler : DestructionObserver
{
public:
    Object_Map_Deletion_Handler(PyObject *object_map) : object_map(object_map) {}

private:
    PyObject *object_map;        // Dictionary from C++ pointer to Python wrapped object having a _c_pointer attribute.

    virtual void  destructors_done(const std::set<void*>& destroyed)
    {
        remove_deleted_objects(destroyed);
    }

    void remove_deleted_objects(const std::set<void*>& destroyed)
    {
        if (PyDict_Size(object_map) == 0)
            return;
        // TODO: Optimize by looping over object_map if it is smaller than destroyed.
        for (auto d: destroyed) {
            PyObject *dp = PyLong_FromVoidPtr(d);
            if (PyDict_Contains(object_map, dp)) {
                PyObject *po = PyDict_GetItem(object_map, dp);
                PyObject_DelAttrString(po, "_c_pointer");
                PyObject_DelAttrString(po, "_c_pointer_ref");
                PyDict_DelItem(object_map, dp);
            }
            Py_DECREF(dp);
        }
    }
};

extern "C" void *object_map_deletion_handler(void *object_map)
{
    try {
	return new Object_Map_Deletion_Handler(static_cast<PyObject *>(object_map));
    } catch (...) {
        molc_error();
	return nullptr;
    }
}

extern "C" void delete_object_map_deletion_handler(void *handler)
{
    try {
        delete static_cast<Object_Map_Deletion_Handler *>(handler);
    } catch (...) {
        molc_error();
    }
}

extern "C" ssize_t pointer_index(void *pointer_array, size_t n, void *pointer)
{
    void **pa = static_cast<void **>(pointer_array);
    try {
        for (size_t i = 0; i != n; ++i)
            if (pa[i] == pointer)
                return i;
        return -1;
    } catch (...) {
        molc_error();
        return -1;
    }
}

extern "C" void pointer_mask(void *pointer_array, size_t n, void *pointer_array2, size_t n2, unsigned char *mask)
{
    void **pa = static_cast<void **>(pointer_array);
    void **pa2 = static_cast<void **>(pointer_array2);
    try {
        std::set<void *> s;
        for (size_t i = 0; i != n2; ++i)
            s.insert(pa2[i]);
        for (size_t i = 0; i != n; ++i)
            mask[i] = (s.find(pa[i]) == s.end() ? 0 : 1);
    } catch (...) {
        molc_error();
    }
}

extern "C" bool pointer_intersects(void *pointer_array, size_t n, void *pointer_array2, size_t n2)
{
    void **pa = static_cast<void **>(pointer_array);
    void **pa2 = static_cast<void **>(pointer_array2);
    try {
        std::set<void *> s;
        for (size_t i = 0; i != n2; ++i)
            s.insert(pa2[i]);
        for (size_t i = 0; i != n; ++i)
            if (s.find(pa[i]) != s.end())
                return true;
        return false;
    } catch (...) {
        molc_error();
        return false;
    }
}

extern "C" void pointer_intersects_each(void *pointer_arrays, size_t na, size_t *sizes,
                                        void *pointer_array, size_t n,
                                        npy_bool *intersects)
{
    void ***pas = static_cast<void ***>(pointer_arrays);
    void **pa = static_cast<void **>(pointer_array);
    try {
        std::set<void *> s;
        for (size_t i = 0; i != n; ++i)
            s.insert(pa[i]);
        for (size_t i = 0 ; i != na; ++i) {
            size_t m = sizes[i];
            void **pai = pas[i];
            intersects[i] = false;
            for (size_t j = 0; j != m; ++j)
                if (s.find(pai[j]) != s.end()) {
                    intersects[i] = true;
                    break;
                }
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" void metadata(void *mols, size_t n, pyobject_t *headers)
{
    AtomicStructure **m = static_cast<AtomicStructure **>(mols);
    PyObject* header_map = NULL;
    try {
        for (size_t i = 0; i < n; ++i) {
            header_map = PyDict_New();
            auto& metadata = m[i]->metadata;
            for (auto& item: metadata) {
                PyObject* key = unicode_from_string(item.first);
                auto& headers = item.second;
                size_t count = headers.size();
                PyObject* values = PyList_New(count);
                for (size_t i = 0; i != count; ++i)
                    PyList_SetItem(values, i, unicode_from_string(headers[i]));
                PyDict_SetItem(header_map, key, values);
            }
            headers[i] = header_map;
            header_map = NULL;
        }
    } catch (...) {
        Py_XDECREF(header_map);
        molc_error();
    }
}
