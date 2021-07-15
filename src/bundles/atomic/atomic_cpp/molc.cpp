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

#include <Python.h>     // Use PyUnicode_FromString

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>      // use PyArray_*(), NPY_*

#include <atomstruct/Atom.h>
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Chain.h>
#include <atomstruct/ChangeTracker.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/destruct.h>     // Use DestructionObserver
#include <atomstruct/MolResId.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/polymer.h>
#include <atomstruct/Pseudobond.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Ring.h>
#include <atomstruct/seq_assoc.h>
#include <atomstruct/Sequence.h>
#include <arrays/pythonarray.h>           // Use python_voidp_array()
#include <pysupport/convert.h>     // Use cset_of_chars_to_pyset

#include <functional>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include <cmath>

#ifndef M_PI
// not defined on Windows
# define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
# define EXPORT __declspec(dllexport)
#else
# define EXPORT __attribute__((__visibility__("default")))
#endif

// Argument delcaration types:
//
// numpy array arguments are sized, so use uint8_t for numpy's uint8,
// float32_t for numpys float32_t, etc.  The integer _t types are from
// <stdint.h>.  Special case is for numpy/C/C++ booleans which are
// processed in all cases as bytes:
//      1 == numpy.bool_().nbytes in Python
//      1 == sizeof (bool) in C++ and in C from <stdbool.h>
//      25 == sizeof (bool [25]) in C++ and C
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

inline PyObject* unicode_from_character(char c)
{
    char buffer[2];
    buffer[0] = c;
    buffer[1] = '\0';
    return unicode_from_string(buffer, 1);
}

inline std::string string_from_unicode(PyObject* obj)
{
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize(obj, &size);
    std::string result(data, size);
    return result;
}

inline const char* CheckedPyUnicode_AsUTF8(PyObject* unicode)
{
    if (PyUnicode_Check(unicode)) {
        return PyUnicode_AsUTF8(unicode);
    }
    throw std::invalid_argument("Not a Unicode string");
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
    } catch (std::regex_error& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
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

// wrap setting "const" (mutable) array elements via member function
template <typename T, typename Elem, typename Elem2 = Elem> void
error_wrap_array_set_mutable(T** instances, size_t n, void (T::*pm)(Elem) const, Elem2* args)
{
    try {
        for (size_t i = 0; i < n; ++i)
            (instances[i]->*pm)(args[i]);
    } catch (...) {
        molc_error();
    }
}

// wrap array calling single argument member function
template <typename T, typename Elem, typename Elem2 = Elem> void
error_wrap_array_1arg(T** instances, size_t n, void (T::*pm)(Elem), Elem2 arg)
{
    try {
        for (size_t i = 0; i < n; ++i)
            (instances[i]->*pm)(arg);
    } catch (...) {
        molc_error();
    }
}


using namespace atomstruct;

// -------------------------------------------------------------------------
// atom functions
//
extern "C" EXPORT void atom_bfactor(void *atoms, size_t n, float32_t *bfactors)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::bfactor, bfactors);
}

extern "C" EXPORT void set_atom_bfactor(void *atoms, size_t n, float32_t *bfactors)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set(a, n, &Atom::set_bfactor, bfactors);
}

extern "C" EXPORT void atom_has_aniso_u(void *atoms, size_t n, npy_bool *has_aniso_u)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::has_aniso_u, has_aniso_u);
}

extern "C" EXPORT void atom_aniso_u(void *atoms, size_t n, float32_t *aniso_u)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const std::vector<float> *ai = a[i]->aniso_u();
            if (ai) {
                // Copy 6 values of symmetric matrix to 3x3 matrix.
                float32_t *ani = aniso_u + 9*i;
                float32_t a00 = (*ai)[0], a01 = (*ai)[1], a02 = (*ai)[2];
                float32_t a11 = (*ai)[3], a12 = (*ai)[4], a22 = (*ai)[5];
                ani[0] = a00; ani[1] = a01; ani[2] = a02;
                ani[3] = a01; ani[4] = a11; ani[5] = a12;
                ani[6] = a02; ani[7] = a12; ani[8] = a22;
            } else {
                PyErr_SetString(PyExc_ValueError, "Atom has no aniso_u value.");
                break;
            }
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_aniso_u6(void *atoms, size_t n, float32_t *aniso_u)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const std::vector<float> *ai = a[i]->aniso_u();
            if (ai) {
                float32_t *ani = aniso_u + 6*i;
                ani[0] = (*ai)[0]; ani[1] = (*ai)[3]; ani[2] = (*ai)[5];
                ani[3] = (*ai)[1]; ani[4] = (*ai)[2]; ani[5] = (*ai)[4];
            } else {
                PyErr_SetString(PyExc_ValueError, "Atom has no aniso_u value.");
                break;
            }
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_aniso_u6(void *atoms, size_t n, float32_t *aniso_u)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
          float32_t *ani = aniso_u + 6*i;
          a[i]->set_aniso_u(ani[0],ani[3],ani[4],ani[1],ani[5],ani[2]);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void clear_atom_aniso_u6(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
          a[i]->clear_aniso_u();
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_occupancy(void *atoms, size_t n, float32_t *occupancies)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::occupancy, occupancies);
}

extern "C" EXPORT void set_atom_occupancy(void *atoms, size_t n, float32_t *occupancies)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set(a, n, &Atom::set_occupancy, occupancies);
}

extern "C" EXPORT void atom_bonds(void *atoms, size_t n, pyobject_t *bonds)
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

// Return list of (structure, chain_id, atoms).
typedef std::pair<Structure *, std::string> StructureChain;
extern "C" EXPORT PyObject *atom_by_chain(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);

    try {
        std::map<StructureChain, Structure::Atoms> sca;
        for (size_t i = 0; i < n; ++i) {
  	    Atom *atom = a[i];
	    Structure *s = atom->structure();
	    const ChainID &cid = atom->residue()->chain_id();
	    Structure::Atoms &catoms = sca[StructureChain(s,cid)];
	    catoms.push_back(atom);
        }
	PyObject *sca_tuple = PyTuple_New(sca.size());
	size_t i = 0;
	for (auto mi = sca.begin() ; mi != sca.end() ; ++mi) {
	    const StructureChain &sc = mi->first;
	    Structure *s = sc.first;
	    const ChainID &cid = sc.second;
	    PyObject *py_cid = PyUnicode_FromString(cid.c_str());
	    Structure::Atoms &atoms = mi->second;
	    const Atom **aa;
	    PyObject *atoms_array = python_voidp_array(atoms.size(), (void***)&aa);
	    for (size_t ai = 0 ; ai < atoms.size() ; ++ai)
	        aa[ai] = atoms[ai];
	    PyObject *sca_item = python_tuple(s->py_instance(true), py_cid, atoms_array);
	    PyTuple_SET_ITEM(sca_tuple, i++, sca_item);
	}
	return sca_tuple;
    } catch (...) {
        molc_error();
    }
    return NULL;
}

extern "C" EXPORT void atom_neighbors(void *atoms, size_t n, pyobject_t *neighbors)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Atom::Neighbors &nb = a[i]->neighbors();
            for (size_t j = 0; j != nb.size(); ++j)
                *neighbors++ = nb[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_py_obj_bonds(void *atoms, size_t n, pyobject_t *bonds)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            PyObject* b_list = PyList_New(a[i]->bonds().size());
            if (b_list == nullptr)
                throw std::bad_alloc();
            bonds[i] = b_list;
            int b_i = 0;
            for (auto b: a[i]->bonds()) {
                PyList_SET_ITEM(b_list, b_i++, b->py_instance(true));
            }
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_py_obj_neighbors(void *atoms, size_t n, pyobject_t *neighbors)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            PyObject* nb_list = PyList_New(a[i]->neighbors().size());
            if (nb_list == nullptr)
                throw std::bad_alloc();
            neighbors[i] = nb_list;
            int nb_i = 0;
            for (auto nb: a[i]->neighbors()) {
                PyList_SET_ITEM(nb_list, nb_i++, nb->py_instance(true));
            }
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_chain_id(void *atoms, size_t n, pyobject_t *cids)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(a[i]->residue()->chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_color(void *atoms, size_t n, uint8_t *rgba)
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

extern "C" EXPORT void set_atom_color(void *atoms, size_t n, uint8_t *rgba)
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

extern "C" EXPORT bool atom_connects_to(pyobject_t atom1, pyobject_t atom2)
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

extern "C" EXPORT void atom_coord(void *atoms, size_t n, float64_t *xyz)
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

extern "C" EXPORT void set_atom_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    std::unordered_map<Structure*, std::vector<Atom*>> s_map;
    std::unordered_set<Structure*> ribbon_changed;
    Coord coord;
    try {
        for (size_t i = 0; i != n; ++i) {
            s_map[(*a)->structure()].push_back(*a);
            coord.set_xyz(*xyz, *(xyz+1), *(xyz+2));
            if ((*a)->in_ribbon())
                ribbon_changed.insert((*a)->structure());
            (*a++)->set_coord(coord, /* track_change */ false);
            xyz += 3;
        }
        for (auto &s_atoms: s_map) {
            auto s = s_atoms.first;
            auto &atoms = s_atoms.second;
            auto ct = s->change_tracker();
            ct->add_modified_set(s, atoms, ChangeTracker::REASON_COORD);
            ct->add_modified(s, s->active_coord_set(), ChangeTracker::REASON_COORDSET);
            s->set_gc_shape();
        }
        for (auto &s: ribbon_changed)
            s->set_gc_ribbon();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_coord_index(void *atoms, size_t n, uint32_t *index)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, unsigned int, unsigned int>(a, n, &Atom::coord_index, index);
}

extern "C" EXPORT void atom_get_coord_crdset(void *atom, int cs_id, float64_t *xyz)
{
    Atom *a = static_cast<Atom *>(atom);
    try {
        auto cs = a->structure()->find_coord_set(cs_id);
        if (cs == nullptr) {
            std::stringstream err_msg;
            err_msg << "Structure has no coordset with ID " << cs_id;
            PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
        } else {
            auto& crd = a->coord(cs);
            *xyz++ = crd[0];
            *xyz++ = crd[1];
            *xyz++ = crd[2];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_get_coord_altloc(void *atom, char altloc, float64_t *xyz)
{
    Atom *a = static_cast<Atom *>(atom);
    try {
        if (a->has_alt_loc(altloc)) {
            auto& crd = a->coord(altloc);
            *xyz++ = crd[0];
            *xyz++ = crd[1];
            *xyz++ = crd[2];
        } else {
            std::stringstream err_msg;
            err_msg << "Atom " << a->str() << " has no altloc " << altloc;
            PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_delete(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        std::map<Structure *, std::vector<Atom *>> matoms;
        for (size_t i = 0; i != n; ++i)
            matoms[a[i]->structure()].push_back(a[i]);

        for (auto ma: matoms)
            ma.first->delete_atoms(ma.second);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_display(void *atoms, size_t n, npy_bool *disp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::display, disp);
}

extern "C" EXPORT void set_atom_display(void *atoms, size_t n, npy_bool *disp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, bool, npy_bool>(a, n, &Atom::set_display, disp);
}

extern "C" EXPORT void atom_hide(void *atoms, size_t n, int32_t *hide)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, int, int>(a, n, &Atom::hide, hide);
}

extern "C" EXPORT void set_atom_hide(void *atoms, size_t n, int32_t *hide)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, int, int>(a, n, &Atom::set_hide, hide);
}

extern "C" EXPORT void set_atom_hide_bits(void *atoms, size_t n, int32_t bit_mask)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_1arg<Atom, int, int>(a, n, &Atom::set_hide_bits, bit_mask);
}

extern "C" EXPORT void clear_atom_hide_bits(void *atoms, size_t n, int32_t bit_mask)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_1arg<Atom, int, int>(a, n, &Atom::clear_hide_bits, bit_mask);
}

extern "C" EXPORT void atom_visible(void *atoms, size_t n, npy_bool *visible)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::visible, visible);
}

extern "C" EXPORT void atom_alt_loc(void *atoms, size_t n, pyobject_t *alt_locs)
{
    Atom **a = static_cast<Atom **>(atoms);
    char buffer[2];
    buffer[1] = '\0';
    try {
        for (size_t i = 0; i != n; ++i) {
            buffer[0] = a[i]->alt_loc();
            alt_locs[i] = unicode_from_string(buffer);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_alt_loc(void *atoms, size_t n, pyobject_t *alt_locs)
{
    Atom **a = static_cast<Atom **>(atoms);
    // can't use error_wrap_array_set because set_alt_loc takes multiple args
    try {
        for (size_t i = 0; i < n; ++i)
            a[i]->set_alt_loc(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(alt_locs[i]))[0]);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_set_alt_loc(void *atom, char alt_loc, bool create, bool from_residue)
{
    // this one used in the Atom class so that the additional args can be supplied,
    // whereas set_atom_alt_loc is used for the setter half of alt_loc properties
    Atom *a = static_cast<Atom *>(atom);
    error_wrap(a, &Atom::set_alt_loc, alt_loc, create, from_residue);
}

extern "C" EXPORT void atom_has_alt_loc(void *atoms, size_t n, char alt_loc, npy_bool *has)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i < n; ++i)
            has[i] = a[i]->has_alt_loc(alt_loc);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *atom_alt_locs(void *atom)
{
    Atom *a = static_cast<Atom *>(atom);
    PyObject *py_alt_locs = nullptr;
    try {
        const auto& alt_locs = a->alt_locs();
        py_alt_locs = PyList_New(alt_locs.size());
        if (py_alt_locs == nullptr)
            return nullptr;
        size_t p = 0;
        for (auto alt_loc: alt_locs) {
            PyObject* py_alt_loc = PyUnicode_FromFormat("%c", (int)alt_loc);
            if (py_alt_loc == nullptr) {
                Py_DECREF(py_alt_locs);
                return nullptr;
            }
            PyList_SET_ITEM(py_alt_locs, p++, py_alt_loc);
        }
        return py_alt_locs;
    } catch (...) {
        Py_XDECREF(py_alt_locs);
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void atom_set_coord(void *atom, void *xyz, int cs_id)
{
    Atom *a = static_cast<Atom *>(atom);
    try {
        auto cs = a->structure()->find_coord_set(cs_id);
        if (cs == nullptr)
            throw std::logic_error("No such coordset ID");
        a->set_coord(Point((double*)xyz), cs);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_draw_mode(void *atoms, size_t n, uint8_t *modes)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            modes[i] = static_cast<uint8_t>(a[i]->draw_mode());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_draw_mode(void *atoms, size_t n, uint8_t *modes)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            a[i]->set_draw_mode(static_cast<Atom::DrawMode>(modes[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_element(void *atoms, size_t n, pyobject_t *resp)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i < n; ++i)
            resp[i] = (pyobject_t*)(&(a[i]->element()));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_element_name(void *atoms, size_t n, pyobject_t *names)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(a[i]->element().name());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_element_number(void *atoms, size_t n, uint8_t *nums)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            nums[i] = a[i]->element().number();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *atom_idatm_info_map()
{
    PyObject* mapping = PyDict_New();
    if (mapping == nullptr)
        molc_error();
    else {
        try {
            // map values are named tuples, set that up...
            PyStructSequence_Field fields[] = {
                { (char*)"geometry", (char*)"arrangement of bonds; 0: no bonds; 1: one bond;"
                    " 2: linear; 3: planar; 4: tetrahedral" },
                { (char*)"substituents", (char*)"number of bond partners" },
                { (char*)"description", (char*)"text description of atom type" },
                { nullptr, nullptr }
            };
            static PyStructSequence_Desc type_desc;
            type_desc.name = (char*)"IdatmInfo";
            type_desc.doc = (char*)"Information about an IDATM type";
            type_desc.fields = fields;
            type_desc.n_in_sequence = 3;
            // Need to disable and enable Python garbage collection around
            // PyStructSequence_NewType, because Py_TPFLAGS_HEAPTYPE isn't
            // set until after the call returns, and then it's too late
            PyObject *mod = PyImport_ImportModule("gc");
            PyObject *mod_dict = mod ? PyModule_GetDict(mod) : NULL;
            PyObject *disable = mod_dict ? PyDict_GetItemString(mod_dict, "disable") : NULL;
            PyObject *enable = mod_dict ? PyDict_GetItemString(mod_dict, "enable") : NULL;
            if (disable == NULL || enable == NULL) {
                disable = enable = NULL;
                std::cerr << "Can't control garbage collection\n";
            }
            if (disable) Py_XDECREF(PyObject_CallNoArgs(disable));
            auto type_obj = PyStructSequence_NewType(&type_desc);
            // As per https://bugs.python.org/issue20066 and https://bugs.python.org/issue15729,
            // the type object isn't completely initialized, so...
            type_obj->tp_flags |= Py_TPFLAGS_HEAPTYPE;
            PyObject *ht_name = PyUnicode_FromString(type_desc.name);
            reinterpret_cast<PyHeapTypeObject*>(type_obj)->ht_name = ht_name;
            reinterpret_cast<PyHeapTypeObject*>(type_obj)->ht_qualname = ht_name;
            for (auto type_info: Atom::get_idatm_info_map()) {
                PyObject* key = PyUnicode_FromString(type_info.first.c_str());
                PyObject* val = PyStructSequence_New(type_obj);
                auto info = type_info.second;
                PyStructSequence_SET_ITEM(val, 0, PyLong_FromLong(info.geometry));
                PyStructSequence_SET_ITEM(val, 1, PyLong_FromLong(info.substituents));
                PyStructSequence_SET_ITEM(val, 2, PyUnicode_FromString(info.description.c_str()));
                PyDict_SetItem(mapping, key, val);
                Py_DECREF(key);
                Py_DECREF(val);
            }
            if (enable) Py_XDECREF(PyObject_CallNoArgs(enable));
        } catch (...) {
            molc_error();
        }
    }
    return mapping;
}

extern "C" EXPORT void atom_idatm_type(void *atoms, size_t n, pyobject_t *idatm_types)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            idatm_types[i] = unicode_from_string(a[i]->idatm_type());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_idatm_type(void *atoms, size_t n, pyobject_t *idatm_types)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            a[i]->set_idatm_type(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(idatm_types[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_in_chain(void *atoms, size_t n, npy_bool *in_chain)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            in_chain[i] = (a[i]->residue()->chain() != NULL);
    } catch (...) {
        molc_error();
    }
}


extern "C" EXPORT void atom_is_backbone(void *atoms, size_t n, int extent, npy_bool *bb)
{
    Atom **a = static_cast<Atom **>(atoms);
    BackboneExtent bbe = static_cast<BackboneExtent>(extent);
    try {
        for (size_t i = 0; i < n; ++i)
            bb[i] = a[i]->is_backbone(bbe);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_is_ribose(void *atoms, size_t n, npy_bool *is_ribose)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            is_ribose[i] = a[i]->is_ribose();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_is_side_chain(void *atoms, size_t n, npy_bool *is_side_chain)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            is_side_chain[i] = a[i]->is_side_chain(false);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_is_side_connector(void *atoms, size_t n, npy_bool *is_side_connector)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            is_side_connector[i] = a[i]->is_side_connector();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_is_side_only(void *atoms, size_t n, npy_bool *is_side_only)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            is_side_only[i] = a[i]->is_side_chain(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_serial_number(void *atoms, size_t n, int32_t *index)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::serial_number, index);
}

extern "C" EXPORT void set_atom_serial_number(void *atoms, size_t n, int32_t *serial)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i < n; ++i)
            a[i]->set_serial_number(serial[i]);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_structure(void *atoms, size_t n, pyobject_t *molp)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i < n; ++i)
            molp[i] = a[i]->structure()->py_instance(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_name(void *atoms, size_t n, pyobject_t *names)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(a[i]->name());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_name(void *atoms, size_t n, pyobject_t *names)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            a[i]->set_name(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(names[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_num_alt_locs(void *atoms, size_t n, size_t *nlocs)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            nlocs[i] = a[i]->alt_locs().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_num_bonds(void *atoms, size_t n, size_t *nbonds)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            nbonds[i] = a[i]->bonds().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_num_explicit_bonds(void *atoms, size_t n, size_t *nbonds)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            nbonds[i] = a[i]->num_explicit_bonds();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT size_t atom_num_residues(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        std::set<Residue *> res;
        for (size_t i = 0; i != n; ++i)
	    res.insert(a[i]->residue());
	return res.size();
    } catch (...) {
        molc_error();
	return 0;
    }
}

extern "C" EXPORT void atom_radius(void *atoms, size_t n, float32_t *radii)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::radius, radii);
}

extern "C" EXPORT void set_atom_radius(void *atoms, size_t n, float32_t *radii)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set(a, n, &Atom::set_radius, radii);
}

extern "C" EXPORT void atom_default_radius(void *atoms, size_t n, float32_t *radii)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::default_radius, radii);
}

extern "C" EXPORT void atom_maximum_bond_radius(void *atoms, size_t n, float32_t default_radius, float32_t *radii)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
          radii[i] = a[i]->maximum_bond_radius(default_radius);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_residue(void *atoms, size_t n, pyobject_t *resp)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get(a, n, &Atom::residue, resp);
}

extern "C" EXPORT PyObject *atom_residue_sums(void *atoms, size_t n, double *atom_values)
{
    Atom **a = static_cast<Atom **>(atoms);
    std::map<Residue *, double> rmap;
    PyObject *result = NULL;
    try {
      for (size_t i = 0; i < n; ++i) {
        Residue *r = a[i]->residue();
        double v = atom_values[i];
        auto ri = rmap.find(r);
        if (ri == rmap.end())
          rmap[r] = v;
        else
          rmap[r] += v;
      }
      void **p;
      double *v;
      PyObject *rp = python_voidp_array(rmap.size(), &p);
      PyObject *rv = python_double_array(rmap.size(), &v);
      Residue **res = (Residue **)p;
      for (auto ri = rmap.begin() ; ri != rmap.end() ; ++ri) {
        *res++ = ri->first;
        *v++ = ri->second;
      }
      result = python_tuple(rp, rv);
    } catch (...) {
        molc_error();
    }
    return result;
}

extern "C" EXPORT void atom_ribbon_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Coord *c = a[i]->ribbon_coord();
	    if (c == NULL) {
	      PyErr_SetString(PyExc_ValueError, "Atom does not hae ribbon coordinate");
	      break;
	    }
            *xyz++ = (*c)[0];
            *xyz++ = (*c)[1];
            *xyz++ = (*c)[2];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_ribbon_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
      Coord coord;
      for (size_t i = 0; i != n; ++i, xyz += 3) {
	  float64_t x = xyz[0], y = xyz[1], z = xyz[2];
	  coord.set_xyz(x, y, z);
	  a[i]->set_ribbon_coord(coord);
      }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_effective_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto c = a[i]->effective_coord();
            *xyz++ = c[0];
            *xyz++ = c[1];
            *xyz++ = c[2];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_effective_scene_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto c = a[i]->effective_scene_coord();
            *xyz++ = c[0];
            *xyz++ = c[1];
            *xyz++ = c[2];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *atom_rings(void *atom, bool cross_residue, int all_size_threshold)
{
    Atom *a = static_cast<Atom *>(atom);
    try {
        auto& rings = a->rings(cross_residue, all_size_threshold);
        const Ring **ra;
        PyObject *r_array = python_voidp_array(rings.size(), (void***)&ra);
        size_t i = 0;
        for (auto r: rings)
            ra[i++] = r;
        return r_array;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void atom_scene_coord(void *atoms, size_t n, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto c = a[i]->scene_coord();
            *xyz++ = c[0];
            *xyz++ = c[1];
            *xyz++ = c[2];
        }
    } catch (...) {
        molc_error();
    }
}


// Apply per-structure transform to atom coordinates.
extern "C" EXPORT void atom_scene_coords(void *atoms, size_t n, void *mols, size_t m, float64_t *mtf, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    Structure **ma = static_cast<Structure **>(mols);

    try {
        std::map<Structure *, double *> tf;
        for (size_t i = 0; i != m; ++i)
            tf[ma[i]] = mtf + 12*i;

        for (size_t i = 0; i != n; ++i) {
            Structure *s = a[i]->structure();
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

// Set atom coordinates after applying a per-structure transform.
extern "C" EXPORT void atom_set_scene_coords(void *atoms, size_t n, void *mols, size_t m, float64_t *mtf, float64_t *xyz)
{
    Atom **a = static_cast<Atom **>(atoms);
    Structure **ma = static_cast<Structure **>(mols);

    try {
        std::map<Structure *, double *> tf;
        for (size_t i = 0; i != m; ++i)
            tf[ma[i]] = mtf + 12*i;

        Point p;
        for (size_t i = 0; i != n; ++i, xyz += 3) {
            Structure *s = a[i]->structure();
            double *t = tf[s];
            double x = xyz[0], y = xyz[1], z = xyz[2];
            p.set_xyz(t[0]*x + t[1]*y + t[2]*z + t[3],
                      t[4]*x + t[5]*y + t[6]*z + t[7],
                      t[8]*x + t[9]*y + t[10]*z + t[11]);
            a[i]->set_coord(p);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_selected(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_get<Atom, bool, npy_bool>(a, n, &Atom::selected, sel);
}

extern "C" EXPORT void atom_structure_category(void *atoms, size_t n, pyobject_t *names)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        const char *cat_name;
        for (size_t i = 0; i != n; ++i) {
            auto cat = a[i]->structure_category();
            if (cat == Atom::StructCat::Main)
                cat_name = "main";
            else if (cat == Atom::StructCat::Solvent)
                cat_name = "solvent";
            else if (cat == Atom::StructCat::Ligand)
                cat_name = "ligand";
            else if (cat == Atom::StructCat::Ions)
                cat_name = "ions";
            else if (cat == Atom::StructCat::Unassigned)
                cat_name = "other";
            else
                throw std::range_error("Unknown structure category");
            names[i] = unicode_from_string(cat_name);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_atom_selected(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    error_wrap_array_set<Atom, bool, npy_bool>(a, n, &Atom::set_selected, sel);
}

extern "C" EXPORT size_t atom_num_selected(void *atoms, size_t n)
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

extern "C" EXPORT void atom_has_selected_bond(void *atoms, size_t n, npy_bool *sel)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Atom::Bonds &b = a[i]->bonds();
        sel[i] = false;
            for (size_t j = 0; j != b.size(); ++j)
          if (b[j]->selected())
        {
          sel[i] = true;
          break;
        }
        }
    } catch (...) {
        molc_error();
    }
}

template <class C, typename T>
void affine_transform(const C& coord, T* tf, C& result)
{
    for (size_t i=0; i<3; ++i)
    {
        result[i] = tf[4*i] * coord[0] + tf[4*i+1] * coord[1] + tf[4*i+2]*coord[2] + tf[4*i+3];
    }
}

template <typename T>
void transform_u_aniso(const std::vector<float>* aup, T* tf, std::vector<float>& result)
{
    // Need to apply only rotation component of transform, as (rot).U.(rot)T
    // aniso_u6 is stored as u11, u12, u13, u22, u23, u33
    const auto& au = *aup;
    std::vector<float> full = {au[0],au[1],au[2],au[1],au[3],au[4],au[2],au[4],au[5]};
    std::vector<float> ir(9);
    for (size_t i=0; i<3; ++i){
        for (size_t j=0; j<3; ++j){
            ir[3*i+j] = tf[4*i] * full[j] + tf[4*i+1] * full[3+j] + tf[4*i+2] * full[6+j];
        }
    }
    for (size_t i=0; i<3; ++i) {
        for (size_t j=0; j<3; ++j){
            result[3*i+j] = ir[3*i] * tf[4*j] + ir[3*i+1] * tf[4*j+1] + ir[3*i+2] * tf[4*j+2];
        }
    }
}

void transform_atom(Atom* atom, double* tf)
{
    Coord transformed;
    affine_transform<Coord, double>(atom->coord(), tf, transformed);
    atom->set_coord(transformed);
    if (atom->has_aniso_u())
    {
        std::vector<float> ua(9);
        transform_u_aniso<double>(atom->aniso_u(), tf, ua);
        atom->set_aniso_u(ua[0],ua[1],ua[2],ua[4],ua[5],ua[8]);
    }
}

extern "C" EXPORT void atom_transform(void* atom, size_t n, double* tf)
{
    try {
        auto a = static_cast<Atom**>(atom);
        char current_altloc;
        for (size_t i=0; i<n; ++i)
        {
            auto atom = *(a++);
            auto altlocs = atom->alt_locs();
            if (altlocs.size())
            {
                current_altloc = atom->alt_loc();
                for (const auto& altloc: altlocs)
                {
                    atom->set_alt_loc(altloc);
                    transform_atom(atom, tf);
                }
                atom->set_alt_loc(current_altloc);
            } else {
                transform_atom(atom, tf);
            }
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_update_ribbon_backbone_atom_visibility(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        // Hide control point atoms as appropriate
        for (size_t i = 0; i != n; ++i) {
            Atom *atom = a[i];
            if (!atom->is_backbone(BBE_RIBBON))
                continue;
            bool hide;
            if (!atom->residue()->ribbon_display() || !atom->residue()->ribbon_hide_backbone())
                hide = false;
            else {
                hide = true;
                for (auto neighbor : atom->neighbors())
                    if (neighbor->visible() && !neighbor->is_backbone(BBE_RIBBON)) {
                        hide = false;
                        break;
                    }
            }
            if (hide)
                atom->set_hide_bits(Atom::HIDE_RIBBON);
            else
                atom->clear_hide_bits(Atom::HIDE_RIBBON);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void atom_use_default_radius(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        for (size_t i = 0; i != n; ++i)
            a[i]->use_default_radius();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *atom_intra_bonds(void *atoms, size_t n)
{
    Atom **a = static_cast<Atom **>(atoms);
    std::set<Atom *> aset;
    std::set<Bond *> bset;
    try {
        for (size_t i = 0; i < n; ++i)
          aset.insert(a[i]);
        for (size_t i = 0; i < n; ++i) {
          const Atom::Bonds &abonds = a[i]->bonds();
            for (auto b = abonds.begin() ; b != abonds.end() ; ++b) {
              Bond *bond = *b;
              const Bond::Atoms &batoms = bond->atoms();
              if (aset.find(batoms[0]) != aset.end() &&
                  aset.find(batoms[1]) != aset.end() &&
                  bset.find(bond) == bset.end())
                bset.insert(bond);
            }
        }
        void **bptr;
        PyObject *ba = python_voidp_array(bset.size(), &bptr);
        int i = 0;
        for (auto b = bset.begin() ; b != bset.end() ; ++b)
          bptr[i++] = *b;
        return ba;
    } catch (...) {
        molc_error();
        return 0;
    }
}

// -------------------------------------------------------------------------
// bond functions
//
extern "C" EXPORT void bond_atoms(void *bonds, size_t n, pyobject_t *atoms)
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

extern "C" EXPORT void bond_color(void *bonds, size_t n, uint8_t *rgba)
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

extern "C" EXPORT void set_bond_color(void *bonds, size_t n, uint8_t *rgba)
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

extern "C" EXPORT PyObject *bond_half_colors(void *bonds, size_t n)
{
    Bond **b = static_cast<Bond **>(bonds);
    uint8_t *rgba1;
    PyObject *colors = python_uint8_array(2*n, 4, &rgba1);
    uint8_t *rgba2 = rgba1 + 4*n;
    try {
        const Rgba *c1, *c2;
        for (size_t i = 0; i < n; ++i) {
          Bond *bond = b[i];
          if (bond->halfbond()) {
              c1 = &bond->atoms()[0]->color();
              c2 = &bond->atoms()[1]->color();
          } else {
              c1 = c2 = &bond->color();
          }
          *rgba1++ = c1->r; *rgba1++ = c1->g; *rgba1++ = c1->b; *rgba1++ = c1->a;
          *rgba2++ = c2->r; *rgba2++ = c2->g; *rgba2++ = c2->b; *rgba2++ = c2->a;
        }
    } catch (...) {
        molc_error();
    }
    return colors;
}

extern "C" EXPORT void bond_display(void *bonds, size_t n, npy_bool *disp)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, bool, npy_bool>(b, n, &Bond::display, disp);
}

extern "C" EXPORT void set_bond_display(void *bonds, size_t n, npy_bool *disp)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, bool, npy_bool>(b, n, &Bond::set_display, disp);
}

extern "C" EXPORT void bond_hide(void *bonds, size_t n, int32_t *hide)
{
    Bond **a = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, int, int>(a, n, &Bond::hide, hide);
}

extern "C" EXPORT void set_bond_hide(void *bonds, size_t n, int32_t *hide)
{
    Bond **a = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, int, int>(a, n, &Bond::set_hide, hide);
}

extern "C" EXPORT void set_bond_hide_bits(void *bonds, size_t n, int32_t bit_mask)
{
    Bond **a = static_cast<Bond **>(bonds);
    error_wrap_array_1arg<Bond, int, int>(a, n, &Bond::set_hide_bits, bit_mask);
}

extern "C" EXPORT void clear_bond_hide_bits(void *bonds, size_t n, int32_t bit_mask)
{
    Bond **a = static_cast<Bond **>(bonds);
    error_wrap_array_1arg<Bond, int, int>(a, n, &Bond::clear_hide_bits, bit_mask);
}

extern "C" EXPORT void bond_visible(void *bonds, size_t n, uint8_t *visible)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i)
            visible[i] = static_cast<uint8_t>(b[i]->visible());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void bond_halfbond(void *bonds, size_t n, npy_bool *halfb)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, bool, npy_bool>(b, n, &Bond::halfbond, halfb);
}

extern "C" EXPORT void bond_in_cycle(void *bonds, size_t n, npy_bool *cycle)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, bool, npy_bool>(b, n, &Bond::in_cycle, cycle);
}

extern "C" EXPORT void set_bond_halfbond(void *bonds, size_t n, npy_bool *halfb)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, bool, npy_bool>(b, n, &Bond::set_halfbond, halfb);
}

extern "C" EXPORT void bond_length(void *bonds, size_t n, float32_t *lengths)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i)
          lengths[i] = b[i]->length();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void bond_radius(void *bonds, size_t n, float32_t *radii)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, float>(b, n, &Bond::radius, radii);
}

extern "C" EXPORT PyObject *bond_rings(void *bond, bool cross_residue, int all_size_threshold)
{
    Bond *b = static_cast<Bond *>(bond);
    try {
        auto& rings = b->rings(cross_residue, all_size_threshold);
        const Ring **ra;
        PyObject *r_array = python_voidp_array(rings.size(), (void***)&ra);
        size_t i = 0;
        for (auto r: rings)
            ra[i++] = r;
        return r_array;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void bond_selected(void *bonds, size_t n, npy_bool *sel)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, bool, npy_bool>(b, n, &Bond::selected, sel);
}

extern "C" EXPORT void set_bond_selected(void *bonds, size_t n, npy_bool *sel)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, bool, npy_bool>(b, n, &Bond::set_selected, sel);
}

extern "C" EXPORT void bond_ends_selected(void *bonds, size_t n, npy_bool *sel)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Bond::Atoms &a = b[i]->atoms();
        sel[i] = (a[0]->selected() && a[1]->selected());
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void bond_shown(void *bonds, size_t n, npy_bool *shown)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_get<Bond, bool, npy_bool>(b, n, &Bond::shown, shown);
}

extern "C" EXPORT int bonds_num_shown(void *bonds, size_t n)
{
    Bond **b = static_cast<Bond **>(bonds);
    int count = 0;
    try {
        for (size_t i = 0; i < n; ++i)
          if (b[i]->shown())
            count += 1;
    } catch (...) {
        molc_error();
    }
    return count;
}

extern "C" EXPORT void* bond_side_atoms(void *bond, void *side_atom)
{
    Bond *b = static_cast<Bond*>(bond);
    Atom *sa = static_cast<Atom*>(side_atom);
    try {
        auto side_atoms = b->side_atoms(sa);
        const Atom **sas;
        PyObject *sa_array = python_voidp_array(side_atoms.size(), (void***)&sas);
        size_t i = 0;
        for (auto s: side_atoms)
            sas[i++] = s;
        return sa_array;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *bond_smaller_side(void *bond)
{
    Bond *b = static_cast<Bond *>(bond);
    try {
        return b->smaller_side()->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *bond_polymeric_start_atom(void *bond)
{
    Bond *b = static_cast<Bond *>(bond);
    try {
        Atom* a = b->polymeric_start_atom();
        if (a == nullptr) {
            Py_RETURN_NONE;
        }
        return a->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT int bonds_num_selected(void *bonds, size_t n)
{
    Bond **b = static_cast<Bond **>(bonds);
    int count = 0;
    try {
        for (size_t i = 0; i < n; ++i)
          if (b[i]->selected())
            count += 1;
    } catch (...) {
        molc_error();
    }
    return count;
}

extern "C" EXPORT void set_bond_radius(void *bonds, size_t n, float32_t *radii)
{
    Bond **b = static_cast<Bond **>(bonds);
    error_wrap_array_set<Bond, float>(b, n, &Bond::set_radius, radii);
}

extern "C" EXPORT void bond_structure(void *bonds, size_t n, pyobject_t *molp)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i < n; ++i)
          molp[i] = b[i]->structure()->py_instance(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *bond_other_atom(void *bond, void *atom)
{
    Bond *b = static_cast<Bond *>(bond);
    Atom *a = static_cast<Atom *>(atom), *oa;
    try {
      oa = b->other_atom(a);
    } catch (...) {
      molc_error();
    }
    return oa;
}

extern "C" EXPORT void bond_delete(void *bonds, size_t n)
{
    auto db = DestructionBatcher(bonds); // don't use structure since delete_bond internally batches on that
    Bond **b = static_cast<Bond **>(bonds);
    try {
        for (size_t i = 0; i != n; ++i)
            b[i]->structure()->delete_bond(b[i]);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void bond_halfbond_cylinder_placements(void *bonds, size_t n, float32_t *m44)
{
    Bond **b = static_cast<Bond **>(bonds);
    try {
      float32_t *m44b = m44 + 16*n;
      for (size_t i = 0; i != n; ++i) {
	Bond *bd = b[i];
	Atom *a0 = bd->atoms()[0], *a1 = bd->atoms()[1];
	const Coord &xyz0 = a0->coord(), &xyz1 = a1->coord();
	float r = bd->radius();

	float x0 = xyz0[0], y0 = xyz0[1], z0 = xyz0[2], x1 = xyz1[0], y1 = xyz1[1], z1 = xyz1[2];
	float vx = x1-x0, vy = y1-y0, vz = z1-z0;
	float h = sqrtf(vx*vx + vy*vy + vz*vz);
	if (h == 0)
	  { vx = vy = 0 ; vz = 1; }
	else
	  { vx /= h; vy /= h; vz /= h; }

	float sx = r, sy = r, sz = h;	// Scale factors

	// Avoid degenerate vz = -1 case.
	if (vz < 0)
	  { vx = -vx; vy = -vy; vz = -vz; sx = -r; sz = -h; }
    
	float c1 = 1.0/(1+vz);
	float vxx = c1*vx*vx, vyy = c1*vy*vy, vxy = c1*vx*vy;
      
	*m44++ = *m44b++ = sx*(vyy + vz);
	*m44++ = *m44b++ = -sx*vxy;
	*m44++ = *m44b++ = -sx*vx;
	*m44++ = *m44b++ = 0;

	*m44++ = *m44b++ = -sy*vxy;
	*m44++ = *m44b++ = sy*(vxx + vz);
	*m44++ = *m44b++ = -sy*vy;
	*m44++ = *m44b++ = 0;

	*m44++ = *m44b++ = sz*vx;
	*m44++ = *m44b++ = sz*vy;
	*m44++ = *m44b++ = sz*vz;
	*m44++ = *m44b++ = 0;

	*m44++ = .75*x0 + .25*x1;
	*m44++ = .75*y0 + .25*y1;
	*m44++ = .75*z0 + .25*z1;
	*m44++ = 1;

	*m44b++ = .25*x0 + .75*x1;
	*m44b++ = .25*y0 + .75*y1;
	*m44b++ = .25*z0 + .75*z1;
	*m44b++ = 1;
      }
    } catch (...) {
        molc_error();
    }
}

// -------------------------------------------------------------------------
// pseudobond functions
//
extern "C" EXPORT void pseudobond_atoms(void *pbonds, size_t n, pyobject_t *atoms)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Pseudobond::Atoms &a = b[i]->atoms();
            *atoms++ = a[0];
            *atoms++ = a[1];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_color(void *pbonds, size_t n, uint8_t *rgba)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
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

extern "C" EXPORT void set_pseudobond_color(void *pbonds, size_t n, uint8_t *rgba)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
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

extern "C" EXPORT void pseudobond_delete(void *pbonds, size_t n)
{
    Pseudobond **pb = static_cast<Pseudobond **>(pbonds);
    try {
        std::map<Proxy_PBGroup *, std::vector<Pseudobond *>> g_pbs;
        for (size_t i = 0; i != n; ++i)
            g_pbs[pb[i]->group()->proxy()].push_back(pb[i]);
        for (auto grp_pbs: g_pbs)
            grp_pbs.first->delete_pseudobonds(grp_pbs.second);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group(void *pbonds, size_t n, pyobject_t *grps)
{
    Pseudobond **pb = static_cast<Pseudobond **>(pbonds);
    try {
        for (size_t i = 0; i != n; ++i) {
            Proxy_PBGroup* grp = pb[i]->group()->proxy();
            *grps++ = grp;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *pseudobond_half_colors(void *pbonds, size_t n)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    uint8_t *rgba1;
    PyObject *colors = python_uint8_array(2*n, 4, &rgba1);
    uint8_t *rgba2 = rgba1 + 4*n;
    try {
        const Rgba *c1, *c2;
        for (size_t i = 0; i < n; ++i) {
          Pseudobond *bond = b[i];
          if (bond->halfbond()) {
              c1 = &bond->atoms()[0]->color();
              c2 = &bond->atoms()[1]->color();
          } else {
              c1 = c2 = &bond->color();
          }
          *rgba1++ = c1->r; *rgba1++ = c1->g; *rgba1++ = c1->b; *rgba1++ = c1->a;
          *rgba2++ = c2->r; *rgba2++ = c2->g; *rgba2++ = c2->b; *rgba2++ = c2->a;
        }
    } catch (...) {
        molc_error();
    }
    return colors;
}

extern "C" EXPORT void pseudobond_get_session_id(void *ptrs, size_t n, int32_t *ses_ids)
{
    Pseudobond **pbonds = static_cast<Pseudobond **>(ptrs);
    try {
        for (size_t i = 0; i < n; ++i) {
            Pseudobond* pb = pbonds[i];
            auto sess_save_pbs = pb->group()->manager()->session_save_pbs;
            if (sess_save_pbs == nullptr)
                throw std::runtime_error("pseudobond session IDs only available during session save");
            ses_ids[i] = static_cast<int32_t>((*sess_save_pbs)[pb]);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *pseudobond_group_resolve_session_id(void *ptr, int ses_id)
{
    Proxy_PBGroup *grp = static_cast<Proxy_PBGroup *>(ptr);
    try {
        return (void*)((*grp->manager()->session_restore_pbs)[ses_id]);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void pseudobond_display(void *pbonds, size_t n, npy_bool *disp)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_get<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::display, disp);
}

extern "C" EXPORT void set_pseudobond_display(void *pbonds, size_t n, npy_bool *disp)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_set<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::set_display, disp);
}

extern "C" EXPORT void pseudobond_halfbond(void *pbonds, size_t n, npy_bool *halfb)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_get<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::halfbond, halfb);
}

extern "C" EXPORT void set_pseudobond_halfbond(void *pbonds, size_t n, npy_bool *halfb)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_set<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::set_halfbond, halfb);
}

extern "C" EXPORT void pseudobond_radius(void *pbonds, size_t n, float32_t *radii)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_get<Pseudobond, float>(b, n, &Pseudobond::radius, radii);
}

extern "C" EXPORT void pseudobond_selected(void *pbonds, size_t n, npy_bool *sel)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_get<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::selected, sel);
}

extern "C" EXPORT void set_pseudobond_selected(void *pbonds, size_t n, npy_bool *sel)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_set<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::set_selected, sel);
}

extern "C" EXPORT int pseudobonds_num_selected(void *bonds, size_t n)
{
    Bond **b = static_cast<Bond **>(bonds);
    int count = 0;
    try {
        for (size_t i = 0; i < n; ++i)
          if (b[i]->selected())
            count += 1;
    } catch (...) {
        molc_error();
    }
    return count;
}

extern "C" EXPORT void pseudobond_shown(void *pbonds, size_t n, npy_bool *shown)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_get<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::shown, shown);
}

extern "C" EXPORT void pseudobond_shown_when_atoms_hidden(void *pbonds, size_t n, npy_bool *shown)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_get<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::shown_when_atoms_hidden, shown);
}

extern "C" EXPORT void set_pseudobond_shown_when_atoms_hidden(void *pbonds, size_t n, npy_bool *shown)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_set<Pseudobond, bool, npy_bool>(b, n, &Pseudobond::set_shown_when_atoms_hidden, shown);
}

extern "C" EXPORT void set_pseudobond_radius(void *pbonds, size_t n, float32_t *radii)
{
    Pseudobond **b = static_cast<Pseudobond **>(pbonds);
    error_wrap_array_set<Pseudobond, float>(b, n, &Pseudobond::set_radius, radii);
}

// -------------------------------------------------------------------------
// pseudobond group functions
//
extern "C" EXPORT void pseudobond_group_category(void *pbgroups, size_t n, void **categories)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (size_t i = 0 ; i < n ; ++i)
            categories[i] = unicode_from_string(pbg[i]->category());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_change_category(void* ptr, const char* cat)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(ptr);
    try {
        std::string scat = cat;
        pbg->change_category(scat);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_change_tracker(void *grps, size_t n, pyobject_t *trackers)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(grps);
    try {
        for (size_t i = 0; i < n; ++i) {
            trackers[i] = pbg[i]->manager()->change_tracker()->py_instance(true);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_group_type(void *pbgroups, size_t n, uint8_t *group_types)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (size_t i = 0; i != n; ++i)
            group_types[i] = static_cast<uint8_t>(g[i]->group_type());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_color(void *groups, size_t n, uint8_t *rgba)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(groups);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Rgba &c = g[i]->color();
            *rgba++ = c.r;
            *rgba++ = c.g;
            *rgba++ = c.b;
            *rgba++ = c.a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_pseudobond_group_color(void *groups, size_t n, uint8_t *rgba)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(groups);
    try {
        Rgba c;
        for (size_t i = 0; i != n; ++i) {
            c.r = *rgba++;
            c.g = *rgba++;
            c.b = *rgba++;
            c.a = *rgba++;
            g[i]->set_color(c);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_halfbond(void *groups, size_t n, npy_bool *halfb)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(groups);
    error_wrap_array_get<Proxy_PBGroup, bool, npy_bool>(g, n, &Proxy_PBGroup::halfbond, halfb);
}

extern "C" EXPORT void set_pseudobond_group_halfbond(void *groups, size_t n, npy_bool *halfb)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(groups);
    error_wrap_array_set<Proxy_PBGroup, bool, npy_bool>(g, n, &Proxy_PBGroup::set_halfbond, halfb);
}

extern "C" EXPORT void pseudobond_group_radius(void *groups, size_t n, float32_t *radii)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(groups);
    error_wrap_array_get<Proxy_PBGroup, float>(g, n, &Proxy_PBGroup::radius, radii);
}

extern "C" EXPORT void set_pseudobond_group_radius(void *groups, size_t n, float32_t *radii)
{
    Proxy_PBGroup **g = static_cast<Proxy_PBGroup **>(groups);
    error_wrap_array_set<Proxy_PBGroup, float>(g, n, &Proxy_PBGroup::set_radius, radii);
}

extern "C" EXPORT void pseudobond_group_graphics_change(void *pbgroups, size_t n, int *changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_get<Proxy_PBGroup, int, int>(pbg, n, &Proxy_PBGroup::get_graphics_changes, changed);
}

extern "C" EXPORT void set_pseudobond_group_graphics_change(void *pbgroups, size_t n, int *changed)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    error_wrap_array_set<Proxy_PBGroup, int, int>(pbg, n, &Proxy_PBGroup::set_graphics_changes, changed);
}

extern "C" EXPORT void pseudobond_group_clear(void *pbgroup)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        pbg->clear();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *pseudobond_group_new_pseudobond(void *pbgroup, void *atom1, void *atom2)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        Pseudobond *b = pbg->new_pseudobond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2));
        return b->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *pseudobond_group_new_pseudobonds(void *pbgroup, void *atoms1, void *atoms2, int natoms)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    Atom **a1 = static_cast<Atom **>(atoms1), **a2 = static_cast<Atom **>(atoms2);
    std::vector<Pseudobond *> pbonds;
    try {
      for (int i = 0 ; i < natoms ; ++i) {
        Pseudobond *b = pbg->new_pseudobond(a1[i], a2[i]);
        pbonds.push_back(b);
      }
      Pseudobond **pbp;
      PyObject *pb = python_voidp_array(pbonds.size(), (void***)&pbp);
      for (size_t i = 0 ; i < pbonds.size() ; ++i)
    pbp[i] = pbonds[i];
      return pb;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *pseudobond_group_new_pseudobond_csid(void *pbgroup,
    void *atom1, void *atom2, int cs_id)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        Pseudobond *b = pbg->new_pseudobond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2),
            pbg->structure()->find_coord_set(cs_id));
        return b->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void pseudobond_group_delete_pseudobond(void *pbgroup, void *pb)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        pbg->delete_pseudobond(static_cast<Pseudobond *>(pb));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_structure(void *pbgroups, size_t n, pyobject_t *resp)
{
    Proxy_PBGroup **pbgs = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (size_t i = 0; i < n; ++i) {
            auto sptr = pbgs[i]->structure();
            if (sptr == nullptr) {
                resp[i] = python_none();
            } else
                resp[i] = sptr->py_instance(true);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_num_pseudobonds(void *pbgroups, size_t n, size_t *num_pseudobonds)
{
    Proxy_PBGroup **pbg = static_cast<Proxy_PBGroup **>(pbgroups);
    try {
        for (size_t i = 0; i != n; ++i)
            *num_pseudobonds++ = pbg[i]->pseudobonds().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_group_pseudobonds(void *pbgroups, size_t n, pyobject_t *pseudobonds)
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

extern "C" EXPORT size_t pseudobond_group_get_num_pseudobonds(void *pbgroup, int cs_id)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        return pbg->pseudobonds(pbg->structure()->find_coord_set(cs_id)).size();
    } catch (...) {
        molc_error();
    }
    return 0;
}

extern "C" EXPORT void pseudobond_group_get_pseudobonds(void *pbgroup, int cs_id,
    Pseudobond **pb_ptrs)
{
    Proxy_PBGroup *pbg = static_cast<Proxy_PBGroup *>(pbgroup);
    try {
        for (auto pb: pbg->pseudobonds(pbg->structure()->find_coord_set(cs_id)))
            *pb_ptrs++ = pb;
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *pseudobond_create_global_manager(void* change_tracker)
{
    try {
        auto pb_manager = new PBManager(static_cast<ChangeTracker*>(change_tracker));
        return pb_manager;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void pseudobond_global_manager_clear(void *manager)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        mgr->clear();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void* pseudobond_global_manager_get_group(void *manager, const char* name, int create)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        return mgr->get_group(name, create);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *pseudobond_global_manager_group_map(void *manager)
{
    PyObject* mapping = PyDict_New();
    if (mapping == nullptr)
        molc_error();
    else {
        try {
            PBManager* mgr = static_cast<PBManager*>(manager);
            for (auto cat_grp: mgr->group_map()) {
                PyObject* key = PyUnicode_FromString(cat_grp.first.c_str());
                PyObject* val = PyLong_FromVoidPtr(cat_grp.second);
                PyDict_SetItem(mapping, key, val);
                Py_DECREF(key);
                Py_DECREF(val);
            }
        } catch (...) {
            molc_error();
        }
    }
    return mapping;
}

extern "C" EXPORT void pseudobond_global_manager_delete_group(void *manager, void *pbgroup)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        Proxy_PBGroup* pbg = static_cast<Proxy_PBGroup*>(pbgroup);
        mgr->delete_group(pbg);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_global_manager_session_restore_setup(void *manager)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        mgr->session_restore_setup();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_global_manager_session_restore_teardown(void *manager)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        mgr->session_restore_teardown();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT int pseudobond_global_manager_session_info(void *manager, PyObject *retvals)
{
    PBManager *mgr = static_cast<PBManager *>(manager);
    if (!PyList_Check(retvals)) {
        molc_error();
    } else {
        try {
            PyObject* ints;
            PyObject* floats;
            PyObject* misc;
            auto version =  mgr->session_info(&ints, &floats, &misc);
            PyList_Append(retvals, ints);
            PyList_Append(retvals, floats);
            PyList_Append(retvals, misc);
            return version;
        } catch (...) {
            molc_error();
        }
    }
    return -1;
}

extern "C" EXPORT void pseudobond_global_manager_session_restore(void *manager, int version,
    PyObject *ints, PyObject *floats, PyObject *misc)
{
    PBManager *mgr = static_cast<PBManager *>(manager);
    try {
        auto iarray = Numeric_Array();
        if (!array_from_python(ints, 1, Numeric_Array::Int, &iarray, false))
            throw std::invalid_argument("Global pseudobond int data is not a one-dimensional"
                " numpy int array");
        int* int_array = static_cast<int*>(iarray.values());
        auto farray = Numeric_Array();
        if (!array_from_python(floats, 1, Numeric_Array::Float, &farray, false))
            throw std::invalid_argument("Global pseudobond float data is not a one-dimensional"
                " numpy float array");
        float* float_array = static_cast<float*>(farray.values());
        mgr->session_restore(version, &int_array, &float_array, misc);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *pseudobond_global_manager_session_save_structure_mapping(void *manager)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        return pysupport::cmap_of_ptr_int_to_pydict(*(mgr->ses_struct_to_id_map()),
            "structure", "session ID");
    } catch (...) {
        molc_error();
    }
    return nullptr;
}

extern "C" EXPORT void pseudobond_global_manager_session_restore_structure_mapping(void *manager,
    PyObject* mapping)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        auto c_map = mgr->ses_id_to_struct_map();
        if (!PyDict_Check(mapping))
            throw std::invalid_argument("session-restore pb structure mapping not a dict!");
        Py_ssize_t index = 0;
        PyObject* ses_id;
        PyObject* ptr;
        while (PyDict_Next(mapping, &index, &ses_id, &ptr)) {
            (*c_map)[PyLong_AsLong(ses_id)] = static_cast<Structure*>(PyLong_AsVoidPtr(ptr));
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_global_manager_session_save_setup(void *manager)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        mgr->session_save_setup();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pseudobond_global_manager_session_save_teardown(void *manager)
{
    try {
        PBManager* mgr = static_cast<PBManager*>(manager);
        mgr->session_save_teardown();
    } catch (...) {
        molc_error();
    }
}

// -------------------------------------------------------------------------
// residue functions
//
PyObject*
_atom_name_frozen_set(const std::set<AtomName>& atom_names)
{
    PyObject* fset = PyFrozenSet_New(nullptr);
    if (fset == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for atom-name frozen set");
        return nullptr;
    }
    for (auto& atom_name: atom_names) {
        const char* n = atom_name.c_str();
        PyObject* py_n = PyUnicode_FromString(n);
        if (py_n == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for atom-name string");
            Py_DECREF(fset);
            return nullptr;
        }
        if (PySet_Add(fset, py_n) < 0) {
            Py_DECREF(fset);
            return nullptr;
        }
    }
    return fset;
}

PyObject*
_atom_name_tuple(const std::vector<AtomName>& atom_names)
{
    PyObject* tuple = PyTuple_New(atom_names.size());
    if (tuple == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for atom-name tuple");
        return nullptr;
    }
    size_t i = 0;
    for (auto& atom_name: atom_names) {
        const char* n = atom_name.c_str();
        PyObject* py_n = PyUnicode_FromString(n);
        if (py_n == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for atom-name string");
            Py_DECREF(tuple);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, i++, py_n);
    }
    return tuple;
}

extern "C" EXPORT PyObject *residue_aa_min_backbone_names()
{
    return _atom_name_frozen_set(Residue::aa_min_backbone_names);
}

extern "C" EXPORT PyObject *residue_aa_max_backbone_names()
{
    return _atom_name_frozen_set(Residue::aa_max_backbone_names);
}

extern "C" EXPORT PyObject *residue_aa_side_connector_names()
{
    return _atom_name_frozen_set(Residue::aa_side_connector_names);
}

extern "C" EXPORT PyObject *residue_na_min_backbone_names()
{
    return _atom_name_frozen_set(Residue::na_min_backbone_names);
}

extern "C" EXPORT PyObject *residue_na_max_backbone_names()
{
    return _atom_name_frozen_set(Residue::na_max_backbone_names);
}

extern "C" EXPORT PyObject *residue_na_side_connector_names()
{
    return _atom_name_frozen_set(Residue::na_side_connector_names);
}

extern "C" EXPORT PyObject *residue_aa_min_ordered_backbone_names()
{
    return _atom_name_tuple(Residue::aa_min_ordered_backbone_names);
}

extern "C" EXPORT PyObject *residue_na_min_ordered_backbone_names()
{
    return _atom_name_tuple(Residue::na_min_ordered_backbone_names);
}

extern "C" EXPORT void residue_atoms(void *residues, size_t n, pyobject_t *atoms)
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

extern "C" EXPORT PyObject *residue_bonds_between(void *res, void* other_res)
{
    Residue *r = static_cast<Residue*>(res);
    Residue *other = static_cast<Residue*>(other_res);
    try {
        auto bonds = r->bonds_between(other);
        const Bond **bb;
        PyObject *b_array = python_voidp_array(bonds.size(), (void***)&bb);
        size_t i = 0;
        for (auto b: bonds)
            bb[i++] = b;
        return b_array;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void residue_center(void *residues, size_t n, float64_t *xyz)
{
    Residue **r = static_cast<Residue **>(residues);  
    try {
      for (size_t i = 0; i != n; ++i) {
        Residue *ri = r[i];
        double x = 0, y = 0, z = 0;
        int na = 0;
        for (auto atom: ri->atoms()) {
          const Coord &c = atom->coord();
          x += c[0]; y += c[1]; z += c[2];
          na += 1;
        }
        if (na > 0) {
          *xyz++ = x/na;  *xyz++ = y/na;  *xyz++ = z/na;
        } else {
          *xyz++ = 0;  *xyz++ = 0;  *xyz++ = 0;
        }
      }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_chain(void *residues, size_t n, pyobject_t *chainp)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::chain, chainp);
}

extern "C" EXPORT void residue_chain_id(void *residues, size_t n, pyobject_t *cids)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(r[i]->chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_residue_chain_id(void *residues, size_t n, pyobject_t *cids)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            r[i]->set_chain_id(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(cids[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT bool residue_connects_to(void *residue, void *other_res)
{
    Residue *r = static_cast<Residue *>(residue);
    Residue *other = static_cast<Residue *>(other_res);
    try {
        return r->connects_to(other);
    } catch (...) {
        molc_error();
        return false;
    }
}

extern "C" EXPORT void residue_mmcif_chain_id(void *residues, size_t n, pyobject_t *cids)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(r[i]->mmcif_chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_delete(void *residues, size_t n)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        Residue::Atoms atoms;
        for (size_t i = 0; i != n; ++i) {
          const Residue::Atoms &ratoms = r[i]->atoms();
          atoms.insert(atoms.end(), ratoms.begin(), ratoms.end());
        }
        atom_delete(atoms.data(), atoms.size());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void* residue_find_atom(void *residue, char *atom_name)
{
    Residue *r = static_cast<Residue*>(residue);
    try {
        return r->find_atom(atom_name);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *residue_ideal_chirality(const char *res_name, const char *atom_name)
{
    std::string error_msg;
    auto i = Residue::ideal_chirality.find(res_name);
    if (i == Residue::ideal_chirality.end()) {
        error_msg.append("mmCIF Chemical Component Dictionary for ");
        error_msg.append(res_name);
        error_msg.append(" has not been read");
        PyErr_SetString(PyExc_KeyError, error_msg.c_str());
        return nullptr;
    }
    auto j = i->second.find(atom_name);
    if (j == i->second.end()) {
        error_msg.append("Atom ");
        error_msg.append(atom_name);
        error_msg.append(" not in mmCIF Chemical Component Dictionary for ");
        error_msg.append(res_name);
        PyErr_SetString(PyExc_KeyError, error_msg.c_str());
        return nullptr;
    }
    return unicode_from_character(j->second);
}

extern "C" EXPORT void residue_insertion_code(void *residues, size_t n, pyobject_t *ics)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto ic = r[i]->insertion_code();
            if (ic == ' ')
                ics[i] = unicode_from_string("", 0);
            else
                ics[i] = unicode_from_character(r[i]->insertion_code());
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_residue_insertion_code(void *residues, size_t n, pyobject_t *ics)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i) {
            PyObject* py_ic = static_cast<PyObject*>(ics[i]);
            auto size = PyUnicode_GET_LENGTH(py_ic);
            if (size > 1)
                throw std::invalid_argument("Insertion code must be one character or empty string");
            char val;
            if (size == 0)
                val = ' ';
            else
                val = (char)PyUnicode_READ_CHAR(py_ic, 0);
            r[i]->set_insertion_code(val);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_principal_atom(void *residues, size_t n, pyobject_t *pas)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            pas[i] = r[i]->principal_atom();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_polymer_type(void *residues, size_t n, uint8_t *polymer_type)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::polymer_type, polymer_type);
}

extern "C" EXPORT void residue_is_helix(void *residues, size_t n, npy_bool *is_helix)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::is_helix, is_helix);
}

extern "C" EXPORT void set_residue_is_helix(void *residues, size_t n, npy_bool *is_helix)
{
    // If true, also unsets is_strand
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_is_helix, is_helix);
}

extern "C" EXPORT void residue_is_missing_heavy_template_atoms(void *residues, size_t n, npy_bool *is_missing)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            is_missing[i] = r[i]->is_missing_heavy_template_atoms(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_is_strand(void *residues, size_t n, npy_bool *is_strand)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::is_strand, is_strand);
}

extern "C" EXPORT void set_residue_is_strand(void *residues, size_t n, npy_bool *is_strand)
{
    // If true, also unsets is_helix
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_is_strand, is_strand);
}

extern "C" EXPORT void residue_ss_id(void *residues, size_t n, int32_t *ss_id)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ss_id, ss_id);
}

extern "C" EXPORT void set_residue_ss_id(void *residues, size_t n, int32_t *ss_id)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ss_id, ss_id);
}

extern "C" EXPORT void residue_ss_type(void *residues, size_t n, int32_t *ss_type)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ss_type, ss_type);
}

extern "C" EXPORT void set_residue_ss_type(void *residues, size_t n, int32_t *ss_type)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i) {
            r[i]->set_ss_type(static_cast<Residue::SSType>(ss_type[i]));
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_ribbon_display(void *residues, size_t n, npy_bool *ribbon_display)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ribbon_display, ribbon_display);
}

extern "C" EXPORT void set_residue_ribbon_display(void *residues, size_t n, npy_bool *ribbon_display)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ribbon_display, ribbon_display);
}

extern "C" EXPORT void residue_ribbon_hide_backbone(void *residues, size_t n, npy_bool *ribbon_hide_backbone)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ribbon_hide_backbone, ribbon_hide_backbone);
}

extern "C" EXPORT void set_residue_ribbon_hide_backbone(void *residues, size_t n, npy_bool *ribbon_hide_backbone)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ribbon_hide_backbone, ribbon_hide_backbone);
}

extern "C" EXPORT void residue_ribbon_adjust(void *residues, size_t n, float32_t *ribbon_adjust)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ribbon_adjust, ribbon_adjust);
}

extern "C" EXPORT void set_residue_ribbon_adjust(void *residues, size_t n, float32_t *ribbon_adjust)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ribbon_adjust, ribbon_adjust);
}

extern "C" EXPORT void residue_selected(void *residues, size_t n, npy_bool *sel)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get<Residue, bool, npy_bool>(r, n, &Residue::selected, sel);
}

extern "C" EXPORT void residue_structure(void *residues, size_t n, pyobject_t *molp)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i < n; ++i)
          molp[i] = r[i]->structure()->py_instance(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_name(void *residues, size_t n, pyobject_t *names)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(r[i]->name());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_residue_name(void *residues, size_t n, pyobject_t *names)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
        r[i]->set_name(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(names[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_num_atoms(void *residues, size_t n, size_t *natoms)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            natoms[i] = r[i]->atoms().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_number(void *residues, size_t n, int32_t *nums)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::number, nums);
}

extern "C" EXPORT void residue_str(void *residues, size_t n, pyobject_t *strs)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i)
            strs[i] = unicode_from_string(r[i]->str().c_str());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_secondary_structure_id(void *residues, size_t n, int32_t *ids)
{
    Residue **res = static_cast<Residue **>(residues);
    std::map<const Residue *, int> sid;
    try {
      int32_t id = 0;
      for (size_t i = 0; i != n; ++i) {
        const Residue *r = res[i];
        if (sid.find(r) != sid.end())
          continue;
        // Scan the chain of this residue to identify secondary structure.
        Chain *c = r->chain();
        if (c == NULL)
          sid[r] = ++id;        // Residue is not part of a chain.
        else {
          const Chain::Residues &cr = c->residues();
          Residue *pres = NULL;
          for (auto cres: cr)
            if (cres) { // Chain residues are null for missing structure.
              sid[cres] = ((pres == NULL ||
                            cres->ss_id() != pres->ss_id() ||
                            cres->is_helix() != pres->is_helix() ||
                            cres->is_strand() != pres->is_strand()) ?
                           ++id : id);
              pres = cres;
            }
        }
      }
      for (size_t i = 0; i != n; ++i)
        ids[i] = sid[res[i]];
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *residue_standard_solvent_names()
{
    PyObject* name_set = PySet_New(nullptr);
    if (name_set == nullptr)
        return nullptr;
    try {
        for (auto name: Residue::std_solvent_names) {
            PyObject* py_name = PyUnicode_FromString(name.c_str());
            if (py_name == nullptr || PySet_Add(name_set, py_name) < 0) {
                Py_DECREF(name_set);
                return nullptr;
            }
        }
    } catch (...) {
        molc_error();
        return nullptr;
    }
    return name_set;
}

extern "C" EXPORT PyObject *residue_standard_water_names()
{
    PyObject* name_set = PySet_New(nullptr);
    if (name_set == nullptr)
        return nullptr;
    try {
        for (auto name: Residue::std_water_names) {
            PyObject* py_name = PyUnicode_FromString(name.c_str());
            if (py_name == nullptr || PySet_Add(name_set, py_name) < 0) {
                Py_DECREF(name_set);
                return nullptr;
            }
        }
    } catch (...) {
        molc_error();
        return nullptr;
    }
    return name_set;
}

extern "C" EXPORT PyObject *residue_unique_sequences(void *residues, size_t n, int *seq_ids)
{
    Residue **r = static_cast<Residue **>(residues);
    PyObject *seqs = PyList_New(1);
    try {
        PyList_SetItem(seqs, 0, unicode_from_string(""));
        std::map<Chain *, int> cmap;
        std::map<std::string, int> smap;
        for (size_t i = 0; i != n; ++i)
          {
            Chain *c = r[i]->chain();
            int si = 0;
            if (c)
              {
                auto ci = cmap.find(c);
                if (ci == cmap.end())
                  {
                    std::string seq(c->begin(), c->end());
                    auto seqi = smap.find(seq);
                    if (seqi == smap.end())
                      {
                        int next_id = smap.size()+1;
                        si = cmap[c] = smap[seq] = next_id;
                        PyList_Append(seqs, unicode_from_string(seq));
                      }
                    else
                      si = cmap[c] = seqi->second;
                  }
                else
                  si = ci->second;
              }
            seq_ids[i] = si;
          }
    } catch (...) {
        molc_error();
    }
    return seqs;
}

extern "C" EXPORT void residue_add_atom(void *res, void *atom)
{
    Residue *r = static_cast<Residue *>(res);
    try {
        r->add_atom(static_cast<Atom *>(atom));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_ribbon_color(void *residues, size_t n, uint8_t *rgba)
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

extern "C" EXPORT void set_residue_ribbon_color(void *residues, size_t n, uint8_t *rgba)
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

extern "C" EXPORT void residue_clear_hide_bits(void *residues, size_t n, int32_t bit_mask, npy_bool atoms_only)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i < n; ++i)
            r[i]->clear_hide_bits(bit_mask, atoms_only);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_set_alt_loc(void *residues, size_t n, char alt_loc)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i < n; ++i)
            r[i]->set_alt_loc(alt_loc);
    } catch (...) {
        molc_error();
    }
}

// -------------------------------------------------------------------------
// ring functions
//
extern "C" EXPORT void ring_aromatic(void *rings, size_t n, npy_bool *aro)
{
    Ring **r = static_cast<Ring **>(rings);
    error_wrap_array_get<Ring, bool, npy_bool>(r, n, &Ring::aromatic, aro);
}

extern "C" EXPORT void ring_atoms(void *rings, size_t n, pyobject_t *atoms)
{
    Ring **r = static_cast<Ring **>(rings);
    try {
        for (size_t i = 0; i != n; ++i) {
            for (auto a: r[i]->atoms())
                *atoms++ = a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void ring_bonds(void *rings, size_t n, pyobject_t *bonds)
{
    Ring **r = static_cast<Ring **>(rings);
    try {
        for (size_t i = 0; i != n; ++i) {
            for (auto b: r[i]->bonds())
                *bonds++ = b;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void ring_ordered_atoms(void *rings, size_t n, pyobject_t *atoms)
{
    Ring **r = static_cast<Ring **>(rings);
    try {
        for (size_t i = 0; i != n; ++i) {
            for (auto a: r[i]->ordered_atoms())
                *atoms++ = a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void ring_ordered_bonds(void *rings, size_t n, pyobject_t *bonds)
{
    Ring **r = static_cast<Ring **>(rings);
    try {
        for (size_t i = 0; i != n; ++i) {
            for (auto b: r[i]->ordered_bonds())
                *bonds++ = b;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void ring_size(void *rings, size_t n, size_t *sizes)
{
    Ring **r = static_cast<Ring **>(rings);
    try {
        for (size_t i = 0; i != n; ++i)
            sizes[i] = r[i]->size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT bool ring_equal(void *ring, void *other)
{
    Ring *r = static_cast<Ring *>(ring);
    Ring *o = static_cast<Ring *>(other);
    bool eq;
    try {
        eq = (*r == *o);
    } catch (...) {
        molc_error();
    }
    return eq;
}

extern "C" EXPORT bool ring_less_than(void *ring, void *other)
{
    Ring *r = static_cast<Ring *>(ring);
    Ring *o = static_cast<Ring *>(other);
    bool lt;
    try {
        lt = (*r < *o);
    } catch (...) {
        molc_error();
    }
    return lt;
}

extern "C" EXPORT void residue_ring_color(void *residues, size_t n, uint8_t *rgba)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Rgba &c = r[i]->ring_color();
            *rgba++ = c.r;
            *rgba++ = c.g;
            *rgba++ = c.b;
            *rgba++ = c.a;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_residue_ring_color(void *residues, size_t n, uint8_t *rgba)
{
    Residue **r = static_cast<Residue **>(residues);
    try {
        Rgba c;
        for (size_t i = 0; i != n; ++i) {
            c.r = *rgba++;
            c.g = *rgba++;
            c.b = *rgba++;
            c.a = *rgba++;
            r[i]->set_ring_color(c);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void residue_ring_display(void *residues, size_t n, npy_bool *ring_display)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::ring_display, ring_display);
}

extern "C" EXPORT void set_residue_ring_display(void *residues, size_t n, npy_bool *ring_display)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_ring_display, ring_display);
}

extern "C" EXPORT void residue_thin_rings(void *residues, size_t n, npy_bool *thin_rings)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::thin_rings, thin_rings);
}

extern "C" EXPORT void set_residue_thin_rings(void *residues, size_t n, npy_bool *thin_rings)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_thin_rings, thin_rings);
}


// -------------------------------------------------------------------------
// structure sequence functions
//
extern "C" EXPORT void sseq_bulk_set(void *sseq_ptr, PyObject *res_ptr_vals, char *seq)
{
    try {
        StructureSeq* sseq = static_cast<StructureSeq*>(sseq_ptr);
        if (!PyList_Check(res_ptr_vals))
            throw std::invalid_argument("sseq_bulk_seq residues arg not a list!");
        auto num_res = PyList_GET_SIZE(res_ptr_vals);
        StructureSeq::Residues residues;
        for (int i = 0; i < num_res; ++i) {
            // since 0 == nullptr, don't need specific conversion...
            residues.push_back(static_cast<Residue*>(
                PyLong_AsVoidPtr(PyList_GET_ITEM(res_ptr_vals, i))));
        }
        // chars not necessarily same length as residues due to gap characters
        Sequence::Contents chars;
        for (auto s = seq; *s != '\0'; ++s)
            chars.push_back(*s);
        sseq->bulk_set(residues, &chars);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void sseq_chain_id(void *chains, size_t n, pyobject_t *cids)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i)
            cids[i] = unicode_from_string(c[i]->chain_id());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_sseq_chain_id(void *chains, size_t n, pyobject_t *cids)
{
    StructureSeq **sseq = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i)
            sseq[i]->set_chain_id(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(cids[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *sseq_copy(void* source)
{
    StructureSeq *sseq = static_cast<StructureSeq*>(source);
    try {
        return sseq->copy();
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void sseq_from_seqres(void *sseqs, size_t n, npy_bool *from_seqres)
{
    StructureSeq **ss = static_cast<StructureSeq **>(sseqs);
    error_wrap_array_get(ss, n, &StructureSeq::from_seqres, from_seqres);
}

extern "C" EXPORT void set_sseq_from_seqres(void *sseqs, size_t n, npy_bool *from_seqres)
{
    StructureSeq **ss = static_cast<StructureSeq **>(sseqs);
    error_wrap_array_set(ss, n, &StructureSeq::set_from_seqres, from_seqres);
}

extern "C" EXPORT void sseq_structure(void *chains, size_t n, pyobject_t *molp)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i < n; ++i)
          molp[i] = c[i]->structure()->py_instance(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *sseq_new(char *chain_id, void *struct_ptr, int polymer_type)
{
    Structure *structure = static_cast<Structure*>(struct_ptr);
    try {
        StructureSeq *sseq = new StructureSeq(chain_id, structure,
            static_cast<atomstruct::PolymerType>(polymer_type));
        return sseq;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void sseq_num_residues(void *chains, size_t n, size_t *nres)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i)
            nres[i] = c[i]->residues().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void sseq_num_existing_residues(void *chains, size_t n, size_t *nres)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i) {
            size_t num_sr = 0;
            for (auto r: c[i]->residues())
                if (r != nullptr) ++num_sr;
            nres[i] = num_sr;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void sseq_polymer_type(void *sseqs, size_t n, uint8_t *polymer_type)
{
    StructureSeq **ss = static_cast<StructureSeq **>(sseqs);
    error_wrap_array_get(ss, n, &StructureSeq::polymer_type, polymer_type);
}

extern "C" EXPORT void sseq_residues(void *chains, size_t n, pyobject_t *res)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i) {
            const StructureSeq::Residues &r = c[i]->residues();
            for (size_t j = 0; j != r.size(); ++j)
                *res++ = r[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void sseq_existing_residues(void *chains, size_t n, pyobject_t *res)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i) {
            const StructureSeq::Residues &r = c[i]->residues();
            for (size_t j = 0; j != r.size(); ++j)
                if (r[j] != nullptr)
                    *res++ = r[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void* sseq_residue_at(void *sseq_ptr, size_t i)
{
    StructureSeq *sseq = static_cast<StructureSeq*>(sseq_ptr);
    try {
        return sseq->residues().at(i);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *sseq_res_map(void *sseq_ptr)
{
    PyObject* mapping = PyDict_New();
    if (mapping == nullptr)
        molc_error();
    else {
        try {
            StructureSeq *sseq = static_cast<StructureSeq*>(sseq_ptr);
            for (auto res_pos: sseq->res_map()) {
                PyObject* key = PyLong_FromVoidPtr(res_pos.first);
                PyObject* val = PyLong_FromSize_t(res_pos.second);
                PyDict_SetItem(mapping, key, val);
                Py_DECREF(key);
                Py_DECREF(val);
            }
        } catch (...) {
            molc_error();
        }
    }
    return mapping;
}

extern "C" EXPORT PyObject *sseq_estimate_assoc_params(void *sseq_ptr)
{
    PyObject* tuple = PyTuple_New(3);
    if (tuple == nullptr)
        molc_error();
    else {
        try {
            StructureSeq *sseq = static_cast<StructureSeq*>(sseq_ptr);
            auto ap = estimate_assoc_params(*sseq);
            PyObject *py_est_len = PyLong_FromSize_t(ap.est_len);
            if (py_est_len == nullptr)
                molc_error();
            else {
                try {
                    PyObject *py_segments = pysupport::cvec_of_cvec_of_char_to_pylist(ap.segments,
                        "continuous sequence segment");
                    PyObject *py_gaps = pysupport::cvec_of_int_to_pylist(ap.gaps,
                        "structure gap size");
                    PyTuple_SET_ITEM(tuple, 0, py_est_len);
                    PyTuple_SET_ITEM(tuple, 1, py_segments);
                    PyTuple_SET_ITEM(tuple, 2, py_gaps);
                } catch (...) {
                    molc_error();
                }
            }
        } catch (...) {
            molc_error();
        }
    }
    return tuple;
}

extern "C" EXPORT PyObject *sseq_try_assoc(void *seq_ptr, void *sseq_ptr, size_t est_len,
    PyObject *py_segments, PyObject *py_gaps, int max_errors)
{
    PyObject* tuple = PyTuple_New(2);
    if (tuple == nullptr)
        molc_error();
    else {
        Sequence *seq = static_cast<Sequence*>(seq_ptr);
        StructureSeq *sseq = static_cast<StructureSeq*>(sseq_ptr);
        std::vector<Sequence::Contents> segments;
        pysupport::pylist_of_string_to_cvec_of_cvec(py_segments, segments,
            "segment residue letter");
        std::vector<int> gaps;
        pysupport::pylist_of_int_to_cvec(py_gaps, gaps, "estimated gap");
        AssocParams ap(est_len, segments.begin(), segments.end(), gaps.begin(), gaps.end());
        AssocRetvals arv;
        try {
            arv = try_assoc(*seq, *sseq, ap, max_errors);
        } catch (SA_AssocFailure& e) {
            // convert to error that maps to ValueError
            PyErr_SetString(PyExc_ValueError, e.what());
            Py_DECREF(tuple);
            return nullptr;
        } catch (...) {
            molc_error();
            Py_DECREF(tuple);
            return nullptr;
        }
        PyObject* map = pysupport::cmap_of_ptr_int_to_pydict(arv.match_map.res_to_pos(),
            "residue", "associated seq position");
        PyTuple_SET_ITEM(tuple, 0, map);
        PyTuple_SET_ITEM(tuple, 1, PyLong_FromLong(static_cast<long>(arv.num_errors)));
    }
    return tuple;
}

// -------------------------------------------------------------------------
// change tracker functions
//
extern "C" EXPORT void *change_tracker_create()
{
    try {
        auto change_tracker = new ChangeTracker();
        return change_tracker;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT npy_bool change_tracker_changed(void *vct)
{
    ChangeTracker* ct = static_cast<ChangeTracker*>(vct);
    try {
        return ct->changed();
    } catch (...) {
        molc_error();
        return false;
    }
}

static PyObject* changes_as_py_dict(const ChangeTracker::ChangesArray& all_changes,
    const std::string* python_class_names)
{
    PyObject* changes_data = PyDict_New();
    for (size_t i = 0; i < all_changes.size(); ++i) {
        auto& class_changes = all_changes[i];
        auto class_name = python_class_names[i];
        PyObject* key = unicode_from_string(class_name);
        PyObject* value = PyTuple_New(4);

        // first tuple item:  created objects
        void **ptrs;
        PyObject *ptr_array = python_voidp_array(class_changes.created.size(), &ptrs);
        size_t j = 0;
        for (auto ptr: class_changes.created)
            ptrs[j++] = const_cast<void*>(ptr);
        PyTuple_SET_ITEM(value, 0, ptr_array);

        // second tuple item:  modified objects
        ptr_array = python_voidp_array(class_changes.modified.size(), &ptrs);
        j = 0;
        for (auto ptr: class_changes.modified)
            ptrs[j++] = const_cast<void*>(ptr);
        PyTuple_SET_ITEM(value, 1, ptr_array);

        // third tuple item:  list of reasons
        PyObject* reasons = PyList_New(class_changes.reasons.size());
        j = 0;
        for (auto reason: class_changes.reasons)
            PyList_SetItem(reasons, j++, unicode_from_string(reason));
        PyTuple_SET_ITEM(value, 2, reasons);

        // fourth tuple item:  total number of deleted objects
        PyTuple_SET_ITEM(value, 3, PyLong_FromLong(class_changes.num_deleted));

        PyDict_SetItem(changes_data, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
    }
    return changes_data;
}

extern "C" EXPORT PyObject* change_tracker_changes(void *vct)
{
    ChangeTracker* ct = static_cast<ChangeTracker*>(vct);
    const std::string* python_class_names = ct->python_class_names;
    PyObject* ret_tuple = PyTuple_New(2);
    try {
        PyTuple_SET_ITEM(ret_tuple, 0,
            changes_as_py_dict(ct->get_global_changes(), python_class_names));
        PyObject* struct_changes_dict = PyDict_New();
        for (auto& s_changes: ct->get_structure_changes()) {
            PyObject* key = PyLong_FromVoidPtr(static_cast<void*>(s_changes.first));
            PyObject* value = changes_as_py_dict(s_changes.second, python_class_names);
            PyDict_SetItem(struct_changes_dict, key, value);
            Py_DECREF(key);
            Py_DECREF(value);
        }
        PyTuple_SET_ITEM(ret_tuple, 1, struct_changes_dict);
    } catch (...) {
        Py_XDECREF(ret_tuple);
        molc_error();
    }
    return ret_tuple;
}

extern "C" EXPORT void change_tracker_clear(void *vct)
{
    ChangeTracker* ct = static_cast<ChangeTracker*>(vct);
    try {
        return ct->clear();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void change_tracker_add_modified(int class_num, void *modded, const char *reason)
{
    try {
        if (class_num == 0) {
            auto atomic_ptr = static_cast<Atom*>(modded);
            auto s = atomic_ptr->structure();
            atomic_ptr->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 1) {
            auto atomic_ptr = static_cast<Bond*>(modded);
            auto s = atomic_ptr->structure();
            atomic_ptr->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 2) {
            auto atomic_ptr = static_cast<Pseudobond*>(modded);
            auto mgr = atomic_ptr->group()->manager();
            auto s_mgr = dynamic_cast<StructureManager*>(mgr);
            Structure* s = (s_mgr == nullptr ? nullptr : s_mgr->structure());
            mgr->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 3) {
            auto atomic_ptr = static_cast<Residue*>(modded);
            auto s = atomic_ptr->structure();
            atomic_ptr->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 4) {
            auto atomic_ptr = static_cast<Chain*>(modded);
            auto s = atomic_ptr->structure();
            s->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 5) {
            auto atomic_ptr = static_cast<AtomicStructure*>(modded);
            auto s = atomic_ptr;
            atomic_ptr->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 6) {
            auto atomic_ptr = static_cast<Proxy_PBGroup*>(modded);
            auto mgr = atomic_ptr->manager();
            auto s_mgr = dynamic_cast<StructureManager*>(mgr);
            Structure* s = (s_mgr == nullptr ? nullptr : s_mgr->structure());
            mgr->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else if (class_num == 7) {
            auto atomic_ptr = static_cast<CoordSet*>(modded);
            auto s = atomic_ptr->structure();
            s->change_tracker()->add_modified(s, atomic_ptr, reason);
        } else {
            throw std::invalid_argument("Bad class value to ChangeTracker.add_modified()");
        }
    } catch (...) {
        molc_error();
    }
}

// -------------------------------------------------------------------------
// coordset functions
//
extern "C" EXPORT void coordset_id(void *coordsets, size_t n, int32_t *index)
{
    CoordSet **a = static_cast<CoordSet **>(coordsets);
    error_wrap_array_get(a, n, &CoordSet::id, index);
}

extern "C" EXPORT void coordset_structure(void *coordsets, size_t n, pyobject_t *molp)
{
    CoordSet **cs = static_cast<CoordSet **>(coordsets);
    try {
        for (size_t i = 0; i < n; ++i)
          molp[i] = cs[i]->structure()->py_instance(true);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject* coordset_xyzs(void *coordset)
{
    CoordSet *cs = static_cast<CoordSet*>(coordset);

    PyObject* ret_val;
    try {
        double *v;
        ret_val = python_double_array(cs->coords().size(), 3, &v);
        for (auto xyz: cs->coords()) {
            *v++ = xyz[0];
            *v++ = xyz[1];
            *v++ = xyz[2];
        }
    } catch (...) {
        molc_error();
        return nullptr;
    }
    return ret_val;
}

// -------------------------------------------------------------------------
// sequence functions
//
extern "C" EXPORT void *sequence_new(const char* name, const char* characters)
{
    try {
        Sequence::Contents chars;
        while (*characters) chars.push_back(*characters++);
        Sequence *seq = new Sequence(chars, name);
        return seq;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void sequence_characters(void *seqs, size_t n, pyobject_t *chars)
{
    Sequence **s = static_cast<Sequence **>(seqs);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto& contents = s[i]->contents();
            char* str = new char[contents.size() + 1];
            auto ptr = str;
            for (auto c: contents)
                *ptr++ = c;
            *ptr = '\0';
            chars[i] = unicode_from_string(str);
            delete[] str;
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_sequence_characters(void *seqs, size_t n, pyobject_t *chars)
{
    Sequence **s = static_cast<Sequence **>(seqs);
    try {
        for (size_t i = 0; i != n; ++i) {
            Sequence::Contents contents;
            auto ptr = CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(chars[i]));
            while (*ptr) contents.push_back(*ptr++);
            s[i]->swap(contents);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void sequence_circular(void *seqs, size_t n, npy_bool *seq_circular)
{
    Sequence **s = static_cast<Sequence **>(seqs);
    error_wrap_array_get(s, n, &Sequence::circular, seq_circular);
}

extern "C" EXPORT void set_sequence_circular(void *seqs, size_t n, npy_bool *seq_circular)
{
    Sequence **s = static_cast<Sequence **>(seqs);
    error_wrap_array_set(s, n, &Sequence::set_circular, seq_circular);
}

extern "C" EXPORT void sequence_extend(void *seq, const char *chars)
{
    Sequence *s = static_cast<Sequence *>(seq);
    try {
        s->extend(chars);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT int sequence_gapped_to_ungapped(void *seq, int32_t index)
{
    Sequence *s = static_cast<Sequence *>(seq);
    try {
        return s->gapped_to_ungapped(index);
    } catch (SeqIndexError& e) {
        return -1;
    } catch (...) {
        molc_error();
        return 0;
    }
}

extern "C" EXPORT size_t sequence_len(void *seq)
{
    Sequence *s = static_cast<Sequence *>(seq);
    try {
        return s->size();
    } catch (...) {
        molc_error();
        return 0;
    }
}

extern "C" EXPORT void sequence_name(void *seqs, size_t n, pyobject_t *names)
{
    Sequence **s = static_cast<Sequence **>(seqs);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = unicode_from_string(s[i]->name().c_str());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_sequence_name(void *seqs, size_t n, pyobject_t *names)
{
    Sequence **s = static_cast<Sequence **>(seqs);
    try {
        for (size_t i = 0; i != n; ++i)
            s[i]->set_name(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(names[i])));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT char sequence_nucleic3to1(const char *rname)
{
    try {
        return Sequence::nucleic3to1(rname);
    } catch (...) {
        molc_error();
        return 'X';
    }
}

extern "C" EXPORT char sequence_protein3to1(const char *rname)
{
    try {
        return Sequence::protein3to1(rname);
    } catch (...) {
        molc_error();
        return 'X';
    }
}

extern "C" EXPORT void sequence_del_pyobj(void *seq_ptr)
{
    Sequence *seq = static_cast<Sequence*>(seq_ptr);
    try {
        seq->python_destroyed();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT char sequence_rname3to1(const char *rname)
{
    try {
        return Sequence::rname3to1(rname);
    } catch (...) {
        molc_error();
        return 'X';
    }
}

extern "C" EXPORT pyobject_t sequence_search(void *seq, const char *pattern, bool case_sensitive)
{
    Sequence *s = static_cast<Sequence *>(seq);
    try {
        auto matches = s->search(pattern, case_sensitive);
        PyObject* py_matches = PyList_New(matches.size());
        int j = 0;
        for (auto index_len: matches) {
            PyObject* match_tuple = PyTuple_New(2);
            PyTuple_SET_ITEM(match_tuple, 0, PyLong_FromLong(index_len.first));
            PyTuple_SET_ITEM(match_tuple, 1, PyLong_FromLong(index_len.second));
            PyList_SetItem(py_matches, j++, match_tuple);
        }
        return py_matches;
    } catch (...) {
        molc_error();
        Py_RETURN_NONE;
    }
}

extern "C" EXPORT pyobject_t sequence_ungapped(void *seq)
{
    Sequence *s = static_cast<Sequence *>(seq);
    try {
        auto ungapped = s->ungapped();
        return unicode_from_string(std::string(ungapped.begin(), ungapped.end()));
    } catch (...) {
        molc_error();
        Py_RETURN_NONE;
    }
}

extern "C" EXPORT int sequence_ungapped_to_gapped(void *seq, int32_t index)
{
    Sequence *s = static_cast<Sequence *>(seq);
    try {
        return s->ungapped_to_gapped(index);
    } catch (...) {
        molc_error();
        return 0;
    }
}

// -------------------------------------------------------------------------
// structure functions
//
extern "C" EXPORT void set_structure_color(void *mol, uint8_t *rgba)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        Rgba c;
        c.r = *rgba++;
        c.g = *rgba++;
        c.b = *rgba++;
        c.a = *rgba++;
        m->set_color(c);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *structure_copy(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->copy();
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void structure_graphics_change(void *mols, size_t n, int *changed)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get<Structure, int, int>(m, n, &Structure::get_all_graphics_changes, changed);
}

extern "C" EXPORT void set_structure_graphics_change(void *mols, size_t n, int *changed)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set<Structure, int, int>(m, n, &Structure::set_all_graphics_changes, changed);
}

extern "C" EXPORT void structure_lower_case_chains(void *mols, size_t n, npy_bool *lower_case_chains)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
            lower_case_chains[i] = m[i]->lower_case_chains;
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_structure_lower_case_chains(void *structures, size_t n, npy_bool *lcc)
{
    Structure **s = static_cast<Structure **>(structures);
    try {
        for (size_t i = 0; i != n; ++i)
            s[i]->lower_case_chains = lcc[i];
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_active_coordset_change_notify(void *structures, size_t n, npy_bool *accn)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_get(s, n, &Structure::active_coord_set_change_notify, accn);
}

extern "C" EXPORT void set_structure_active_coordset_change_notify(void *structures, size_t n, npy_bool *accn)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_set(s, n, &Structure::set_active_coord_set_change_notify, accn);
}

extern "C" EXPORT void structure_alt_loc_change_notify(void *structures, size_t n, npy_bool *alcn)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_get(s, n, &Structure::alt_loc_change_notify, alcn);
}

extern "C" EXPORT void set_structure_alt_loc_change_notify(void *structures, size_t n, npy_bool *alcn)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_set_mutable(s, n, &Structure::set_alt_loc_change_notify, alcn);
}

extern "C" EXPORT void structure_idatm_valid(void *structures, size_t n, npy_bool *valid)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_get(s, n, &Structure::idatm_valid, valid);
}

extern "C" EXPORT void set_structure_idatm_valid(void *structures, size_t n, npy_bool *valid)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_set(s, n, &Structure::set_idatm_valid, valid);
}

extern "C" EXPORT void structure_num_atoms(void *mols, size_t n, size_t *natoms)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
            natoms[i] = m[i]->atoms().size();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_num_atoms_visible(void *mols, size_t n, size_t *natoms)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
          {
            const Structure::Atoms &atoms = m[i]->atoms();
            int c = 0;
            for (auto a: atoms)
              if (a->visible())
                c += 1;
            natoms[i] = c;
          }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_atoms(void *mols, size_t n, pyobject_t *atoms)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Structure::Atoms &a = m[i]->atoms();
            for (size_t j = 0; j != a.size(); ++j)
                *atoms++ = a[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ball_scale(void *mols, size_t n, float32_t *bscales)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ball_scale, bscales);
}

extern "C" EXPORT void set_structure_ball_scale(void *mols, size_t n, float32_t *bscales)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set(m, n, &Structure::set_ball_scale, bscales);
}
extern "C" EXPORT void structure_num_bonds(void *mols, size_t n, size_t *nbonds)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::num_bonds, nbonds);
}

extern "C" EXPORT void structure_num_bonds_visible(void *mols, size_t n, size_t *nbonds)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
          {
            const Structure::Bonds &bonds = m[i]->bonds();
            int c = 0;
            for (auto b: bonds)
              if (b->shown())
                c += 1;
            nbonds[i] = c;
          }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_bonds(void *mols, size_t n, pyobject_t *bonds)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Structure::Bonds &b = m[i]->bonds();
            for (size_t j = 0; j != b.size(); ++j)
                *bonds++ = b[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_num_ribbon_residues(void *mols, size_t n, size_t *nres)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::num_ribbon_residues, nres);
}

extern "C" EXPORT void structure_num_residues(void *mols, size_t n, size_t *nres)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::num_residues, nres);
}

extern "C" EXPORT void structure_residues(void *mols, size_t n, pyobject_t *res)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Structure::Residues &r = m[i]->residues();
            for (size_t j = 0; j != r.size(); ++j)
                *res++ = r[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *structure_rings(void *mol, bool cross_residue, int all_size_threshold)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        auto& rings = m->rings(cross_residue, all_size_threshold);
        const Ring **ra;
        PyObject *r_array = python_voidp_array(rings.size(), (void***)&ra);
        size_t i = 0;
        for (auto& r: rings)
            ra[i++] = &r;
        return r_array;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void structure_active_coordset(void *mols, size_t n, pyobject_t *resp)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            resp[i] = (pyobject_t*)(m[i]->active_coord_set());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_active_coordset_id(void *mols, size_t n, int32_t *coordset_ids)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
      for (size_t i = 0; i != n; ++i) {
        CoordSet *cs = m[i]->active_coord_set();
        coordset_ids[i] = (cs ? cs->id() : -1);
      }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_structure_active_coordset_id(void *mols, size_t n, int32_t *coordset_ids)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            CoordSet *cs = m[i]->find_coord_set(coordset_ids[i]);
            if (cs == NULL)
              PyErr_Format(PyExc_IndexError, "No coordset id %d", coordset_ids[i]);
            else
              m[i]->set_active_coord_set(cs);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_add_coordset(void *mol, int id, void *xyz, size_t n)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        CoordSet *cs = m->new_coord_set(id);
        cs->set_coords((double *)xyz, n);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_remove_coordsets(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->clear_coord_sets();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_add_coordsets(void *mol, bool replace, void *xyz, size_t n_sets, size_t n_coords)
{
    Structure *m = static_cast<Structure *>(mol);
    double* xyzs = (double*)xyz;
    try {
        if (replace)
            m->clear_coord_sets();
        for (size_t i = 0; i < n_sets; ++i) {
            CoordSet *cs = m->new_coord_set();
            cs->set_coords((double *)xyzs, n_coords);
            xyzs += n_coords * 3;
        }
        if (replace)
            m->set_active_coord_set(m->coord_sets()[0]);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *structure_ribbon_orient(void *mol, void *residues, size_t n)
{
    Structure *m = static_cast<Structure *>(mol);
    Residue **r = static_cast<Residue **>(residues);
    PyObject *o = NULL;
    try {
        std::vector<int> orients;
        for (size_t i = 0; i != n; ++i)
            orients.push_back(m->ribbon_orient(r[i]));
        o = c_array_to_python(orients);
    } catch (...) {
        molc_error();
    }
    return o;
}

extern "C" EXPORT void structure_coordset_ids(void *mols, size_t n, int32_t *coordset_ids)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i)
          for (auto cs: m[i]->coord_sets())
            *coordset_ids++ = cs->id();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_coordset_size(void *mols, size_t n, int32_t *coordset_size)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
      for (size_t i = 0; i != n; ++i) {
        CoordSet *cs = m[i]->active_coord_set();
        *coordset_size++ = (cs ? cs->coords().size() : 0);
      }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_num_coordsets(void *mols, size_t n, size_t *ncoord_sets)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::num_coord_sets, ncoord_sets);
}

extern "C" EXPORT void structure_num_chains(void *mols, size_t n, size_t *nchains)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::num_chains, nchains);
}

extern "C" EXPORT void structure_chains(void *mols, size_t n, pyobject_t *chains)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Structure::Chains &c = m[i]->chains();
            for (size_t j = 0; j != c.size(); ++j)
                *chains++ = c[j];
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_change_tracker(void *mols, size_t n, pyobject_t *trackers)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i) {
            trackers[i] = m[i]->change_tracker()->py_instance(true);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ribbon_tether_scale(void *mols, size_t n, float32_t *ribbon_tether_scale)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_tether_scale, ribbon_tether_scale);
}

extern "C" EXPORT void set_structure_ribbon_tether_scale(void *mols, size_t n, float32_t *ribbon_tether_scale)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set(m, n, &Structure::set_ribbon_tether_scale, ribbon_tether_scale);
}

extern "C" EXPORT void structure_ribbon_tether_shape(void *mols, size_t n, int32_t *ribbon_tether_shape)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_tether_shape, ribbon_tether_shape);
}

extern "C" EXPORT void set_structure_ribbon_tether_shape(void *mols, size_t n, int32_t *ribbon_tether_shape)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            m[i]->set_ribbon_tether_shape(static_cast<Structure::TetherShape>(ribbon_tether_shape[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ribbon_tether_sides(void *mols, size_t n, int32_t *ribbon_tether_sides)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_tether_sides, ribbon_tether_sides);
}

extern "C" EXPORT void set_structure_ribbon_tether_sides(void *mols, size_t n, int32_t *ribbon_tether_sides)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set(m, n, &Structure::set_ribbon_tether_sides, ribbon_tether_sides);
}

extern "C" EXPORT void structure_ribbon_tether_opacity(void *mols, size_t n, float32_t *ribbon_tether_opacity)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_tether_opacity, ribbon_tether_opacity);
}

extern "C" EXPORT void set_structure_ribbon_tether_opacity(void *mols, size_t n, float32_t *ribbon_tether_opacity)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set(m, n, &Structure::set_ribbon_tether_opacity, ribbon_tether_opacity);
}

extern "C" EXPORT void structure_ribbon_orientation(void *mols, size_t n, int32_t *ribbon_orientation)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_orientation, ribbon_orientation);
}

extern "C" EXPORT void set_structure_ribbon_orientation(void *mols, size_t n, int32_t *ribbon_orientation)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            m[i]->set_ribbon_orientation(static_cast<Structure::RibbonOrientation>(ribbon_orientation[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ribbon_mode_helix(void *mols, size_t n, int32_t *ribbon_mode_helix)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_mode_helix, ribbon_mode_helix);
}

extern "C" EXPORT void set_structure_ribbon_mode_helix(void *mols, size_t n, int32_t *ribbon_mode_helix)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            m[i]->set_ribbon_mode_helix(static_cast<Structure::RibbonMode>(ribbon_mode_helix[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ribbon_mode_strand(void *mols, size_t n, int32_t *ribbon_mode_strand)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_mode_strand, ribbon_mode_strand);
}

extern "C" EXPORT void set_structure_ribbon_mode_strand(void *mols, size_t n, int32_t *ribbon_mode_strand)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            m[i]->set_ribbon_mode_strand(static_cast<Structure::RibbonMode>(ribbon_mode_strand[i]));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ribbon_show_spine(void *mols, size_t n, npy_bool *ribbon_show_spine)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_show_spine, ribbon_show_spine);
}

extern "C" EXPORT void set_structure_ribbon_show_spine(void *mols, size_t n, npy_bool *ribbon_show_spine)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set(m, n, &Structure::set_ribbon_show_spine, ribbon_show_spine);
}

extern "C" EXPORT void set_structure_ss_assigned(void *structures, size_t n, npy_bool *ss_assigned)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_set(s, n, &Structure::set_ss_assigned, ss_assigned);
}

extern "C" EXPORT void set_structure_display(void *structures, size_t n, npy_bool *display)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_set(s, n, &Structure::set_display, display);
}

extern "C" EXPORT PyObject* structure_bonded_groups(void *structure, bool consider_missing_structure)
{
    Structure *s = static_cast<Structure *>(structure);
    std::vector<std::vector<Atom*>> groups;
    try {
        s->bonded_groups(&groups, consider_missing_structure);
        PyObject* grps_list = PyList_New(groups.size());
        if (grps_list == nullptr)
            throw std::bad_alloc();
        int grps_i = 0;
        for (auto grp: groups) {
            PyObject* grp_list = PyList_New(grp.size());
            PyList_SET_ITEM(grps_list, grps_i++, grp_list);
            int grp_i = 0;
            for (auto atom: grp) {
                PyList_SET_ITEM(grp_list, grp_i++, PyLong_FromVoidPtr(atom));
            }
        }
        return grps_list;
    } catch (...) {
        molc_error();
        return NULL;
    }
}

extern "C" EXPORT void structure_change_chain_ids(void *structure, PyObject *py_chains, PyObject *py_chain_ids, bool non_polymeric)
{
    Structure *s = static_cast<Structure *>(structure);
    std::vector<StructureSeq*> changing;
    std::vector<ChainID> chain_ids;
    auto size = PyList_GET_SIZE(py_chains);
    try {
        if (PyList_GET_SIZE(py_chain_ids) != size)
            throw std::logic_error("Chain ID list must be same size as chain list");
        for (int i = 0; i < size; ++i) {
            changing.push_back(
                static_cast<StructureSeq*>(PyLong_AsVoidPtr(PyList_GET_ITEM(py_chains, i))));
            chain_ids.push_back(static_cast<ChainID>(CheckedPyUnicode_AsUTF8(PyList_GET_ITEM(py_chain_ids, i))));
        }
        s->change_chain_ids(changing, chain_ids, non_polymeric);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_renumber_residues(void *structure, PyObject *py_residues, int start)
{
    Structure *s = static_cast<Structure *>(structure);
    std::vector<Residue*> renumbered;
    auto size = PyList_GET_SIZE(py_residues);
    for (int i = 0; i < size; ++i) {
        renumbered.push_back(
            static_cast<Residue*>(PyLong_AsVoidPtr(PyList_GET_ITEM(py_residues, i))));
    }
    try {
        s->renumber_residues(renumbered, start);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_reorder_residues(void *structure, PyObject *py_new_order)
{
    Structure *s = static_cast<Structure *>(structure);
    Structure::Residues new_order;
    auto size = PyList_GET_SIZE(py_new_order);
    for (int i = 0; i < size; ++i) {
        new_order.push_back(
            static_cast<Residue*>(PyLong_AsVoidPtr(PyList_GET_ITEM(py_new_order, i))));
    }
    try {
        s->reorder_residues(new_order);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ribbon_display_count(void *mols, size_t n, int32_t *ribbon_display_count)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ribbon_display_count, ribbon_display_count);
}

extern "C" EXPORT void structure_ring_display_count(void *mols, size_t n, int32_t *ring_display_count)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::ring_display_count, ring_display_count);
}

extern "C" EXPORT void structure_pbg_map(void *mols, size_t n, pyobject_t *pbgs)
{
    Structure **m = static_cast<Structure **>(mols);
    PyObject* pbg_map = NULL;
    try {
        for (size_t i = 0; i != n; ++i) {
            pbg_map = PyDict_New();
            for (auto grp_info: m[i]->pb_mgr().group_map()) {
                PyObject* name = unicode_from_string(grp_info.first.c_str());
                PyObject *pbg = PyLong_FromVoidPtr(grp_info.second);
                PyDict_SetItem(pbg_map, name, pbg);
                Py_DECREF(name);
                Py_DECREF(pbg);
            }
            pbgs[i] = pbg_map;
            pbg_map = NULL;
        }
    } catch (...) {
        Py_XDECREF(pbg_map);
        molc_error();
    }
}

extern "C" EXPORT const char *structure_PBG_METAL_COORDINATION()
{
    return Structure::PBG_METAL_COORDINATION;
}
extern "C" EXPORT const char *structure_PBG_MISSING_STRUCTURE()
{
    return Structure::PBG_MISSING_STRUCTURE;
}
extern "C" EXPORT const char *structure_PBG_HYDROGEN_BONDS()
{
    return Structure::PBG_HYDROGEN_BONDS;
}

extern "C" EXPORT PyObject *structure_pseudobond_group(void *mol, const char *name, int create_type)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        Proxy_PBGroup *pbg = m->pb_mgr().get_group(name, create_type);
        if (pbg == nullptr) {
            Py_RETURN_NONE;
        }
        return pbg->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void structure_delete_pseudobond_group(void *mol, void *pbgroup)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        Proxy_PBGroup* pbg = static_cast<Proxy_PBGroup*>(pbgroup);
        m->pb_mgr().delete_group(pbg);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject* structure_py_obj_coordset(void* ptr, int csid)
{
    Structure *s = static_cast<Structure*>(ptr);
    try {
        CoordSet* cs = s->find_coord_set(csid);
        if (cs == nullptr) // raise IndexError
            throw std::out_of_range("No such coordset");
        return cs->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT size_t structure_session_atom_to_id(void *mol, void* atom)
{
    Structure *m = static_cast<Structure *>(mol);
    Atom *a = static_cast<Atom *>(atom);
    try {
        return (*m->session_save_atoms)[a];
    } catch (...) {
        molc_error();
        return -1;
    }
}

extern "C" EXPORT size_t structure_session_bond_to_id(void *mol, void* bond)
{
    Structure *m = static_cast<Structure *>(mol);
    Bond *b = static_cast<Bond *>(bond);
    try {
        return (*m->session_save_bonds)[b];
    } catch (...) {
        molc_error();
        return -1;
    }
}

extern "C" EXPORT size_t structure_session_chain_to_id(void *mol, void* chain)
{
    Structure *m = static_cast<Structure *>(mol);
    Chain *c = static_cast<Chain *>(chain);
    try {
        return (*m->session_save_chains)[c];
    } catch (...) {
        molc_error();
        return -1;
    }
}

extern "C" EXPORT size_t structure_session_residue_to_id(void *mol, void* res)
{
    Structure *m = static_cast<Structure *>(mol);
    Residue *r = static_cast<Residue *>(res);
    try {
        return (*m->session_save_residues)[r];
    } catch (...) {
        molc_error();
        return -1;
    }
}

extern "C" EXPORT void* structure_session_id_to_atom(void *mol, size_t i)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->atoms()[i];
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void* structure_session_id_to_bond(void *mol, size_t i)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->bonds()[i];
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void* structure_session_id_to_chain(void *mol, size_t i)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->chains()[i];
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void* structure_session_id_to_residue(void *mol, size_t i)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->residues()[i];
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT int structure_session_info(void *mol, PyObject *ints, PyObject *floats, PyObject *misc)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->session_info(ints, floats, misc);
    } catch (...) {
        molc_error();
        return -1;
    }
}

extern "C" EXPORT void structure_session_restore(void *mol, int version,
    PyObject *ints, PyObject *floats, PyObject *misc)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->session_restore(version, ints, floats, misc);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_session_restore_setup(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
            m->session_restore_setup();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_session_restore_teardown(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
            m->session_restore_teardown();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_session_save_setup(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
            m->session_save_setup();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_session_save_teardown(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
            m->session_save_teardown();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_set_position(void *mol, void *pos)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->set_position_matrix((double*)pos);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_ss_assigned(void *structures, size_t n, npy_bool *ss_assigned)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_get(s, n, &Structure::ss_assigned, ss_assigned);
}

extern "C" EXPORT void structure_display(void *structures, size_t n, npy_bool *display)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_get(s, n, &Structure::display, display);
}

extern "C" EXPORT void structure_start_change_tracking(void *mol, void *vct)
{
    Structure *m = static_cast<Structure *>(mol);
    ChangeTracker* ct = static_cast<ChangeTracker*>(vct);
    try {
            m->start_change_tracking(ct);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_use_default_atom_radii(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->use_default_atom_radii();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *structure_molecules(void *mol)
{
    Structure *s = static_cast<Structure *>(mol);
    PyObject *mols = NULL;
    try {
        std::vector<std::vector<Atom*>> molecules;
        s->bonded_groups(&molecules, true);
        mols = PyTuple_New(molecules.size());
        size_t p = 0;
        for (auto atomvec: molecules) {
            void **aa;
            PyObject *a_array = python_voidp_array(atomvec.size(), &aa);
            size_t i = 0;
            for (auto a: atomvec)
                aa[i++] = static_cast<void *>(a);
            PyTuple_SetItem(mols, p++, a_array);
        }
        return mols;
    } catch (...) {
        Py_XDECREF(mols);
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *structure_nonstandard_residue_names(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    auto rnames = m->nonstd_res_names();
    PyObject* py_rnames = PyList_New(rnames.size());
    size_t index = 0;
    for (auto rname: rnames) {
        PyList_SET_ITEM(py_rnames, index++, unicode_from_string(rname));
    }
    return py_rnames;
}

extern "C" EXPORT PyObject *structure_polymers(void *mol, int missing_structure_treatment, int consider_chains_ids)
{
    Structure *m = static_cast<Structure *>(mol);
    PyObject *poly = nullptr;
    try {
        std::vector<std::pair<Chain::Residues,PolymerType>> polymers = m->polymers(static_cast<Structure::PolymerMissingStructure>(missing_structure_treatment), consider_chains_ids);
        poly = PyList_New(polymers.size());
        size_t p = 0;
        for (auto residues_ptype: polymers) {
            auto& resvec = residues_ptype.first;
            auto pt = residues_ptype.second;
            void **ra;
            PyObject *r_array = python_voidp_array(resvec.size(), &ra);
            size_t i = 0;
            for (auto r: resvec)
                ra[i++] = static_cast<void *>(r);
            PyObject *ptype = PyLong_FromLong((long)pt);
            PyObject *vals = PyTuple_New(2);
            PyTuple_SET_ITEM(vals, 0, r_array);
            PyTuple_SET_ITEM(vals, 1, ptype);
            PyList_SET_ITEM(poly, p++, vals);
        }
        return poly;
    } catch (...) {
        Py_XDECREF(poly);
        molc_error();
        return nullptr;
    }
}

inline static bool chain_trace_connection(const Residue *r0, const Residue *r1, PolymerType ptype,
                      const AtomName &trace_atom, const AtomName &connect_atom_0,
                      const AtomName &connect_atom_1, Atom **atom0, Atom **atom1)
{
  if (r0->polymer_type() != ptype || r1->polymer_type() != ptype || !r0->connects_to(r1))
    return false;
  if (r0->ribbon_display() && r1->ribbon_display())
    return false;
  Atom *ta0 = r0->find_atom(trace_atom);
  if (ta0 == NULL || !ta0->display() || ta0->hide())
    return false;
  Atom *c0 = r0->find_atom(connect_atom_0);
  if (c0 && c0->display())
    return false;
  Atom *c1 = (ptype == PT_AMINO ? r1 : r0)->find_atom(connect_atom_1);
  if (c1 && c1->display())
    return false;
  Atom *ta1 = r1->find_atom(trace_atom);
  if (ta1 == NULL || !ta1->display() || ta1->hide())
    return false;
  *atom0 = ta0;
  *atom1 = ta1;
  return true;
}

extern "C" EXPORT PyObject *structure_chain_trace_atoms(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    PyObject *atom_pairs;
    try {
      // Find neighbor CA that are shown but intervening C and N are not shown.
      const Structure::Residues &res = m->residues();
      size_t nr = res.size();
      std::vector<Atom *> cta0, cta1;
      Atom *ta0, *ta1;
      for (size_t i = 0 ; i < nr-1 ; ++i) {
        Residue *r0 = res[i], *r1 = res[i+1];
        if (chain_trace_connection(r0, r1, PT_AMINO, "CA", "C", "N", &ta0, &ta1) ||
            chain_trace_connection(r0, r1, PT_NUCLEIC, "P", "O5'", "O3'", &ta0, &ta1)) {
          cta0.push_back(ta0);
          cta1.push_back(ta1);
        }
      }
      int na = cta0.size();
      if (na == 0)
        atom_pairs = python_none();
      else {
        void **ap0, **ap1;
        PyObject *a0 = python_voidp_array(cta0.size(), &ap0);
        PyObject *a1 = python_voidp_array(cta1.size(), &ap1);
        for (int i = 0 ; i < na ; ++i) {
          ap0[i] = static_cast<void *>(cta0[i]);
          ap1[i] = static_cast<void *>(cta1[i]);
        }
        atom_pairs = python_tuple(a0, a1);
      }
    } catch (...) {
        molc_error();
    }
    return atom_pairs;
}

extern "C" EXPORT void *structure_new(PyObject* logger)
{
    try {
        Structure *g = new Structure(logger);
        return g;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void *atomic_structure_new(PyObject* logger)
{
    try {
        AtomicStructure *m = new AtomicStructure(logger);
        return m;
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void structure_combine_sym_atoms(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->combine_sym_atoms();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_delete_alt_locs(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->delete_alt_locs();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_delete(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        delete m;
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *structure_new_atom(void *mol, const char *atom_name, void *element)
{
    Structure *m = static_cast<Structure *>(mol);
    Element *e = static_cast<Element *>(element);
    try {
        Atom *a = m->new_atom(atom_name, *e);
        return a->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void structure_delete_atom(void *mol, void *atom)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->delete_atom(static_cast<Atom *>(atom));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_delete_bond(void *mol, void *bond)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->delete_bond(static_cast<Bond *>(bond));
    } catch (...) {
        molc_error();
    }
}


extern "C" EXPORT void structure_delete_residue(void *mol, void *res)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->delete_residue(static_cast<Residue *>(res));
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *structure_new_bond(void *mol, void *atom1, void *atom2)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        Bond *b = m->new_bond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2));
        return b->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void structure_new_coordset_default(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->new_coord_set();
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_new_coordset_index(void *mol, int32_t index)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->new_coord_set(index);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_new_coordset_index_size(void *mol, int32_t index, int32_t size)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->new_coord_set(index, size);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject *structure_new_residue(void *mol, const char *residue_name, const char *chain_id, int pos, char insert, void* precedes)
{
    Structure *m = static_cast<Structure *>(mol);
    Residue *nb = static_cast<Residue *>(precedes);
    try {
        Residue *r = m->new_residue(residue_name, chain_id, pos, insert, nb, false);
        return r->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT PyObject *structure_find_residue(void *mol, const char *chain_id, int pos, char insert)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        Residue *r = m->find_residue(chain_id, pos, insert);
        if (r == nullptr) {
            Py_RETURN_NONE;
        }
        return r->py_instance(true);
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void metadata(void *mols, size_t n, pyobject_t *headers)
{
    Structure **m = static_cast<Structure **>(mols);
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

extern "C" EXPORT void set_metadata_entry(void* mols, size_t n, PyObject* key, PyObject* values)
{
    if (!PyUnicode_Check(key)) {
        PyErr_Format(PyExc_ValueError, "Expected key to be a string");
        return;
    }
    PyObject* fast_values = PySequence_Fast(values, "Expected values to be a sequence");
    if (fast_values == NULL)
        return;
    try {
        std::vector<std::string> cpp_values;
        Py_ssize_t size = PySequence_Fast_GET_SIZE(fast_values);
        cpp_values.reserve(size);
        PyObject **fast_array = PySequence_Fast_ITEMS(fast_values);
        for (auto i = 0; i < size; ++i) {
            if (!PyUnicode_Check(fast_array[i]))
                throw std::logic_error("Expected values to be sequence of strings");
            cpp_values.push_back(string_from_unicode(fast_array[i]));
        }
        std::string cpp_key = string_from_unicode(key);
        Structure **m = static_cast<Structure **>(mols);
        for (size_t i = 0; i < n; ++i) {
            if (m == nullptr)
                continue;
            auto& metadata = m[i]->metadata;
            metadata[cpp_key] = cpp_values;
        }
    } catch (...) {
        molc_error();
    }
    Py_DECREF(fast_values);
}

extern "C" EXPORT void pdb_version(void *mols, size_t n, int32_t *version)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            version[i] = m[i]->pdb_version;
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_pdb_version(void *mols, size_t n, int32_t *version)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            m[i]->pdb_version = version[i];
    } catch (...) {
        molc_error();
    }
}


// -------------------------------------------------------------------------
// element functions
//
extern "C" EXPORT float element_bond_length(void *element1, void *element2)
{
    Element *e1 = static_cast<Element *>(element1);
    Element *e2 = static_cast<Element *>(element2);
    try {
        return Element::bond_length(*e1, *e2);
    } catch (...) {
        molc_error();
        return 0.0;
    }
}

extern "C" EXPORT float element_bond_radius(void *element)
{
    Element *e = static_cast<Element *>(element);
    try {
        return Element::bond_radius(*e);
    } catch (...) {
        molc_error();
        return 0.0;
    }
}

extern "C" EXPORT void element_name(void *elements, size_t n, pyobject_t *names)
{
    Element **e = static_cast<Element **>(elements);
    try {
        for (size_t i = 0; i != n; ++i)
            names[i] = PyUnicode_FromString(e[i]->name());
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT PyObject* element_names()
{
    PyObject* e_names = NULL;
    try {
        e_names = pysupport::cset_of_chars_to_pyset(Element::names(), "element names");
    } catch (...) {
        molc_error();
    }
    return e_names;
}

extern "C" EXPORT void element_number(void *elements, size_t n, uint8_t *number)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::number, number);
}

extern "C" EXPORT size_t element_NUM_SUPPORTED_ELEMENTS()
{
    return static_cast<size_t>(Element::NUM_SUPPORTED_ELEMENTS);
}

extern "C" EXPORT void element_mass(void *elements, size_t n, float *mass)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::mass, mass);
}

extern "C" EXPORT void *element_number_get_element(int en)
{
    try {
        return (void*)(&Element::get_element(en));
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void *element_name_get_element(const char *en)
{
    try {
        return (void*)(&Element::get_element(en));
    } catch (...) {
        molc_error();
        return nullptr;
    }
}

extern "C" EXPORT void element_is_alkali_metal(void *elements, size_t n, npy_bool *a_metal)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::is_alkali_metal, a_metal);
}

extern "C" EXPORT void element_is_halogen(void *elements, size_t n, npy_bool *halogen)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::is_halogen, halogen);
}

extern "C" EXPORT void element_is_metal(void *elements, size_t n, npy_bool *metal)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::is_metal, metal);
}

extern "C" EXPORT void element_is_noble_gas(void *elements, size_t n, npy_bool *ngas)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::is_noble_gas, ngas);
}

extern "C" EXPORT void element_valence(void *elements, size_t n, uint8_t *valence)
{
    Element **e = static_cast<Element **>(elements);
    error_wrap_array_get(e, n, &Element::valence, valence);
}

// -------------------------------------------------------------------------
// initialization functions
//
static void *init_numpy()
{
    import_array(); // Initialize use of numpy
    return NULL;
}

// ---------------------------------------------------------------------------
// array updater functions
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
            // TODO: Resizing the array in place is not possible from Python numpy API,
            // so this breaks assumptions about numpy.  Cause of subtle ChimeraX bug #1096.
            /*
            // Numpy array can't be resized in place with PyArray_Resize() if references to
            // the array (different views?) exist as described in PyArray_Resize() documentation.
            // This is because PyArray_Resize() reallocates the array to the new size.
            // Won't work anyways because array will reallocate while looping over old array
            // of atoms being deleted.
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

extern "C" EXPORT void remove_deleted_c_pointers(PyObject *numpy_array)
{
    try {
        if (array_updater == NULL)
            array_updater = new Array_Updater();

        array_updater->add_array(numpy_array);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pointer_array_freed(void *numpy_array)
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


// -------------------------------------------------------------------------
// pointer array functions
extern "C" EXPORT ssize_t pointer_index(void *pointer_array, size_t n, void *pointer)
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

extern "C" EXPORT void pointer_mask(void *pointer_array, size_t n, void *pointer_array2, size_t n2, unsigned char *mask)
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

extern "C" EXPORT void pointer_indices(void *pointer_array, size_t n, void *pointer_array2, size_t n2, int *indices)
{
    void **pa = static_cast<void **>(pointer_array);
    void **pa2 = static_cast<void **>(pointer_array2);
    try {
        std::map<void *,int> s;
        for (size_t i = 0; i != n2; ++i)
            s[pa2[i]] = i;
        for (size_t i = 0; i != n; ++i) {
            std::map<void *,int>::iterator si = s.find(pa[i]);
            indices[i] = (si == s.end() ? -1 : si->second);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT bool pointer_intersects(void *pointer_array, size_t n, void *pointer_array2, size_t n2)
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

extern "C" EXPORT void pointer_intersects_each(void *pointer_arrays, size_t na, size_t *sizes,
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

typedef std::map<void *, int> PointerTable;
extern "C" EXPORT void *pointer_table_create(void *pointer_array, size_t n)
{
    void **pa = static_cast<void **>(pointer_array);
    PointerTable *t = new PointerTable;
    try {
      for (int i = n-1; i >= 0; --i)
	(*t)[pa[i]] = i;
    } catch (...) {
        molc_error();
    }
    return t;
}

extern "C" EXPORT void pointer_table_delete(void *pointer_table)
{
    PointerTable *t = static_cast<PointerTable *>(pointer_table);
    try {
        delete t;
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT bool pointer_table_includes_any(void *pointer_table, void *pointer_array, size_t n)
{
    PointerTable *t = static_cast<PointerTable *>(pointer_table);
    void **pa = static_cast<void **>(pointer_array);
    try {
        for (size_t i = 0; i < n; ++i)
	  if (t->find(pa[i]) != t->end())
	    return true;
    } catch (...) {
        molc_error();
    }
    return false;
}

extern "C" EXPORT void pointer_table_includes_each(void *pointer_table, void *pointer_array, size_t n,
						   unsigned char *mask)
{
    PointerTable *t = static_cast<PointerTable *>(pointer_table);
    void **pa = static_cast<void **>(pointer_array);
    try {
        for (size_t i = 0; i < n; ++i)
	  mask[i] = (t->find(pa[i]) != t->end() ? 1 : 0);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void pointer_table_indices(void *pointer_table, void *pointer_array, size_t n,
					     int *indices)
{
    PointerTable *t = static_cast<PointerTable *>(pointer_table);
    void **pa = static_cast<void **>(pointer_array);
    try {
      for (size_t i = 0; i < n; ++i) {
	PointerTable::iterator ti = t->find(pa[i]);
	indices[i] = (ti == t->end() ? -1 : ti->second);
      }
    } catch (...) {
        molc_error();
    }
}

// inform C++ about relevant class objects
//
#include "ctypes_pyinst.h"
SET_PYTHON_CLASS(atom, Atom)
SET_PYTHON_CLASS(bond, Bond)
SET_PYTHON_CLASS(changetracker, ChangeTracker)
SET_PYTHON_CLASS(coordset, CoordSet)
SET_PYTHON_CLASS(element, Element)
SET_PYTHON_CLASS(pseudobond, Pseudobond)
SET_PYTHON_CLASS(pseudobondgroup, PBGroup)
SET_PYTHON_CLASS(residue, Residue)
SET_PYTHON_CLASS(ring, Ring)

SET_PYTHON_INSTANCE(changetracker, ChangeTracker)
SET_PYTHON_INSTANCE(pseudobondgroup, Proxy_PBGroup)
SET_PYTHON_INSTANCE(sequence, Sequence)
SET_PYTHON_INSTANCE(structure, Structure)

GET_PYTHON_INSTANCES(atom, Atom)
GET_PYTHON_INSTANCES(bond, Bond)
GET_PYTHON_INSTANCES(chain, Chain)
GET_PYTHON_INSTANCES(changetracker, ChangeTracker)
GET_PYTHON_INSTANCES(coordset, CoordSet)
GET_PYTHON_INSTANCES(element, Element)
GET_PYTHON_INSTANCES(pseudobond, Pseudobond)
GET_PYTHON_INSTANCES(pseudobondgroup, Proxy_PBGroup)
GET_PYTHON_INSTANCES(residue, Residue)
GET_PYTHON_INSTANCES(ring, Ring)
GET_PYTHON_INSTANCES(sequence, Sequence)
GET_PYTHON_INSTANCES(structure, Structure)
GET_PYTHON_INSTANCES(structureseq, StructureSeq)

#include <pyinstance/PythonInstance.declare.h>
extern "C" EXPORT PyObject *python_instances_of_class(PyObject* cls)
{
    PyObject *obj_list = nullptr;
    try {
        obj_list = PyList_New(0);
        for (auto ptr_obj: pyinstance::_pyinstance_object_map) {
            auto is_inst = PyObject_IsInstance(ptr_obj.second, cls);
            if (is_inst < 0)
                return nullptr;
            if (!is_inst)
                continue;
            if (PyList_Append(obj_list, ptr_obj.second) < 0)
                return nullptr;
        }
        return obj_list;
    } catch (...) {
        Py_XDECREF(obj_list);
        molc_error();
        return nullptr;
    }
}
