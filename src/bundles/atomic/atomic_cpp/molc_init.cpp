// vi: set expandtab shiftwidth=4 softtabstop=4:

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


#include "molc.h"

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
extern "C" EXPORT Py_ssize_t pointer_index(void *pointer_array, size_t n, void *pointer)
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
      if (n > INT_MAX)
	throw std::range_error("pointer_table_create: array size exceeds maximum integer size");
      for (size_t i = 0; i < n; ++i)
	(*t)[pa[i]] = (int)i;
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

// need a special setter for Sequences created in the Python layer that does not add a reference
extern "C" EXPORT void set_pysequence_py_instance(void* pysequence, PyObject* py_inst)
{
   Sequence *s = static_cast<Sequence *>(pysequence);
   try {
       s->set_py_instance(py_inst);
       Py_DECREF(py_inst);
   } catch (...) {
       molc_error();
   }
}

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
