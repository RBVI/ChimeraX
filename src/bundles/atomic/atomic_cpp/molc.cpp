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

extern "C" EXPORT PyObject *atom_average_ribbon_color(void *atoms, size_t n)
{
    Atom **aa = static_cast<Atom **>(atoms);
    try {
      unsigned char *rgba;
      PyObject *color = python_uint8_array(4, &rgba);
      if (n > 0)
	{
	  double r = 0, g = 0, b = 0, a = 0;
	  for (size_t i = 0; i < n; ++i) {
	    const Rgba &c = aa[i]->residue()->ribbon_color();
	    r += c.r;
	    g += c.g;
	    b += c.b;
	    a += c.a;
	  }
	  rgba[0] = (int)(r/n + .5);
	  rgba[1] = (int)(g/n + .5);
	  rgba[2] = (int)(b/n + .5);
	  rgba[3] = (int)(a/n + .5);
	}
      else
	{ rgba[0] = rgba[1] = rgba[2] = 180; rgba[3] = 255; }
      return color;
    } catch (...) {
        molc_error();
        return 0;
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

extern "C" EXPORT void atom_delete(void *atoms, size_t n, npy_bool compact_coordsets)
{
    Atom **a = static_cast<Atom **>(atoms);
    try {
        std::map<Structure *, std::vector<Atom *>> matoms;
        for (size_t i = 0; i != n; ++i)
            matoms[a[i]->structure()].push_back(a[i]);

        for (auto ma: matoms) {
            // Deleting all the atoms of a parent model can cause
            // child models to be destroyed before this loop gets
            // to them (e.g. parent model of Find Cavities);
            // guard against that
            if (ma.first->py_instance(false) == Py_None)
                continue;
            ma.first->delete_atoms(ma.second, compact_coordsets);
        }
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

extern "C" EXPORT PyObject *structure_frag_sel(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    PyObject *o = NULL;
    std::set<Atom *> current_sel;
    std::set<Atom *> next_layer, prev_layer;
    try {
        for (auto a: m->atoms()) {
            if (a->selected()) {
                current_sel.insert(a);
                prev_layer.insert(a);
            }
        }
        do {
            // lacks the Chimera 1 logic where selected chain-trace pseudobonds
            // select their residues; not sure it's necessary (and can be added if so)
            next_layer.clear();
            for (auto la: prev_layer) {
                for (auto nb: la->neighbors()) {
                    if (la->residue()->chain_id() != nb->residue()->chain_id())
                        continue;
                    if (current_sel.find(nb) != current_sel.end())
                        continue;
                    next_layer.insert(nb);
                }
            }
            current_sel.insert(next_layer.begin(), next_layer.end());
            prev_layer = next_layer;
        } while (!next_layer.empty());

        unsigned char *sels;
        o = python_bool_array(m->atoms().size(), &sels);
        for (auto a: m->atoms())
            *sels++ = current_sel.find(a) != current_sel.end();
    } catch (...) {
        molc_error();
    }
    return o;
}
