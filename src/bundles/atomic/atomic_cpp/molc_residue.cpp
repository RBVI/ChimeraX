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
        atom_delete(atoms.data(), atoms.size(), false);
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

extern "C" EXPORT PyObject *residue_find_existing_atoms(void *residues, size_t n, char *atom_name)
{
    Residue **r = static_cast<Residue **>(residues);
    std::vector<Atom *> atoms;
    try {
      for (size_t i = 0 ; i < n ; ++i) {
	Atom *a = r[i]->find_atom(atom_name);
	if (a)
	  atoms.push_back(a);
      }
      Atom **ap;
      PyObject *a = python_voidp_array(atoms.size(), (void***)&ap);
      for (size_t i = 0 ; i < atoms.size() ; ++i)
	ap[i] = atoms[i];
      return a;
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

extern "C" EXPORT void set_residue_number(void *residues, size_t n, int32_t *num)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_number, num);
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
                        auto py_seq = unicode_from_string(seq);
                        PyList_Append(seqs, py_seq);
                        Py_DECREF(py_seq);
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

extern "C" EXPORT void residue_worm_radius(void *residues, size_t n, float32_t *radii)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_get(r, n, &Residue::worm_radius, radii);
}

extern "C" EXPORT void set_residue_worm_radius(void *residues, size_t n, float32_t *radii)
{
    Residue **r = static_cast<Residue **>(residues);
    error_wrap_array_set(r, n, &Residue::set_worm_radius, radii);
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

extern "C" EXPORT void sseq_description(void *chains, size_t n, pyobject_t *descripts)
{
    StructureSeq **c = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto& descript = c[i]->description();
            if (descript.empty()) {
                descripts[i] = Py_None;
                Py_INCREF(Py_None);
            } else
                descripts[i] = unicode_from_string(descript);
        }
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void set_sseq_description(void *chains, size_t n, pyobject_t *descripts)
{
    StructureSeq **sseq = static_cast<StructureSeq **>(chains);
    try {
        for (size_t i = 0; i != n; ++i) {
            auto descript = static_cast<PyObject *>(descripts[i]);
            if (descript == Py_None)
                sseq[i]->set_description("");
            else
                sseq[i]->set_description(CheckedPyUnicode_AsUTF8(static_cast<PyObject *>(descripts[i])));
        }
    } catch (...) {
        molc_error();
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
        for (size_t i = 0; i < n; ++i) {
          auto s = c[i]->structure();
          if (s == nullptr) {
              molp[i] = Py_None;
              Py_INCREF(Py_None);
          } else
              molp[i] = c[i]->structure()->py_instance(true);
        }
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

extern "C" EXPORT char sequence_is_gap_character(const char *c)
{
    try {
        return Sequence::is_gap_character(c[0]);
    } catch (...) {
        molc_error();
        return true;
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
