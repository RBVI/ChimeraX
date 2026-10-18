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

extern "C" EXPORT void structure_combine(void *combination, void *s, void *pos, PyObject* chain_id_map)
{
    Structure *combo = static_cast<Structure *>(combination);
    Structure *addition = static_cast<Structure *>(s);
    try {
        if (!PyDict_Check(chain_id_map))
            throw std::invalid_argument("Structure.combine() chain_id_map is not a dict!");
        Py_ssize_t index = 0;
        PyObject* from_cid;
        PyObject* to_cid;
        std::map<ChainID, ChainID> cid_mapping;
        while (PyDict_Next(chain_id_map, &index, &from_cid, &to_cid)) {
            cid_mapping[string_from_unicode(from_cid)] = string_from_unicode(to_cid);
        }
        if (pos == nullptr) {
            combo->combine(addition, &cid_mapping);
            return;
        }
        PositionMatrix pm;
        auto dptr = static_cast<double*>(pos);
        for (int row=0; row < 3; ++row)
            for (int col=0; col < 4; ++col)
                pm[row][col] = *dptr++;
        combo->combine(addition, &cid_mapping, pm);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void *structure_break_triangle_waters(void *mol)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        for (auto r: m->residues()) {
            auto& atoms = r->atoms();
            if (atoms.size() != 3)
                continue;
            Atom* o = nullptr;
            Atom* h1 = nullptr;
            Atom* h2 = nullptr;
            for (auto a: atoms) {
                if (a->element() == Element::O) {
                    if (o == nullptr)
                        o = a;
                    else
                        break;
                } else if (a->element() == Element::H) {
                    if (h1 == nullptr)
                        h1 = a;
                    else if (h2 == nullptr)
                        h2 = a;
                    else
                        break;
                } else
                    break;
            }
            if (o != nullptr && h1 != nullptr && h2 != nullptr) {
                for (auto& b: h1->bonds())
                    if (b->other_atom(h1) == h2) {
                        m->delete_bond(b);
                        break;
                    }
            }
        }
    } catch (...) {
        molc_error();
    }
    return nullptr;
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

extern "C" EXPORT void structure_ss_change_notify(void *structures, size_t n, npy_bool *alcn)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_get(s, n, &Structure::ss_change_notify, alcn);
}

extern "C" EXPORT void set_structure_ss_change_notify(void *structures, size_t n, npy_bool *alcn)
{
    Structure **s = static_cast<Structure **>(structures);
    error_wrap_array_set_mutable(s, n, &Structure::set_ss_change_notify, alcn);
}

extern "C" EXPORT void structure_idatm_failed(void *structures, size_t n, npy_bool *failed)
{
    Structure **s = static_cast<Structure **>(structures);
    try {
        for (size_t i = 0; i < n; ++i)
            failed[i] = s[i]->idatm_failed();
    } catch (...) {
        molc_error();
    }
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

extern "C" EXPORT void structure_coordsets(void *mols, size_t n, pyobject_t *coordsets)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i != n; ++i) {
            const Structure::CoordSets &cs = m[i]->coord_sets();
            for (size_t j = 0; j != cs.size(); ++j)
                *coordsets++ = cs[j];
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

extern "C" EXPORT bool structure_res_numbering_valid(void *mol, int res_numbering)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        return m->res_numbering_valid(static_cast<atomstruct::ResNumbering>(res_numbering));
    } catch (...) {
        molc_error();
        return false;
    }
}

extern "C" EXPORT void set_structure_res_numbering_valid(void *mol, int res_numbering, bool valid)
{
    Structure *m = static_cast<Structure *>(mol);
    try {
        m->set_res_numbering_valid(static_cast<atomstruct::ResNumbering>(res_numbering), valid);
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

extern "C" EXPORT void structure_worm_ribbon(void *mols, size_t n, npy_bool *worm_ribbon)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::worm_ribbon, worm_ribbon);
}

extern "C" EXPORT void set_structure_worm_ribbon(void *mols, size_t n, npy_bool *worm_ribbon)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_set(m, n, &Structure::set_worm_ribbon, worm_ribbon);
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

extern "C" EXPORT void structure_res_numbering(void *mols, size_t n, int32_t *res_numbering)
{
    Structure **m = static_cast<Structure **>(mols);
    error_wrap_array_get(m, n, &Structure::res_numbering, res_numbering);
}

extern "C" EXPORT void set_structure_res_numbering(void *mols, size_t n, int32_t *res_numbering)
{
    Structure **m = static_cast<Structure **>(mols);
    try {
        for (size_t i = 0; i < n; ++i)
            m[i]->set_res_numbering(static_cast<atomstruct::ResNumbering>(res_numbering[i]));
    } catch (...) {
        molc_error();
    }
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

static DestructionBatcher* destruction_batcher = nullptr;
extern "C" EXPORT void structure_begin_destructor_batching()
{
    try {
        if (destruction_batcher == nullptr)
            destruction_batcher = new DestructionBatcher(&destruction_batcher);
    } catch (...) {
        molc_error();
    }
}

extern "C" EXPORT void structure_end_destructor_batching()
{
    try {
        if (destruction_batcher != nullptr) {
            delete destruction_batcher;
            destruction_batcher = nullptr;
        }
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
                Py_DECREF(key);
                Py_DECREF(values);
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
    if (values == Py_None) {
        // remove key from metadata
        try {
            std::string cpp_key = string_from_unicode(key);
            Structure **m = static_cast<Structure **>(mols);
            for (size_t i = 0; i < n; ++i) {
                if (m == nullptr)
                    continue;
                auto& metadata = m[i]->metadata;
                metadata.erase(cpp_key);
            }
        } catch (...) {
            molc_error();
        }
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

