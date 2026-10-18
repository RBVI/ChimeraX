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
    PyObject *ints, PyObject *floats, PyObject *misc, bool combine)
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
        mgr->session_restore(version, &int_array, &float_array, misc, combine);
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
