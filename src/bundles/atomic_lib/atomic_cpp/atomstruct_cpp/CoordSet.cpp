// vi: set expandtab ts=4 sw=4:

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

#include <utility>  // for pair

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "ChangeTracker.h"
#include "CoordSet.h"
#include "destruct.h"
#include "Structure.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::CoordSet>;

namespace atomstruct {

CoordSet::CoordSet(Structure* as, int cs_id):
    _cs_id(cs_id), _structure(as)
{
    as->change_tracker()->add_created(as, this);
}

CoordSet::CoordSet(Structure* as, int cs_id, int size):
    _cs_id(cs_id), _structure(as)
{
    _coords.reserve(size);
    as->change_tracker()->add_created(as, this);
}

CoordSet::~CoordSet()
{
    if (DestructionCoordinator::destruction_parent() != _structure)
        _structure->pb_mgr().remove_cs(this);
    _structure->change_tracker()->add_deleted(_structure, this);
}

void
CoordSet::set_coords(Real *xyz, size_t n)
{
    size_t nc = _coords.size();
    size_t c = 0;
    for (size_t i = 0 ; i < nc ; ++i, c += 3)
        _coords[i].set_xyz(xyz[c], xyz[c+1], xyz[c+2]);
    for (size_t i = nc ; i < n ; ++i, c += 3)
        add_coord(Coord(xyz[c], xyz[c+1], xyz[c+2]));

    _structure->change_tracker()->add_modified(_structure, this, ChangeTracker::REASON_COORDSET);
    if (_structure->active_coord_set() == this)
        _structure->change_tracker()->add_modified(_structure, _structure,
            ChangeTracker::REASON_SCENE_COORD);
}

float
CoordSet::get_bfactor(const Atom *a) const
{
    std::unordered_map<const Atom *, float>::const_iterator bfi = _bfactor_map.find(a);
    if (bfi == _bfactor_map.end())
        return 0.0;
    return (*bfi).second;
}

float
CoordSet::get_occupancy(const Atom *a) const
{
    std::unordered_map<const Atom *, float>::const_iterator bfi = _occupancy_map.find(a);
    if (bfi == _occupancy_map.end())
        return 1.0;
    return (*bfi).second;
}

void
CoordSet::session_restore(int /*version*/, int** ints, float** floats)
{
    auto& int_ptr = *ints;
    auto& float_ptr = *floats;

    auto& atoms = structure()->atoms();

    auto num_bfactor = *int_ptr++;
    for (decltype(num_bfactor) i = 0; i < num_bfactor; ++i) {
        _bfactor_map[atoms[*int_ptr++]] = *float_ptr++;
    }

    auto num_occupancy = *int_ptr++;
    for (decltype(num_occupancy)i = 0; i < num_occupancy; ++i) {
        _occupancy_map[atoms[*int_ptr++]] = *float_ptr++;
    }

    auto num_coords = *int_ptr++;
    for (decltype(num_coords)i = 0; i < num_coords; ++i) {
        _coords.emplace_back(float_ptr[0], float_ptr[1], float_ptr[2]);
        float_ptr += 3;
    }
}

void
CoordSet::session_save(int** ints, float** floats) const
{
    auto& int_ptr = *ints;
    auto& float_ptr = *floats;
    auto& atom_map = *structure()->session_save_atoms;

    int_ptr[0] = _bfactor_map.size();
    int_ptr++;
    for (auto atom_bf : _bfactor_map) {
        int_ptr[0] = atom_map[atom_bf.first];
        float_ptr[0]  = atom_bf.second;
        int_ptr++; float_ptr++;
    }

    int_ptr[0] = _occupancy_map.size();
    int_ptr++;
    for (auto atom_occ : _occupancy_map) {
        int_ptr[0] = atom_map[atom_occ.first];
        float_ptr[0]  = atom_occ.second;
        int_ptr++; float_ptr++;
    }

    int_ptr[0] = _coords.size();
    int_ptr++;
    for (auto crd: _coords) {
        float_ptr[0] = crd[0];
        float_ptr[1] = crd[1];
        float_ptr[2] = crd[2];
        float_ptr += 3;
    }
}

void
CoordSet::xform(PositionMatrix mat)
{
    size_t nc = _coords.size();
    for (size_t i = 0 ; i < nc ; ++i)
        _coords[i].xform(mat);
}

}  // namespace atomstruct
