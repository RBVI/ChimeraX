// vi: set expandtab ts=4 sw=4:

#include <utility>  // for pair

#include "Atom.h"
#include "AtomicStructure.h"
#include "CoordSet.h"
#include <basegeom/destruct.h>

namespace atomstruct {

CoordSet::CoordSet(AtomicStructure* as, int cs_id):
    _cs_id(cs_id), _structure(as)
{
}

CoordSet::CoordSet(AtomicStructure* as, int cs_id, int size):
    _cs_id(cs_id), _structure(as)
{
    _coords.reserve(size);
}

CoordSet::~CoordSet()
{
    if (basegeom::DestructionCoordinator::destruction_parent() != _structure)
        _structure->pb_mgr().remove_cs(this);
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
CoordSet::set_bfactor(const Atom *a, float val)
{
    _bfactor_map.insert(std::pair<const Atom *, float>(a, val));
}

void
CoordSet::set_occupancy(const Atom *a, float val)
{
    _occupancy_map.insert(std::pair<const Atom *, float>(a, val));
}

}  // namespace atomstruct
