// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

#include <math.h>

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "Bond.h"
#include "Coord.h"
#include "destruct.h"
#include "Real.h"
#include "Ring.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::Ring>;

namespace atomstruct {

bool Ring::_temporary_rings = false;

bool
Ring::operator<(const Ring &other) const
{
    if (_bonds.size() < other.bonds().size())
        return true;
    
    if (_bonds.size() > other.bonds().size())
        return false;
    
    for (Bonds::const_iterator oi = other.bonds().begin(),
    i = _bonds.begin(); i != _bonds.end(); ++oi, ++i) {
        if (*i < *oi)
            return true;
        if (*i > *oi)
            return false;
    }

    return false;
}

bool
Ring::operator==(const Ring &other) const
{
    if (_bonds.size() != other.bonds().size())
        return false;
    
    for (Bonds::const_iterator oi = other.bonds().begin(),
    i = _bonds.begin(); i != _bonds.end(); ++oi, ++i) {
        if (*i != *oi)
            return false;
    }

    return true;
}

// determine plane equation Ax+By+Cz+D=0 for ring vertices and the
// average distance to plane and maximum distance
void
Ring::planarity(double ABCD[4], double* avg_err, double* max_err) const
{
    // need to arrange vertices into x/y/z arrays, in bonded order
    Bonds::size_type num_atoms = _bonds.size();
    Bonds remaining = _bonds;
    Bond *cur_bond = *(remaining.begin());
    remaining.erase(remaining.begin());

    std::vector<Coord> coords(num_atoms + 1);
    coords[0] = cur_bond->atoms()[0]->coord();
    coords[1] = cur_bond->atoms()[1]->coord();
    // put first atom on end of list as well to simplify later loops
    coords[num_atoms] = coords[0];
    int cur_xyz = 2;

    Atom *cur_atom = cur_bond->atoms()[1];
    while (remaining.size() > 1) { // don't care about ring-closing bond
        for (auto b: cur_atom->bonds()) {
            if (remaining.find(b) == remaining.end())
                continue;
            remaining.erase(b);
            cur_bond = b;
            cur_atom = b->other_atom(cur_atom);
            coords[cur_xyz++] = cur_atom->coord();
            break;
        }
    }

    Real A = 0, B = 0, C = 0, D = 0;
    Real avg_x = 0, avg_y = 0, avg_z = 0;
    for (Bonds::size_type i = 0; i < num_atoms; ++i) {
        Point &cur = coords[i], &next = coords[i+1];
        Real x = cur[0], y = cur[1], z = cur[2];
        Real nx = next[0], ny = next[1], nz = next[2];
        A += (y - ny) * (z + nz);
        B += (z - nz) * (x + nx);
        C += (x - nx) * (y + ny);
        avg_x += x;
        avg_y += y;
        avg_z += z;
    }
    avg_x /= num_atoms;
    avg_y /= num_atoms;
    avg_z /= num_atoms;
    // original code:
    // Vector N(A, B, C);
    // D = 0 - avg.point().toVector() * N;
    D = 0.0 - (avg_x * A + avg_y * B + avg_z * C);

    // "normalize" the equation
    double normF = sqrt(A * A + B * B + C * C);
    A /= normF;
    B /= normF;
    C /= normF;
    D /= normF;

    ABCD[0] = A;
    ABCD[1] = B;
    ABCD[2] = C;
    ABCD[3] = D;

    // compute errors
    if (avg_err == nullptr)
        return;
    *avg_err = 0;
    *max_err = 0;
    for (Bonds::size_type i = 0; i < num_atoms; ++i) {
        Point &cur = coords[i];
        Real err = fabs(A*cur[0] + B*cur[1] + C*cur[2] + D);
        *avg_err += err;
        if (err > *max_err)
            *max_err = err;
    }
    *avg_err /= num_atoms;
}

const Ring::Atoms&
Ring::atoms() const
{
    if (_atoms.size() == 0) {
        for (auto b: _bonds) {
            _atoms.insert(b->atoms()[0]);
            _atoms.insert(b->atoms()[1]);
        }
    }

    return _atoms;
}

const std::vector<Atom*>
Ring::ordered_atoms() const
{
    std::vector<Atom *> ordered;
    Ring::Bonds raw_bonds(bonds());
    Atom *cur_atom = (*raw_bonds.begin())->atoms()[0];
    raw_bonds.erase(raw_bonds.begin());
    ordered.push_back(cur_atom);
    while (raw_bonds.size() > 0) {
        for (auto i = raw_bonds.begin(); i != raw_bonds.end(); ++i) {
            Bond* b = *i;
            if (b->contains(cur_atom)) {
                cur_atom = b->other_atom(cur_atom);
                ordered.push_back(cur_atom);
                raw_bonds.erase(i);
                break;
            }
        }
    }
    return ordered;
}

const std::vector<Bond*>
Ring::ordered_bonds() const
{
    std::vector<Bond *> ordered;
    Ring::Bonds raw_bonds(bonds());
    Atom *cur_atom = (*raw_bonds.begin())->atoms()[0];
    ordered.push_back(*raw_bonds.begin());
    raw_bonds.erase(raw_bonds.begin());
    while (raw_bonds.size() > 0) {
        for (auto i = raw_bonds.begin(); i != raw_bonds.end(); ++i) {
            Bond* b = *i;
            if (b->contains(cur_atom)) {
                cur_atom = b->other_atom(cur_atom);
                ordered.push_back(b);
                raw_bonds.erase(i);
                break;
            }
        }
    }
    return ordered;
}


bool
Ring::aromatic() const {
    // Python layer performs this function on its own, since idatm_type() may cause the Ring to
    // get deleted!
    for (auto a: atoms()) {
        if (a->element() == Element::C && a->idatm_type() != "Car")
            return false;
    }
    return true;
}


long
Ring::hash() const
{
    // Python compatible hash function
    long value = 0;
    for (auto b: bonds()) {
        long v = reinterpret_cast<long>(b);
        value ^= v;
    }
    if (value == -1)
        value = -2;
    return value;
}

void
Ring::add_bond(Bond* element)
{
    _bonds.insert(element);
}

void
Ring::remove_bond(Bond* element)
{
    _bonds.erase(element);
}

Ring::Ring(std::set<Bond*>& ring_bonds): _temporary(Ring::_temporary_rings)
{
{
    _bonds = ring_bonds;
}
}

Ring::~Ring()
{
    if (!_temporary)
        DestructionUser(this);
}

} // namespace molecule

