// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_Ring
#define atomstruct_Ring

#include <pyinstance/PythonInstance.declare.h>
#include <set>

namespace atomstruct {

class Bond;
class Structure;

class ATOMSTRUCT_IMEX Ring: public pyinstance::PythonInstance<Ring> {
public:
    typedef std::set<Atom*>  Atoms;
    typedef std::set<Bond*>  Bonds;
private:
    Bonds  _bonds;
    mutable Atoms  _atoms;
    bool _temporary;
    friend class Structure; // to access _temporary_rings
    static bool _temporary_rings;
public:
    Ring(std::set<Bond*>& ring_bonds);
    ~Ring();

    void  add_bond(Bond* element);
    void  remove_bond(Bond* element);
    const Bonds&  bonds() const { return _bonds; }
    Bonds::size_type  size() const { return _bonds.size(); }

    // Only bonds, not atoms, are stored "naturally" in the ring.
    // Nonetheless, it is convenient to get the atoms easily...
    const Atoms&  atoms() const;

    // atoms()/bonds() don't return their values in ring order;
    // these do...
    const std::vector<Bond*>  ordered_bonds() const;
    const std::vector<Atom*>  ordered_atoms() const;

    bool  aromatic() const;

    bool  operator<(const Ring&) const;
    bool  operator==(const Ring&) const;
    long  hash() const;

    // determine plane equation Ax+By+Cz+D=0 using algorithm in
    // Foley and van Damm 2nd edition, pp. 476-477
    // avgErr is average distance from plane to ring vertex,
    // maxErr is the largest such distance
    void  planarity(double plane_coeffs[4], double* avg_err = nullptr,
      double* max_err = nullptr) const;
};

} // namespace atomstruct

namespace std {

template <> struct hash<atomstruct::Ring>
{
    size_t operator()(const atomstruct::Ring& r) const { return r.hash(); }
};

} // namespace std

#endif  // atomstruct_Ring
