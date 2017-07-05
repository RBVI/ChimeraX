// vi: set expandtab ts=4 sw=4:

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

#ifndef atomstruct_Ring
#define atomstruct_Ring

#include <set>

namespace atomstruct {

class Bond;

class ATOMSTRUCT_IMEX Ring {
public:
    typedef std::set<Atom*>  Atoms;
    typedef std::set<Bond*>  Bonds;
private:
    Bonds  _bonds;
    mutable Atoms  _atoms;
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
