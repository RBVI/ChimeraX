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

#ifndef atomstruct_Bond
#define atomstruct_Bond

#include <pyinstance/PythonInstance.declare.h>
#include <set>
#include <vector>

#include "destruct.h"
#include "Connection.h"
#include "imex.h"
#include "session.h"

namespace atomstruct {

class Atom;
class ChangeTracker;
class Residue;
class Ring;
class Structure;

class ATOMSTRUCT_IMEX Bond: public UniqueConnection, public pyinstance::PythonInstance<Bond> {
    friend class Structure;
public:
    // use Atom::HIDE_* constants for hide bits
    typedef std::vector<const Ring*>  Rings;
private:
    Bond(Structure*, Atom*, Atom*, bool);
    void  add_to_atoms() { atoms()[0]->add_bond(this); atoms()[1]->add_bond(this); }
    const char*  err_msg_exists() const
        { return "Bond already exists between these atoms"; }
    const char*  err_msg_loop() const
        { return "Can't bond an atom to itself"; }
    mutable Rings  _rings;

    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }
public:
    virtual ~Bond() {
        DestructionUser(this);
        change_tracker()->add_deleted(structure(), this);
    }
    const Rings&  all_rings(bool cross_residues = false, int size_threshold = 0,
        std::set<const Residue*>* ignore = nullptr) const;
    bool  in_cycle() const;
    bool  is_backbone() const;
    // length() inherited from UniqueConnection
    const Rings&  minimum_rings(bool cross_residues = false,
            std::set<const Residue*>* ignore = nullptr) const {
        return rings(cross_residues, 0, ignore);
    }
    static bool  polymer_bond_atoms(Atom* first, Atom* second);
    Atom*  polymeric_start_atom() const;
    const Rings&  rings(bool cross_residues = false, int all_size_threshold = 0,
        std::set<const Residue*>* ignore = nullptr) const;
    std::vector<Atom*>  side_atoms(const Atom*) const;
    virtual bool shown() const;
    Atom*  smaller_side() const; // considers missing structure, returns nullptr if in a cycle
    // sqlength() inherited from UniqueConnection
    Structure*  structure() const;

    // session related
    static int  session_num_floats(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_FLOATS(version) + UniqueConnection::session_num_floats(version);
    }
    static int  session_num_ints(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_INTS(version) + UniqueConnection::session_num_ints(version);
    }
    // session_restore and session_save simply inherited from UniqueConnection

    // change tracking
    ChangeTracker*  change_tracker() const;
    void track_change(const std::string& reason) const {
        change_tracker()->add_modified(structure(), this, reason);
    }

    // graphics related
    GraphicsChanges*  graphics_changes() const;
};

}  // namespace atomstruct

#include "Atom.h"
#include "Structure.h"

namespace atomstruct {

inline bool Bond::shown() const {
    return Connection::shown() &&
      (atoms()[0]->draw_mode() != Atom::DrawMode::Sphere ||
       atoms()[1]->draw_mode() != Atom::DrawMode::Sphere);
}

inline ChangeTracker*
Bond::change_tracker() const { return atoms()[0]->change_tracker(); }

inline const Bond::Rings&
Bond::all_rings(bool cross_residues, int size_threshold,
    std::set<const Residue*>* ignore) const
{
    int max_ring_size = size_threshold;
    if (max_ring_size == 0)
        max_ring_size = atoms()[0]->structure()->num_atoms();
    return rings(cross_residues, max_ring_size, ignore);
}

inline GraphicsChanges*
Bond::graphics_changes() const {
    return reinterpret_cast<GraphicsChanges*>(atoms()[0]->structure());
}

inline const Bond::Rings&
Bond::rings(bool cross_residues, int all_size_threshold,
    std::set<const Residue*>* ignore) const
{
    atoms()[0]->structure()->rings(cross_residues, all_size_threshold, ignore);
    return _rings;
}

inline Structure* Bond::structure() const { return atoms()[0]->structure(); }

}  // namespace atomstruct

#endif  // atomstruct_Bond
