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

#ifndef atomstruct_ChangeTracker
#define atomstruct_ChangeTracker

#include <algorithm>
#include <array>
#include <map>
#include <pyinstance/PythonInstance.declare.h>
#include <set>
#include <string>
#include <vector>

#include "imex.h"

namespace atomstruct {

class Atom;
class AtomicStructure;
class Bond;
class Chain;
class CoordSet;
class Proxy_PBGroup;
class Pseudobond;
class Residue;
class Structure;
    
}

namespace atomstruct {

class ATOMSTRUCT_IMEX Changes {
public:
    // plain "set" (rather than "unordered_set") empirically faster to add_created() and clear()
    std::set<const void*>  created; // use set so that deletions can be easily found
    std::set<const void*>  modified;
    std::set<std::string>   reasons;
    long  num_deleted = 0;

    bool  changed() const { return !(created.empty() && modified.empty() && reasons.empty() && num_deleted==0); }
    void  clear() { created.clear(); modified.clear(); reasons.clear(); num_deleted=0; }
};

class ATOMSTRUCT_IMEX ChangeTracker: public pyinstance::PythonInstance<ChangeTracker> {
protected:
    static const int  _num_types = 8;

public:
    typedef std::array<Changes, _num_types>  ChangesArray;

protected:
    template<class C>
    int  _ptr_to_type(C*);

    bool  _discarding;
    // array much faster than map...
    mutable ChangesArray  _global_type_changes;
    mutable std::map<Structure*, ChangesArray>  _structure_type_changes;
    std::set<Structure*> _dead_structures;
    bool  _structure_okay(Structure* s) {
        return s != nullptr && _dead_structures.find(s) == _dead_structures.end();
    }

public:
    ChangeTracker() : _discarding(false) {}
    virtual ~ChangeTracker() {}

    static const std::string  REASON_ACTIVE_COORD_SET;
    static const std::string  REASON_ALT_LOC;
    static const std::string  REASON_ANISO_U;
    static const std::string  REASON_BALL_SCALE;
    static const std::string  REASON_BFACTOR;
    static const std::string  REASON_CHAIN_ID;
    static const std::string  REASON_COLOR;
    static const std::string  REASON_COORD;
    static const std::string  REASON_COORDSET;
    static const std::string  REASON_DISPLAY;
    static const std::string  REASON_DRAW_MODE;
    static const std::string  REASON_ELEMENT;
    static const std::string  REASON_HALFBOND;
    static const std::string  REASON_HIDE;
    static const std::string  REASON_IDATM_TYPE;
    static const std::string  REASON_INSERTION_CODE;
    static const std::string  REASON_NAME;
    static const std::string  REASON_NUMBER;
    static const std::string  REASON_OCCUPANCY;
    static const std::string  REASON_RADIUS;
    static const std::string  REASON_RESIDUES;
    static const std::string  REASON_RIBBON_ADJUST;
    static const std::string  REASON_RIBBON_COLOR;
    static const std::string  REASON_RIBBON_DISPLAY;
    static const std::string  REASON_RIBBON_HIDE_BACKBONE;
    static const std::string  REASON_RIBBON_TETHER;
    static const std::string  REASON_RIBBON_ORIENTATION;
    static const std::string  REASON_RIBBON_MODE;
    static const std::string  REASON_RING_COLOR;
    static const std::string  REASON_RING_DISPLAY;
    static const std::string  REASON_RING_MODE;
    static const std::string  REASON_SCENE_COORD;
    static const std::string  REASON_SELECTED;
    static const std::string  REASON_SEQUENCE;
    static const std::string  REASON_SERIAL_NUMBER;
    static const std::string  REASON_STRUCTURE_CATEGORY;
    static const std::string  REASON_SS_ID;
    static const std::string  REASON_SS_TYPE;
    
    template<class C>
    void  add_created(Structure* s, C* ptr) {
        if (_discarding)
            return;
        if (_structure_okay(s)) {
            auto& s_changes = _structure_type_changes[s][_ptr_to_type(ptr)];
            s_changes.created.insert(ptr);
        } else if (s == nullptr)
            _global_type_changes[_ptr_to_type(ptr)].created.insert(ptr);
    }

    // this aggregate routine seemingly *slower* than calling the single-pointer version in a loop,
    //   possibly due to inlining chicanery
    template<class C>
    void  add_created(Structure* s, const std::set<C*>& ptrs) {
        if (_discarding)
            return;
        if (_structure_okay(s)) {
            auto& s_changes = _structure_type_changes[s]
                [_ptr_to_type(static_cast<typename std::set<C*>::value_type>(nullptr))];
            // looping through and inserting individually empirically faster than the commented-out
            //   single call below, possibly due to the generic nature of that call
            for (auto ptr: ptrs)
                s_changes.created.insert(ptr);
            //s_changes.created.insert(ptrs.begin(), ptrs.end());
        } else if (s == nullptr) {
            auto& g_changes = _global_type_changes
                [_ptr_to_type(static_cast<typename std::set<C*>::value_type>(nullptr))];
            for (auto ptr: ptrs)
                g_changes.created.insert(ptr);
        }
    }

    template<class C>
    void  add_modified(Structure* s, C* ptr, const std::string& reason) {
        if (_discarding)
            return;
        if (_structure_okay(s)) {
            auto& s_changes = _structure_type_changes[s][_ptr_to_type(ptr)];
            if (ptr == nullptr) {
                // If ptr is null the object is not included in the modified list.
                // This is to improve speed with large structures, see ticket #3000.
                s_changes.reasons.insert(reason);
            } else if (s_changes.created.find(static_cast<const void*>(ptr)) == s_changes.created.end()) {
                // newly created objects don't also go in modified set
                s_changes.modified.insert(ptr);
                s_changes.reasons.insert(reason);
            }
        } else if (s == nullptr) {
            auto& g_changes = _global_type_changes[_ptr_to_type(ptr)];
            g_changes.modified.insert(ptr);
            g_changes.reasons.insert(reason);
        }
    }

    template<class C>
    void  add_modified(Structure* s, C* ptr, const std::string& reason, const std::string& reason2) {
        if (_discarding)
            return;
        if (_structure_okay(s)) {
            auto& s_changes = _structure_type_changes[s][_ptr_to_type(ptr)];
            if (s_changes.created.find(static_cast<const void*>(ptr)) == s_changes.created.end()) {
                // newly created objects don't also go in modified set
                s_changes.modified.insert(ptr);
                s_changes.reasons.insert(reason);
                s_changes.reasons.insert(reason2);
            }
        } else if (s == nullptr) {
            auto& g_changes = _global_type_changes[_ptr_to_type(ptr)];
            g_changes.modified.insert(ptr);
            g_changes.reasons.insert(reason);
            g_changes.reasons.insert(reason2);
        }
    }

    template<class C>
    void  add_deleted(Structure* s, C* ptr) {
        if (_discarding)
            return;
        if (s == static_cast<void*>(ptr)) {
            _structure_type_changes.erase(s);
            _dead_structures.insert(s);
        }
        if (_structure_okay(s)) {
            auto& s_changes = _structure_type_changes[s][_ptr_to_type(ptr)];
            ++s_changes.num_deleted;
            s_changes.created.erase(ptr);
            s_changes.modified.erase(ptr);
        } else { // also put deletions for dead structures in global changes
            auto& g_changes = _global_type_changes[_ptr_to_type(ptr)];
            ++g_changes.num_deleted;
            g_changes.created.erase(ptr);
            g_changes.modified.erase(ptr);
        }
    }

    template<class C>
    void add_modified_set(Structure* s, const std::vector<C*>& ptrs, const std::string& reason) {
        if (_discarding)
            return;
        if (_structure_okay(s)) {
            auto& s_changes = _structure_type_changes[s][
                _ptr_to_type(static_cast<typename std::set<C*>::value_type>(nullptr))];
            auto& s_created = s_changes.created;
            auto& s_modified = s_changes.modified;
            if (s_created.size()) {
                for (auto ptr: ptrs) {
                    if (s_created.find(static_cast<const void*>(ptr)) == s_created.end())
                        s_modified.insert(ptr);
                }
            } else {
                for (auto ptr: ptrs)
                    s_modified.insert(ptr);
            }
            s_changes.reasons.insert(reason);
        } else if (s == nullptr) {
            auto& g_changes = _global_type_changes[
                _ptr_to_type(static_cast<typename std::set<C*>::value_type>(nullptr))];
            auto& g_created = g_changes.created;
            auto& g_modified = g_changes.modified;
            if (g_created.size()) {
                for (auto ptr: ptrs) {
                    if (g_created.find(static_cast<const void*>(ptr)) == g_created.end())
                        g_modified.insert(ptr);
                }
            } else {
                for (auto ptr: ptrs)
                    g_modified.insert(ptr);
            }
            g_changes.reasons.insert(reason);
        }
    }

    bool  changed() const {
        for (auto& s_changes: _structure_type_changes) {
            auto& structure_changes = s_changes.second;
            for (auto& c: structure_changes)
                if (c.changed())
                    return true;
        }
        for (auto& changes: _global_type_changes)
            if (changes.changed())
                return true;
        return false;
    }
    void  clear() {
        for (auto& changes: _global_type_changes) changes.clear();
        _structure_type_changes.clear();
        _dead_structures.clear();
    }
    const ChangesArray&  get_global_changes() const {
        // global type changes only initially holds the non-structure-associated changes
        // (global pseudobonds and groups); supplement with structure changes
        for (auto& s_changes: _structure_type_changes) {
            auto &structure_changes = s_changes.second;
            for (int i = 0; i < _num_types; ++i) {
                auto &target = _global_type_changes[i];
                auto &source = structure_changes[i];
                for (auto ptr: source.created)
                    target.created.insert(ptr);
                for (auto ptr: source.modified)
                    target.modified.insert(ptr);
                for (auto &reason: source.reasons)
                    target.reasons.insert(reason);
                target.num_deleted += source.num_deleted;
            }
        }
        return _global_type_changes;
    }
    const std::map<Structure*, ChangesArray>&  get_structure_changes() const {
        return _structure_type_changes;
    }
    const std::string  python_class_names[_num_types] = {
        "Atom", "Bond", "Pseudobond", "Residue", "Chain",
        "StructureData", "PseudobondGroupData", "CoordSet"
    };
};

// Before structures are opened in ChimeraX, they don't generate change-tracking
// events.  This class enables that by being the "change tracker" until the
// point that actual change tracking is turned on.
class ATOMSTRUCT_IMEX DiscardingChangeTracker : public ChangeTracker {
public:
    DiscardingChangeTracker() : ChangeTracker() { _discarding = true; }
    static DiscardingChangeTracker*  discarding_change_tracker();
};

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Atom*) { return 0; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::Atom*) { return 0; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Bond*) { return 1; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::Bond*) { return 1; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Pseudobond*) { return 2; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::Pseudobond*) { return 2; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Residue*) { return 3; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::Residue*) { return 3; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Chain*) { return 4; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::Chain*) { return 4; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Structure*) { return 5; }
template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::AtomicStructure*) { return 5; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::AtomicStructure*) { return 5; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Proxy_PBGroup*) { return 6; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::Proxy_PBGroup*) { return 6; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::CoordSet*) { return 7; }

template <>
inline int
ChangeTracker::_ptr_to_type(const atomstruct::CoordSet*) { return 7; }

}  // namespace atomstruct

#endif  // atomstruct_ChangeTracker
