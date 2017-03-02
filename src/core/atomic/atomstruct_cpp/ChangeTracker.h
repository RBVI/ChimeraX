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

#ifndef atomstruct_ChangeTracker
#define atomstruct_ChangeTracker

#include <set>
#include <string>
#include <vector>

#include "imex.h"

namespace atomstruct {

class Atom;
class Bond;
class Structure;
class Pseudobond;
class Residue;
class Chain;
class AtomicStructure;
class Proxy_PBGroup;
    
}

namespace atomstruct {

class ATOMSTRUCT_IMEX Changes {
public:
    // plain "set" (rather than "unordered_set") empirically faster to add_created() and clear()
    std::set<const void*>  created; // use set so that deletions can be easily found
    std::set<const void*>  modified;
    std::set<std::string>   reasons;
    long  num_deleted = 0;

    bool  changed() const { return !(created.empty() && modified.empty() && num_deleted==0); }
    void  clear() { created.clear(); modified.clear(); reasons.clear(); num_deleted=0; }
};

class ATOMSTRUCT_IMEX ChangeTracker {
protected:
    static const int  _num_types = 7;
    template<class C>
    int  _ptr_to_type(C*);

    bool  _discarding;
    // vector much faster than map...
    std::vector<Changes>  _type_changes;

public:
    ChangeTracker() : _discarding(false), _type_changes(_num_types) {};

    static const std::string  REASON_ACTIVE_COORD_SET;
    static const std::string  REASON_ALT_LOC;
    static const std::string  REASON_ANISO_U;
    static const std::string  REASON_BALL_SCALE;
    static const std::string  REASON_BFACTOR;
    static const std::string  REASON_COLOR;
    static const std::string  REASON_COORD;
    static const std::string  REASON_DISPLAY;
    static const std::string  REASON_DRAW_MODE;
    static const std::string  REASON_HALFBOND;
    static const std::string  REASON_HIDE;
    static const std::string  REASON_IDATM_TYPE;
    static const std::string  REASON_IS_BACKBONE;
    static const std::string  REASON_IS_HET;
    static const std::string  REASON_NAME;
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
    static const std::string  REASON_SELECTED;
    static const std::string  REASON_SEQUENCE;
    static const std::string  REASON_SERIAL_NUMBER;
    static const std::string  REASON_STRUCTURE_CATEGORY;
    static const std::string  REASON_SS_ID;
    static const std::string  REASON_SS_TYPE;
    
    template<class C>
    void  add_created(C* ptr) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(ptr)];
        changes.created.insert(ptr);
    }

    // this aggregate routine seemingly *slower* than calling the single-pointer version in a loop,
    //   possibly due to inlining chicanery
    template<class C>
    void  add_created(const std::set<C*>& ptrs) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(static_cast<typename std::set<C*>::value_type>(nullptr))];
        // looping through and inserting individually empirically faster than the commented-out
        //   single call below, possibly due to the generic nature of that call
        for (auto ptr: ptrs)
            changes.created.insert(ptr);
        //changes.created.insert(ptrs.begin(), ptrs.end());
    }

    // this aggregate routine seemingly *slower* than calling the single-pointer version in a loop,
    //   possibly due to inlining chicanery
    template<class C>
    void  add_created(const std::vector<C*>& ptrs) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(static_cast<typename std::vector<C*>::value_type>(nullptr))];
        // looping through and inserting individually empirically faster than the commented-out
        //   single call below, possibly due to the generic nature of that call
        for (auto ptr: ptrs)
            changes.created.insert(ptr);
        //changes.created.insert(ptrs.begin(), ptrs.end());
    }

    template<class C>
    void  add_modified(C* ptr, const std::string& reason) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(ptr)];
        if (changes.created.find(static_cast<const void*>(ptr)) == changes.created.end()) {
            // newly created objects don't also go in modified set
            changes.modified.insert(ptr);
            changes.reasons.insert(reason);
        }
    }

    template<class C>
    void  add_modified(C* ptr, const std::string& reason, const std::string& reason2) {
        auto& changes = _type_changes[_ptr_to_type(ptr)];
        if (_discarding)
            return;
        if (changes.created.find(static_cast<const void*>(ptr)) == changes.created.end()) {
            // newly created objects don't also go in modified set
            changes.modified.insert(ptr);
            changes.reasons.insert(reason);
            changes.reasons.insert(reason2);
        }
    }

    template<class C>
    void  add_deleted(C* ptr) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(ptr)];
        ++changes.num_deleted;
        changes.created.erase(ptr);
        changes.modified.erase(ptr);
    }

    bool  changed() const {
        for (auto& changes: _type_changes) {
            if (changes.changed())
                return true;
        }
        return false;
    }
    void  clear() { for (auto& changes: _type_changes) changes.clear(); }
    const std::vector<Changes>&  get_changes() const { return _type_changes; }
    const std::string  python_class_names[_num_types] = {
        "Atom", "Bond", "Pseudobond", "Residue", "Chain",
        "StructureData", "PseudobondGroupData"
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

}  // namespace atomstruct

#endif  // atomstruct_ChangeTracker
