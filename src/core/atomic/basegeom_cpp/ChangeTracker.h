// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_ChangeTracker
#define basegeom_ChangeTracker

#include <set>
#include <string>
#include <vector>

#include "imex.h"

namespace atomstruct {

class Atom;
class Bond;
class PBond;
class Residue;
class Chain;
class AtomicStructure;
class Proxy_PBGroup;
    
}

namespace basegeom {

class Changes {
public:
    std::set<void*>  created; // use set so that deletions can be easily found
    std::set<void*>  modified;
    std::set<std::string>   reasons;
    long  num_deleted = 0;

    void  clear() { created.clear(); modified.clear(); reasons.clear(); num_deleted=0; }
};

class BASEGEOM_IMEX ChangeTracker {
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
    static const std::string  REASON_IS_HELIX;
    static const std::string  REASON_IS_HET;
    static const std::string  REASON_IS_SHEET;
    static const std::string  REASON_OCCUPANCY;
    static const std::string  REASON_RADIUS;
    static const std::string  REASON_RESIDUES;
    static const std::string  REASON_RIBBON_DISPLAY;
    static const std::string  REASON_RIBBON_COLOR;
    static const std::string  REASON_SELECTED;
    static const std::string  REASON_SEQUENCE;
    static const std::string  REASON_SERIAL_NUMBER;
    static const std::string  REASON_SS_ID;
    
    template<class C>
    void  add_created(C* ptr) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(ptr)];
        changes.created.insert(ptr);
    }

    template<class C>
    void  add_modified(C* ptr, const std::string& reason) {
        if (_discarding)
            return;
        auto& changes = _type_changes[_ptr_to_type(ptr)];
        if (changes.created.find(ptr) == changes.created.end()) {
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
        if (changes.created.find(ptr) == changes.created.end()) {
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

    void  clear() { for (auto& changes: _type_changes) changes.clear(); }
    const std::vector<Changes>&  get_changes() const { return _type_changes; }
    const std::string  python_class_names[_num_types] = {
        "Atom", "Bond", "Pseudobond", "Residue", "Chain",
        "AtomicStructureData", "PseudobondGroupData"
    };
};

// Before structures are opened in Chimera, they don't generate change-tracking
// events.  This class enables that by being the "change tracker" until the
// point that actual change tracking is turned on.
class BASEGEOM_IMEX DiscardingChangeTracker : public ChangeTracker {
public:
    DiscardingChangeTracker() : ChangeTracker() { _discarding = true; }
    static DiscardingChangeTracker*  discarding_change_tracker();
};

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Atom*) { return 0; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Bond*) { return 1; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::PBond*) { return 2; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Residue*) { return 3; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Chain*) { return 4; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::AtomicStructure*) { return 5; }

template <>
inline int
ChangeTracker::_ptr_to_type(atomstruct::Proxy_PBGroup*) { return 6; }

}  // namespace basegeom

#endif  // atomstruct_ChangeTracker
