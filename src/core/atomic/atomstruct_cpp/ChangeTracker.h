// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_ChangeTracker
#define atomstruct_ChangeTracker

#include <set>
#include <string>
#include <vector>

#include "imex.h"

namespace atomstruct {

class Changes {
public:
    std::set<void*>  created; // use set so that deletions can be easily found
    std::set<void*>  modified;
    std::set<std::string>   reasons;
    long  num_deleted = 0;

    void  clear() { created.clear(); modified.clear(); reasons.clear(); num_deleted=0; }
};

class ATOMSTRUCT_IMEX ChangeTracker {
private:
    static const int  _num_types = 7;
    template<class C>
    int  _ptr_to_type(C*);

    // vector much faster than map...
    std::vector<Changes>  _type_changes;

public:
    ChangeTracker() : _type_changes(_num_types) {};

    template<class C>
    void  add_created(C* ptr) {
        auto changes = _type_changes[_ptr_to_type(ptr)];
        changes.created.insert(ptr);
    }

    template<class C>
    void  add_modified(C* ptr, const std::string &reason) {
        auto changes = _type_changes[_ptr_to_type(ptr)];
        if (changes.created.find(ptr) == changes.created.end()) {
            // newly created objects don't also go in modified set
            changes.modified.insert(ptr);
            changes.reasons.insert(reason);
        }
    }

    template<class C>
    void  add_deleted(C* ptr) {
        auto changes = _type_changes[_ptr_to_type(ptr)];
        ++changes.num_deleted;
        changes.created.erase(ptr);
        changes.modified.erase(ptr);
    }

    void  clear() { for (auto changes: _type_changes) changes.clear(); }
    const std::vector<Changes>&  get_changes() const { return _type_changes; }
    const std::string  python_class_names[_num_types] = {
        "Atom", "Bond", "PBond", "Residue", "Chain", "AtomicStructure", "Proxy_PBGroup"
    };
};

class Atom;
template <>
inline int
ChangeTracker::_ptr_to_type(Atom*) { return 0; }

class Bond;
template <>
inline int
ChangeTracker::_ptr_to_type(Bond*) { return 1; }

class PBond;
template <>
inline int
ChangeTracker::_ptr_to_type(PBond*) { return 2; }

class Residue;
template <>
inline int
ChangeTracker::_ptr_to_type(Residue*) { return 3; }

class Chain;
template <>
inline int
ChangeTracker::_ptr_to_type(Chain*) { return 4; }

class AtomicStructure;
template <>
inline int
ChangeTracker::_ptr_to_type(AtomicStructure*) { return 5; }

class Proxy_PBGroup;
template <>
inline int
ChangeTracker::_ptr_to_type(Proxy_PBGroup*) { return 6; }

}  // namespace atomstruct

#endif  // atomstruct_ChangeTracker
