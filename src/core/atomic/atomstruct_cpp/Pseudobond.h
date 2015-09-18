// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Pseudobond
#define atomstruct_Pseudobond

#include <set>
#include <unordered_map>

#include <basegeom/Connection.h>
#include <basegeom/destruct.h>
#include "imex.h"
#include <pseudobond/Manager.h>

namespace atomstruct {

class Atom;
class AtomicStructure;
class CoordSet;

using basegeom::ChangeTracker;
using basegeom::GraphicsContainer;

class ATOMSTRUCT_IMEX PBond: public basegeom::Connection<Atom, PBond>
{
    friend class PBGroup;
    friend class Owned_PBGroup;
    friend class CS_PBGroup;
private:
    GraphicsContainer*  _gc;

    PBond(Atom* a1, Atom* a2, GraphicsContainer* gc):
        basegeom::Connection<Atom, PBond>(a1, a2), _gc(gc) {};
protected:
    const char*  err_msg_loop() const
        { return "Can't form pseudobond to itself"; }
    const char*  err_msg_not_end() const
        { return "Atom given to other_end() not in pseudobond!"; }
public:
    virtual ~PBond() {}
    typedef End_points  Atoms;
    const Atoms&  atoms() const { return end_points(); }
    ChangeTracker*  change_tracker() const;
    GraphicsContainer*  graphics_container() const { return _gc; }
    GraphicsContainer*  group() const { return graphics_container(); }
};

typedef std::set<PBond*>  PBonds;

class Proxy_PBGroup;

// global pseudobond manager
// Though for C++ purposes it could be templated off of PBGroup instead
// of Proxy_PBGroup, this allows groups to be treated uniformly on the
// Python side
class PBManager: public pseudobond::Base_Manager<Proxy_PBGroup> {
    ChangeTracker*  _ct;
public:
    PBManager(ChangeTracker* ct): _ct(ct) {}

    ChangeTracker*  change_tracker() const { return _ct; }
    void  delete_group(Proxy_PBGroup*);
    Proxy_PBGroup*  get_group(const std::string& name, int create = GRP_NONE);
};

// in per-AtomicStructure groups there are per-CoordSet groups
// and overall groups...
class Owned_PBGroup_Base: public pseudobond::Owned_Group<AtomicStructure, Atom, PBond> {
protected:
    friend class AS_PBManager;
    void  _check_ownership(Atom* a1, Atom* a2);
    Owned_PBGroup_Base(const std::string& cat, AtomicStructure* as):
        Owned_Group<AtomicStructure, Atom, PBond>(cat, as) {};
};

class Owned_PBGroup: public Owned_PBGroup_Base {
private:
    friend class Proxy_PBGroup;
    PBonds  _pbonds;
protected:
    Owned_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) {}
    ~Owned_PBGroup() { dtor_code(); }
public:
    void  check_destroyed_atoms(const std::set<void*>& destroyed);
    void  clear() { for (auto pb : _pbonds) delete pb; _pbonds.clear(); }
    PBond*  new_pseudobond(Atom* a1, Atom* a2) {
        _check_ownership(a1, a2);
        PBond* pb = new PBond(a1, a2, this);
        pb->finish_construction();
        _pbonds.insert(pb); return pb;
    }
    const PBonds&  pseudobonds() const { return _pbonds; }
};

class CS_PBGroup: public Owned_PBGroup_Base
{
private:
    friend class Proxy_PBGroup;
    mutable std::unordered_map<const CoordSet*, PBonds>  _pbonds;
    void  remove_cs(const CoordSet* cs) { _pbonds.erase(cs); }
protected:
    CS_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) {}
    ~CS_PBGroup() {
        _destruction_relevant = false;
        auto du = basegeom::DestructionUser(this);
        for (auto name_pbs: _pbonds) {
            for (auto pb: name_pbs.second)
                delete pb;
        }
    }
public:
    void  check_destroyed_atoms(const std::set<void*>& destroyed);
    void  clear() {
        for (auto cat_set : _pbonds)
            for (auto pb: cat_set.second) delete pb;
        _pbonds.clear();
    }
    PBond*  new_pseudobond(Atom* a1, Atom* a2);
    PBond*  new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs);
    const PBonds&  pseudobonds() const;
    const PBonds&  pseudobonds(const CoordSet* cs) const {
        return _pbonds[cs];
    }
};

class AS_PBManager:
    public pseudobond::Owned_Manager<AtomicStructure, Proxy_PBGroup>
{
public:
    static const int  GRP_PER_CS = GRP_NORMAL + 1;
private:
    friend class AtomicStructure;
    friend class CoordSet;
    AS_PBManager(AtomicStructure* as):
        pseudobond::Owned_Manager<AtomicStructure, Proxy_PBGroup>(as) {}
    void  remove_cs(const CoordSet* cs);
public:
    ChangeTracker*  change_tracker() const;
    void  delete_group(Proxy_PBGroup*);
    Proxy_PBGroup*  get_group(const std::string& name, int create = GRP_NONE);
    AtomicStructure*  structure() const { return owner(); }
};

// Need a proxy class that can be contained/returned by the pseudobond
// manager and that will dispatch calls to the appropriate contained class
class Proxy_PBGroup: public Owned_PBGroup_Base
{
public:
    typedef pseudobond::Base_Manager<Proxy_PBGroup>  BaseManager;
private:
    friend class AS_PBManager;
    friend class pseudobond::Owned_Manager<AtomicStructure, Proxy_PBGroup>;
    friend class pseudobond::Base_Manager<Proxy_PBGroup>;
    friend class PBManager;
    int  _group_type;
    BaseManager*  _manager;
    void*  _proxied;

    Proxy_PBGroup(BaseManager* manager, const std::string& cat):
        Owned_PBGroup_Base(cat, nullptr), _manager(manager) { init(AS_PBManager::GRP_NORMAL); }
    Proxy_PBGroup(BaseManager* manager, const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as), _manager(manager) { init(AS_PBManager::GRP_NORMAL); }
    Proxy_PBGroup(BaseManager* manager, const std::string& cat, AtomicStructure* as, int grp_type):
        Owned_PBGroup_Base(cat, as), _manager(manager) { init(grp_type); }
    ~Proxy_PBGroup() {
        auto du = basegeom::DestructionUser(this);
        if (_group_type == AS_PBManager::GRP_NORMAL)
            delete static_cast<Owned_PBGroup*>(_proxied);
        else
            delete static_cast<CS_PBGroup*>(_proxied);
    }
    void  init(int grp_type) {
        _group_type = grp_type;
        if (grp_type == AS_PBManager::GRP_NORMAL)
            _proxied = new Owned_PBGroup(_category, _owner);
        else
            _proxied = new CS_PBGroup(_category, _owner);
    }
    void  remove_cs(const CoordSet* cs) {
        if (_group_type == AS_PBManager::GRP_PER_CS)
            static_cast<CS_PBGroup*>(_proxied)->remove_cs(cs);
    }
public:
    const std::string&  category() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<Owned_PBGroup*>(_proxied)->category();
        return static_cast<CS_PBGroup*>(_proxied)->category();
    }
    void  check_destroyed_atoms(const std::set<void*>& destroyed) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<Owned_PBGroup*>(_proxied)->check_destroyed_atoms(destroyed);
        static_cast<CS_PBGroup*>(_proxied)->check_destroyed_atoms(destroyed);
    }
    void  clear() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<Owned_PBGroup*>(_proxied)->clear();
        static_cast<CS_PBGroup*>(_proxied)->clear();
    }
    void  destroy() {
        if (owner() == nullptr)
            static_cast<PBManager*>(_manager)->delete_group(this);
        else
            static_cast<AS_PBManager*>(_manager)->delete_group(this);
    }
    int  group_type() const { return _group_type; }
    PBond*  new_pseudobond(Atom* a1, Atom* a2) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<Owned_PBGroup*>(_proxied)->new_pseudobond(a1, a2);
        return static_cast<CS_PBGroup*>(_proxied)->new_pseudobond(a1, a2);
    }
    PBond*  new_pseudobond(Atom* const ends[2]) {
        // should be in base class, but C++ won't look in base
        // classes for overloads!
        return new_pseudobond(ends[0], ends[1]);
    }
    PBond*  new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            throw std::invalid_argument("Not a per-coordset pseudobond group");
        return static_cast<CS_PBGroup*>(_proxied)->new_pseudobond(a1, a2, cs);
    }
    PBond*  new_pseudobond(Atom* const ends[2], CoordSet* cs) {
        return new_pseudobond(ends[0], ends[1], cs);
    }
    const std::set<PBond*>&  pseudobonds() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<Owned_PBGroup*>(_proxied)->pseudobonds();
        return static_cast<CS_PBGroup*>(_proxied)->pseudobonds();
    }
    const std::set<PBond*>&  pseudobonds(const CoordSet* cs) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            throw std::invalid_argument("Not a per-coordset pseudobond group");
        return static_cast<CS_PBGroup*>(_proxied)->pseudobonds(cs);
    }
    decltype(_owner)  structure() const { return owner(); }

};

}  // namespace atomstruct

#endif  // atomstruct_Pseudobond
