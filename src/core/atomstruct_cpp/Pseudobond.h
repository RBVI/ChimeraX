// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Pseudobond
#define atomstruct_Pseudobond

#include <set>
#include <unordered_map>

#include <basegeom/Connection.h>
#include "imex.h"
#include <pseudobond/Manager.h>

namespace atomstruct {

class Atom;
class AtomicStructure;
class CoordSet;

class ATOMSTRUCT_IMEX PBond: public basegeom::Connection<Atom>
{
    friend class PBGroup;
    friend class Owned_PBGroup;
    friend class CS_PBGroup;
private:
    PBond(Atom* a1, Atom* a2): basegeom::Connection<Atom>(a1, a2) {};
protected:
    const char*  err_msg_loop() const
        { return "Can't form pseudobond to itself"; }
    const char*  err_msg_not_end() const
        { return "Atom given to other_end() not in pseudobond!"; }
public:
    typedef End_points  Atoms;
    const Atoms&  atoms() const { return end_points(); }
};

// "global" pseudobond groups...
class PBGroup: pseudobond::Group<Atom, PBond>
{
private:
    std::set<PBond*>  _pbonds;
public:
    void  clear() { for (auto pb: _pbonds) delete pb; _pbonds.clear(); }
    PBond*  new_pseudobond(Atom* a1, Atom* a2) {
        PBond* pb = new PBond(a1, a2);
        pb->finish_construction();
        _pbonds.insert(pb);
        return pb;
    }
    PBGroup(const std::string& cat): pseudobond::Group<Atom, PBond>(cat) {}
    const std::set<PBond*>&  pseudobonds() const { return _pbonds; }
};

// global pseudobond manager
typedef pseudobond::Global_Manager<PBGroup>  PBManager;

// in per-AtomicStructure groups there are per-CoordSet groups
// and overall groups...
class Owned_PBGroup_Base: public pseudobond::Owned_Group<AtomicStructure, Atom, PBond> {
protected:
    friend class AS_PBManager;
    friend class pseudobond::Owned_Manager<AtomicStructure, Owned_PBGroup_Base>;
    void  _check_ownership(Atom* a1, Atom* a2);
    Owned_PBGroup_Base(const std::string& cat, AtomicStructure* as):
        Owned_Group<AtomicStructure, Atom, PBond>(cat, as) {};
};

class Owned_PBGroup: public Owned_PBGroup_Base {
private:
    std::set<PBond*>  _pbonds;
public:
    Owned_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) {}
    ~Owned_PBGroup() { clear(); }
    void  clear() { for (auto pb : _pbonds) delete pb; _pbonds.clear(); }
    PBond*  new_pseudobond(Atom* a1, Atom* a2) {
        _check_ownership(a1, a2);
        PBond* pb = new PBond(a1, a2);
        pb->finish_construction();
        _pbonds.insert(pb); return pb;
    }
    const std::set<PBond*>&  pseudobonds() const { return _pbonds; }
};

class Proxy_PBGroup;

class CS_PBGroup: public Owned_PBGroup_Base
{
private:
    friend class Proxy_PBGroup;
    mutable std::unordered_map<const CoordSet*, std::set<PBond*>>  _pbonds;
    void  remove_cs(const CoordSet* cs) { _pbonds.erase(cs); }
public:
    CS_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) {}
    ~CS_PBGroup() { clear(); }
    void  clear() {
        for (auto cat_set : _pbonds)
            for (auto pb: cat_set.second) delete pb;
        _pbonds.clear();
    }
    PBond*  new_pseudobond(Atom* a1, Atom* a2);
    PBond*  new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs);
    const std::set<PBond*>&  pseudobonds() const;
    const std::set<PBond*>&  pseudobonds(const CoordSet* cs) const {
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
    Proxy_PBGroup*  get_group(const std::string& name, int create = GRP_NONE) const;
};

// Need a proxy class that can be contained/returned by the pseudobond
// manager and that will dispatch calls to the appropriate contained class
class Proxy_PBGroup: public Owned_PBGroup_Base
{
private:
    friend class AS_PBManager;
    int  _group_type;
    void*  _proxied;
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
    void  clear() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<Owned_PBGroup*>(_proxied)->clear();
        else
            static_cast<CS_PBGroup*>(_proxied)->clear();
    }
    int  group_type() const { return _group_type; }
    PBond*  new_pseudobond(Atom* a1, Atom* a2) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<Owned_PBGroup*>(_proxied)->new_pseudobond(a1, a2);
        else
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
    Proxy_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) { init(AS_PBManager::GRP_NORMAL); }
    Proxy_PBGroup(const std::string& cat, AtomicStructure* as, int grp_type):
        Owned_PBGroup_Base(cat, as) { init(grp_type); }
    const std::set<PBond*>&  pseudobonds() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<Owned_PBGroup*>(_proxied)->pseudobonds();
        else
            return static_cast<CS_PBGroup*>(_proxied)->pseudobonds();
    }
    const std::set<PBond*>&  pseudobonds(const CoordSet* cs) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            throw std::invalid_argument("Not a per-coordset pseudobond group");
        return static_cast<CS_PBGroup*>(_proxied)->pseudobonds(cs);
    }
};

}  // namespace atomstruct

#endif  // atomstruct_Pseudobond
