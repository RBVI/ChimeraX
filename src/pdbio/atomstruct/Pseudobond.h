// vim: set expandtab ts=4 sw=4:
#ifndef atomic_Pseudobond
#define atomic_Pseudobond

#include <set>
#include <unordered_map>

#include "basegeom/Connection.h"
#include "imex.h"
#include "pseudobond/Manager.h"

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
    PBond*  newPseudoBond(Atom* a1, Atom* a2) {
        PBond* pb = new PBond(a1, a2);
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
    void  _check_ownership(Atom* a1, Atom* a2);
    Owned_PBGroup_Base(const std::string& cat, AtomicStructure* as):
        Owned_Group<AtomicStructure, Atom, PBond>(cat, as) {};
};

class Owned_PBGroup: public Owned_PBGroup_Base {
private:
    std::set<PBond*>  _pbonds;
public:
    void  clear() { for (auto pb : _pbonds) delete pb; _pbonds.clear(); }
    PBond*  newPseudoBond(Atom* a1, Atom* a2) {
        _check_ownership(a1, a2);
        PBond* pb = new PBond(a1, a2);
        _pbonds.insert(pb); return pb;
    }
    PBond*  newPseudoBond(Atom* const ends[2]) {
        // should be in base class, but C++ won't look in base
        // classes for overloads!
        return newPseudoBond(ends[0], ends[1]);
    }
    Owned_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) {}
    const std::set<PBond*>&  pseudobonds() const { return _pbonds; }
};

class CS_PBGroup: public Owned_PBGroup_Base
{
private:
    friend class CoordSet;
    friend class AS_CS_PBManager;
    mutable std::unordered_map<const CoordSet*, std::set<PBond*>>  _pbonds;
    void  remove_cs(const CoordSet* cs) { _pbonds.erase(cs); }
public:
    void  clear() {
        for (auto cat_set : _pbonds)
            for (auto pb: cat_set.second) delete pb;
        _pbonds.clear();
    }
    CS_PBGroup(const std::string& cat, AtomicStructure* as):
        Owned_PBGroup_Base(cat, as) {}
    PBond*  newPseudoBond(Atom* a1, Atom* a2);
    PBond*  newPseudoBond(Atom* const ends[2]) {
        // should be in base class, but C++ won't look in base
        // classes for overloads!
        return newPseudoBond(ends[0], ends[1]);
    }
    PBond*  newPseudoBond(Atom* a1, Atom* a2, CoordSet* cs);
    PBond*  newPseudoBond(Atom* const ends[2], CoordSet* cs) {
        // should be in base class, but C++ won't look in base
        // classes for overloads!
        return newPseudoBond(ends[0], ends[1], cs);
    }
    const std::set<PBond*>&  pseudobonds() const;
    const std::set<PBond*>&  pseudobonds(const CoordSet* cs) const {
        return _pbonds[cs];
    }
};

// per-AtomicStructure pseudobond manager(s)..
class AS_CS_PBManager:
    public pseudobond::Owned_Manager<AtomicStructure, CS_PBGroup>
{
private:
    friend class AtomicStructure;
    friend class CoordSet;
    AS_CS_PBManager(AtomicStructure* as):
        pseudobond::Owned_Manager<AtomicStructure, CS_PBGroup>(as) {}
    void  remove_cs(const CoordSet* cs) {
        for (auto pbg_info: _groups) pbg_info.second->remove_cs(cs);
    }
};

class AS_PBManager:
    public pseudobond::Owned_Manager<AtomicStructure, Owned_PBGroup>
{
    friend class AtomicStructure;
    AS_PBManager(AtomicStructure* as):
        pseudobond::Owned_Manager<AtomicStructure, Owned_PBGroup>(as) {}
};

}  // namespace atomstruct

#endif  // atomic_Pseudobond
