// vim: set expandtab ts=4 sw=4:
#ifndef atomic_Pseudobond
#define atomic_Pseudobond

#include <set>
#include <map>

#include "pseudobond/Manager.h"
#include "imex.h"

namespace atomstruct {

class Atom;
class AtomicStructure;
class CoordSet;

typedef pseudobond::Link<Atom>  PBond;

// "global" pseudobond groups...
class PBGroup: pseudobond::Group<Atom>
{
private:
    std::set<PBond*>  _pbonds;
public:
    void  clear() { for (auto pb: _pbonds) delete pb; _pbonds.clear(); }
    PBond*  newPseudoBond(Atom* a1, Atom* a2) {
        PBond* pb = makeLink(a1, a2);
        _pbonds.insert(pb);
        return pb;
    }
    PBGroup(std::string& cat): pseudobond::Group<Atom>(cat) {}
};

// global pseudobond manager
typedef pseudobond::Global_Manager<PBGroup>  PBManager;

// per-AtomicStructure pseudobond manager(s)..
class AS_PBManager: public pseudobond::Owned_Manager<AtomicStructure, Atom> {};

// in per-AtomicStructure groups there are per-CoordSet groups
// and overall groups...
class Owned_PBGroup_Base: public pseudobond::Owned_Group<AtomicStructure, Atom> {
protected:
    virtual PBond*  addPseudoBond(PBond *) = 0;
public:
    PBond*  newPseudoBond(Atom* a1, Atom* a2);
};

class Owned_PBGroup: public Owned_PBGroup_Base {
private:
    std::set<PBond*>  _pbonds;
    PBond*  addPseudoBond(PBond* pb) { _pbonds.insert(pb); return pb; }
public:
    void  clear() { for (auto pb : _pbonds) delete pb; _pbonds.clear(); }
};
//class CS_PBGroup: public PBGroup_Base {};

}  // namespace atomstruct

#endif  // atomic_Pseudobond
