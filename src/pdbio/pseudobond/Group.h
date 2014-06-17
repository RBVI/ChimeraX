// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Group
#define pseudobonds_Group

#include <stdexcept>
#include <string>

#include "PBond.h"

namespace pseudobond {

template <class PBondContainer, class EndPoint>
class Group {
protected:
    std::string  _category;
    PBondContainer  _pbonds;
public:
    virtual void  clear()  { _pbonds.clear(); }
    Group(std::string& cat): _category(cat) {}
    virtual  ~Group() { for (auto i : _pbonds) delete i; }
    virtual PBond<EndPoint>*  newPseudoBond(EndPoint* e1, EndPoint* e2);
};

template <class Owner, class PBondContainer, class EndPoint>
class Owned_Group: Group<PBondContainer, EndPoint> {
protected:
    Owner*  _owner;
public:
    virtual PBond<EndPoint>*  newPseudoBond(EndPoint* e1, EndPoint* e2,
        const char* err_msg = "Pseudobond endpoints not owned by same object");
    Owned_Group(std::string& cat, Owner* owner):
        Group<PBondContainer, EndPoint>(cat), _owner(owner) {}
    virtual  ~Owned_Group() {};
};

template <class PBondContainer, class EndPoint>
PBond<EndPoint>*
Group<PBondContainer, EndPoint>::newPseudoBond(EndPoint* e1, EndPoint* e2)
{
    PBond<EndPoint>* pb = new PBond<EndPoint>(e1, e2);
    _pbonds.insert(pb);
    return pb;
}

template <class Owner, class PBondContainer, class EndPoint>
PBond<EndPoint>*
Owned_Group<Owner, PBondContainer, EndPoint>::newPseudoBond(
    EndPoint* e1, EndPoint* e2, const char* err_msg)
{
    if (e1->owner() != e2->owner())
        throw std::invalid_argument(err_msg);
    return Group<PBondContainer, EndPoint>::newPseudoBond(e1, e2);
}

}  // namespace pseudobond

#endif  // pseudobonds_Group
