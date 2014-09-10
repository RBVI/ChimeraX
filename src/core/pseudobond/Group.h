// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Group
#define pseudobonds_Group

#include <set>
#include <stdexcept>
#include <string>

namespace pseudobond {

template <class EndPoint, class PBond>
class Group {
protected:
    std::string  _category;
public:
    virtual void  clear() = 0;
    Group(const std::string& cat): _category(cat) {}
    virtual  ~Group() {}
    virtual PBond*  newPseudoBond(EndPoint* e1, EndPoint* e2) = 0;
    virtual const std::set<PBond*>&  pseudobonds() const = 0;
};

template <class Owner, class EndPoint, class PBond>
class Owned_Group: public Group<EndPoint, PBond> {
protected:
    Owner*  _owner;
public:
    virtual PBond*  newPseudoBond(EndPoint* e1, EndPoint* e2) = 0;
    Owned_Group(const std::string& cat, Owner* owner):
            Group<EndPoint, PBond>(cat), _owner(owner) {}
    virtual  ~Owned_Group() {};
};

}  // namespace pseudobond

#endif  // pseudobonds_Group
