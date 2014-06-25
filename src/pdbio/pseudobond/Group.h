// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Group
#define pseudobonds_Group

#include <set>
#include <stdexcept>
#include <string>

#include "Link.h"

namespace pseudobond {

template <class EndPoint>
class Group {
protected:
    std::string  _category;
    Link<EndPoint>*  makeLink(EndPoint* e1, EndPoint* e2) {
        // since friendship isn't inherited, make it possible
        // to create Link instances from derived Groups
        return new Link<EndPoint>(e1, e2);
    }
public:
    virtual void  clear() = 0;
    Group(std::string& cat): _category(cat) {}
    virtual  ~Group() {}
    virtual Link<EndPoint>*  newPseudoBond(EndPoint* e1, EndPoint* e2) = 0;
    virtual const std::set<Link<EndPoint>*>&  pseudobonds() const = 0;
};

template <class Owner, class EndPoint>
class Owned_Group: public Group<EndPoint> {
protected:
    Owner*  _owner;
public:
    virtual Link<EndPoint>*  newPseudoBond(EndPoint* e1, EndPoint* e2) = 0;
    Owned_Group(std::string& cat, Owner* owner):
            Group<EndPoint>(cat), _owner(owner) {}
    virtual  ~Owned_Group() {};
};

}  // namespace pseudobond

#endif  // pseudobonds_Group
