// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Manager
#define pseudobonds_Manager

#include <map>
#include <memory>
#include <set>
#include <string>

#include "Group.h"
#include "PBond.h"

namespace pseudobond {

template <class Grp_Class>
class Base_Manager {
protected:
    std::map<std::string, Grp_Class>  _groups;
public:
    virtual  ~Base_Manager() {}
    virtual Grp_Class*  get_group(std::string& name, bool create = false) = 0;
};

template <class EndPoint>
class Global_Manager: Base_Manager<Group<std::set<PBond<EndPoint>*>, EndPoint>> {
private:
    static Global_Manager  _manager;
public:
    virtual Group<std::set<PBond<EndPoint>*>, EndPoint>*  get_group(std::string& name, bool create = false);
    virtual  ~Global_Manager() {};
    static Global_Manager&  manager() { return _manager; }
};

template <class Owner, class PBondContainer, class EndPoint>
class Owned_Manager: Base_Manager<Owned_Group<Owner, PBondContainer, EndPoint>> {
public:
    virtual Owned_Group<Owner, PBondContainer, EndPoint>*  get_group(
            std::string& name, bool create = false);
    virtual  ~Owned_Manager() {};
};

template <class Owner, class PBondContainer, class EndPoint>
Owned_Group<Owner, PBondContainer, EndPoint>*
Owned_Manager<Owner, PBondContainer, EndPoint>::get_group(std::string& name, bool create)
{
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end())
        return &(*gmi).second;

    if (!create)
        return nullptr;

    this->_groups.emplace(name, name);
    return &(*this->_groups.find(name)).second;
}

template <class EndPoint>
Group<std::set<PBond<EndPoint>*>, EndPoint>*
Global_Manager<EndPoint>::get_group(std::string& name, bool create)
{
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end())
        return &(*gmi).second;

    if (!create)
        return nullptr;

    this->_groups.emplace(name, name);
    return &(*(this->_groups).find(name)).second;
}

}  // namespace pseudobond

#endif  // pseudobonds_Manager
