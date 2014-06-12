// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Manager
#define pseudobonds_Manager

#include <map>
#include <string>

#include "Group.h"

template <class Grp_Class>
class Base_Manager {
protected:
    std::map<std::string, Grp_Class>  _groups;
public:
    virtual  ~Base_Manager() {}
    virtual Grp_Class*  get_group(std::string& name, bool create = false) = 0;
};

class Global_Manager: Base_Manager<Group> {
public:
    virtual Group*  get_group(std::string& name, bool create = false);
    virtual  ~Global_Manager() {};
};

template <class Owner>
class Owned_Manager: Base_Manager<Owned_Group<Owner>> {
public:
    virtual Owned_Group<Owner>*  get_group(std::string& name, bool create = false);
    virtual  ~Owned_Manager() {};
};

template <class Owner>
Owned_Group<Owner>*
Owned_Manager<Owner>::get_group(std::string& name, bool create)
{
    auto gmi = this->_groups.find(name);
    if (gmi != this->_groups.end())
        return &(*gmi).second;

    if (!create)
        return nullptr;

    this->_groups.emplace(name, name);
    return &(*this->_groups.find(name)).second;
}

#endif  // pseudobonds_Manager
