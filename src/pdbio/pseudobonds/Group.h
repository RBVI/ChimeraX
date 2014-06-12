// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Group
#define pseudobonds_Group

#include <string>

class Group {
protected:
    std::string  _category;
public:
    Group(std::string& cat): _category(cat) {}
    virtual  ~Group() {}
};

template <class Owner>
class Owned_Group: Group {
protected:
    Owner*  _owner;
public:
    Owned_Group(std::string& cat, Owner* owner): Group(cat), _owner(owner) {}
    virtual  ~Owned_Group() {};
};

#endif  // pseudobonds_Group
