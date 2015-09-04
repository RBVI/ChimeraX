// vi: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Manager
#define pseudobonds_Manager

#include <map>
#include <memory>
#include <string>

#include "Group.h"
#include <basegeom/destruct.h>

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace pseudobond {

//classes
template <class Grp_Class>
class Base_Manager {
public:
    // so that subclasses can create multiple types of groups...
    static const int GRP_NONE = 0;
    static const int GRP_NORMAL = GRP_NONE + 1;
    typedef std::map<std::string, Grp_Class*>  GroupMap;
protected:
    GroupMap  _groups;
public:
    Base_Manager() {}
    virtual  ~Base_Manager() {}
    virtual Grp_Class*  get_group(
            const std::string& name, int create = GRP_NONE) = 0;
    const GroupMap&  group_map() const { return _groups; }
};

template <class Owner, class Grp_Class>
class Owned_Manager: public Base_Manager<Grp_Class> {
protected:
    Owner*  _owner;
public:
    Owned_Manager(Owner* owner): _owner(owner) {}
    virtual  ~Owned_Manager() {
        // assign to var so it lives to end of destructor,
        // and only do this in owned managers and not the
        // global manager since the global manager is only
        // destroyed at program exit (and therefore races
        // against the destruction of the DestructionCoordinator)
        // [so move this into ~Base_Manager once global managers
        // are per session]
        auto du = basegeom::DestructionUser(this);
        // delete groups while DestructionUser active
        for (auto name_grp: this->_groups) {
            delete name_grp.second;
        }
    };

    Owner*  owner() const { return _owner; }
    virtual int  session_info(PyObject* ints, PyObject* floats, PyObject* misc) const = 0;
};

}  // namespace pseudobond

#endif  // pseudobonds_Manager
