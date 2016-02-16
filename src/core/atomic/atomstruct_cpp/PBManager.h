// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_PBManager
#define atomstruct_PBManager

#include <map>
#include <string>
#include <unordered_map>

#include "imex.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class ChangeTracker;
class CoordSet;
class Graph;
class Proxy_PBGroup;
class Pseudobond;

class BaseManager {
public:
    // so that subclasses can create multiple types of groups...
    static const int GRP_NONE = 0;
    static const int GRP_NORMAL = GRP_NONE + 1;
    typedef std::map<std::string, Proxy_PBGroup*>  GroupMap;
    typedef std::map<Graph*, int>  SessionStructureToIDMap;
    typedef std::map<int, Graph*>  SessionIDToStructureMap;
protected:
    ChangeTracker*  _change_tracker;
    GroupMap  _groups;
    // these maps need to be in the Base Manager (despite the global manager only
    // really using them) since Pseudobond adds to them and only distinguishes
    // global from per-structure at run time
    mutable SessionStructureToIDMap*  _ses_struct_to_id_map;
    mutable SessionIDToStructureMap*  _ses_id_to_struct_map;
public:
    BaseManager(ChangeTracker* ct): _change_tracker(ct) {}
    virtual  ~BaseManager();

    ChangeTracker*  change_tracker() { return _change_tracker; }
    void  clear();
    void  delete_group(Proxy_PBGroup*);
    virtual Proxy_PBGroup*  get_group(
            const std::string& name, int create = GRP_NONE) = 0;
    const GroupMap&  group_map() const { return _groups; }
    SessionStructureToIDMap*  ses_struct_to_id_map() const { return _ses_struct_to_id_map; }
    SessionIDToStructureMap*  ses_id_to_struct_map() const { return _ses_id_to_struct_map; }
    int  session_info(PyObject** ints, PyObject** floats, PyObject** misc) const;
    typedef std::unordered_map<const Pseudobond*, int> SessionSavePbMap;
    mutable SessionSavePbMap* session_save_pbs;
    void  session_save_setup() const {
        session_save_pbs = new SessionSavePbMap;
        _ses_struct_to_id_map = new SessionStructureToIDMap;
    }
    void  session_save_teardown() const {
        delete session_save_pbs;
        delete _ses_struct_to_id_map;
    }
    void  session_restore(int version, int** ints, float** floats, PyObject* misc);
    typedef std::unordered_map<int, const Pseudobond*> SessionRestorePbMap;
    mutable SessionRestorePbMap* session_restore_pbs;
    void  session_restore_setup() const {
        session_restore_pbs = new SessionRestorePbMap;
        _ses_id_to_struct_map = new SessionIDToStructureMap;
    }
    void  session_restore_teardown() const {
        delete session_restore_pbs;
        delete _ses_id_to_struct_map;
    }
};

class StructureManager: public BaseManager {
protected:
    Graph*  _structure;
public:
    StructureManager(Graph* structure);
    virtual  ~StructureManager() {}

    Graph*  structure() const { return _structure; }
};

// global pseudobond manager
// Though for C++ purposes it could use PBGroup instead of Proxy_PBGroup,
// using proxy groups allows them to be treated uniformly on the Python side
class PBManager: public BaseManager {
public:
    PBManager(ChangeTracker* ct): BaseManager(ct) {}

    Proxy_PBGroup*  get_group(const std::string& name, int create = GRP_NONE);
};

class AS_PBManager: public StructureManager
{
public:
    static const int  GRP_PER_CS = GRP_NORMAL + 1;
private:
    friend class Graph;
    friend class CoordSet;
    AS_PBManager(Graph* as): StructureManager(as) {}

    void  remove_cs(const CoordSet* cs);
public:
    ChangeTracker*  change_tracker() const;
    Proxy_PBGroup*  get_group(const std::string& name, int create = GRP_NONE);
};

}  // namespace atomstruct

#endif  // atomstruct_PBManager
