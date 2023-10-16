// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_PBManager
#define atomstruct_PBManager

#include <map>
#include <pyinstance/PythonInstance.declare.h>
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
class Structure;
class Proxy_PBGroup;
class Pseudobond;

class ATOMSTRUCT_IMEX BaseManager: public pyinstance::PythonInstance<BaseManager> {
public:
    // so that subclasses can create multiple types of groups...
    static const int GRP_NONE = 0;
    static const int GRP_NORMAL = GRP_NONE + 1;
    typedef std::map<std::string, Proxy_PBGroup*>  GroupMap;
    typedef std::map<Structure*, int>  SessionStructureToIDMap;
    typedef std::map<int, Structure*>  SessionIDToStructureMap;
protected:
    ChangeTracker*  _change_tracker;
    GroupMap  _groups;
    // these maps need to be in the Base Manager (despite the global manager only
    // really using them) since Pseudobond adds to them and only distinguishes
    // global from per-structure at run time
    mutable SessionStructureToIDMap*  _ses_struct_to_id_map = nullptr;
    mutable SessionIDToStructureMap*  _ses_id_to_struct_map = nullptr;
public:
    BaseManager(ChangeTracker* ct): _change_tracker(ct) {}
    virtual  ~BaseManager();

    ChangeTracker*  change_tracker() { return _change_tracker; }
    void  clear();
    void  change_category(Proxy_PBGroup*, std::string&);
    void  delete_group(Proxy_PBGroup*);
    virtual Proxy_PBGroup*  get_group(const std::string& name) const;
    virtual Proxy_PBGroup*  get_group(const std::string& name, int create) = 0;
    const GroupMap&  group_map() const { return _groups; }
    SessionStructureToIDMap*  ses_struct_to_id_map() const { return _ses_struct_to_id_map; }
    SessionIDToStructureMap*  ses_id_to_struct_map() const { return _ses_id_to_struct_map; }
    int  session_info(PyObject** ints, PyObject** floats, PyObject** misc) const;
    typedef std::unordered_map<const Pseudobond*, int> SessionSavePbMap;
    mutable SessionSavePbMap* session_save_pbs = nullptr;
    void  session_save_setup() const;
    void  session_save_teardown() const {
        delete session_save_pbs;
        delete _ses_struct_to_id_map;
        session_save_pbs = nullptr;
        _ses_struct_to_id_map = nullptr;
    }
    void  session_restore(int version, int** ints, float** floats, PyObject* misc);
    typedef std::unordered_map<int, const Pseudobond*> SessionRestorePbMap;
    mutable SessionRestorePbMap* session_restore_pbs = nullptr;
    void  session_restore_setup() const {
        session_restore_pbs = new SessionRestorePbMap;
        _ses_id_to_struct_map = new SessionIDToStructureMap;
    }
    void  session_restore_teardown() const {
        delete session_restore_pbs;
        delete _ses_id_to_struct_map;
        session_restore_pbs = nullptr;
        _ses_id_to_struct_map = nullptr;
    }
    void  start_change_tracking(ChangeTracker* ct) { _change_tracker = ct; }
};

class ATOMSTRUCT_IMEX StructureManager: public BaseManager {
protected:
    Structure*  _structure;
public:
    StructureManager(Structure* structure);
    virtual  ~StructureManager() {}

    Structure*  structure() const { return _structure; }
};

// global pseudobond manager
// Though for C++ purposes it could use PBGroup instead of Proxy_PBGroup,
// using proxy groups allows them to be treated uniformly on the Python side
class ATOMSTRUCT_IMEX PBManager: public BaseManager {
public:
    PBManager(ChangeTracker* ct): BaseManager(ct) {}

    Proxy_PBGroup*  get_group(const std::string& name) const {
        return BaseManager::get_group(name);
    }
    Proxy_PBGroup*  get_group(const std::string& name, int create);
};

class ATOMSTRUCT_IMEX AS_PBManager: public StructureManager
{
public:
    static const int  GRP_PER_CS = GRP_NORMAL + 1;
private:
    friend class Structure;
    friend class CoordSet;
    AS_PBManager(Structure* as): StructureManager(as) {}

    void  change_cs(const CoordSet* cs);
    void  remove_cs(const CoordSet* cs);
public:
    ChangeTracker*  change_tracker() const;
    Proxy_PBGroup*  get_group(const std::string& name) const {
        return BaseManager::get_group(name);
    }
    Proxy_PBGroup*  get_group(const std::string& name, int create);
};

}  // namespace atomstruct

#endif  // atomstruct_PBManager
