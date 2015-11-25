// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_PBGroup
#define atomstruct_PBGroup

#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <basegeom/destruct.h>
#include <basegeom/Graph.h>
#include <basegeom/Rgba.h>

#include "imex.h"
#include "PBManager.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class Atom;
class AtomicStructure;
class CoordSet;
class Pseudobond;

using basegeom::Rgba;

class Group: public basegeom::DestructionObserver, public basegeom::GraphicsContainer {
public:
    typedef std::set<Pseudobond*>  Pseudobonds;

    static const int  SESSION_NUM_INTS = 1;
    static const int  SESSION_NUM_FLOATS = 0;
protected:
    std::string  _category;
    Rgba  _default_color = {255,255,0,255}; // yellow
    bool  _default_halfbond = false;
    bool  _destruction_relevant;
    BaseManager*  _manager;

    // the manager will need to be declared as a friend...
    Group(const std::string& cat, BaseManager* manager):
        _category(cat), _destruction_relevant(true), _manager(manager) { }
    virtual  ~Group() {}

    // can't call pure virtuals from base class destructors, so
    // make the code easily available to derived classes...
    void  dtor_code();
public:
    virtual void  clear() = 0;
    virtual const std::string&  category() const { return _category; }
    virtual void  check_destroyed_atoms(const std::set<void*>& destroyed) = 0;
    virtual void  destructors_done(const std::set<void*>& destroyed) {
        if (!_destruction_relevant)
            return;
        check_destroyed_atoms(destroyed);
    }
    virtual const Rgba&  get_default_color() const { return _default_color; }
    virtual bool  get_default_halfbond() const { return _default_halfbond; }
    BaseManager*  manager() const { return _manager; }
    virtual Pseudobond*  new_pseudobond(Atom* e1, Atom* e2) = 0;
    virtual const std::set<Pseudobond*>&  pseudobonds() const = 0;
    static int  session_num_floats(bool /*global*/ = false) {
        return SESSION_NUM_FLOATS + Rgba::session_num_floats();
    }
    static int  session_num_ints(bool /*global*/ = false) {
        return SESSION_NUM_INTS + Rgba::session_num_ints();
    }
    virtual void  session_save(int**, float**, PyObject*, bool /*global*/ = false) const;
    virtual void  set_default_color(const Rgba& rgba) { _default_color = rgba; }
    virtual void  set_default_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a = 255) { this->set_default_color(Rgba(r,g,b,a)); }
    virtual void  set_default_halfbond(bool hb) { _default_halfbond = hb; }
};

// in per-AtomicStructure groups there are per-CoordSet groups
// and overall groups...
class StructurePBGroupBase: public Group {
public:
    static const int  SESSION_NUM_INTS = 0;
    static const int  SESSION_NUM_FLOATS = 0;
protected:
    friend class AS_PBManager;
    void  _check_structure(Atom* a1, Atom* a2);
    AtomicStructure*  _structure;
    StructurePBGroupBase(const std::string& cat, AtomicStructure* as, BaseManager* manager):
        Group(cat, manager), _structure(as) {}
    virtual  ~StructurePBGroupBase() {}
public:
    virtual Pseudobond*  new_pseudobond(Atom* e1, Atom* e2) = 0;
    static int  session_num_floats(bool /*global*/ = false) {
        return SESSION_NUM_FLOATS + Group::session_num_floats();
    }
    static int  session_num_ints(bool /*global*/ = false) {
        return SESSION_NUM_INTS + Group::session_num_ints();
    }
    virtual void  session_save(int** ints, float** floats, PyObject* misc,
        bool global = false) const { Group::session_save(ints, floats, misc, global); }
    AtomicStructure*  structure() const { return _structure; }
};

class StructurePBGroup: public StructurePBGroupBase {
public:
    static const int  SESSION_NUM_INTS = 0;
    static const int  SESSION_NUM_FLOATS = 0;
private:
    friend class Proxy_PBGroup;
    Pseudobonds  _pbonds;
protected:
    StructurePBGroup(const std::string& cat, AtomicStructure* as, BaseManager* manager):
        StructurePBGroupBase(cat, as, manager) {}
    ~StructurePBGroup() { dtor_code(); }
public:
    void  check_destroyed_atoms(const std::set<void*>& destroyed);
    void  clear();
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2);
    const Pseudobonds&  pseudobonds() const { return _pbonds; }
    int  session_num_ints(bool global = false) const;
    int  session_num_floats(bool global = false) const;
    virtual void  session_save(int** , float** , PyObject* , bool global = false) const;
};

class CS_PBGroup: public StructurePBGroupBase
{
private:
    friend class Proxy_PBGroup;
    mutable std::unordered_map<const CoordSet*, Pseudobonds>  _pbonds;
    void  remove_cs(const CoordSet* cs) { _pbonds.erase(cs); }
protected:
    CS_PBGroup(const std::string& cat, AtomicStructure* as, BaseManager* manager):
        StructurePBGroupBase(cat, as, manager) {}
    ~CS_PBGroup();
public:
    void  check_destroyed_atoms(const std::set<void*>& destroyed);
    void  clear();
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2);
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs);
    const Pseudobonds&  pseudobonds() const;
    const Pseudobonds&  pseudobonds(const CoordSet* cs) const { return _pbonds[cs]; }
    int  session_num_ints(bool global = false) const;
    int  session_num_floats(bool global = false) const;
    virtual void  session_save(int** , float** , PyObject* , bool global = false) const;
};

// Need a proxy class that can be contained/returned by the pseudobond
// manager and that will dispatch calls to the appropriate contained class
class Proxy_PBGroup: public StructurePBGroupBase
{
private:
    friend class AS_PBManager;
    friend class StructureManager;
    friend class BaseManager;
    friend class PBManager;
    int  _group_type;
    void*  _proxied;

    Proxy_PBGroup(BaseManager* manager, const std::string& cat):
        StructurePBGroupBase(cat, nullptr, manager) { init(AS_PBManager::GRP_NORMAL); }
    Proxy_PBGroup(BaseManager* manager, const std::string& cat, AtomicStructure* as):
        StructurePBGroupBase(cat, as, manager) { init(AS_PBManager::GRP_NORMAL); }
    Proxy_PBGroup(BaseManager* manager, const std::string& cat, AtomicStructure* as, int grp_type):
        StructurePBGroupBase(cat, as, manager) { init(grp_type); }
    ~Proxy_PBGroup() {
        auto du = basegeom::DestructionUser(this);
        if (_group_type == AS_PBManager::GRP_NORMAL)
            delete static_cast<StructurePBGroup*>(_proxied);
        else
            delete static_cast<CS_PBGroup*>(_proxied);
        manager()->change_tracker()->add_deleted(this);
    }
    void  init(int grp_type) {
        _group_type = grp_type;
        if (grp_type == AS_PBManager::GRP_NORMAL)
            _proxied = new StructurePBGroup(_category, _structure, _manager);
        else
            _proxied = new CS_PBGroup(_category, _structure, _manager);
    }
    void  remove_cs(const CoordSet* cs) {
        if (_group_type == AS_PBManager::GRP_PER_CS)
            static_cast<CS_PBGroup*>(_proxied)->remove_cs(cs);
    }

public:
    const std::string&  category() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->category();
        return static_cast<CS_PBGroup*>(_proxied)->category();
    }
    void  check_destroyed_atoms(const std::set<void*>& destroyed) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->check_destroyed_atoms(destroyed);
        static_cast<CS_PBGroup*>(_proxied)->check_destroyed_atoms(destroyed);
    }
    void  clear() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->clear();
        static_cast<CS_PBGroup*>(_proxied)->clear();
    }
    void  destroy() {
        if (structure() == nullptr)
            static_cast<PBManager*>(_manager)->delete_group(this);
        else
            static_cast<AS_PBManager*>(_manager)->delete_group(this);
    }
    const Rgba&  get_default_color() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->get_default_color();
        return static_cast<CS_PBGroup*>(_proxied)->get_default_color();
    }
    bool  get_default_halfbond() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->get_default_halfbond();
        return static_cast<CS_PBGroup*>(_proxied)->get_default_halfbond();
    }
    int  group_type() const { return _group_type; }
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->new_pseudobond(a1, a2);
        return static_cast<CS_PBGroup*>(_proxied)->new_pseudobond(a1, a2);
    }
    Pseudobond*  new_pseudobond(Atom* const ends[2]) {
        // should be in base class, but C++ won't look in base
        // classes for overloads!
        return new_pseudobond(ends[0], ends[1]);
    }
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            throw std::invalid_argument("Not a per-coordset pseudobond group");
        return static_cast<CS_PBGroup*>(_proxied)->new_pseudobond(a1, a2, cs);
    }
    Pseudobond*  new_pseudobond(Atom* const ends[2], CoordSet* cs) {
        return new_pseudobond(ends[0], ends[1], cs);
    }
    const Pseudobonds&  pseudobonds() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->pseudobonds();
        return static_cast<CS_PBGroup*>(_proxied)->pseudobonds();
    }
    const Pseudobonds&  pseudobonds(const CoordSet* cs) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            throw std::invalid_argument("Not a per-coordset pseudobond group");
        return static_cast<CS_PBGroup*>(_proxied)->pseudobonds(cs);
    }
    int  session_num_ints(bool global = false) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_num_ints(global);
        return static_cast<CS_PBGroup*>(_proxied)->session_num_ints(global);
    }
    int  session_num_floats(bool global = false) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_num_floats(global);
        return static_cast<CS_PBGroup*>(_proxied)->session_num_floats(global);
    }
    virtual void  session_save(int** ints, float** floats, PyObject* misc, bool global = false) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_save(ints, floats, misc, global);
        return static_cast<CS_PBGroup*>(_proxied)->session_save(ints, floats, misc, global);
    }
    void  set_default_color(const Rgba& rgba) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_default_color(rgba);
        static_cast<CS_PBGroup*>(_proxied)->set_default_color(rgba);
    }
    void  set_default_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a = 255) { set_default_color(Rgba(r,g,b,a)); }
    void  set_default_halfbond(bool hb) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_default_halfbond(hb);
        static_cast<CS_PBGroup*>(_proxied)->set_default_halfbond(hb);
    }
    AtomicStructure*  structure() const { 
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->structure();
        return static_cast<CS_PBGroup*>(_proxied)->structure();
    }

    virtual void  gc_clear() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->gc_clear();
        static_cast<CS_PBGroup*>(_proxied)->gc_clear();
    }
    virtual bool  get_gc_color() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->get_gc_color();
        return static_cast<CS_PBGroup*>(_proxied)->get_gc_color();
    }
    virtual bool  get_gc_select() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->get_gc_select();
        return static_cast<CS_PBGroup*>(_proxied)->get_gc_select();
    }
    virtual bool  get_gc_shape() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->get_gc_shape();
        return static_cast<CS_PBGroup*>(_proxied)->get_gc_shape();
    }
    virtual void  set_gc_color(bool gc = true) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_color(gc);
        static_cast<CS_PBGroup*>(_proxied)->set_gc_color(gc);
    }
    virtual void  set_gc_select(bool gc = true) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_select(gc);
        static_cast<CS_PBGroup*>(_proxied)->set_gc_select(gc);
    }
    virtual void  set_gc_shape(bool gc = true) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_shape(gc);
        static_cast<CS_PBGroup*>(_proxied)->set_gc_shape(gc);
    }

};

}  // namespace atomstruct

#endif  // atomstruct_PBGroup
