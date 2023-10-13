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

#ifndef atomstruct_PBGroup
#define atomstruct_PBGroup

#include <pyinstance/PythonInstance.declare.h>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ChangeTracker.h"
#include "destruct.h"
#include "imex.h"
#include "PBManager.h"
#include "Rgba.h"
#include "session.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class Atom;
class CoordSet;
class Pseudobond;
class Proxy_PBGroup;
class Structure;

class ATOMSTRUCT_IMEX PBGroup: public DestructionObserver, public GraphicsChanges,
        public pyinstance::PythonInstance<PBGroup> {
public:
    typedef std::set<Pseudobond*>  Pseudobonds;

    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 1; }
    static int  SESSION_NUM_FLOATS(int version=CURRENT_SESSION_VERSION) {
        return version < 7 ? 0: 1;
    }
protected:
    std::string  _category;
    Rgba  _color = {255,215,0,255}; // gold
    bool  _destruction_relevant;
    bool  _halfbond = false;
    BaseManager*  _manager;
    friend class Proxy_PBGroup;
    Proxy_PBGroup* _proxy; // the proxy for this group
    float  _radius = 0.1f;
    Structure*  _structure = nullptr;

    // the manager will need to be declared as a friend...
    PBGroup(const std::string& cat, BaseManager* manager);
    virtual  ~PBGroup() { }
    // can't call pure virtuals from base class destructors, so
    // make the code easily available to derived classes...
    void  dtor_code();

    void _check_destroyed_atoms(PBGroup::Pseudobonds& pbonds, const std::set<void*>& destroyed);
    void delete_pbs_check(const std::set<Pseudobond*>& pbs) const;
public:
    virtual void  clear() = 0;
    virtual const Rgba&  color() const { return _color; }
    virtual const std::string&  category() const { return _category; }
    virtual void  change_category(std::string& category) {
        _manager->change_category(this->_proxy, category); // may throw invalid_argument
        _category = category;
    }
    virtual void  check_destroyed_atoms(const std::set<void*>& destroyed) = 0;
    virtual void  delete_pseudobond(Pseudobond* pb) = 0;
    virtual void  delete_pseudobonds(const std::set<Pseudobond*>& pbs) = 0;
    virtual void  destructors_done(const std::set<void*>& destroyed) {
        if (!_destruction_relevant)
            return;
        check_destroyed_atoms(destroyed);
    }
    virtual bool  halfbond() const { return _halfbond; }
    BaseManager*  manager() const { return _manager; }
    virtual Pseudobond*  new_pseudobond(Atom* e1, Atom* e2) = 0;
    Proxy_PBGroup*  proxy() const { return _proxy; }
    virtual const Pseudobonds&  pseudobonds() const = 0;
    virtual float  radius() const { return _radius; }
    static int  session_num_floats(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_FLOATS(version) + Rgba::session_num_floats();
    }
    static int  session_num_ints(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_INTS(version) + Rgba::session_num_ints();
    }
    virtual void  session_restore(int version, int**, float**);
    virtual void  session_save(int**, float**) const;
    virtual void  session_save_setup() const = 0;
    virtual void  set_color(const Rgba& rgba);
    virtual void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a = 255) { this->set_color(Rgba(r,g,b,a)); }
    virtual void  set_halfbond(bool hb);
    virtual void  set_radius(float r);
    Structure*  structure() const { return _structure; }

    // change tracking
    void  track_change(const std::string& reason) const {
        manager()->change_tracker()->add_modified(structure(), proxy(), reason);
    }
};

// in per-AtomicStructure groups there are per-CoordSet groups
// and overall groups...
class ATOMSTRUCT_IMEX StructurePBGroupBase: public PBGroup {
public:
    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }
protected:
    friend class AS_PBManager;
    void  _check_structure(Atom* a1, Atom* a2);
    StructurePBGroupBase(const std::string& cat, Structure* as, BaseManager* manager):
        PBGroup(cat, manager) { _structure = as; }
    virtual  ~StructurePBGroupBase() {}
public:
    virtual Pseudobond*  new_pseudobond(Atom* e1, Atom* e2) = 0;
    std::pair<Atom*, Atom*> session_get_pb_ctor_info(int** ints) const;
    void  session_note_pb_ctor_info(Pseudobond* pb, int** ints) const;
    static int  session_num_floats(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_FLOATS(version) + PBGroup::session_num_floats();
    }
    static int  session_num_ints(int version=CURRENT_SESSION_VERSION) {
        return SESSION_NUM_INTS(version) + PBGroup::session_num_ints();
    }
    virtual void  session_restore(int version, int** ints, float** floats) {
        PBGroup::session_restore(version, ints, floats);
    }
    virtual void  session_save(int** ints, float** floats) const {
        PBGroup::session_save(ints, floats);
    }
};

class ATOMSTRUCT_IMEX StructurePBGroup: public StructurePBGroupBase {
public:
    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 1; }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }
private:
    friend class Proxy_PBGroup;
    Pseudobonds  _pbonds;
protected:
    StructurePBGroup(const std::string& cat, Structure* as, BaseManager* manager):
        StructurePBGroupBase(cat, as, manager) {}
    ~StructurePBGroup() { dtor_code(); }
public:
    void  check_destroyed_atoms(const std::set<void*>& destroyed);
    void  clear();
    void  delete_pseudobond(Pseudobond* pb);
    void  delete_pseudobonds(const std::set<Pseudobond*>& pbs);
    void  delete_pseudobonds(const std::vector<Pseudobond*>& pbs) {
        delete_pseudobonds(std::set<Pseudobond*>(pbs.begin(), pbs.end()));
    }
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2);
    const Pseudobonds&  pseudobonds() const { return _pbonds; }
    int  session_num_ints(int version=CURRENT_SESSION_VERSION) const;
    int  session_num_floats(int version=CURRENT_SESSION_VERSION) const;
    void  session_restore(int version, int** , float**);
    void  session_save(int** , float**) const;
    void  session_save_setup() const;
    mutable std::unordered_map<const Pseudobond*, size_t>  *session_save_pbs;
};

class ATOMSTRUCT_IMEX CS_PBGroup: public StructurePBGroupBase
{
public:
    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 1; }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }
private:
    friend class Proxy_PBGroup;
    mutable std::unordered_map<const CoordSet*, Pseudobonds>  _pbonds;
    void  change_cs(const CoordSet* cs);
    void  remove_cs(const CoordSet* cs);
protected:
    CS_PBGroup(const std::string& cat, Structure* as, BaseManager* manager):
        StructurePBGroupBase(cat, as, manager) {}
    ~CS_PBGroup();
public:
    void  check_destroyed_atoms(const std::set<void*>& destroyed);
    void  clear();
    void  delete_pseudobond(Pseudobond* pb);
    void  delete_pseudobonds(const std::set<Pseudobond*>& pbs);
    void  delete_pseudobonds(const std::vector<Pseudobond*>& pbs) {
        delete_pseudobonds(std::set<Pseudobond*>(pbs.begin(), pbs.end()));
    }
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2);
    Pseudobond*  new_pseudobond(Atom* a1, Atom* a2, CoordSet* cs);
    const Pseudobonds&  pseudobonds() const;
    const Pseudobonds&  pseudobonds(const CoordSet* cs) const { return _pbonds[cs]; }
    int  session_num_ints(int version=CURRENT_SESSION_VERSION) const;
    int  session_num_floats(int version=CURRENT_SESSION_VERSION) const;
    void  session_restore(int, int** , float**);
    void  session_save(int** , float**) const;
    void  session_save_setup() const;
    mutable std::unordered_map<const Pseudobond*, size_t>  *session_save_pbs;
    void  set_color(const Rgba& rgba);
    void  set_radius(float r);
};

// Need a proxy class that can be contained/returned by the pseudobond
// manager and that will dispatch calls to the appropriate contained class
class ATOMSTRUCT_IMEX Proxy_PBGroup: public StructurePBGroupBase
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
    Proxy_PBGroup(BaseManager* manager, const std::string& cat, Structure* as):
        StructurePBGroupBase(cat, as, manager) { init(AS_PBManager::GRP_NORMAL); }
    Proxy_PBGroup(BaseManager* manager, const std::string& cat, Structure* as, int grp_type):
        StructurePBGroupBase(cat, as, manager) { init(grp_type); }
    ~Proxy_PBGroup() {
        _destruction_relevant = false;
        auto du = DestructionUser(this);
        manager()->change_tracker()->add_deleted(structure(), this);
        if (_group_type == AS_PBManager::GRP_NORMAL)
            delete static_cast<StructurePBGroup*>(_proxied);
        else
            delete static_cast<CS_PBGroup*>(_proxied);
    }
    void  init(int grp_type) {
        _group_type = grp_type;
        if (grp_type == AS_PBManager::GRP_NORMAL)
            _proxied = new StructurePBGroup(_category, _structure, _manager);
        else
            _proxied = new CS_PBGroup(_category, _structure, _manager);
        _proxy = this;
        static_cast<PBGroup*>(_proxied)->_proxy = this;
        _manager->change_tracker()->add_created(structure(), this);
    }
    void  change_cs(const CoordSet* cs) {
        if (_group_type == AS_PBManager::GRP_PER_CS)
            static_cast<CS_PBGroup*>(_proxied)->change_cs(cs);
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
    void  change_category(std::string& category) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->change_category(category);
        return static_cast<CS_PBGroup*>(_proxied)->change_category(category);
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
    const Rgba&  color() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->color();
        return static_cast<CS_PBGroup*>(_proxied)->color();
    }
    void  delete_pseudobond(Pseudobond* pb) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->delete_pseudobond(pb);
        return static_cast<CS_PBGroup*>(_proxied)->delete_pseudobond(pb);
    }
    void  delete_pseudobonds(const std::set<Pseudobond*>& pbs) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->delete_pseudobonds(pbs);
        return static_cast<CS_PBGroup*>(_proxied)->delete_pseudobonds(pbs);
    }
    void  delete_pseudobonds(const std::vector<Pseudobond*>& pbs) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->delete_pseudobonds(pbs);
        return static_cast<CS_PBGroup*>(_proxied)->delete_pseudobonds(pbs);
    }
    void  destroy() {
        if (structure() == nullptr)
            static_cast<PBManager*>(_manager)->delete_group(this);
        else
            static_cast<AS_PBManager*>(_manager)->delete_group(this);
    }
    int  group_type() const { return _group_type; }
    bool  halfbond() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->halfbond();
        return static_cast<CS_PBGroup*>(_proxied)->halfbond();
    }
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
    float  radius() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->radius();
        return static_cast<CS_PBGroup*>(_proxied)->radius();
    }
    int  session_num_ints() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_num_ints();
        return static_cast<CS_PBGroup*>(_proxied)->session_num_ints();
    }
    int  session_num_floats() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_num_floats();
        return static_cast<CS_PBGroup*>(_proxied)->session_num_floats();
    }
    void  session_restore(int version, int** ints, float** floats) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_restore(version, ints, floats);
        return static_cast<CS_PBGroup*>(_proxied)->session_restore(version, ints, floats);
    }
    void  session_save(int** ints, float** floats) const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_save(ints, floats);
        return static_cast<CS_PBGroup*>(_proxied)->session_save(ints, floats);
    }
    void  session_save_setup() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->session_save_setup();
        return static_cast<CS_PBGroup*>(_proxied)->session_save_setup();
    }
    void  set_color(const Rgba& rgba) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_color(rgba);
        else
            static_cast<CS_PBGroup*>(_proxied)->set_color(rgba);
    }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a = 255) { set_color(Rgba(r,g,b,a)); }
    void  set_halfbond(bool hb) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_halfbond(hb);
        else
            static_cast<CS_PBGroup*>(_proxied)->set_halfbond(hb);
    }
    void  set_radius(float r) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_radius(r);
        else
            static_cast<CS_PBGroup*>(_proxied)->set_radius(r);
    }
    Structure*  structure() const { 
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->structure();
        return static_cast<CS_PBGroup*>(_proxied)->structure();
    }

    void  gc_clear() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->gc_clear();
    else
        static_cast<CS_PBGroup*>(_proxied)->gc_clear();
    }
    int   get_graphics_changes() const {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            return static_cast<StructurePBGroup*>(_proxied)->get_graphics_changes();
        return static_cast<CS_PBGroup*>(_proxied)->get_graphics_changes();
    }
    void  set_gc_color() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_color();
    else
        static_cast<CS_PBGroup*>(_proxied)->set_gc_color();
    }
    void  set_gc_select() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_select();
    else
        static_cast<CS_PBGroup*>(_proxied)->set_gc_select();
    }
    void  set_gc_shape() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_shape();
    else
        static_cast<CS_PBGroup*>(_proxied)->set_gc_shape();
    }
    void  set_gc_ribbon() {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_gc_ribbon();
    else
        static_cast<CS_PBGroup*>(_proxied)->set_gc_ribbon();
    }
    void  set_graphics_changes(int change) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_graphics_changes(change);
    else
        static_cast<CS_PBGroup*>(_proxied)->set_graphics_changes(change);
    }
    void  set_graphics_change(ChangeType type) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->set_graphics_change(type);
    else
        static_cast<CS_PBGroup*>(_proxied)->set_graphics_change(type);
    }
    void  clear_graphics_change(ChangeType type) {
        if (_group_type == AS_PBManager::GRP_NORMAL)
            static_cast<StructurePBGroup*>(_proxied)->clear_graphics_change(type);
    else
        static_cast<CS_PBGroup*>(_proxied)->clear_graphics_change(type);
    }
};

}  // namespace atomstruct

#include "Pseudobond.h"

namespace atomstruct {
    
inline void  PBGroup::set_color(const Rgba& rgba) {
    _color = rgba;
    for (auto pb: pseudobonds())
        pb->set_color(rgba);
    track_change(ChangeTracker::REASON_COLOR);
}

inline void  PBGroup::set_halfbond(bool hb) {
    _halfbond = hb;
    for (auto pb: pseudobonds())
        pb->set_halfbond(hb);
    track_change(ChangeTracker::REASON_HALFBOND);
}

inline void  PBGroup::set_radius(float r) {
    _radius = r;
    for (auto pb: pseudobonds())
        pb->set_radius(r);
    track_change(ChangeTracker::REASON_RADIUS);
}

}  // namespace atomstruct

#endif  // atomstruct_PBGroup
