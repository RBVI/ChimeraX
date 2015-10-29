// vi: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Group
#define pseudobonds_Group

#include <set>
#include <stdexcept>
#include <string>

#include <basegeom/destruct.h>
#include <basegeom/Graph.h>
#include <basegeom/Rgba.h>

namespace pseudobond {

using basegeom::Rgba;

template <class EndPoint, class PBond>
class Group:
    public basegeom::DestructionObserver, public basegeom::GraphicsContainer {
protected:
    std::string  _category;
    Rgba  _default_color = {255,255,0,255}; // yellow
    bool  _default_halfbond = false;
    bool  _destruction_relevant;

    // the manager will need to be declared as a friend...
    Group(const std::string& cat): _category(cat), _destruction_relevant(true) { }
    virtual  ~Group() {}

    // can't call pure virtuals from base class destructors, so
    // make the code easily available to derived classes...
    void  dtor_code() {
        _destruction_relevant = false;
        auto du = basegeom::DestructionUser(this);
        for (auto pb: pseudobonds())
            delete pb;
    }
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
    virtual PBond*  new_pseudobond(EndPoint* e1, EndPoint* e2) = 0;
    virtual const std::set<PBond*>&  pseudobonds() const = 0;
    virtual void  set_default_color(const Rgba& rgba) { _default_color = rgba; }
    virtual void  set_default_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b,
        Rgba::Channel a = 255) { this->set_default_color(Rgba(r,g,b,a)); }
    virtual void  set_default_halfbond(bool hb) { _default_halfbond = hb; }
};

template <class Owner, class EndPoint, class PBond>
class Owned_Group: public Group<EndPoint, PBond> {
protected:
    Owner*  _owner;
public:
    Owned_Group(const std::string& cat, Owner* owner):
            Group<EndPoint, PBond>(cat), _owner(owner) {}
    virtual  ~Owned_Group() {};

    virtual PBond*  new_pseudobond(EndPoint* e1, EndPoint* e2) = 0;
    Owner*  owner() const { return _owner; }
};

}  // namespace pseudobond

#endif  // pseudobonds_Group
