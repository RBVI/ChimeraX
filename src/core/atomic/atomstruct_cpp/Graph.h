// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Graph
#define atomstruct_Graph

#include <set>
#include <vector>

#include "ChangeTracker.h"
#include "destruct.h"
#include "Rgba.h"

namespace atomstruct {

class Atom;
class Bond;
    
class GraphicsContainer {
private:
    bool  _gc_color:1;
    bool  _gc_select:1;
    bool  _gc_shape:1;
    
public:
    GraphicsContainer(): _gc_color(false), _gc_select(false),
        _gc_shape(false) {}
    virtual  ~GraphicsContainer() {}
    virtual void  gc_clear()
        { _gc_color = false; _gc_select = false; _gc_shape = false; }
    virtual bool  get_gc_color() const { return _gc_color; }
    virtual bool  get_gc_select() const { return _gc_select; }
    virtual bool  get_gc_shape() const { return _gc_shape; }
    virtual void  set_gc_color(bool gc = true) { _gc_color = gc; }
    virtual void  set_gc_select(bool gc = true) { _gc_select = gc; }
    virtual void  set_gc_shape(bool gc = true) { _gc_shape = gc; }
};

class Graph: public GraphicsContainer {
public:
    typedef std::vector<Atom*>  Atoms;
    typedef std::vector<Bond*>  Bonds;

protected:
    Atoms  _atoms;
    Bonds  _bonds;

    float  _ball_scale = 0.25;
    ChangeTracker*  _change_tracker;
    bool  _display = true;

    void  add_bond(Bond* b) { _bonds.emplace_back(b); }
    void  add_atom(Atom* a) { _atoms.emplace_back(a); }
    void  delete_bond(Bond* b);
    void  delete_atom(Atom* a);
    void  delete_atoms(const std::set<Atom*>& atoms);

public:
    Graph(): _change_tracker(DiscardingChangeTracker::discarding_change_tracker()) {}
    virtual  ~Graph();

    const Atoms&  atoms() const { return _atoms; }
    const Bonds&  bonds() const { return _bonds; }

    // graphics related
    float  ball_scale() const { return _ball_scale; }
    ChangeTracker*  change_tracker() { return _change_tracker; }
    bool  display() const { return _display; }
    void  set_ball_scale(float bs) {
        if (bs == _ball_scale)
            return;
        set_gc_shape();
        _ball_scale = bs;
        change_tracker()->add_modified(this, ChangeTracker::REASON_BALL_SCALE);
    }
    virtual void  set_color(const Rgba&);
    void  set_display(bool d) {
        if (d == _display)
            return;
        set_gc_shape();
        _display = d;
        change_tracker()->add_modified(this, ChangeTracker::REASON_DISPLAY);
    }
    virtual void  start_change_tracking(ChangeTracker* ct) { _change_tracker = ct; }
};

} //  namespace atomstruct

#endif  // atomstruct_Graph
