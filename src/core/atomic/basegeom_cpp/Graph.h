// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Graph
#define basegeom_Graph

#include <set>
#include <vector>

#include "ChangeTracker.h"
#include "destruct.h"
#include "Rgba.h"

namespace basegeom {

using ::basegeom::ChangeTracker;
    
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

template <class FinalGraph, class Node, class Edge>
class Graph: public GraphicsContainer {
protected:
    typedef std::vector<Node*>  Nodes;
    typedef std::vector<Edge*>  Edges;
private:
    Nodes  _nodes;
    Edges  _edges;

    float  _ball_scale = 0.25;
    ChangeTracker*  _change_tracker;
    bool  _display = true;

protected:
    void  add_edge(Edge *e) { _edges.emplace_back(e); }
    void  add_node(Node *v) { _nodes.emplace_back(v); }
    void  delete_edge(Edge *e);
    void  delete_node(Node *v);
    void  delete_nodes(const Nodes& ns) {
        delete_nodes(std::set<Node*>(ns.begin(), ns.end()));
    }
    void  delete_nodes(const std::set<Node*>& ns);
    const Edges &  edges() const { return _edges; }
    const Nodes &  nodes() const { return _nodes; }

public:
    Graph(): _change_tracker(DiscardingChangeTracker::discarding_change_tracker()) {}
    virtual  ~Graph() {
        // need to assign to variable make it live to end of destructor
        auto du = DestructionUser(this);
        for (auto e: _edges)
            delete e;
        for (auto n: _nodes)
            delete n;
    }

    // graphics related
    float  ball_scale() const { return _ball_scale; }
    ChangeTracker*  change_tracker() { return _change_tracker; }
    bool  display() const { return _display; }
    void  set_ball_scale(float bs) {
        if (bs == _ball_scale)
            return;
        set_gc_shape();
        _ball_scale = bs;
        change_tracker()->add_modified(dynamic_cast<FinalGraph*>(this),
            ChangeTracker::REASON_BALL_SCALE);
    }
    virtual void  set_color(const Rgba&);
    void  set_display(bool d) {
        if (d == _display)
            return;
        set_gc_shape();
        _display = d;
        change_tracker()->add_modified(dynamic_cast<FinalGraph*>(this),
            ChangeTracker::REASON_DISPLAY);
    }
    virtual void  start_change_tracking(ChangeTracker* ct) { _change_tracker = ct; }
};

} //  namespace basegeom

#endif  // basegeom_Graph
