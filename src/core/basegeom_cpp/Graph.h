// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Graph
#define basegeom_Graph

#include <set>
#include <vector>

#include "destruct.h"

namespace basegeom {
    
class GraphicsContainer {
private:
    bool  _gc_redraw:1;
    bool  _gc_select:1;
    bool  _gc_shape:1;
    
public:
    GraphicsContainer(): _gc_redraw(false), _gc_select(false),
        _gc_shape(false) {}
    virtual  ~GraphicsContainer() {}
    void  gc_clear()
        { _gc_redraw = false; _gc_select = false; _gc_shape = false; }
    bool  get_gc_redraw() const { return _gc_redraw; }
    bool  get_gc_select() const { return _gc_select; }
    bool  get_gc_shape() const { return _gc_shape; }
    void  set_gc_redraw(bool gc = true) { _gc_redraw = gc; }
    void  set_gc_select(bool gc = true) { _gc_select = gc; }
    void  set_gc_shape(bool gc = true) { _gc_shape = gc; }
};

template <class Vertex, class Edge>
class Graph: public GraphicsContainer {
protected:
    typedef std::vector<Vertex*>  Vertices;
    typedef std::vector<Edge*>  Edges;
private:
    Vertices  _vertices;
    Edges  _edges;

    float  _ball_scale;
    bool  _display = true;

protected:
    void  add_edge(Edge *e) { _edges.emplace_back(e); }
    void  add_vertex(Vertex *v) { _vertices.emplace_back(v); }
    void  delete_edge(Edge *e);
    void  delete_vertex(Vertex *v);
    void  delete_vertices(const Vertices& vs) {
        delete_vertices(std::set<Vertex*>(vs.begin(), vs.end()));
    }
    void  delete_vertices(const std::set<Vertex*>& vs);
    const Edges &  edges() const { return _edges; }
    const Vertices &  vertices() const { return _vertices; }

public:
    virtual  ~Graph() {
        // need to assign to variable make it live to end of destructor
        auto du = DestructionUser(this);
        for (auto e: _edges)
            delete e;
        for (auto v: _vertices)
            delete v;
    }

    // graphics related
    float  ball_scale() const { return _ball_scale; }
    bool  display() const { return _display; }
    void  set_ball_scale(float bs) { set_gc_shape(); _ball_scale = bs; }
    void  set_display(bool d) { set_gc_shape(); _display = d; }
};

} //  namespace basegeom

#endif  // basegeom_Graph
