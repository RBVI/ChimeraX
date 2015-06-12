// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Graph
#define basegeom_Graph

#include <set>
#include <vector>

#include "destruct.h"

namespace basegeom {
    
template <class Vertex, class Edge>
class Graph {
protected:
    typedef std::vector<Vertex*>  Vertices;
    typedef std::vector<Edge*>  Edges;
private:
    Vertices  _vertices;
    Edges  _edges;

    float  _ball_scale;

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
    void  set_ball_scale(float bs) { _ball_scale = bs; }

    // temporary until a Model class exists
private:
    bool  _display = true;
public:
    bool  display() const { return _display; }
    void  set_display(bool d) { _display = d; }
};

} //  namespace basegeom

#endif  // basegeom_Graph
