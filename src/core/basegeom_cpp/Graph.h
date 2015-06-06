// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Graph
#define basegeom_Graph

#include <vector>
#include <algorithm>

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

template <class Vertex, class Edge>
void
Graph<Vertex, Edge>::delete_edge(Edge *e)
{
    typename Edges::iterator i = std::find_if(_edges.begin(), _edges.end(),
        [&e](Edge* ue) { return ue == e; });
    if (i == _edges.end())
        throw std::invalid_argument("delete_edge called for Edge not in Graph");
    _edges.erase(i);
}

template <class Vertex, class Edge>
void
Graph<Vertex, Edge>::delete_vertex(Vertex *v)
{
    typename Vertices::iterator i = std::find_if(_vertices.begin(), _vertices.end(),
        [&v](Vertex* uv) { return uv == v; });
    if (i == _vertices.end())
        throw std::invalid_argument("delete_vertex called for Vertex not in Graph");
    _vertices.erase(i);
}

} //  namespace basegeom

#endif  // basegeom_Graph
