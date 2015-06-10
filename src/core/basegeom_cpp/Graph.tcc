// vi: set expandtab ts=4 sw=4:

#include <algorithm>
#include <set>

#include "Connectible.tcc"
#include "destruct.h"
#include "Graph.h"

namespace basegeom {
    
template <class Vertex, class Edge>
void
Graph<Vertex, Edge>::delete_edge(Edge *e)
{
    typename Edges::iterator i = std::find_if(_edges.begin(), _edges.end(),
        [&e](Edge* ue) { return ue == e; });
    if (i == _edges.end())
        throw std::invalid_argument("delete_edge called for Edge not in Graph");
    auto db = DestructionBatcher(this);
    for (auto v: e->end_points())
        v->remove_connection(e);
    _edges.erase(i);
    delete e;
}

template <class Vertex, class Edge>
void
Graph<Vertex, Edge>::delete_vertex(Vertex *v)
{
    typename Vertices::iterator i = std::find_if(_vertices.begin(), _vertices.end(),
        [&v](Vertex* uv) { return uv == v; });
    if (i == _vertices.end())
        throw std::invalid_argument("delete_vertex called for Vertex not in Graph");
    auto db = DestructionBatcher(this);
    for (auto e: v->connections())
        e->other_end(v)->remove_connection(e);
    _vertices.erase(i);
    delete v;
}

template <class Vertex, class Edge>
void
Graph<Vertex, Edge>::delete_vertices(const std::set<Vertex*>& vertices)
{
    auto db = DestructionBatcher(this);
    // remove_if doesn't swap the removed items into the end of the vector,
    // so can't just go through the tail of the vector and delete things,
    // need to delete them as part of the lambda
    auto new_v_end = std::remove_if(_vertices.begin(), _vertices.end(),
        [&vertices](Vertex* v) { 
            bool rm = vertices.find(v) != vertices.end();
            if (rm) delete v; return rm;
        });
    _vertices.erase(new_v_end, _vertices.end());

    for (auto v: _vertices) {
        std::vector<Edge*> removals;
        for (auto e: v->connections()) {
            if (vertices.find(e->other_end(v)) != vertices.end())
                removals.push_back(e);
        }
        for (auto e: removals)
            v->remove_connection(e);
    }

    auto new_e_end = std::remove_if(_edges.begin(), _edges.end(),
        [&vertices](Edge* e) {
            bool rm = vertices.find(e->end_points()[0]) != vertices.end()
            || vertices.find(e->end_points()[1]) != vertices.end();
            if (rm) delete e; return rm;
        });
    _edges.erase(new_e_end, _edges.end());
}

} //  namespace basegeom
