// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Graph_tcc
#define basegeom_Graph_tcc

#include <algorithm>
#include <set>

#include "Connection.h"
#include "destruct.h"
#include "Graph.h"

namespace basegeom {
    
template <class FinalGraph, class Node, class Edge>
void
Graph<FinalGraph, Node, Edge>::delete_edge(Edge *e)
{
    typename Edges::iterator i = std::find_if(_edges.begin(), _edges.end(),
        [&e](Edge* ue) { return ue == e; });
    if (i == _edges.end())
        throw std::invalid_argument("delete_edge called for Edge not in Graph");
    auto db = DestructionBatcher(this);
    for (auto n: e->end_points())
        n->remove_bond(e);
    _edges.erase(i);
    set_gc_shape();
    delete e;
}

template <class FinalGraph, class Node, class Edge>
void
Graph<FinalGraph, Node, Edge>::delete_node(Node *n)
{
    typename Nodes::iterator i = std::find_if(_nodes.begin(), _nodes.end(),
        [&n](Node* un) { return un == n; });
    if (i == _nodes.end())
        throw std::invalid_argument("delete_node called for Node not in Graph");
    auto db = DestructionBatcher(this);
    for (auto e: n->bonds())
        e->other_end(n)->remove_bond(e);
    _nodes.erase(i);
    set_gc_shape();
    delete n;
}

template <class FinalGraph, class Node, class Edge>
void
Graph<FinalGraph, Node, Edge>::delete_nodes(const std::set<Node*>& nodes)
{
    auto db = DestructionBatcher(this);
    // remove_if doesn't swap the removed items into the end of the vector,
    // so can't just go through the tail of the vector and delete things,
    // need to delete them as part of the lambda
    auto new_n_end = std::remove_if(_nodes.begin(), _nodes.end(),
        [&nodes](Node* n) { 
            bool rm = nodes.find(n) != nodes.end();
            if (rm) delete n; return rm;
        });
    _nodes.erase(new_n_end, _nodes.end());

    for (auto n: _nodes) {
        std::vector<Edge*> removals;
        for (auto e: n->bonds()) {
            if (nodes.find(e->other_end(n)) != nodes.end())
                removals.push_back(e);
        }
        for (auto e: removals)
            n->remove_bond(e);
    }

    auto new_e_end = std::remove_if(_edges.begin(), _edges.end(),
        [&nodes](Edge* e) {
            bool rm = nodes.find(e->end_points()[0]) != nodes.end()
            || nodes.find(e->end_points()[1]) != nodes.end();
            if (rm) delete e; return rm;
        });
    _edges.erase(new_e_end, _edges.end());
    set_gc_shape();
}

template <class FinalGraph, class Node, class Edge>
void
Graph<FinalGraph, Node, Edge>::set_color(const Rgba& rgba)
{
    for (auto n: _nodes)
        n->set_color(rgba);
    for (auto e: _edges)
        e->set_color(rgba);
}

} //  namespace basegeom

#endif  // basegeom_Graph_tcc
