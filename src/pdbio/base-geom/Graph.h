// vim: set expandtab ts=4 sw=4:
#ifndef base_geom_Graph
#define base_geom_Graph

#include <vector>
#include <memory>
#include <algorithm>

template <class Vertex, class Edge>
class Graph {
protected:
    typedef std::vector<std::unique_ptr<Vertex>>  Vertices;
    typedef std::vector<std::unique_ptr<Edge>>  Edges;
private:
    Vertices  _vertices;
    Edges  _edges;
protected:
    void  add_edge(Edge *e) { _edges.emplace_back(e); }
    void  add_vertex(Vertex *v) { _vertices.emplace_back(v); }
    void  delete_edge(Edge *e);
    const Edges &  edges() const { return _edges; }
    const Vertices &  vertices() const { return _vertices; }
};

template <class Vertex, class Edge>
void
Graph<Vertex, Edge>::delete_edge(Edge *e)
{
    typename Edges::iterator i = std::find_if(_edges.begin(), _edges.end(),
        [&e](std::unique_ptr<Edge>& ve) { return ve.get() == e; });
    if (i == _edges.end())
        throw std::invalid_argument("delete_edge called for Edge not in Graph");
    _edges.erase(i);
}
#endif  // base_geom_Graph
