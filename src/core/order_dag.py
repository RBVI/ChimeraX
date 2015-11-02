# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
order_dag: Ordered DAG Traversal
================================

This module defines a generator, 'order_dag', for traversing a
directed acyclic graph (DAG) where nodes are visited such that if
there is a path from node A to node B, then node B will be visited
before node A.  If there is a cycle, then the function raises a
OrderDAGError exception.

Assumptions
===========

- The keys are nodes.
- All nodes appear as keys.
- The values are list of nodes.

Notes
=====

The dictionary argument is modified.  On a successful call, the
dictionary argument will be empty on return.

The generated order for the same graph may differ in different runs
because the algorithm process keys in dictionary order, i.e., random.

Usage
=====

    edge_dict = {
        'a': ['b'],
        'b': ['c','d'],
        'c': ['e'],
        'd': ['e', 'f'],
        'e': [],
        'f': [],
    }
    from order_dag import order_dag
    for node in order_dag(edge_dict):
        print(node)

"""

class OrderDAGError(ValueError):

    def __init__(self, msg, path):
        super().__init__(msg)
        self.path = path


def order_dag(d):
    "Generator returning nodes from a post-order DAG traversal."
    while d:
        node = next(iter(d.keys()))
        path = [node]
        yield from _postorder_traversal(path, d, node)


def _postorder_traversal(path, d, node):
        try:
            depends_on = d.pop(node)
        except KeyError:
            # Node must have been visited before
            return
        for dnode in depends_on:
            if dnode in path:
                raise OrderDAGError("cycle detected", path)
            path.append(dnode)
            yield from _postorder_traversal(path, d, dnode)
            path.pop()
        yield node


if __name__ == "__main__":
    def dump_dag(g):
        for n in order_dag(g):
            print(n, end=' ')
        print()
    dump_dag({
        'a': ['b'],
        'b': ['c','d'],
        'c': ['e'],
        'd': ['e', 'f'],
        'e': [],
        'f': [],
    })
    dump_dag({
        'a': ['b'],
        'b': ['c'],
        'c': ['d'],
        'd': ['a'],
    })
