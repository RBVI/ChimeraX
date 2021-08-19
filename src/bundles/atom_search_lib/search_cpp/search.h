// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomsearch_search
#define atomsearch_search

#include "imex.h"

#include <vector>

#include "Python.h"

#include "atomstruct/Atom.h"
#include "atomstruct/destruct.h"

using atomstruct::Atom;
using atomstruct::Coord;
using atomstruct::Real;

class _Node {
private:
    void  _make_leaf(const std::vector<Atom*>&, bool);
public:
    enum NodeType { Leaf, Interior };

    _Node(const std::vector<Atom*>&, bool, double);
    virtual ~_Node();

    int  axis;
    double  bbox[3][2];
    std::vector<Atom*>  leaf_atoms;
    _Node*  left;
    double  median;
    _Node*  right;
    std::vector<Atom*>  search(const Coord&, double, double*);
    NodeType  type;
};

class ATOM_SEARCH_IMEX AtomSearchTree: public atomstruct::DestructionObserver {
    // AtomSearchTree is a specialization of an 'adaptive k-d tree'
    // as per "The Design and Analysis of Spatial Data Structures" pp. 70-71.
    // Basically, given a set of k-dimensional points (each dimension referred
    // to as an "attribute") with associated data, they are partitioned into
    // leaf nodes.  Each leaf nodes hold lists of associated data whose
    // corresponding attributes vary by less than an initially-supplied threshold
    // ('sep_val').  Also, each leaf node holds a bounding box of the leaf data.

    // The interior nodes of the tree contain details of the partitioning.
    // In particular, what attribute this node partitions along ('axis'),
    // and what value ('median') partitions left child node from right child node.
    // Whether a node is interior or leaf is stored in 'type'.

    // The specialization is that this is a 3D tree of Atoms, and node-associated data
    // is not supported.
private:
    std::vector<Atom*>  _atoms;
    double  _sep_val;
    bool  _transformed;

    void  init_root();

public:
    AtomSearchTree(const std::vector<Atom*>& atoms, bool transformed, double sep_val);
    virtual ~AtomSearchTree();
    virtual void  destructors_done(const std::set<void*>& destroyed);
    std::vector<Atom*>  search(Atom*, double);
    std::vector<Atom*>  search(const Coord&, double);
    _Node  *root;
};

#endif  // atomsearch_search
