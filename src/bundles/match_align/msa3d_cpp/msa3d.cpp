// vi: set expandtab shiftwidth=4 softtabstop=4:

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

#include <Python.h>
#include <algorithm>    // std::min
#include <math.h>
#include <memory>
#include <string>
#include <vector>

#include <atomstruct/Chain.h>
#include <atomstruct/Residue.h>
#include <atomstruct/search.h>
#include <logger/logger.h>

using atomstruct::Atom;
using atomstruct::Chain;
using atomstruct::AtomSearchTree;

namespace { // so these class declarations are not visible outside this file

class EndPoint
{
public:
    Chain*  chain;
    Chain::SeqPos  pos;

    EndPoint(Chain* _chain, Chain::SeqPos _pos): chain(_chain), pos(_pos) {};
};

class Link
{
public:
    std::vector<EndPoint*> info;
    float val;
    float penalty = 0.0;
    std::vector<std::shared_ptr<Link>> cross_links;

    Link(EndPoint* e1, EndPoint* e2, float _val) {
        info.push_back(e1);
        info.push_back(e2);
        val = _val;
    }
    ~Link() {
        for (auto ep: info)
            delete ep;
    }
};

} // end namespace

typedef std::vector<std::shared_ptr<Link>> LinkList;

class CircularLinkData
{
public:
    std::vector<LinkList> links1, links2;
    LinkList link_list;
};

static void
find_prune_crosslinks(
    LinkList& all_links,
    std::map<const Chain*, std::vector<LinkList>>& pairings,
    const Chain* seq1, const Chain* seq2,
    LinkList& link_list, std::vector<LinkList>& links1, std::vector<LinkList>& links2,
    std::string& tag, const char* status_prefix, PyObject* py_logger)
{
    logger::status(py_logger, status_prefix, "Finding crosslinks ", tag);
    std::vector<LinkList::size_type> ends;
    LinkList seq2_links;
    for (auto& links: links2) {
        ends.push_back(seq2_links.size());
        seq2_links.insert(seq2_links.end(), links.begin(), links.end());
    }
    std::vector<LinkList> l2_lists;
    for (auto end: ends)
        l2_lists.insert(l2_lists.end(), seq2_links.begin(), seq2_links.begin() + end);
    for (auto& link1: link_list) {
        auto i1 = link1->info[0]->pos;
        auto i2 = link1->info[1]->pos;
        for (auto& link2: l2_lists[i2]) {
            if (link2->info[0]->pos <= i1)
                continue;
            link1->crosslinks.push_back(link2);
            link2->crosslinks.push_back(link1);
            link1->penalty += link2->val;
            link2->penalty += link1->val;
        }
    }
    
    logger::status(py_logger, status_prefix, "Pruning crosslinks ", tag);
    while (link_list.size() > 0) {
        LinkList::iterator pos;
        decltype(Link::penalty) pen;
        for (auto iter = link_list.begin(); iter != link_list.end(); ++iter) {
            auto x_pen = (*iter)->penalty;
            if (iter == link_list.begin() || x_pen > pen) {
                pos = iter;
                pen = x_pen;
            }
        }
        if (pen <= 0.0001))
            break;
        auto& link = *pos;
        auto& l1_list = links1[link->info[0]->pos];
        l1_list.erase(std::find(l1_list.begin(), l1_list.end(), link);
        auto& l2_list = links2[link->info[1]->pos];
        l2_list.erase(std::find(l2_list.begin(), l2_list.end(), link);
        for (auto& clink: link->crosslinks)
            clink->penalty -= link->val;
        link_list.erase(pos);
    }

    auto& p1 = pairings[seq1];
    for (auto iter = links1.begin(); iter != links1.end(); ++iter) {
        auto& p1_ll = p1[iter - links1.begin()];
        p1_ll.insert(p1_ll.end(), (*iter)->begin(), (*iter)->end());
        all_links.insert(all_links.begin(), (*iter)->begin(), (*iter)->end());
    }
    auto& p2 = pairings[seq2];
    for (auto iter = links2.begin(); iter != links2.end(); ++iter) {
        auto& p2_ll = p2[iter - links2.begin()];
        p2_ll.insert(p2_ll.end(), (*iter)->begin(), (*iter)->end());
    }
}

PyObject *
multi_align(std::vector<const Chain*>& chains, double dist_cutoff, bool col_all, char gap_char,
    bool circular, const char* status_prefix, PyObject* py_logger)
{
    // Create list of pairings between chains and prune to be monotonic
    if (circular) {
        PyErr_SetString(PyExc_NotImplementedError, "C++ multi_align cicular permutation support not implemented");
        return nullptr;
    }

    // For each pair, go through the second chain residue by residue
    // and compile crosslinks to other chain.  As links are compiled,
    // figure out what previous links are crossed and keep a running
    // "penalty" for links based on what they cross.  Sort links by
    // penalty and keep pruning worst link until no links cross.
    std::vector<LinkList> all_links;

    std::map<const Chain*, std::vector<Atom*>> pas;
    std::map<const Chain*, std::vector<LinkList>> pairings;
    logger::status(py_logger, status_prefix, "Finding residue principal atoms");
    for (auto chain: chains) {
        decltype(pas)::mapped_type seq_pas;
        decltype(pairings)::mapped_type pairing;
        Atom* pa;
        for (auto res: chain->residues()) {
            pa = (res == nullptr ? nullptr : res->principal_atom());
            pairing.emplace_back();
            if (circular)
                pairing.emplace_back();
            if (pa == nullptr) {
                if (res != nullptr)
                    logger::warning(py_logger, "Cannot determine principal atom for residue ", res->str());
                seq_pas.push_back(nullptr);
                continue;
            }
            seq_pas.push_back(pa);
        }
        pas[chain] = seq_pas;
        pairings[chain] = pairing;
    }

    //TODO: circular permutation support
    //std::map<std::pair<const Chain*, const Chain*>, std::pair<int, int>> circular_pairs;
    //std::map<std::pair<const Chain*, const Chain*>, CircularLinkData> hold_data;

    std::map<const Chain*, AtomSearchTree*> trees;
    auto num_chains = chains.size();
    decltype(num_chains) loop_count = 0, loop_limit = (num_chains * (num_chains-1))/2;
    std::map<const Chain*, std::map<Atom*, decltype(num_chains)>> datas;
    for (decltype(num_chains) i = 0; i < num_chains; ++i) {
        auto seq1 = chains[i];
        auto len1 = pairings[seq1].size();
        auto& seq1_pas = pas[seq1];
        auto num_pas1 = seq1_pas.size();
        for (decltype(i) j = i+1; j < num_chains; ++j) {
            loop_count += 1;
            std::string tag;
            tag << "(" << j-i << "/" << num_chains-1 << ")";
            auto seq2 = chains[j];
            auto len2 = pairings[seq2].size();
            std::vector<LinkList> links1(len1);
            std::vector<LinkList> links2(len2);
            LinkList link_list;
            auto tree_i = trees.find(seq2);
            decltype(trees)::mapped_type tree;
            if (tree_i == trees.end()) {
                logger::status(py_logger, status_prefix, "Building search tree ", tag);
                auto& seq2_pas = pas[seq2];
                auto num_pas = seq2_pas.size();
                std::vector<Atom*> atoms;
                auto& data = datas[seq2];
                for (decltype(num_pas) k = 0; k < num_pas; ++k) {
                    auto pa = seq2_pas[k];
                    if (pa == nullptr)
                        continue;
                    atoms.push_back(pa);
                    data[pa] = k;
                }
                tree = new AtomSearchTree(atoms, true, dist_cutoff);
                trees[seq2] = tree;
            } else {
                tree = (*tree_i).second;
            }
            logger::status(py_logger, status_prefix,
                    "Searching tree, building links (", loop_count, "/", loop_limit, ")");
            for (decltype(num_pas1) k = 0; k < num_pas1; ++k) {
                auto pa1 = seq1_pas[k];
                if (pa1 == nullptr)
                    continue;
                auto crd1 = pa1->scene_coord();
                auto matches = tree->search(pa1, dist_cutoff);
                auto& data = datas[seq2];
                for (auto pa2: matches) {
                    auto dist = crd1.distance(pa2->scene_coord());
                    auto val = dist_cutoff - dist;
                    if (val <= 0.0)
                        continue;
                    auto i2 = data[pa2];
                    auto link = Link(EndPoint(seq1, k), EndPoint(seq2, i2), val);
                    links1[k].push_back(&link);
                    links2[i2].push_back(&link);
                    link_list.push_back(&link);
                }
            }
            if (circular) {
                //TODO
            } else {
                find_prune_crosslinks(all_links, pairings, seq1, seq2, link_list, links1, links2, tag,
                    status_prefix);
            }
        }
    }
    for (auto chain_tree: trees) {
        delete chain_tree.second;
    }

    if (circular) {
        //TODO
    }

    //TODO: column collation

    PyErr_SetString(PyExc_NotImplementedError, "C++ multi_align not implemented");
    return nullptr;
    // return Python map from original sequence to realigned sequence ;
}

static PyObject*
py_multi_align(PyObject*, PyObject* args)
{
    PyObject* chain_ptrs_list;
    PyObject* py_logger;
    double dist_cutoff;
    int col_all, circular, py_gap_char;
    const char* status_prefix;
    if (!PyArg_ParseTuple(args, const_cast<char *>("OfpCpsO"),
            &chain_ptrs_list, &dist_cutoff, &col_all, &py_gap_char, &circular, &status_prefix, &py_logger))
        return NULL;
    char gap_char = (char)py_gap_char;
    if (!PySequence_Check(chain_ptrs_list)) {
        PyErr_SetString(PyExc_TypeError, "First arg is not a sequence of Chain pointers");
        return nullptr;
    }
    auto num_chains = PySequence_Size(chain_ptrs_list);
    if (num_chains < 2) {
        PyErr_SetString(PyExc_ValueError, "First arg (sequence of chain pointers) must contain at least"
            " two chains");
        return nullptr;
    }
    std::vector<const Chain*> chains;
    for (decltype(num_chains) i = 0; i < num_chains; ++i) {
        PyObject* py_ptr = PySequence_GetItem(chain_ptrs_list, i);
        if (!PyLong_Check(py_ptr)) {
            std::stringstream err_msg;
            err_msg << "Item at index " << i << " of first arg is not an int (chain pointer)";
            PyErr_SetString(PyExc_TypeError, err_msg.str().c_str());
            return nullptr;
        }
        chains.push_back(static_cast<const Chain*>(PyLong_AsVoidPtr(py_ptr)));
    }
    return multi_align(chains, dist_cutoff, (bool)col_all, gap_char, (bool)circular,
        status_prefix, py_logger);
}

static struct PyMethodDef msa3d_methods[] =
{
  {const_cast<char*>("multi_align"), py_multi_align, METH_VARARGS, NULL},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef msa3d_def = {
        PyModuleDef_HEAD_INIT,
        "_msa3d",
        "Compute alignment from 3D superposition",
        -1,
        msa3d_methods,
        nullptr,
        nullptr,
        nullptr,
        nullptr
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__msa3d()
{
    return PyModule_Create(&msa3d_def);
}
