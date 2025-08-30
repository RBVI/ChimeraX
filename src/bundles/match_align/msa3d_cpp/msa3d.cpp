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
 *=== UCSF ChimeraX Copyright ===
 */

#include <Python.h>
#include <algorithm>    // std::min, std::max
#include <climits>      // INT_MIN
#include <iterator>     // std::next
#include <list>
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

typedef std::map<Chain*, std::vector<Atom*>> PrincipalAtomMap;

// for my personal sanity, define local min/max functions, so that I don't have to try to
// figure out how to get references/pointers to template functions
static int min(int a, int b) { return std::min(a, b); }
static int max(int a, int b) { return std::max(a, b); }
static int (*val_func)(int, int);

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
    // info is really a pair of end points, but much easier to iterate through a vector
    // than a pair, so...
    std::vector<EndPoint*> info;
    double val;
    double penalty = 0.0;
    std::vector<std::shared_ptr<Link>> cross_links;

    Link(EndPoint* e1, EndPoint* e2, double _val) {
        info.push_back(e1);
        info.push_back(e2);
        val = _val;
    }
    ~Link() {
        for (auto ep: info)
            delete ep;
    }
};

class Column
{
public:
    std::map<Chain*, Chain::SeqPos> positions;

    bool  contains(Chain* seq, Chain::SeqPos pos) const {
        return positions.find(seq) != positions.end() && positions.at(seq) == pos;
    };
    double  participation(PrincipalAtomMap& pas, double dist_cutoff) const;
    double  value(PrincipalAtomMap& pas, double dist_cutoff) const;
};

double Column::participation(PrincipalAtomMap& pas, double dist_cutoff) const
{
    double p = 0;
    for (auto i = positions.begin(); i != positions.end(); ++i) {
        auto seq1 = i->first;
        auto pos1 = i->second;
        //TODO: if circular...
        auto pa1 = pas[seq1][pos1];
        if (pa1 == nullptr)
            continue;
        for (auto j = std::next(i); j != positions.end(); ++j) {
            auto seq2 = j->first;
            auto pos2 = j->second;
            //TODO: if circular...
            auto pa2 = pas[seq2][pos2];
            if (pa2 == nullptr)
                continue;
            p += dist_cutoff - pa1->scene_coord().distance(pa2->scene_coord());
        }
    }
    return p;
}

double Column::value(PrincipalAtomMap& pas, double dist_cutoff) const
{   
    bool val_set = false;
    double value;
    for (auto i = positions.begin(); i != positions.end(); ++i) {
        auto seq1 = i->first;
        auto pos1 = i->second;
        //TODO: if circular...
        auto pa1 = pas[seq1][pos1];
        if (pa1 == nullptr)
            continue;
        for (auto j = std::next(i); j != positions.end(); ++j) {
            auto seq2 = j->first;
            auto pos2 = j->second;
            //TODO: if circular...
            auto pa2 = pas[seq2][pos2];
            if (pa2 == nullptr)
                continue;
            double val = dist_cutoff - pa1->scene_coord().distance(pa2->scene_coord());
            if (!val_set) {
                value = val;
                val_set = true;
                continue;
            }
            if (val_func == &min && value < 0.0)
                break;
        }
        if (val_func == &min && value < 0.0)
            break;
    }
    return value;
}

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
    std::map<Chain*, std::vector<LinkList>>& pairings,
    Chain* seq1, Chain* seq2,
    LinkList& link_list, std::vector<LinkList>& links1, std::vector<LinkList>& links2,
    std::string tag, const char* status_prefix, PyObject* py_logger)
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
        l2_lists.push_back(LinkList(seq2_links.begin(), seq2_links.begin() + end));
    for (auto& link1: link_list) {
        auto i1 = link1->info[0]->pos;
        auto i2 = link1->info[1]->pos;
        for (auto& link2: l2_lists[i2]) {
            if (link2->info[0]->pos <= i1)
                continue;
            link1->cross_links.push_back(link2);
            link2->cross_links.push_back(link1);
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
        if (pen <= 0.0001)
            break;
        auto& link = *pos;
        auto& l1_list = links1[link->info[0]->pos];
        l1_list.erase(std::find(l1_list.begin(), l1_list.end(), link));
        auto& l2_list = links2[link->info[1]->pos];
        l2_list.erase(std::find(l2_list.begin(), l2_list.end(), link));
        for (auto& clink: link->cross_links)
            clink->penalty -= link->val;
        link_list.erase(pos);
    }

    auto& p1 = pairings[seq1];
    for (auto iter = links1.begin(); iter != links1.end(); ++iter) {
        auto& p1_ll = p1[iter - links1.begin()];
        p1_ll.insert(p1_ll.end(), iter->begin(), iter->end());
        all_links.insert(all_links.begin(), iter->begin(), iter->end());
    }
    auto& p2 = pairings[seq2];
    for (auto iter = links2.begin(); iter != links2.end(); ++iter) {
        auto& p2_ll = p2[iter - links2.begin()];
        p2_ll.insert(p2_ll.end(), iter->begin(), iter->end());
    }
}

bool
_check(std::map<Chain*, Chain::SeqPos>& info, std::map<Chain*, std::vector<Column*>>& order,
    std::vector<Chain*>& chains)
{
    std::map<Chain*, std::vector<Chain::SeqPos>> equiv;
    std::vector<Chain::SeqPos> null_init = { INT_MAX, INT_MAX, INT_MAX };
    for (auto chain: chains)
        equiv[chain] = null_init;
    std::vector<std::pair<std::vector<Column*>, int>> todo;
    for (auto& seq_pos: info) {
        auto seq = seq_pos.first;
        auto pos = seq_pos.second;
        auto& pos_vec = equiv[seq];
        pos_vec[0] = pos - 1;
        pos_vec[1] = pos;
        pos_vec[2] = pos + 1;
        auto seq_cols = order[seq];
        if (seq_cols.empty())
            continue;
        auto num_cols = seq_cols.size();
        bool added_todo = false;
        decltype(num_cols) i, j;
        for (i = 0; i < num_cols; ++i) {
            auto col = seq_cols[i];
            if (col->positions[seq] >= pos) {
                todo.emplace_back(std::vector<Column*>(seq_cols.begin(), seq_cols.begin() + i), -1);
                added_todo = true;
                break;
            }
        }
        if (!added_todo) {
            todo.emplace_back(seq_cols, -1);
            continue;
        }
        added_todo = false;
        for (j = i; j < num_cols; ++j) {
            auto col = seq_cols[j];
            if (col->positions[seq] > pos) {
                if (j > 1)
                    todo.emplace_back(std::vector<Column*>(seq_cols.begin() + i, seq_cols.begin() + j), 0);
                added_todo = true;
                break;
            }
        }
        if (!added_todo) {
            todo.emplace_back(std::vector<Column*>(seq_cols.begin() + i, seq_cols.end()), 0);
            continue;
        }
        todo.emplace_back(std::vector<Column*>(seq_cols.begin() + j, seq_cols.end()), 1);
    }
    while (todo.size() > 0) {
        auto& cols_rel = todo.back();
        auto cols = cols_rel.first;
        auto rel = cols_rel.second;
        todo.pop_back();
        for (auto col: cols) {
            for (auto& cseq_cpos: col->positions) {
                auto cseq = cseq_cpos.first;
                auto cpos = cseq_cpos.second;
                auto eqseq = equiv[cseq];
                auto eq = eqseq[rel+1];
                if (eq != INT_MAX && (cpos < eq ? -1 : (cpos > eq ? 1 : 0)) == rel)
                    continue;
                auto seq_cols = order[cseq];
                if (rel == 0) {
                    auto num_seq_cols = seq_cols.size();
                    decltype(num_seq_cols) i, j;
                    if (eq != INT_MAX)
                        return false;
                    if ((eqseq[0] != INT_MAX && eqseq[0] >= cpos)
                    || (eqseq[2] != INT_MAX && eqseq[2] <= cpos))
                        return false;
                    eqseq[1] = cpos;
                    bool broke = false;
                    for (i = 0; i < num_seq_cols; ++i) {
                        auto ccol = seq_cols[i];
                        auto ccol_pos = ccol->positions[cseq];
                        if (ccol_pos > cpos) {
                            i = num_seq_cols;
                            broke = true;
                            break;
                        }
                        if (ccol_pos == cpos) {
                            broke = true;
                            break;
                        }
                    }
                    if (!broke)
                        i = num_seq_cols;
                    broke = false;
                    for (j = i; j < num_seq_cols; ++j) {
                        auto ccol = seq_cols[j];
                        if (ccol->positions[cseq] > cpos) {
                            broke = true;
                            break;
                        }
                    }
                    if (!broke)
                        j = num_seq_cols;
                    if (j > i) {
                        auto td_list = std::vector<Column*>(seq_cols.begin() + i, seq_cols.begin() + j);
                        td_list.erase(std::find(td_list.begin(), td_list.end(), col));
                        if (!td_list.empty())
                            todo.emplace_back(td_list, 0);
                    }
                    continue;
                }
                auto test = equiv[cseq][1];
                if (test == INT_MAX)
                    test = equiv[cseq][1-rel];
                if (test != INT_MAX && (cpos < test ? -1 : (cpos > test ? 1 : 0)) != rel)
                    return false;
                if (rel < 0) {
                    auto num_seq_cols = seq_cols.size();
                    decltype(num_seq_cols) i, j;
                    if (eq == INT_MAX)
                        i = 0;
                    else {
                        bool broke = false;
                        for (i = 0; i < num_seq_cols; ++i) {
                            auto ccol = seq_cols[i];
                            if (ccol->positions[cseq] > eq) {
                                broke = true;
                                break;
                            }
                        }
                        if (!broke)
                            i = num_seq_cols;
                    }
                    bool broke = false;
                    for (j = i; j < num_seq_cols; ++j) {
                        auto ccol = seq_cols[j];
                        if (ccol->positions[cseq] > cpos) {
                            broke = true;
                            break;
                        }
                    }
                    if (!broke)
                        j = num_seq_cols;
                    equiv[cseq][rel+1] = cpos;
                    if (j > 1) {
                        auto td_list = std::vector<Column*>(seq_cols.begin() + i, seq_cols.begin() + j);
                        td_list.erase(std::find(td_list.begin(), td_list.end(), col));
                        if (!td_list.empty())
                            todo.emplace_back(td_list, rel);
                    }

                } else {
                    int num_seq_cols = seq_cols.size();
                    decltype(num_seq_cols) i, j;
                    if (eq == INT_MAX)
                        i = num_seq_cols - 1;
                    else {
                        bool broke = false;
                        for (i = num_seq_cols-1; i >= 0; --i) {
                            auto ccol = seq_cols[i];
                            if (ccol->positions[cseq] < eq) {
                                broke = true;
                                break;
                            }
                        }
                        if (!broke)
                            i = -1;
                        broke = false;
                        for (j = i; j >= 0; --j) {
                            auto ccol = seq_cols[j];
                            if (ccol->positions[cseq] < cpos) {
                                j += 1;
                                broke = true;
                                break;
                            }
                        }
                        if (!broke)
                            j = 0;
                        equiv[cseq][rel+1] = cpos;
                        if (j < i+1) {
                            auto td_list = std::vector<Column*>(seq_cols.begin()+j, seq_cols.begin()+i+1);
                            td_list.erase(std::find(td_list.begin(), td_list.end(), col));
                            if (!td_list.empty())
                                todo.emplace_back(td_list, rel);
                        }
                    }
                }
            }
        }
    }
    return true;
}

PyObject *
multi_align(std::vector<Chain*>& chains, double dist_cutoff, bool col_all, char gap_char,
    bool circular, const char* status_prefix, PyObject* py_logger)
{
    // Create list of pairings between chains and prune to be monotonic
    if (circular) {
        PyErr_SetString(PyExc_NotImplementedError, "C++ multi_align cicular permutation support not implemented");
        return nullptr;
    }

    if (col_all)
        val_func = &min;
    else
        val_func = &max;

    // For each pair, go through the second chain residue by residue
    // and compile crosslinks to other chain.  As links are compiled,
    // figure out what previous links are crossed and keep a running
    // "penalty" for links based on what they cross.  Sort links by
    // penalty and keep pruning worst link until no links cross.

    PrincipalAtomMap pas;
    std::map<Chain*, std::vector<LinkList>> pairings;
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
    //std::map<std::pair<Chain*, Chain*>, std::pair<int, int>> circular_pairs;
    //std::map<std::pair<Chain*, Chain*>, CircularLinkData> hold_data;

    LinkList all_links;
    std::map<Chain*, AtomSearchTree*> trees;
    auto num_chains = chains.size();
    decltype(num_chains) loop_count = 0, loop_limit = (num_chains * (num_chains-1))/2;
    std::map<Chain*, std::map<Atom*, decltype(num_chains)>> datas;
    for (decltype(num_chains) i = 0; i < num_chains; ++i) {
        auto seq1 = chains[i];
        auto len1 = pairings[seq1].size();
        auto& seq1_pas = pas[seq1];
        auto num_pas1 = seq1_pas.size();
        for (decltype(i) j = i+1; j < num_chains; ++j) {
            loop_count += 1;
            std::stringstream tag;
            tag << "(" << j-i << "/" << num_chains-1 << ")";
            auto seq2 = chains[j];
            auto len2 = pairings[seq2].size();
            std::vector<LinkList> links1(len1);
            std::vector<LinkList> links2(len2);
            LinkList link_list;
            auto tree_i = trees.find(seq2);
            decltype(trees)::mapped_type tree;
            if (tree_i == trees.end()) {
                logger::status(py_logger, status_prefix, "Building search tree ", tag.str());
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
            // To avoid having the logging slow us down a bunch, only update every 1%
            int before = (loop_count-1) * 100.0 / loop_limit;
            int after = loop_count * 100.0 / loop_limit;
            if (after > before)
                logger::status(py_logger, status_prefix, "Searching tree, building links: ", after, "%");
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
                    auto end1 = new EndPoint(seq1, k);
                    auto end2 = new EndPoint(seq2, i2);
                    auto link = std::shared_ptr<Link>(new Link(end1, end2, val));
                    links1[k].push_back(link);
                    links2[i2].push_back(link);
                    link_list.push_back(link);
                }
            }
            if (circular) {
                //TODO
            } else {
                find_prune_crosslinks(all_links, pairings, seq1, seq2, link_list, links1, links2, tag.str(),
                    status_prefix, py_logger);
            }
        }
    }
    for (auto chain_tree: trees) {
        delete chain_tree.second;
    }

    if (circular) {
        //TODO
    }

    // column collation
    std::map<Chain*, std::map<Column*, std::vector<int>::size_type>> columns;
    std::map<Chain*, std::vector<Column*>> partial_order;

    std::set<std::pair<EndPoint*, EndPoint*>> seen;
    while (all_links.size() > 0) {
        if (all_links.size() % 100 == 0)
            logger::status(py_logger, status_prefix,
                    "Forming columns (", all_links.size(), " links to check)");
        auto back_val = all_links.back()->val;
        for (auto link: all_links) {
            if (link->val > back_val) {
                std::sort(all_links.begin(), all_links.end(),
                    [](std::shared_ptr<Link>& l1, std::shared_ptr<Link>& l2) { return l1->val < l2->val; });
                if (val_func == &min) {
                    // Since all_links is a vector, try to make only one erase call...
                    auto erasable = all_links.begin();
                    for (auto i = all_links.begin(); i != all_links.end(); ++i) {
                        if (i+1 == all_links.end())
                            break;
                        if ((*i)->val > 0)
                            break;
                        erasable = i+1;
                    }
                    if (erasable != all_links.begin())
                        all_links.erase(all_links.begin(), erasable);
                }
                break;
            }
        }
        auto link = all_links.back();
        all_links.pop_back();
        if (link->val < 0)
            break;

        std::pair<EndPoint*, EndPoint*> key(link->info[0], link->info[1]);
        if (seen.find(key) != seen.end())
            continue;
        seen.insert(key);

        std::map<Chain*, Chain::SeqPos> check_info;
        for (auto endp: link->info) {
            auto& list = pairings[endp->chain][endp->pos];
            list.erase(std::find(list.begin(), list.end(), link));
            check_info[endp->chain] = endp->pos;
        }

        // AFAICT links are _always_ between different chains, so the whole "okay check"
        // in the Chimera code seems superfluous

        if (!_check(check_info, partial_order, chains))
            continue;
        //TODO
    }
        

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
    std::vector<Chain*> chains;
    for (decltype(num_chains) i = 0; i < num_chains; ++i) {
        PyObject* py_ptr = PySequence_GetItem(chain_ptrs_list, i);
        if (!PyLong_Check(py_ptr)) {
            std::stringstream err_msg;
            err_msg << "Item at index " << i << " of first arg is not an int (chain pointer)";
            PyErr_SetString(PyExc_TypeError, err_msg.str().c_str());
            return nullptr;
        }
        chains.push_back(static_cast<Chain*>(PyLong_AsVoidPtr(py_ptr)));
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
