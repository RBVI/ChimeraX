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
#include <climits>      // INT_MAX
#include <cmath>        // std::isnan
#include <iterator>     // std::next
#include <limits>       // quiet_NaN
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
using atomstruct::Sequence;
using atomstruct::StructureSeq;
using atomstruct::AtomSearchTree;

typedef std::map<Chain*, std::vector<Atom*>> PrincipalAtomMap;

// for my personal sanity, define local min/max functions, so that I don't have to try to
// figure out how to get references/pointers to template functions
static int min(int a, int b) { return std::min(a, b); }
static int max(int a, int b) { return std::max(a, b); }
static int (*val_func)(int, int);

namespace { // so these class declarations are not visible outside this file

class Column
{
public:
    std::map<Chain*, Chain::SeqPos> positions;

    Column(decltype(positions) pos_info): positions(pos_info) {};
    Column(const std::shared_ptr<Column>& _col): positions(_col->positions) {};

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

class EndPoint
{
public:
    Chain*  chain;
    Chain::SeqPos  pos;
    decltype(Column::positions) positions;
    EndPoint(Chain* _chain, Chain::SeqPos _pos): chain(_chain), pos(_pos) { positions[_chain] = _pos; };

    bool contains(Chain* _seq, Chain::SeqPos _pos) { return chain == _seq && pos == _pos; }
};

// Link.info can contain end points or columns...
class EndPointOrColumn
{
public:
    bool is_column;
    std::shared_ptr<EndPoint> ep;
    std::shared_ptr<Column> col;

    EndPointOrColumn(std::shared_ptr<EndPoint>& _ep): is_column(false), ep(_ep) {}
    EndPointOrColumn(std::shared_ptr<Column>& _col): is_column(true), col(_col) {}

    void operator=(std::shared_ptr<EndPoint>& _ep) { ep = _ep; is_column = false; }
    void operator=(std::shared_ptr<Column>& _col) { col = _col; is_column = true; }

    bool contains(Chain* _seq, Chain::SeqPos _pos) {
        if (is_column)
            return col->contains(_seq, _pos);
        return ep->contains(_seq, _pos);
    }
    decltype(Column::positions) positions() {
        if (is_column)
            return col->positions;
        return ep->positions;
    }
};

class Link
{
public:
    // info is really a pair of end points or column + end point, but much easier to iterate through a vector
    // than a pair, so...
    std::vector<EndPointOrColumn> info;
    double val;
    double penalty = 0.0;
    std::vector<std::shared_ptr<Link>> cross_links;

    Link(std::shared_ptr<EndPoint>& e1, std::shared_ptr<EndPoint>& e2, double _val) {
        info.emplace_back(e1);
        info.emplace_back(e2);
        val = _val;
    }
    Link(std::shared_ptr<Column>& col, std::shared_ptr<EndPoint>& ep, double _val) {
        info.emplace_back(col);
        info.emplace_back(ep);
        val = _val;
    }

    void evaluate(PrincipalAtomMap& pas, double dist_cutoff);
};

void Link::evaluate(PrincipalAtomMap& pas, double dist_cutoff)
{
    val = std::numeric_limits<double>::quiet_NaN();
    for (auto seq_pos1: info[0].positions()) {
        auto s1 = seq_pos1.first;
        auto p1 = seq_pos1.second;
        //TODO: circular
        auto pa1 = pas[s1][p1];
        if (pa1 == nullptr)
            continue;
        for (auto seq_pos2: info[1].positions()) {
            auto s2 = seq_pos2.first;
            auto p2 = seq_pos2.second;
            //TODO: circular
            auto pa2 = pas[s2][p2];
            if (pa2 == nullptr)
                continue;
            auto _val = dist_cutoff - pa1->scene_coord().distance(pa2->scene_coord());
            if (std::isnan(val)) {
                val = _val;
                continue;
            }
            val = (val_func)(val, _val);
            if (val_func == &min && val < 0.0)
                break;
        }
    }
}

class GapInfo
{
public:
    bool  in_gap;
    Chain::SeqPos  pos;
    unsigned int  num_gaps;

    GapInfo(bool _in_gap, Chain::SeqPos _pos, unsigned int _num_gaps):
        in_gap(_in_gap), pos(_pos), num_gaps(_num_gaps) {};
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
        auto i1 = link1->info[0].ep->pos;
        auto i2 = link1->info[1].ep->pos;
        for (auto& link2: l2_lists[i2]) {
            if (link2->info[0].ep->pos <= i1)
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
        auto& l1_list = links1[link->info[0].ep->pos];
        l1_list.erase(std::find(l1_list.begin(), l1_list.end(), link));
        auto& l2_list = links2[link->info[1].ep->pos];
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
_check(std::map<Chain*, Chain::SeqPos>& info, std::map<Chain*, std::vector<std::shared_ptr<Column>>>& order,
    std::vector<Chain*>& chains)
{
    std::map<Chain*, std::vector<Chain::SeqPos>> equiv;
    std::vector<Chain::SeqPos> null_init = { INT_MAX, INT_MAX, INT_MAX };
    for (auto chain: chains)
        equiv[chain] = null_init;
    std::vector<std::pair<std::vector<std::shared_ptr<Column>>, int>> todo;
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
                todo.emplace_back(std::vector<std::shared_ptr<Column>>(seq_cols.begin(),
                    seq_cols.begin() + i), -1);
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
                    todo.emplace_back(std::vector<std::shared_ptr<Column>>(seq_cols.begin() + i,
                        seq_cols.begin() + j), 0);
                added_todo = true;
                break;
            }
        }
        if (!added_todo) {
            todo.emplace_back(std::vector<std::shared_ptr<Column>>(seq_cols.begin() + i, seq_cols.end()), 0);
            continue;
        }
        todo.emplace_back(std::vector<std::shared_ptr<Column>>(seq_cols.begin() + j, seq_cols.end()), 1);
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
                        auto td_list = std::vector<std::shared_ptr<Column>>(seq_cols.begin() + i,
                            seq_cols.begin() + j);
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
                        auto td_list = std::vector<std::shared_ptr<Column>>(seq_cols.begin() + i,
                            seq_cols.begin() + j);
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
                            auto td_list = std::vector<std::shared_ptr<Column>>(seq_cols.begin()+j,
                                seq_cols.begin()+i+1);
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
    bool circular, const char* status_prefix, PyObject* py_logger, PyObject* error_class)
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
                    auto end1 = std::shared_ptr<EndPoint>(new EndPoint(seq1, k));
                    auto end2 = std::shared_ptr<EndPoint>(new EndPoint(seq2, i2));
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
    std::map<Chain*, std::map<std::shared_ptr<Column>, std::vector<int>::size_type>> columns;
    std::map<Chain*, std::vector<std::shared_ptr<Column>>> partial_order;

    std::set<std::pair<std::shared_ptr<EndPoint>, std::shared_ptr<EndPoint>>> seen;
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

        std::pair<std::shared_ptr<EndPoint>, std::shared_ptr<EndPoint>>
            key(link->info[0].ep, link->info[1].ep);
        if (seen.find(key) != seen.end())
            continue;
        seen.insert(key);

        std::map<Chain*, Chain::SeqPos> check_info;
        for (auto endp: link->info) {
            auto& list = pairings[endp.ep->chain][endp.ep->pos];
            list.erase(std::find(list.begin(), list.end(), link));
            check_info[endp.ep->chain] = endp.ep->pos;
        }

        // AFAICT links are _always_ between different chains, so the whole "okay check"
        // in the Chimera code seems superfluous

        if (!_check(check_info, partial_order, chains))
            continue;

        auto col = std::shared_ptr<Column>(new Column(check_info));
        for (auto seq_pos: check_info) {
            auto seq = seq_pos.first;
            auto pos = seq_pos.second;
            auto po = partial_order[seq];
            auto num_po = po.size();
            decltype(num_po) i;
            bool broke = false;
            for (i = 0; i < num_po; ++i) {
                auto pcol = po[i];
                if (pcol->positions[seq] > pos) {
                    broke = true;
                    break;
                }
            }
            if (!broke)
                i = num_po;
            po.insert(po.begin()+i, col);
            auto cols = columns[seq];
            cols[col] = i;
            for (auto ncol_i = po.begin()+i+1; ncol_i != po.end(); ++ncol_i) {
                auto ncol = *ncol_i;
                cols[ncol] += 1;
            }
        }

        // From here forward, link.info might contain Columns, so don't just use its
        // .ep attribute, which is what the preceding code does
        for (auto col_or_ep: link->info) {
            for (auto seq_pos: col_or_ep.positions()) {
                auto seq = seq_pos.first;
                auto pos = seq_pos.second;
                for (auto l: pairings[seq][pos]) {
                    EndPointOrColumn *base, *connect;
                    if (l->info[0].contains(seq, pos)) {
                        base = &l->info[0];
                        connect = &l->info[1];
                    } else {
                        connect = &l->info[0];
                        base = &l->info[1];
                    }
                    l->info[0] = col;
                    l->info[1] = *connect;
                    l->evaluate(pas, dist_cutoff);
                    for (auto c_seq_pos: col->positions) {
                        auto cseq = c_seq_pos.first;
                        auto cpos = c_seq_pos.second;
                        if (base->contains(cseq, cpos))
                            continue;
                        pairings[cseq][cpos].push_back(l);
                    }
                }
            }
            if (col_or_ep.is_column) {
                for (auto seq_pos: col_or_ep.positions()) {
                    auto seq = seq_pos.first;
                    auto seq_cols = columns[seq];
                    auto opos = seq_cols[col_or_ep.col];
                    auto po = partial_order[seq];
                    auto new_po = decltype(po)(po.begin(), po.begin()+opos);
                    auto new_po_back = decltype(po)(po.begin()+opos+1, po.end());
                    new_po.insert(new_po.end(), new_po_back.begin(), new_po_back.end());
                    partial_order[seq] = new_po;
                    for (auto pcol: new_po_back)
                        seq_cols[pcol] -= 1;
                    seq_cols.erase(col_or_ep.col);
                }
            }
        }
    }
        
    logger::status(py_logger, status_prefix, "Collating columns");

    std::vector<std::shared_ptr<Column>> ordered_columns;
    std::shared_ptr<Column> col;
    while (true) {
        // find an initial sequence column that can lead
        bool broke_po = false;
        for (auto seq_cols: partial_order) {
            auto seq = seq_cols.first;
            auto cols = seq_cols.second;
            if (cols.empty()) {
                auto py_struct = seq->structure()->py_instance(true);
                auto py_struct_name = PyObject_Str(py_struct);
                Py_DECREF(py_struct);
                if (py_struct_name == nullptr)
                    return nullptr;
                std::stringstream err_msg;
                err_msg << "Cannot generate alignment with " << PyUnicode_AsUTF8(py_struct_name)
                    << " " << seq->name() << " because it is not superimposed on the other structures";
                Py_DECREF(py_struct_name);
                PyErr_SetString(error_class, err_msg.str().c_str());
                return nullptr;
            }
            col = cols[0];
            bool broke_cseq = false;
            for (auto c_seq_pos: col->positions) {
                auto cseq = c_seq_pos.first;
                if (partial_order[cseq][0] != col) {
                    broke_cseq = true;
                    break;
                }
            }
            if (!broke_cseq) {
                broke_po = true;
                break;
            }
        }
        if (!broke_po)
            break;
    }
    if (ordered_columns.empty()) {
        logger::status(py_logger, "");
        PyErr_SetString(error_class, "No residues satisfy distance constraint for column!");
        return nullptr;
    }

    // Make the clone in the C++ layer, so that it is easier/faster to access its functions
    std::map<Chain*, StructureSeq*> cpp_clones;
    std::map<Chain*, int> current;
    std::map<StructureSeq*, Sequence::Contents> working_seqs;
    for (auto seq: chains) {
        auto cpp_clone = seq->copy();
        // clear() is private, so need to track assembled sequence characters separately
        // (working_seqs map) and use bulk_set at the end to finish
        cpp_clone->set_description(seq->description());
        auto py_s = seq->structure()->py_instance(true);
        auto py_struct_name = PyObject_Str(py_s);
        Py_DECREF(py_s);
        if (py_struct_name == nullptr) {
            PyErr_SetString(error_class, "Could not access structure name");
            return nullptr;
        }
        cpp_clone->set_name(PyUnicode_AsUTF8(py_struct_name));
        cpp_clones[seq] = cpp_clone;
        current[seq] = -1;
    }

    // For maximum benefit from the "column squeezing" step that follows, we
    // need to add in the one-residue columns whose position is well-determined.
    decltype(ordered_columns) new_ordered;
    for (auto col: ordered_columns) {
        if (new_ordered.empty()) {
            new_ordered.push_back(col);
            continue;
        }
        Chain* gap = nullptr;
        for (auto seq_pos: new_ordered.back()->positions) {
            auto seq = seq_pos.first;
            if (col->positions.find(seq) == col->positions.end())
                continue;
            auto pos = seq_pos.second;
            if (col->positions[seq] == pos + 1)
                continue;
            if (gap != nullptr) {
                // not well-determined
                gap = nullptr;
                break;
            }
            gap = seq;
        }
        if (gap != nullptr) {
            for (auto pos = new_ordered.back()->positions[gap]+1; pos < col->positions[gap]; ++pos) {
                if (gap->residues()[pos % gap->residues().size()] == nullptr)
                    continue;
                new_ordered.emplace_back(new Column({{gap, pos}}));
            }
        }
        new_ordered.push_back(col);
    }
    ordered_columns = new_ordered;

    // Squeeze column where possible:
    //
    //   Find pairs of columns where the left-hand one could accept
    //   one or more residues from the right-hand one
    //
    //   Keep looking right (if necessary) until each row has at
    //   least one gap, but no more than one
    //
    //   Squeeze
    decltype(ordered_columns)::size_type col_index = 0;
    while (col_index < ordered_columns.size() - 1) {
        logger::status(py_logger, status_prefix,
            "Merging columns (", col_index, "/", ordered_columns.size()-1, ")");
        auto l = ordered_columns[col_index];
        auto r = ordered_columns[col_index+1];
        bool squeezable = false;
        for (auto seq_pos: r->positions) {
            auto seq = seq_pos.first;
            if (l->positions.find(seq) == l->positions.end()) {
                squeezable = true;
                break;
            }
        }
        if (!squeezable) {
            col_index += 1;
            continue;
        }

        std::map<Chain*, std::shared_ptr<GapInfo>> gap_info;
        for (auto seq: chains) {
            // Couldn't figure out how to emplace() in a map where the second arg is a class
            // with constructor args, so...
            // (after more Googling, maybe:
            //    .emplace(seq, GapInfo{...args..})
            // or
            //    .emplace<Chain*, GapInfo>(seq, {...args...})
            if (l->positions.find(seq) != l->positions.end())
                gap_info[seq] = std::shared_ptr<GapInfo>(new GapInfo(false, l->positions[seq], 0u));
            else
                gap_info[seq] = std::shared_ptr<GapInfo>(new GapInfo(true, INT_MAX, 1u));
        }

        squeezable = false;
        bool redo = false;
        int rcols = 0;
        for (auto ri = ordered_columns.begin()+col_index+1; ri != ordered_columns.end(); ++ri) {
            auto r = *ri;
            rcols += 1;
            // look for indeterminate residues first, so we can potentially
            // form a single-residue column to complete the squeeze
            bool indeterminates = false;
            for (auto seq_rpos: r->positions) {
                auto seq = seq_rpos.first;
                auto right_pos = seq_rpos.second;
                auto gi = gap_info[seq];
                auto left_pos = gi->pos;
                if (gi->pos == INT_MAX || right_pos == left_pos + 1)
                    continue;
                if (gi->num_gaps == 0) {
                    indeterminates = true;
                    continue;
                }
                bool broke_gaps = false;
                for (auto chain_gi: gap_info) {
                    auto oseq = chain_gi.first;
                    if (oseq == seq)
                        continue;
                    auto info = chain_gi.second;
                    if (info->in_gap)
                        continue;
                    if (info->num_gaps != 0) {
                        broke_gaps = true;
                        break;
                    }
                }
                if (!broke_gaps) {
                    // squeezable
                    ordered_columns.insert(ordered_columns.begin() + col_index + rcols,
                        std::shared_ptr<Column>(new Column({{seq, left_pos}})));
                    redo = true;
                    break;
                }
                indeterminates = true;
            }

            if (redo)
                break;

            if (indeterminates)
                break;

            bool broke_gaps = false;
            for (auto seq_info: gap_info) {
                auto seq = seq_info.first;
                auto info = seq_info.second;
                auto in_gap = info->in_gap;
                auto num_gaps = info->num_gaps;
                if (r->positions.find(seq) != r->positions.end()) {
                    auto right_pos = r->positions[seq];
                    if (in_gap)
                        // closing a gap
                        gap_info[seq] = std::shared_ptr<GapInfo>(new GapInfo(false, right_pos, 1));
                    else
                        // non gap
                        gap_info[seq] = std::shared_ptr<GapInfo>(new GapInfo(false, right_pos, num_gaps));
                } else {
                    if (!in_gap && num_gaps > 0) {
                        // two gaps: no-no
                        broke_gaps = true;
                        break;
                    }
                    auto left_pos = info->pos;
                    gap_info[seq] = std::shared_ptr<GapInfo>(new GapInfo(true, left_pos, 1));
                }
            }
            if (!broke_gaps) {
                // check if squeeze criteria fulfilled
                bool broke_squeeze = false;
                for (auto seq_info: gap_info) {
                    auto info = seq_info.second;
                    if (info->num_gaps == 0) {
                        broke_squeeze = true;
                        break;
                    }
                }
                if (!broke_squeeze) {
                    squeezable = true;
                    break;
                }
                l = r;
                continue;
            }
            break;
        }

        if (redo)
            continue;

        if (!squeezable) {
            col_index += 1;
            continue;
        }

        // squeeze
        std::vector<std::shared_ptr<Column>> replace_cols;
        auto rc_end = ordered_columns.begin() + col_index + rcols + 1;
        for (auto oi = ordered_columns.begin() + col_index; oi != rc_end; ++oi)
            replace_cols.emplace_back(*oi);
        bool broke_col_value = false;
        for (decltype(replace_cols)::size_type i = 0; i < replace_cols.size()-1; ++i) {
            auto col = replace_cols[i];
            auto rcol = replace_cols[i+1];
            for (auto seq_pos: rcol->positions) {
                auto seq = seq_pos.first;
                if (col->positions.find(seq) != col->positions.end())
                    continue;
                auto pos = seq_pos.second;
                col->positions[seq] = pos;
                rcol->positions.erase(seq);
            }
            if (col->value(pas, dist_cutoff) < 0.0) {
                broke_col_value = true;
                break;
            }
        }
        if (!broke_col_value) {
            if (!replace_cols.back()->positions.empty()) {
                PyErr_SetString(PyExc_AssertionError, "Final replacement column not empty");
                return nullptr;
            }
            double ov = 0.0;
            auto rc_end = ordered_columns.begin() + col_index + rcols + 1;
            for (auto oi = ordered_columns.begin() + col_index; oi != rc_end; ++oi)
                ov += (*oi)->participation(pas, dist_cutoff);
            decltype(ov) nv = 0.0;
            for (decltype(replace_cols)::size_type i = 0; i < replace_cols.size()-1; ++i)
                nv += replace_cols[i]->participation(pas, dist_cutoff);
            if (ov >= nv) {
                col_index += 1;
                continue;
            }
            for (decltype(rcols) i = 0; i < rcols; ++i)
                ordered_columns[col_index + i] = replace_cols[i];
            ordered_columns.erase(ordered_columns.begin() + col_index + rcols);
            if (col_index > 0)
                col_index -= 1;
            continue;
        }
        col_index += 1;
    }

    logger::status(py_logger, status_prefix, "Composing alignment");
    for (auto col: ordered_columns) {
        for (auto seq_offset: col->positions) {
            auto seq = seq_offset.first;
            auto offset = seq_offset.second;
            auto cur_pos = current[seq];
            auto diff = offset - cur_pos;
            if (diff < 2)
                continue;
            //TODO: circular
            auto start_frag = cur_pos+1;
            decltype(start_frag) end_frag = offset;
            for (auto ci = start_frag; ci <= end_frag; ++ci)
                working_seqs[seq].push_back(seq->characters()[ci]);
            Sequence::Contents gap(diff-1, gap_char);
            for (auto wseq_chars: working_seqs) {
                auto wseq = wseq_chars.first;
                if (wseq == seq)
                    continue;
                auto& chars = wseq_chars.second;
                chars.insert(chars.end(), gap.begin(), gap.end());
            }
        }
        for (auto seq: chains) {
            if (col->positions.find(seq) != col->positions.end()) {
                auto offset = col->positions[seq];
                //TODO: circular
                auto c = seq->characters()[offset];
                working_seqs[seq].push_back(c);
                current[seq] = offset;
            } else
                working_seqs[seq].push_back(gap_char);
        }
    }

    for (auto seq_offset: current) {
        //TODO: circular
        auto seq = seq_offset.first;
        auto offset = seq_offset.second;
        if (offset == static_cast<int>(seq->size())-1)
            continue;
        Sequence::Contents frag;
        for (Sequence::Contents::size_type ci = offset+1; ci < seq->size(); ++ci)
            frag.push_back(seq->characters()[ci]);
        Sequence::Contents gap(frag.size(), gap_char);
        for (auto wseq_chars: working_seqs) {
            auto wseq = wseq_chars.first;
            auto& chars = wseq_chars.second;
            if (wseq == seq)
                chars.insert(chars.end(), frag.begin(), frag.end());
            else
                chars.insert(chars.end(), gap.begin(), gap.end());
        }
    }

    // Put the sequences in the clones
    for (auto chain_clone: cpp_clones) {
        auto chain = chain_clone.first;
        auto& clone = chain_clone.second;
        auto& clone_seq = working_seqs[clone];
        auto& residues = chain->residues();
        clone->bulk_set(residues, &clone_seq);
    }

    auto py_clones = PyList_New(cpp_clones.size());
    if (py_clones == nullptr) {
        PyErr_SetString(PyExc_AssertionError, "Could not make list for aligned sequences");
        return nullptr;
    }
    Py_ssize_t li = 0;
    for (auto chain_clone: cpp_clones)
        PyList_SET_ITEM(py_clones, li++, chain_clone.second->py_instance(true));

    logger::status(py_logger, status_prefix, "Done");
    return py_clones;
}

static PyObject*
py_multi_align(PyObject*, PyObject* args)
{
    PyObject* chain_ptrs_list;
    PyObject* py_logger;
    PyObject* error_class;
    double dist_cutoff;
    int col_all, circular, py_gap_char;
    const char* status_prefix;
    if (!PyArg_ParseTuple(args, const_cast<char *>("OfpCpsOO"), &chain_ptrs_list, &dist_cutoff,
            &col_all, &py_gap_char, &circular, &status_prefix, &py_logger, &error_class))
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
        status_prefix, py_logger, error_class);
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
