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

#include <algorithm>  // std::sort, mix/max_element
#include <utility>  // std::make_pair

#define ATOMSTRUCT_EXPORT
#include "search.h"

namespace atomstruct {

AtomSearchTree::AtomSearchTree(const std::vector<Atom*>& atoms, bool transformed, double sep_val):
    _atoms(atoms), _sep_val(sep_val), _transformed(transformed)
{
    init_root();
}

AtomSearchTree::~AtomSearchTree()
{
    if (root != nullptr)
        delete root;
}

void
AtomSearchTree::destructors_done(const std::set<void*>& destroyed)
{
    std::vector<Atom*> survivors;
    for (auto a: _atoms)
        if (destroyed.find(a) == destroyed.end())
            survivors.push_back(a);
    if (survivors.size() < _atoms.size()) {
        delete root;
        _atoms.swap(survivors);
        init_root();
    }
}

std::vector<Atom*>
AtomSearchTree::search(Atom* a, double window)
{
    if (_transformed)
        return search(a->scene_coord(), window);
    return search(a->coord(), window);
}

std::vector<Atom*>
AtomSearchTree::search(const Coord &target, double window)
{
    // Search tree for all leaves within 'window' of target.
    //
    // The cumulative distance along all axes must be less than 'window'.
    //
    // Note that unlike the Python AdaptiveTree implementation,
    // the leaf atoms are pruned down to just those that
    // satisfy the 'window' criteria.
    
    std::vector<Atom*> ret_val;
    if (root != nullptr) {
        double window_sq = window * window;
        double diffs_sq[3] = { 0.0, 0.0, 0.0 };
        for (auto a: root->search(target, window_sq, diffs_sq)) {
            if (_transformed) {
                if (a->scene_coord().sqdistance(target) <= window_sq)
                    ret_val.push_back(a);
            } else {
                if (a->coord().sqdistance(target) <= window_sq)
                    ret_val.push_back(a);
            }
        }
    }
    return ret_val;
}

void
AtomSearchTree::init_root()
{
    if (_atoms.size() > 0)
        root = new _Node(_atoms, _transformed, _sep_val);
    else
        root = nullptr;
}

_Node::_Node(const std::vector<Atom*>& atoms, bool transformed, double sep_val)
{
    if (atoms.size() < 2) {
        // leaf node
        _make_leaf(atoms, transformed);
        return;
    }

    std::vector<Real> axes_data[3];
    for (auto& ad: axes_data)
        ad.reserve(atoms.size());
    if (transformed) {
        for (auto a: atoms) {
            auto crd = a->scene_coord();
            for (int i = 0; i < 3; ++i)
                axes_data[i].push_back(crd[i]);
        }
    } else {
        for (auto a: atoms) {
            auto crd = a->coord();
            for (int i = 0; i < 3; ++i)
                axes_data[i].push_back(crd[i]);
        }
    }

    double max_var = -1.0;
    int last_index = atoms.size() - 1;
    int max_axis;
    double max_median;
    std::vector<std::pair<Real,Atom*>> max_atom_axis_data;
    std::vector<double> max_axis_data;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<std::pair<Real,Atom*>> atom_axis_data;
        atom_axis_data.reserve(atoms.size());
        for (auto a: atoms) {
            if (transformed)
                atom_axis_data.push_back(std::make_pair(a->scene_coord()[axis], a));
            else
                atom_axis_data.push_back(std::make_pair(a->coord()[axis], a));
        }
        std::sort(atom_axis_data.begin(), atom_axis_data.end());
        std::vector<double> axis_data;
        axis_data.reserve(atoms.size());
        for (auto crd_atom: atom_axis_data)
            axis_data.push_back(crd_atom.first);
        double var = axis_data.back() - axis_data.front();
        if (var < sep_val)
            continue;

        // want axis that caries most from the median rather
        // than the one that varies most from end to end
        double median = (axis_data[last_index/2] + axis_data[1+(last_index/2)]) / 2.0;
        double var1 = median - axis_data.front();
        double var2 = axis_data.back() - median;
        if (var1 > var2)
            var = var1;
        else
            var = var2;
        if (var > max_var) {
            max_var = var;
            max_axis = axis;
            // there can be freak cases where the median is the same as an end value
            // (e.g. [a a b]), so we need to tweak the median in these cases so that
            // the left and right nodes both receive data
            if (axis_data.front() == median) {
                for (auto ad: axis_data) {
                    if (ad > median) {
                        auto next_median = (median + ad) / 2.0;
                        if (next_median == median)
                            // difference so small that you cannot halve it
                            median = ad;
                        else
                            median = next_median;
                        break;
                    }
                }
            } else if (axis_data.back() == median) {
                for (int i = last_index; i >= 0; --i) {
                    if (axis_data[i] < median) {
                        auto next_median = (median + axis_data[i]) / 2.0;
                        if (next_median == median)
                            // difference so small that you cannot halve it
                            median = axis_data[i];
                        else
                            median = next_median;
                        break;
                    }
                }
            }
            max_median = median;
            max_atom_axis_data.swap(atom_axis_data);
            max_axis_data.swap(axis_data);
        }
    }

    if (max_var < 0) {
        // leaf node
        _make_leaf(atoms, transformed);
        return;
    }

    type = Interior;
    axis = max_axis;
    median = max_median;

    // less than median goes into the left node,
    // greater-than-or-equal-to goes into the right node
    int left_index = 0;
    for (int index = last_index/2; index >= 0; --index) {
        if (max_axis_data[index] < median) {
            left_index = index + 1;
            break;
        }
    }
    std::vector<Atom*> left_atoms, right_atoms;
    for (int i = 0; i < left_index; ++i)
        left_atoms.push_back(max_atom_axis_data[i].second);
    left = new _Node(left_atoms, transformed, sep_val);
    for (int i = left_index; i <= last_index; ++i)
        right_atoms.push_back(max_atom_axis_data[i].second);
    right = new _Node(right_atoms, transformed, sep_val);
}

_Node::~_Node()
{
    if (type == Interior) {
        delete left;
        delete right;
    }
}

void
_Node::_make_leaf(const std::vector<Atom*>& atoms, bool transformed)
{
    type = Leaf;
    leaf_atoms = atoms;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<double>  axis_data;
        axis_data.reserve(atoms.size());
        for (auto a: atoms) {
            if (transformed)
                axis_data.push_back(a->scene_coord()[axis]);
            else
                axis_data.push_back(a->coord()[axis]);
        }
        bbox[axis][0] = *std::min_element(axis_data.begin(), axis_data.end());
        bbox[axis][1] = *std::max_element(axis_data.begin(), axis_data.end());
    }
        
}

std::vector<Atom*>
_Node::search(const Coord& target, double window_sq, double* diffs_sq)
{
    if (type == Leaf) {
        double diff_sq_sum = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            double min = bbox[axis][0];
            double max = bbox[axis][1];
            double target_val = target[axis];
            if (target_val < min) {
                double diff = min - target_val;
                diff_sq_sum += diff * diff;
            } else if (target_val > max) {
                double diff = target_val - max;
                diff_sq_sum += diff * diff;
            }
            if (diff_sq_sum > window_sq)
                return std::vector<Atom*>();
        }
        return leaf_atoms;
    }

    // interior
    double target_val = target[axis];
    double diff_sq_sum = diffs_sq[0] + diffs_sq[1] + diffs_sq[2];
    double remaining_window_sq = window_sq - diff_sq_sum;

    std::vector<Atom*> leaves;
    if (target_val < median) {
        leaves = left->search(target, window_sq, diffs_sq);
        double diff = median - target_val;
        double diff_sq = diff * diff;
        if (diff_sq <= remaining_window_sq) {
            double prev_diff_sq = diffs_sq[axis];
            auto new_leaves = right->search(target, window_sq, diffs_sq);
            leaves.insert(leaves.end(), new_leaves.begin(), new_leaves.end());
            diffs_sq[axis] = prev_diff_sq;
        }
    } else {
        leaves = right->search(target, window_sq, diffs_sq);
        double diff = target_val - median;
        double diff_sq = diff * diff;
        if (diff_sq <= remaining_window_sq) {
            double prev_diff_sq = diffs_sq[axis];
            auto new_leaves = left->search(target, window_sq, diffs_sq);
            leaves.insert(leaves.end(), new_leaves.begin(), new_leaves.end());
            diffs_sq[axis] = prev_diff_sq;
        }
    }
    return leaves;
}

}  // namespace atomstruct
