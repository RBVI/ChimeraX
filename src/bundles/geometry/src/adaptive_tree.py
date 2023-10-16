# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

""" adaptive_tree.py: Define AdaptiveTree class to simplify 3D space
    partitioning (though class is not limited to three dimensions).
    Defines an 'adaptive k-d tree' as per Friedman, Bentley, and
    Finkel and as described in "The Design and Analysis of Spatial
    Data Structures", H. Samet (1989), pp. 70-71.
"""
from numpy import array, object_
from operator import add
from functools import reduce

import sys
leaf = sys.intern('leaf')
interior = sys.intern('interior')

class AdaptiveTree:
    """Define an 'adaptive k-d tree' as per "The Design and Analysis of
        Spatial Data Structures" pp. 70-71.  Basically, given a set
        of k-dimensional points (each dimension referred to as an
        "attribute") with associated data, they are partitioned into
        leaf nodes.  Each leaf nodes hold lists of associated data
        whose corresponding attributes vary by less than an initially-
        supplied threshold ('sep_val').  Also, each leaf node holds a
        bounding box of the leaf data.

        The interior nodes of the tree contain details of the
        partitioning.  In particular, what attribute this node
        partitions along ('axis'), and what value ('median')
        partitions left child node from right child node.  Whether
        a node is interior or leaf is stored in 'type'.
    """

    def __init__(self, attribute_data, leaf_data, sep_val):
        """attribute_data is a sequence of sequences.  Each individual
           sequence is attribute data.  For example, in a 3D space
           partitioning, the attribute data is x, y, and z values.

           leaf_data ia a sequence of the same length as attribute_data.
           Each item is what is to put into leaf nodes after tree
           partitioning.

           sep_val is the value at which a tree will no longer be
           decomposed, i.e. if the maximum variance of each attribute
           is less than sep_val, then a leaf node is created."""

        if len(attribute_data) > 0: # because numpy, can't just "if attribute_data"; ugh
            attr_data = array(attribute_data)
            leaf_data = array(leaf_data, object_)
            self.root = Node(attr_data, leaf_data, sep_val)
        else:
            self.root = None

    def search_tree(self, target, window, zero=0.0):
        """Search tree for all leaves within 'window' of target.

        The cumulative difference of all attributes from target must
        be less than 'window'.

        Note that this search only identifies leaf nodes that could
        satisfy the window criteria and returns all leaf data in
        those nodes.  Each individual leaf data may not satisfy the
        window criteria.

        For attributes that aren't floats or ints but that otherwise
        obey numeric operations and comparisons, the 'zero' parameter
        may be specified so that the searches know how to initialize
        their difference totals.
        """

        if not self.root:
            return []
        return _search_node(self.root, target, window * window, [zero]*len(target), zero)

    def bbox_search_tree(self, bbox):
        """Search tree for all leaves within a bounding box.

        Mostly similar to 'search_tree'.  'bbox' is a sequence of
        lower-bound / upper-bound pairs defining the bounds on
        each axis.
        """

        if not self.root:
            return []
        leaves = []
        _bbox_search_node(self.root, bbox, leaves)
        return leaves

class Node:
    def __init__(self, attr_data, leaf_data, sep_val):
        if len(attr_data) < 2:
            # leaf node
            self._make_leaf(leaf_data, attr_data)
            return

        max_var = -1
        last_index = len(attr_data) - 1
        for axis in range(len(attr_data[0])):
            axis_data = attr_data[:,axis]
            sort = axis_data.argsort()
            var = axis_data[sort[last_index]] - axis_data[sort[0]]
            if var < sep_val:
                continue

            # want axis that varies most from the median rather
            # than the one that varies most from end to end
            median = (axis_data[sort[last_index//2]] + axis_data[sort[1+last_index//2]]) / 2.0
            var1 = median - axis_data[sort[0]]
            var2 = axis_data[sort[last_index]] - median
            if var1 > var2:
                var = var1
            else:
                var = var2
            if var > max_var:
                max_var = var
                max_axis = axis
                max_sort = sort
                max_axis_data = axis_data
                # there can be freak cases where the median
                # is the same as an end value (e.g. [a a b]),
                # so we need to tweak the median in these
                # cases so that the left and right nodes both
                # receive data
                if axis_data[sort[0]] == median:
                    for ad in axis_data[sort]:
                        if ad > median:
                            median = (median + ad) / 2.0
                            break
                elif axis_data[sort[-1]] == median:
                    for ad in axis_data[sort[-1::-1]]:
                        if ad < median:
                            median = (median + ad) / 2.0
                            break
                max_median = median

        if max_var < 0:
            # leaf_node
            self._make_leaf(leaf_data, attr_data)
            return

        self.type = interior
        self.axis = max_axis
        self.median = max_median

        # less than median goes into left node, greater-than-or-
        # equal-to goes into right node
        left_index = 0
        for index in range(last_index//2, -1, -1):
            if max_axis_data[max_sort[index]] < max_median:
                left_index = index + 1
                break
        self.left = Node(attr_data.take(max_sort[:left_index], 0),
                leaf_data.take(max_sort[:left_index], 0), sep_val)
        self.right = Node(attr_data.take(max_sort[left_index:], 0),
                leaf_data.take(max_sort[left_index:], 0), sep_val)

    def _make_leaf(self, leaf_data, attr_data):
        self.type = leaf
        if isinstance(leaf_data, list):
            self.leaf_data = leaf_data
        else:
            self.leaf_data = leaf_data.tolist()
        self.bbox = []
        last_index = len(attr_data) - 1
        for axis in range(len(attr_data[0])):
            axis_data = attr_data[:,axis]
            sort = axis_data.argsort()
            self.bbox.append(([axis_data[sort[0]], axis_data[sort[last_index]]]))

def _search_node(node, target, window_sq, diffs_sq, zero):
    if node.type == leaf:
        diff_sq_sum = zero
        for axis in range(len(target)):
            min, max = node.bbox[axis]
            target_val = target[axis]
            if target_val < min:
                diff = min - target_val
                diff_sq_sum = diff_sq_sum + diff * diff
            elif target_val > max:
                diff = target_val - max
                diff_sq_sum = diff_sq_sum + diff * diff
            if diff_sq_sum > window_sq:
                return []
        return node.leaf_data

    # interior
    target_val = target[node.axis]
    diff_sq_sum = reduce(add, diffs_sq)
    diff_sq_sum = diff_sq_sum - diffs_sq[node.axis]
    remaining_window_sq = window_sq - diff_sq_sum

    if target_val < node.median:
        leaves = _search_node(node.left, target, window_sq, diffs_sq, zero)
        diff = node.median - target_val
        diff_sq = diff * diff
        if diff_sq <= remaining_window_sq:
            prev_diff_sq = diffs_sq[node.axis]
            diffs_sq[node.axis] = diff_sq
            leaves = leaves + _search_node(node.right, target, window_sq, diffs_sq, zero)
            diffs_sq[node.axis] = prev_diff_sq
    else:
        leaves = _search_node(node.right, target, window_sq, diffs_sq, zero)
        diff = target_val - node.median
        diff_sq = diff * diff
        if diff_sq <= remaining_window_sq:
            prev_diff_sq = diffs_sq[node.axis]
            diffs_sq[node.axis] = diff_sq
            leaves = leaves + _search_node(node.left, target, window_sq, diffs_sq, zero)
            diffs_sq[node.axis] = prev_diff_sq
    return leaves

def _bbox_search_node(node, bbox, leaf_list):
    if node.type == leaf:
        for axis in range(len(bbox)):
            nmin, nmax = node.bbox[axis]
            bbmin, bbmax = bbox[axis]
            if nmin > bbmax or nmax < bbmin:
                return
        for datum in node.leaf_data:
            leaf_list.append(datum)
        return

    # interior node
    bbmin, bbmax = bbox[node.axis]
    if bbmin < node.median:
        _bbox_search_node(node.left, bbox, leaf_list)
    if bbmax >= node.median:
        _bbox_search_node(node.right, bbox, leaf_list)
