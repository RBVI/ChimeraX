# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Read 3d array data from a HDF5 file using PyTables (table).
# The entire directory tree in the HDF5 file is searched for all 3D numeric arrays.
#
class HDFData:

    def __init__(self, path):

        self.path = path

        import os.path
        self.name = os.path.basename(path)
    
        import tables
        f = tables.open_file(path)

        arrays = self.find_arrays(f.root)
        if len(arrays) == 0:
            raise SyntaxError('HDF5 file %s contains no 3d arrays' % path)

        imlist = [HDFImage(a) for a in arrays]
        imlist.sort(key = lambda i: i.name)
        self.images = imlist

        f.close()

    # --------------------------------------------------------------------------
    # Return list of grouped arrays.  Each element is a tuple containing a
    # group and a list of 2-d and 3-d arrays that are children of the group.
    #
    def find_arrays(self, parent, arrays = None):

        if arrays is None:
            arrays = []

        garrays = []
        groups = []
        from tables.array import Array
        from tables.group import Group
        for node in parent._f_iter_nodes():
            if isinstance(node, Array):
                dims = len(node.shape)
                if dims == 3:
                    garrays.append(node)
            elif isinstance(node, Group):
                groups.append(node)

        if garrays:
            arrays.extend(garrays)

        for g in groups:
            self.find_arrays(g, arrays = arrays)

        return arrays

    # --------------------------------------------------------------------------
    # Reads a submatrix returning 3D NumPy matrix with zyx index order.
    # array_path can be a HDF5 path to a 3d array or a list of paths to
    # a stack of 2d arrays.
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step,
                    array_path, array, progress):

        import tables
        f = tables.open_file(self.path)
        if progress:
            progress.close_on_cancel(f)
        a = f.get_node(array_path)
        from ..cmap import copy_hdf5_array
        copy_hdf5_array(a, ijk_origin, ijk_size, ijk_step, array, progress)
        if progress:
            progress.done()

        f.close()
        return array

    # --------------------------------------------------------------------------
    #
    def find_attribute(self, attribute_name):
        '''Used for finding segmentation attributes.'''
        values = []
        import tables
        f = tables.open_file(self.path)
        from tables.array import Array
        from tables.group import Group
        for node in f.root._f_walknodes():
            if isinstance(node, Array) and node.name == attribute_name:
                values.append((node._v_pathname, node.read()))
            elif isinstance(node, Group) and attribute_name in node._v_attrs:
                a = getattr(node._v_attrs, attribute_name)
                values.append(('%s/%s' % (node._v_pathname, attribute_name), a))
        f.close()
        if len(values) > 1:
            paths = ','.join(path for path,value in values[:5])
            if len(paths) > 5:
                paths += '...'
            raise LookupError('HDFGrid.find_attribute(): More than one'
                              ' attribute with name "%s" in file %s (%s)'
                              % (attribute_name, self.path, paths))
        value = values[0][1] if len(values) == 1 else None
        return value

# -----------------------------------------------------------------------------
#
class HDFImage:

    def __init__(self, array):

        # TODO: Handle error when group has only subsample arrays
        self.array_path = array._v_pathname
        from os.path import basename
        self.name = basename(self.array_path)
        self.size = tuple(reversed(array.shape))
        self.value_type = array.atom.dtype
