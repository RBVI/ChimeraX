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
# Read 3d array data from a Chimera HDF5 file using PyTables (table).
#
# Example HDF5 format written by Chimera.
#
#  /image (group, any name allowed)
#    name "centriole" (attribute)
#    step (1.2, 1.2, 1.2) (attribute)
#    origin (-123.4, -522, 34.5) (attribute)
#    cell_angles (90.0, 90.0, 90.0) (attribute)
#    rotation_axis (0.0, 0.0, 1.0) (attribute)
#    rotation_angle 45.0 (attribute, degrees)
#    color (1.0, 1.0, 0, 1.0) (attribute, rgba 0-1 float)
#    time 5 (attribute, time series frame number)
#    channel 0 (attribute, integer for multichannel data)
#    /data (3d array of uint8 (123,542,82)) (dataset, any name allowed)
#    /data_x (3d array of uint8 (123,542,82), alternate chunk shape) (dataset, any name allowed)
#    /data_2 (3d array of uint8 (61,271,41)) (dataset, any name allowed)
#        subsample_spacing (2, 2, 2) (attribute)
#    (more subsampled or alternate chunkshape versions of same data)
#
class Chimera_HDF_Data:

    def __init__(self, path):

        self.path = path

        import os.path
        self.name = os.path.basename(path)
    
        import tables
        f = tables.open_file(path)

        agroups = self.find_arrays(f.root)
        if len(agroups) == 0:
            raise SyntaxError('Chimera HDF5 file %s contains no 3d arrays' % path)

        imlist = [Chimera_HDF_Image(g,a) for g,a in agroups]
        imlist.sort(key = lambda i: i.name)
        self.images = imlist

        f.close()

    # --------------------------------------------------------------------------
    # Return list of grouped arrays.  Each element is a tuple containing a
    # group and a list of 2-d and 3-d arrays that are children of the group.
    #
    def find_arrays(self, parent, anodes = None):

        if anodes is None:
            anodes = []

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
            anodes.append((parent, garrays))

        for g in groups:
            self.find_arrays(g, anodes)

        return anodes

    # --------------------------------------------------------------------------
    # Reads a submatrix returning 3D NumPy matrix with zyx index order.
    # array_path can be a HDF5 path to a 3d array or a list of paths to
    # a stack of 2d arrays.
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step,
                    array_paths, array, progress):

        import tables
        f = tables.open_file(self.path)
        if progress:
            progress.close_on_cancel(f)
        array_path = choose_chunk_size(f, array_paths, ijk_size)
        a = f.get_node(array_path)
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
            raise LookupError('Chimera_HDF_Grid.find_attribute(): More than one'
                              ' attribute with name "%s" in file %s (%s)'
                              % (attribute_name, self.path, paths))
        value = values[0][1] if len(values) == 1 else None
        return value
            
# -----------------------------------------------------------------------------
#
def copy_hdf5_array(a, ijk_origin, ijk_size, ijk_step, array,
                    progress = None, block_size = 2**26):
    i0,j0,k0 = ijk_origin
    isz,jsz,ksz = ijk_size
    istep,jstep,kstep = ijk_step

    if array.nbytes <= block_size:
        array[:,:,:] = a[k0:k0+ksz:kstep,j0:j0+jsz:jstep,i0:i0+isz:istep]
        return

    # Read in blocks along axis with smallest chunk size.
    cshape = a._v_chunkshape
    if cshape is None:
        axis = 0
    else:
        csmin = min(cshape)
        axis = cshape.index(csmin)
    bf = block_size / array.nbytes
    if axis == 0:
        n = ksz // kstep
        pstep = max(1, int(bf*n))
        kpstep = kstep*pstep
        kmax = k0 + ksz
        for p in range(0,n,pstep):
            k = k0+p*kstep
            array[p:p+pstep,:,:] = a[k:min(k+kpstep,kmax):kstep,j0:j0+jsz:jstep,i0:i0+isz:istep]
            if progress:
                progress.plane(p)
    elif axis == 1:
        n = jsz // jstep
        pstep = max(1, int(bf*n))
        jpstep = jstep*pstep
        jmax = j0 + jsz
        for p in range(0,n,pstep):
            j = j0+p*jstep
            array[:,p:p+pstep,:] = a[k0:k0+ksz:kstep,j:min(j+jpstep,jmax):jstep,i0:i0+isz:istep]
            if progress:
                progress.plane(p)
    elif axis == 2:
        n = isz // istep
        pstep = max(1, int(bf*n))
        ipstep = istep*pstep
        imax = i0 + isz
        for p in range(0,n,pstep):
            i = i0+p*istep
            array[:,p:p+pstep,:] = a[k0:k0+ksz:kstep,j0:j0+jsz:jstep,i:min(i+ipstep,imax):istep]
            if progress:
                progress.plane(p)

# -----------------------------------------------------------------------------
#
class Chimera_HDF_Image:

    def __init__(self, group, arrays):

        parrays = []    # Primary data arrays.  If more than one they should
                        #  have different chunkshapes for efficient access
                        #  of different subregions, e.g. x, y, and z planes.
        sarrays = []    # Subsample arrays
        for a in arrays:
            if 'subsample_spacing' in a._v_attrs:
                sarrays.append(a)
            else:
                parrays.append(a)

        # TODO: Handle error when group has only subsample arrays
        self.array_paths = [a._v_pathname for a in parrays]
        self.size = self.check_array_sizes(parrays)
        self.name = self.find_name(group)
        self.step = self.find_plane_spacing(group)
        self.origin = self.find_origin(group)
        self.value_type = self.check_array_types(parrays + sarrays)
        self.cell_angles = self.find_cell_angles(group)
        self.rotation = self.find_rotation(group)
        self.symmetries = self.find_symmetries(group)
        self.default_color = self.find_color(group)
        self.time = self.find_time(group)
        self.channel = self.find_channel(group)

        subsamples = []
        stable = {}
        for a in sarrays:
            step = tuple(a._v_attrs.subsample_spacing)
            size = tuple(reversed(a.shape))
            if not (step,size) in stable:
                array_paths = [a._v_pathname]
                subsamples.append((step, size, array_paths))
                stable[(step,size)] = array_paths
            else:
                stable[(step,size)].append(a._v_pathname)
        self.subsamples = subsamples

    # --------------------------------------------------------------------------
    #
    def find_name(self, group):

        va = group._v_attrs
        if 'name' in va:
            name = va.name
            if isinstance(name, bytes):
                # This was needed in Python 2
                name = name.decode('utf-8')
            return name
        return ''

    # --------------------------------------------------------------------------
    #
    def find_plane_spacing(self, group):

        va = group._v_attrs
        if 'step' in va:
            step = tuple(float(s) for s in va.step)
        else:
            step = (1.0, 1.0, 1.0)
        return step

    # --------------------------------------------------------------------------
    #
    def find_origin(self, group):

        va = group._v_attrs
        if 'origin' in va:
            origin = tuple(float(x) for x in va.origin)
        else:
            origin = (0.0, 0.0, 0.0)
        return origin

    # --------------------------------------------------------------------------
    #
    def find_cell_angles(self, group):

        va = group._v_attrs
        if 'cell_angles' in va:
            cell_angles = tuple(float(a) for a in va.cell_angles)
        else:
            cell_angles = (90.0, 90.0, 90.0)
        return cell_angles

    # --------------------------------------------------------------------------
    #
    def find_rotation(self, group):

        va = group._v_attrs
        if 'rotation_axis' in va and 'rotation_angle' in va:
            axis = va.rotation_axis
            angle = va.rotation_angle
            from chimerax.geometry import rotation
            rot = rotation(axis, angle)
            r = tuple(tuple(row) for row in rot.matrix[:,:3])
        else:
            r = ((1,0,0),(0,1,0),(0,0,1))
        return r

    # --------------------------------------------------------------------------
    #
    def find_symmetries(self, group):

        va = group._v_attrs
        if 'symmetries' in va:
            from chimerax.geometry import Places
            from numpy import array, float64
            sym = Places(place_array = array(va.symmetries, float64))
        else:
            sym = None
        return sym

    # --------------------------------------------------------------------------
    #
    def find_color(self, group):

        va = group._v_attrs
        if 'color' in va:
            color = tuple(float(r) for r in va.color)
        else:
            color = None
        return color

    # --------------------------------------------------------------------------
    #
    def find_time(self, group):

        va = group._v_attrs
        t = va.time if 'time' in va else None
        return t

    # --------------------------------------------------------------------------
    #
    def find_channel(self, group):

        va = group._v_attrs
        c = va.channel if 'channel' in va else None
        return c

    # --------------------------------------------------------------------------
    #
    def check_array_sizes(self, arrays):

        shape = arrays[0].shape
        for a in arrays[1:]:
            if a.shape != shape:
                self.mismatched_arrays(arrays)
        size = list(reversed(shape))
        if len(size) == 2:
            size += [1]
        size = tuple(size)
        return size

    # --------------------------------------------------------------------------
    #
    def check_array_types(self, arrays):

        dtype = arrays[0].atom.dtype
        for a in arrays[1:]:
            if a.atom.dtype != dtype:
                self.mismatched_arrays(arrays)
        return dtype

    # --------------------------------------------------------------------------
    #
    def mismatched_arrays(self, arrays):

        a0 = arrays[0]
        file_path = a0._v_file.filename
        group_path = a0._v_parent._v_pathname
        message = 'Chimera HDF5 file %s has a group %s containing arrays of different sizes or value types\n' % (file_path, group_path)
        sizes = '\n'.join(['  %s  (%d,%d,%d)  %s' %
                           ((a._v_name,) + tuple(a.shape) + (a.atom.dtype.name,))
                           for a in arrays])
        message += sizes
        raise SyntaxError(message)

# -----------------------------------------------------------------------------
#
def choose_chunk_size(f, array_paths, ijk_size):

    if len(array_paths) == 1:
        return array_paths[0]

    alist = []
    shape = tuple(reversed(ijk_size))
    for p in array_paths:
        a = f.get_node(p)
        cshape = a._v_chunkshape
        pad = [(cshape[i] - (shape[i]%cshape[i]))%cshape[i] for i in (0,1,2)]
        extra = sum([pad[i] * shape[(i+1)%3] * shape[(i+2)%3] for i in (0,1,2)])
        alist.append((extra, p))
    alist.sort()
    array_path = alist[0][1]
    return array_path
