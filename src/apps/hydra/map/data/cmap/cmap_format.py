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
        f = tables.openFile(path)

        agroups = self.find_arrays(f.root)
        if len(agroups) == 0:
            raise ValueError('Chimera HDF5 file %s contains no 3d arrays' % path)

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
        for node in parent._f_iterNodes():
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

        i0,j0,k0 = ijk_origin
        isz,jsz,ksz = ijk_size
        istep,jstep,kstep = ijk_step
        import tables
        f = tables.openFile(self.path)
        if progress:
            progress.close_on_cancel(f)
        array_path = choose_chunk_size(f, array_paths, ijk_size)
        a = f.getNode(array_path)
        cshape = a._v_chunkshape
        csmin = min(cshape)
        if cshape[0] == csmin:
            for k in range(k0,k0+ksz,kstep):
                array[(k-k0)/kstep,:,:] = a[k,j0:j0+jsz:jstep,i0:i0+isz:istep]
                if progress:
                    progress.plane((k-k0)/kstep)
        elif cshape[1] == csmin:
            for j in range(j0,j0+jsz,jstep):
                array[:,(j-j0)/jstep,:] = a[k0:k0+ksz:kstep,j,i0:i0+isz:istep]
                if progress:
                    progress.plane((j-j0)/jstep)
        else:
            for i in range(i0,i0+isz,istep):
                array[:,:,(i-i0)/istep] = a[k0:k0+ksz:kstep,j0:j0+jsz:jstep,i]
                if progress:
                    progress.plane((i-i0)/istep)
        if progress:
            progress.done()

        f.close()
        return array

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
            step = tuple(map(float, va.step))
        else:
            step = (1.0, 1.0, 1.0)
        return step

    # --------------------------------------------------------------------------
    #
    def find_origin(self, group):

        va = group._v_attrs
        if 'origin' in va:
            origin = tuple(map(float, va.origin))
        else:
            origin = (0.0, 0.0, 0.0)
        return origin

    # --------------------------------------------------------------------------
    #
    def find_cell_angles(self, group):

        va = group._v_attrs
        if 'cell_angles' in va:
            cell_angles = tuple(map(float, va.cell_angles))
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
            from ....geometry import matrix
            r = matrix.rotation_from_axis_angle(axis, angle)
        else:
            r = ((1,0,0),(0,1,0),(0,0,1))
        return r

    # --------------------------------------------------------------------------
    #
    def find_symmetries(self, group):

        va = group._v_attrs
        if 'symmetries' in va:
            sym = va.symmetries
        else:
            sym = None
        return sym

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
        raise ValueError(message)

# -----------------------------------------------------------------------------
#
def choose_chunk_size(f, array_paths, ijk_size):

    if len(array_paths) == 1:
        return array_paths[0]

    alist = []
    shape = tuple(reversed(ijk_size))
    for p in array_paths:
        a = f.getNode(p)
        cshape = a._v_chunkshape
        pad = [(cshape[i] - (shape[i]%cshape[i]))%cshape[i] for i in (0,1,2)]
        extra = sum([pad[i] * shape[(i+1)%3] * shape[(i+2)%3] for i in (0,1,2)])
        alist.append((extra, p))
    alist.sort()
    array_path = alist[0][1]
    return array_path
