# -----------------------------------------------------------------------------
# Read 3-d array data from an EMAN HDF5 file using PyTables (table).
#
# Example EMAN micrograph micrograph_616003.hdf file layout.
# Command: h5dump -A micrograph_616003.hdf 
#
# /MDF (group)
#  /images (group)
#    image_id_max 0 (attribute)
#    /0
#      EMAN.HostEndian "little" (attribute)
#      EMAN.ImageEndian "big"
#      EMAN.apix_x 1  ("EMAN." is part of attribute name).
#      EMAN.apix_y 1
#      EMAN.apix_z 1
#      EMAN.is_complex 0
#      EMAN.is_complex_ri 1
#      EMAN.is_complex_x 0
#      EMAN.maximum 0.315815
#      EMAN.mean 0.189981
#      EMAN.mean_nonzero 0.189981
#      EMAN.minimum -0.704721
#      EMAN.nx 2916
#      EMAN.ny 4374
#      EMAN.nz 1
#      EMAN.sigma 0.0325618
#      EMAN.sigma_nonzero 0.0325618
#      EMAN.square_sum 473871
#      EMAN.xform.align3d
#        matrix elements named "00", "01", "02", "03", "10", "11", "12", "13"
#                               "20", "21", "22", "23"
#      image (2d array of floats (2916,4374)) (dataset)
#
class EMAN_HDF_Data:

    def __init__(self, path):

        self.path = path

        import os.path
        self.name = os.path.basename(path)
    
        import tables
        f = tables.openFile(path)

        arrays = self.find_arrays(f.root)
        if len(arrays) == 0:
            raise ValueError, 'EMAN HDF5 file %s contains no arrays' % path

        self.images = [EMAN_HDF_Image(a) for a in arrays]

        f.close()

    # --------------------------------------------------------------------------
    # Return list of arrays.
    #
    def find_arrays(self, parent):

        try:
            i = parent.MDF.images
        except:
            return []

        arrays = []
        from tables.array import Array
        from tables.group import Group
        for g in i._f_iterNodes():
            if isinstance(g, Group) and 'image' in g:
                a = g.image
                if isinstance(a, Array) and len(a.shape) in (2,3):
                    arrays.append(a)

        return arrays

    # --------------------------------------------------------------------------
    # Reads a submatrix returning 3D NumPy matrix with zyx index order.
    # array_path can be a HDF5 path to a 3d array or a list of paths to
    # a stack of 2d arrays.
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step,
                    array_path, array, progress):

        i0,j0,k0 = ijk_origin
        isz,jsz,ksz = ijk_size
        istep,jstep,kstep = ijk_step
        import tables
        f = tables.openFile(self.path)
        if progress:
            progress.close_on_cancel(f)
        a = f.getNode(array_path)
        for k in range(k0,k0+ksz,kstep):
            if len(a.shape) == 3:
                plane = a[k,j0:j0+jsz:jstep,i0:i0+isz:istep]
            else:
                plane = a[j0:j0+jsz:jstep,i0:i0+isz:istep]  # 2d array
            array[(k-k0)/kstep,:,:] = plane
            if progress:
                progress.plane((k-k0)/kstep)
        f.close()
        return array

# -----------------------------------------------------------------------------
#
class EMAN_HDF_Image:

    def __init__(self, array):

        self.array_path = array._v_pathname
        self.size = tuple(reversed(array.shape))
        if len(self.size) == 2:
            self.size += (1,)   # Add third dimension to 2d plane.

        g = array._v_parent._v_attrs
        self.value_type = array.atom.dtype

        eman_step = ('EMAN.apix_x', 'EMAN.apix_y', 'EMAN.apix_z')
        if eman_step[0] in g and eman_step[1] in g and eman_step[2] in g:
            self.step = tuple([float(getattr(g, n)[0]) for n in eman_step])
        else:
            self.step = (1.0, 1.0, 1.0)

        self.origin = (0,0,0)
        eman_align3d = 'EMAN.xform.align3d'
        if eman_align3d in g:
            a = getattr(g, eman_align3d)
            import numpy
            if isinstance(a, numpy.ndarray) and len(a) == 1:
                t = a[0]
                tf = [[float(t['%d%d' % (r,c)]) for c in (0,1,2,3)]
                      for r in (0,1,2)]
                self.origin = [tf[i][3]*self.step[i] for i in (0,1,2)]
