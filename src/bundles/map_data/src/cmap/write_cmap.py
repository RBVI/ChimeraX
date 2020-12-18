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
# Write volume data in hdf5 format.
#
# Example HDF5 format written by Chimera.
#
# /Chimera
#  /image1
#    chimera_map_version 1
#    chimera_version "1.2422"
#    name "centriole"
#    step (1.2, 1.2, 1.2)
#    origin (-123.4, -522, 34.5)
#    cell_angles (90.0, 90.0, 90.0)
#    rotation_axis (0.0, 0.0, 1.0)
#    rotation_angle 45.0
#    symmetries (((0,-1,0,0),(1,0,0,0),(0,0,1,0)),...)
#    color (1.0, 1.0, 0, 1.0) (attribute, rgba 0-1 float)
#    data_zyx (3d array of uint8 (123,542,82))
#    data_yzx (3d array of uint8 (123,542,82), alternate chunk shape)
#    data_zyx_2 (3d array of uint8 (61,271,41))
#        subsample_spacing (2, 2, 2)
#    (more subsampled or alternate chunkshape versions of same data)
#
# Names "Chimera", "chimera_version", "name", "step", "origin", "cell_angles",
# "rotation_axis", "rotation_angle", "symmetries", "subsample_spacing"
# are fixed while "image1", "data_zyx", "data_yzx" and "data_zyx_2" can
# be any name.
#
# In the example "Chimera" and "image1" are HDF groups,
# "chimera_version", "name", "step", "origin", "cell_angles",
# "rotation_axis", "rotation_angle", "symmetries", "color" are group
# attributes, "data_zyx", "data_yzx" and "data_zyx_2" are hdf datasets
# (arrays), and "subsample_step" is a dataset attribute.
#
# All data sets within the group represent the same data, such as optional
# subsampled arrays or alternate chunk shape for efficient disk access.
#
# Cell angles need not be included if they are 90,90,90.  They are
# included for handling crystallographic density maps.  An identity
# rotation need not be included.  The rotation angle is in degrees.
# Symmetries need not be included.
#
# The file is saved with the Python PyTables modules which includes
# additional attributes "VERSION", "CLASS", "TITLE", "PYTABLES_FORMAT_VERSION".
#
def write_grid_as_chimera_map(grid_data, path, options = {}, progress = None):

    settings = {
        'subsamples': [],	# e.g. [(2,2,1),(4,4,1)]
        'min_subsample_elements': 2**17,
        'chunk_shapes': ['zyx'],
        'chunk_size': 2**22,       # 4 Mbytes
        'append': False,
        'compress': False,
        'compress_method': 'blosc:lz4',  # 'zlib', 'lzo', 'bzip2', 'blosc', 'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy', 'blosc:zlib', 'blosc:zstd'
        'compress_level': 5,	# 1-9
        'compress_shuffle': True,
        }
    settings.update(options)

    from .. import GridData
    if isinstance(grid_data, GridData):
        data_sets = [grid_data]
    else:
        data_sets = grid_data

    if settings['append']:      mode = 'a'
    else:                       mode = 'w'
    import tables
    h5file = tables.open_file(path, mode = mode)
    try:
        if progress:
            progress.close_on_cancel(h5file)

        if '/Chimera' in h5file:
            cg = h5file.get_node('/Chimera')
        else:
            cg = h5file.create_group(h5file.root, 'Chimera')

        ioffset = next_suffix_number(cg, 'image')
        for i, d in enumerate(data_sets):
            g = h5file.create_group(cg, 'image%d' % (i+ioffset))
            write_grid_data(h5file, d, g, settings, progress)
    finally:
        h5file.close()

# -----------------------------------------------------------------------------
# Used to produce new group name when adding maps to an existing hdf file.
#
def next_suffix_number(group, prefix):

    imax = 0
    for gname in group._v_groups.keys():
        if gname.startswith(prefix):
            suffix = gname[len(prefix):]
            try:
                imax = max(imax, int(suffix))
            except ValueError:
                pass
    return imax + 1

# -----------------------------------------------------------------------------
# Write volume data in Chimera hdf5 format.
#
def write_grid_data(h5file, grid_data, g, settings, progress):

    if progress:
        from os.path import basename
        wpath = basename(h5file.filename)
        progress.operation = 'Writing %s to %s' % (grid_data.name, wpath)
        progress.array_size(grid_data.size, grid_data.value_type.itemsize)

    g._v_attrs.chimera_map_version = 1

    from chimerax import app_dirs
    g._v_attrs.chimera_version = '%s %s' % (app_dirs.appname, app_dirs.version)        # string

    if grid_data.name:
        g._v_attrs.name = data_name(grid_data)

    # Need to use numpy arrays for attributes otherwise pytables 2.0 uses
    # python pickle strings in the hdf file.  Want hdf file to be readable
    # by other software without Python.
    from numpy import array, float32
    g._v_attrs.origin = array(grid_data.origin, float32)
    g._v_attrs.step = array(grid_data.step, float32)
    if grid_data.cell_angles != (90,90,90):
        g._v_attrs.cell_angles = array(grid_data.cell_angles, float32)
    if grid_data.rotation != ((1,0,0),(0,1,0),(0,0,1)):
        from chimerax.geometry import matrix
        axis, angle = matrix.rotation_axis_angle(grid_data.rotation)
        g._v_attrs.rotation_axis = array(axis, float32)
        g._v_attrs.rotation_angle = array(angle, float32)
    if grid_data.symmetries:
        g._v_attrs.symmetries = array(grid_data.symmetries.array(), float32)
    if grid_data.rgba is not None:
        g._v_attrs.color = array(grid_data.rgba, float32)
    if hasattr(grid_data, 'time') and grid_data.time is not None:
        g._v_attrs.time = grid_data.time
    if hasattr(grid_data, 'channel') and grid_data.channel is not None:
        g._v_attrs.channel = grid_data.channel
        
    # Determine data type.
    import tables
    atom = tables.Atom.from_dtype(grid_data.value_type)

    # Create data array, subsample arrays, and alternate chunk shape arrays.
    arrays = make_arrays(h5file, g, grid_data.size, atom, settings)

    # Write values to primary and subsample arrays.
    isz,jsz,ksz = grid_data.size
    # Read multiple planes so reading HDF5 with multiplane chunks is more efficient
    # Read a slab of planes up to 1 Gbytes.  Take power of 2 number of planes (common file chunk size).
    max_slab_bytes = 2**30
    plane_bytes = isz * jsz * grid_data.value_type.itemsize
    kchunk = floor_power_of_2(max(1, max_slab_bytes // plane_bytes))
    for k in range(ksz):
        if progress:
            progress.plane(k)
        # Read a slab at a time to handle data sets that do not fit in memory.
        if k % kchunk == 0:
            kc = kchunk if k + kchunk <= ksz else ksz-k
            mc = grid_data.matrix((0,0,k), (isz,jsz,kc))
        mk = mc[k%kchunk,:,:]
        for (istep,jstep,kstep), a in arrays:
            if kstep == 1 or k % kstep == 0:
                a[k//kstep,:,:] = mk[::jstep,::istep]

    # TODO: Use subsample arrays if available.
    # TODO: Optimize read write depending on chunk shapes.

# -----------------------------------------------------------------------------
# Largest power of 2 less than or equal to n.
#
def floor_power_of_2(n):
  p = 1
  while 2*p <= n:
    p *= 2
  return p
    
# -----------------------------------------------------------------------------
# Determine name used to distinguish multiple maps in file.
#
def data_name(grid_data):

    return grid_data.name                       # often is file name

# -----------------------------------------------------------------------------
#
def make_arrays(h5file, g, size, atom, settings):

    chunk_elements = settings['chunk_size'] // atom.itemsize
    chunk_shapes = settings['chunk_shapes']
    subsamples = settings['subsamples']
    min_subsample_elements = settings['min_subsample_elements']

    if 'compress' in settings and settings['compress']:
        method = settings.get('compress_method', 'zlib')
        level = settings.get('compress_level', 5)
        shuffle = settings.get('compress_shuffle', True)
        from tables import Filters
        filters = Filters(complevel = level, complib = method, shuffle = shuffle)
    else:
        filters = None
    
    arrays = []
    isize, jsize, ksize = size
    shape = (ksize,jsize,isize)
    cshapes = {}    # Avoid duplicating chunk shapes
    for csname in chunk_shapes:
        cshape = chunk_shape(shape, csname, chunk_elements)
        if not cshape in cshapes:
            a = h5file.create_carray(g, 'data_' + csname, atom, shape,
                                     chunkshape = cshape, filters = filters)
            arrays.append(((1,1,1),a))
            cshapes[cshape] = True

    # Compute step sizes to use.
    steps = list(subsamples)
    istep,jstep,kstep = tuple(2*s for s in subsamples[-1]) if subsamples else (2,2,2)
    from numpy import array, int32
    while (isize >= istep and jsize >= jstep and ksize >= kstep and
           (isize//istep)*(jsize//jstep)*(ksize//kstep) >= min_subsample_elements):
        steps.append((istep,jstep,kstep))
        istep *= 2
        jstep *= 2
        kstep *= 2

    # Make subsample arrays.
    for step in steps:
        istep,jstep,kstep = step
        shape = (1+(ksize-1)//kstep, 1+(jsize-1)//jstep, 1+(isize-1)//istep)
        cshapes = {}    # Avoid duplicating chunk shapes
        for csname in chunk_shapes:
            cshape = chunk_shape(shape, csname, chunk_elements)
            if not cshape in cshapes:
                sstep = '%d_%d_%d' % tuple(step)
                a = h5file.create_carray(g, 'data_%s_%s' % (csname,sstep), atom,
                                         shape, chunkshape = cshape,
                                         filters = filters)
                a._v_attrs.subsample_spacing = array(step, int32)
                arrays.append((step, a))
                cshapes[cshape] = True

    return arrays

# -----------------------------------------------------------------------------
# Name corresponds to slow-medium-fast axes.
#
def chunk_shape(shape, name, chunk_elements):

    if name in ('zyx', 'zxy', 'yxz', 'yzx', 'xzy', 'xyz'):
        axes = {'x':2, 'y':1, 'z':0}
        smf_axes = [axes[a] for a in name]      # slow, medium, fast axes
        fms_axes = reversed(smf_axes)
        cshape = [1,1,1]
        csize = 1
        for a in fms_axes:
            # Avoid chunk shapes like (1,n-1,n) for n^3 data set that make
            # the file size twice as large as needed.
            s = min(shape[a], max(1, int(chunk_elements//csize)))
            c = (shape[a] + s - 1) // s  # chunks needed to cover
            ms = (shape[a] + c - 1) // c # min size to cover with c chunks.
            cshape[a] = ms
            csize *= cshape[a]
        cshape = tuple(cshape)
        return cshape

    return None                 # Use hdf default chunk shape.
