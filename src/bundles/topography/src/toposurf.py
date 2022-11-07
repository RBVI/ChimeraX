# -----------------------------------------------------------------------------
# Create a surface with z height from a 2d array.
#

# -----------------------------------------------------------------------------
# Interpolation modes are 'isotropic' or 'none'.
# Mesh patterns are 'isotropic', 'slash', or 'backslash'.
# Colormap is 'rainbow' or 'none'.
#
def topography(session, volume, height = None, interpolate = 'none',
               mesh_type = 'isotropic', palette = None, range = None,
               smoothing_factor = 0.3, smoothing_iterations = 0,
               color = (180,180,180,255), replace = True):

    m = volume.matrix()
    axes = [a for a in (0,1,2) if m.shape[2-a] == 1]
    if len(axes) != 1:
        session.logger.warning('Volume %s has more than one plane shown (%d,%d,%d)' %
                               ((volume,) + tuple(reversed(m.shape))))
        return
    axis = axes[0]
    m = m.squeeze()     # Convert 3d array to 2d

    perm = {0: ((0,0,1,0),(1,0,0,0),(0,1,0,0)),     # 2d matrix xyh -> 3d yzx
            1: ((1,0,0,0),(0,0,1,0),(0,1,0,0)),     # 2d matrix xyh -> 3d xzy
            2: ((1,0,0,0),(0,1,0,0),(0,0,1,0))}[axis]
    from chimerax.geometry import Place
    tf = volume.matrix_indices_to_xyz_transform() * Place(matrix = perm)
    
    s = _create_surface(session, m, height, tf, color, interpolate, mesh_type,
                        smoothing_factor, smoothing_iterations)
    s.name = volume.name + ' height'
    s.position = volume.scene_position
    s.topography_volume = volume

    if palette is not None:
        normal = volume.data.ijk_to_xyz_transform.matrix[:,axis].copy()
        invert = not height is None and height < 0
        if invert:
            normal *= -1
        center = tf.origin()
        from chimerax.surface import color_height
        from chimerax.core.commands import Axis, Center
        color_height(session, [s],
                     axis = Axis(coords=normal),
                     center = Center(coords=center),
                     coordinate_system = volume.scene_position,
                     palette = palette, range = range)

    if replace:
        session.models.close([m for m in session.models
                              if getattr(m, 'topography_volume', None) is volume])
    session.models.add([s])

    return s
    
# -----------------------------------------------------------------------------
#
def register_topography_command(logger):
    from chimerax.core.commands import register, CmdDesc, IntArg, FloatArg, EnumOf, BoolArg
    from chimerax.core.commands import ColormapArg, ColormapRangeArg, Color8Arg
    from chimerax.map import MapArg
    desc = CmdDesc(required=[('volume', MapArg)],
                   keyword=[('height', FloatArg),
                            ('interpolate', EnumOf(['cubic', 'none'])),
                            ('mesh_type', EnumOf(['isotropic','slash','backslash'])),
                            ('color', Color8Arg),
                            ('palette', ColormapArg),
                            ('range', ColormapRangeArg),
                            ('smoothing_iterations', IntArg),
                            ('smoothing_factor', FloatArg),
                            ('replace', BoolArg),
                            ],
                   synopsis='Calculate topographic surface for 2D image')
    register('topography', desc, topography, logger=logger)
    
# -----------------------------------------------------------------------------
#
def _create_surface(session, matrix, height, transform, color, interpolate, mesh,
                    smoothing_factor, smoothing_iterations):

    if interpolate == 'cubic':
        matrix = _cubic_interpolated_2d_array(matrix)
        from chimerax.geometry import scale
        transform = transform * scale(.5)

    if mesh == 'isotropic':
        vertices, triangles = _isotropic_surface_geometry(matrix)
    else:
        cell_diagonal_direction = mesh
        vertices, triangles = _surface_geometry(matrix, cell_diagonal_direction)

    # Adjust vertex z range.
    x, z = vertices[:,0], vertices[:,2]
    zmin = z.min()
    xsize, zextent = x.max() - x.min(), z.max() - zmin
#    z += -zmin
    if zextent > 0:
        if height is None:
            # Use 1/10 of grid voxels in x dimension.
            zscale = 0.1 * xsize / zextent
        else:
            # Convert zscale physical units to volume index z size.
            from chimerax.geometry import length
            zstep = length(transform.transform_vector((0,0,1)))
            zscale = (height/zstep) / zextent
        z *= zscale

    # Transform vertices from index units to physical units.
    transform.transform_points(vertices, in_place = True)

    if smoothing_factor != 0 and smoothing_iterations > 0:
        from chimerax.surface import smooth_vertex_positions
        smooth_vertex_positions(vertices, triangles,
                                smoothing_factor, smoothing_iterations)

    from chimerax.core.models import Surface
    sm = Surface('topography', session)
    from chimerax.surface import calculate_vertex_normals
    normals = calculate_vertex_normals(vertices, triangles)
    sm.set_geometry(vertices, normals, triangles)
    sm.color = color
        
    return sm

# -----------------------------------------------------------------------------
#
def _surface_geometry(matrix, cell_diagonal_direction):

    vertices = _surface_vertices(matrix)
    grid_size = tuple(reversed(matrix.shape))
    triangles = _surface_triangles(grid_size, cell_diagonal_direction)
    return vertices, triangles
    
# -----------------------------------------------------------------------------
#
def _surface_vertices(data):

    ysize, xsize = data.shape
    from numpy import zeros, reshape, float32
    vgrid = zeros((ysize, xsize, 3), float32)
    vgrid[:,:,2] = data
    for j in range(ysize):
        vgrid[j,:,1] = j
    for i in range(xsize):
        vgrid[:,i,0] = i
    vertices = reshape(vgrid, (xsize*ysize, 3))
    return vertices
    
# -----------------------------------------------------------------------------
#
def _surface_triangles(grid_size, cell_diagonal_direction = 'slash'):

    xsize, ysize = grid_size
    from numpy import zeros, intc, arange, add, reshape, array
    tgrid = zeros((xsize*ysize, 6), intc)
    i = arange(0, xsize*ysize)
    for k in range(6):
        tgrid[:,k] = i
    if cell_diagonal_direction == 'slash':
        add(tgrid[:,1], 1, tgrid[:,1])
        add(tgrid[:,2], xsize+1, tgrid[:,2])
        add(tgrid[:,4], xsize+1, tgrid[:,4])
        add(tgrid[:,5], xsize, tgrid[:,5])
    else:
        add(tgrid[:,1], 1, tgrid[:,1])
        add(tgrid[:,2], xsize, tgrid[:,2])
        add(tgrid[:,3], xsize, tgrid[:,3])
        add(tgrid[:,4], 1, tgrid[:,4])
        add(tgrid[:,5], xsize+1, tgrid[:,5])
    tgrid = reshape(tgrid, (ysize, xsize, 6))
    tgrid = array(tgrid[:ysize-1, :xsize-1, :])
    triangles = reshape(tgrid, (2*(ysize-1)*(xsize-1),3))
    return triangles

# -----------------------------------------------------------------------------
#
def _isotropic_surface_geometry(matrix):

    vertices = _isotropic_surface_vertices(matrix)
    grid_size = tuple(reversed(matrix.shape))
    triangles = _isotropic_surface_triangles(grid_size)
    return vertices, triangles

# -----------------------------------------------------------------------------
# Include midpoint of every cell.
#
def _isotropic_surface_vertices(data):

    ysize, xsize = data.shape
    gsize = ysize*xsize
    msize = (ysize-1)*(xsize-1)
    from numpy import zeros, reshape, add, multiply, float32
    v = zeros((gsize+msize,3), float32)

    # Create grid point vertices
    vg = reshape(v[:gsize,:], (ysize, xsize, 3))
    vg[:,:,2] = data
    for j in range(ysize):
        vg[j,:,1] = j
    for i in range(xsize):
        vg[:,i,0] = i

    # Create cell midpoint vertices
    mg = reshape(v[gsize:,:], (ysize-1, xsize-1, 3))
    mg[:,:,:2] = vg[:ysize-1,:xsize-1,:2]
    add(mg[:,:,:2], .5, mg[:,:,:2])
    mz = mg[:,:,2]
    add(mz, data[:ysize-1,:xsize-1], mz)
    add(mz, data[1:ysize,:xsize-1], mz)
    add(mz, data[:ysize-1,1:xsize], mz)
    add(mz, data[1:ysize,1:xsize], mz)
    multiply(mz, .25, mz)
    
    return v
    
# -----------------------------------------------------------------------------
# Use midpoint of every cell.
#
def _isotropic_surface_triangles(grid_size):

    xsize, ysize = grid_size
    gsize = ysize*xsize
    from numpy import zeros, intc, reshape
    t = zeros((4*(ysize-1)*(xsize-1),3), intc)

    # Each cell is divided into 4 triangles using cell midpoint
    tg = reshape(t, (ysize-1, xsize-1, 12))
    tg[:,:,::3] += gsize
    for i in range(xsize-1):
        tg[:,i,0] += i          # Bottom triangle
        tg[:,i,1] += i
        tg[:,i,2] += i+1
        tg[:,i,3] += i          # Right triangle
        tg[:,i,4] += i+1
        tg[:,i,5] += i+1
        tg[:,i,6] += i          # Top triangle
        tg[:,i,7] += i+1
        tg[:,i,8] += i
        tg[:,i,9] += i          # Left triangle
        tg[:,i,10] += i
        tg[:,i,11] += i
    for j in range(ysize-1):
        tg[j,:,0] += j*(xsize-1)
        tg[j,:,1] += j*xsize
        tg[j,:,2] += j*xsize
        tg[j,:,3] += j*(xsize-1)
        tg[j,:,4] += j*xsize
        tg[j,:,5] += (j+1)*xsize
        tg[j,:,6] += j*(xsize-1)
        tg[j,:,7] += (j+1)*xsize
        tg[j,:,8] += (j+1)*xsize
        tg[j,:,9] += j*(xsize-1)
        tg[j,:,10] += (j+1)*xsize
        tg[j,:,11] += j*xsize

    return t

# -----------------------------------------------------------------------------
#
def _cubic_interpolated_2d_array(matrix):

    new_size = [2*n-1 for n in matrix.shape]
    from numpy import zeros, float32
    new_matrix = zeros(new_size, float32)
    new_matrix[::2,::2] = matrix
    temp = zeros(max(matrix.shape), float32)
    for r in range(matrix.shape[0]):
        _cubic_interpolate_1d_array(new_matrix[2*r, :], temp)
    for c in range(new_matrix.shape[1]):
        _cubic_interpolate_1d_array(new_matrix[:,c], temp)

    return new_matrix

# -----------------------------------------------------------------------------
# Even elements of array are cubic interpolated to set odd elements.
#
def _cubic_interpolate_1d_array(array, temp):

    from numpy import multiply, add
    a = (-1.0/16, 9.0/16, 9.0/16, -1.0/16)
    ae = array[::2]
    n = ae.shape[0]
    ao = array[3:-3:2]
    t = temp[:ao.shape[0]]
    for k in range(4):
        multiply(ae[k:n-3+k], a[k], t)
        add(ao, t, ao)

    # Quadratic interpolation for end positions
    c = (3.0/8, 6.0/8, -1.0/8)
    array[1] = c[0]*array[0] + c[1]*array[2] + c[2]*array[4]
    array[-2] = c[0]*array[-1] + c[1]*array[-3] + c[2]*array[-5]
