#
# ******************************************************************************
# *                                                                            *
# *     DUMP A 3D VOLUME TO AN IMAGIC FILE                                     *
# *                                                                            *
# ******************************************************************************
# *                                                                            *
# *     An IMAGIC file actually consists of TWO files:                         *
# *                                                                            *
# *        1> <file name>.hed  -  file with header information for             *
# *                               each image/each volume section               *
# *        2> <file name>.img  -  file with all image/volume densities         *
# *                                                                            *
# ******************************************************************************
# *                                                                            *
# *     Image Science Software GmbH                                            *
# *     Lutterbacher Str. 22                                                   *
# *     14167 Berlin                                                           *
# *     Germany                                                                *
# *     imagic@ImageScience.de                                                 *
# *                                                                            *
# ******************************************************************************
#
#
# ******************************************************************************
# *                                                                            *
# *     IMAGIC HEADER                                                          *
# *     (see: imagic_format.html or www.ImageScience.de/formats)               *
# *                                                                            *
# *     Header values needed (description see below):                          *
# *                                                                            *
# *     IDAT1 (1)               - IMN                                          *
# *     IDAT1 (2)               - IFOL                                         *
# *     IDAT1 (4)  - IDAT1 (7)  - Creation date                                *
# *     IDAT1 (8)  - IDAT1(10)  - Creation time                                *
# *     IDAT1(13)               - IXLP                                         *
# *     IDAT1(14)               - IYLP                                         *
# *     IDAT1(15)               - TYPE                                         *
# *     IDAT1(18)               - AVDENS                                       *
# *     IDAT1(19)               - SIGMA                                        *
# *     IDAT1(22)               - DENSMAX                                      *
# *     IDAT1(23)               - DENSMIN                                      *
# *     IDAT1(61)               - IZLP                                         *
# *     IDAT1(62)               - I4LP                                         *
# *     IDAT1(68)               - IMAVERS                                      *
# *     IDAT1(69)               - REALTYPE                                     *
# *     DAT1(123)               - PIXSIZE                                      *
# *     IDAT1(200) - IDAT1(256) - HISTORY                                      *
# *                                                                            *
# ******************************************************************************
#

# -----------------------------------------------------------------------------
#
# Write value(s) to file
# (copied and modified from writemrc.py)
#

def write_imagic_grid_data(grid_data, path, options = {}, progress = None):

    import os
    import numpy as np

    #
    # IMAGIC header and data files
    #
    split = os.path.splitext(path)
    hed_path = split[0] + '.hed'
    img_path = split[0] + '.img'

    #
    # Open HEADER file
    #
    hed = open(hed_path, 'wb')
    if progress:
        progress.close_on_cancel(hed)

    #
    # Volume dimensions
    #
    ix, iy, iz = grid_data.size

    #
    # Data type
    #
    mtype = grid_data.value_type.type
    type  = volume_data_type(mtype)

    #
    # Global matrix statistics
    #
    matrix  = grid_data.matrix((0,0,0), (ix,iy,iz))
    densmin = matrix.min()
    densmax = matrix.max()
    avdens  = np.mean(matrix, dtype=np.float64)
    sigma   = np.sqrt(np.var(matrix, dtype=np.float64))

    #
    # Put matrix statistics into header
    #
    for k in range(iz):
        loc = k + 1
        header = imagic_header(grid_data, type, loc, densmin, densmax, avdens, sigma)
        hed.write(header)
        if progress:
            progress.plane(k)

    #
    # Close HEADER file
    #
    hed.close()

    #
    # Open DENSITIES file
    #
    img = open(img_path, 'wb')
    if progress:
        progress.close_on_cancel(img)

    matrix = grid_data.matrix((0,0,0), (ix,iy,iz))
    if type != mtype:
        matrix = matrix.astype(type)

    #
    # From IMAGIC to CHIMERA coordinate system
    #
    matrix = matrix.copy()[:,::-1,:]

    #
    # Dump matrix
    #
    img.write(matrix.tostring())

    #
    # Close DENSITIES file
    #
    img.close()

#
# -----------------------------------------------------------------------------
#
# Write header file
#
def imagic_header(grid_data, value_type, location, densmin, densmax, avdens, sigma):

    import datetime
    d = datetime.datetime.now()

    #
    # Volume dimensions
    #
    size       = grid_data.size
    ix, iy, iz = grid_data.size

    #
    # Pixel/Voxel size
    #
    step                = grid_data.step
    stepx, stepy, stepz = grid_data.step

    if stepx != stepy:
        if location == 1:
                print('**WARNING: Step x = ', stepx, ' and step y = ', stepy, 'are different. Stepx is used.')
    if stepx != stepz:
        if location == 1:
                print('**WARNING: Step x = ', stepx, ' and step z = ', stepz, 'are different. Stepx is used.')
    
    #
    # From ChimeraX to IMAGIC coordinate system
    #
    ixlp       = iy
    iylp       = ix
    izlp       = iz

    imn        = location

    ifol       = 0
    if location == 1:
        ifol   = izlp - 1

    from numpy import float32, int16, int8, int32
    if value_type == float32:         type = b'REAL'
    elif value_type == int32:         type = b'LONG'
    elif value_type == int16:         type = b'INTG'
    elif value_type == int8:          type = b'PACK'

    pixsize    = stepx

    cell_size  = map(lambda a,b: a*b, grid_data.step, size)

    from numpy import little_endian
    if little_endian:
        endian      = 0x2020202
    else:
        endian      = 0x4040404

    imavers         = 20190319

    from chimerax.core import version
    from time import asctime

    name_string     = ('3D VOLUME EXPORTED BY CHIMERAX %s' % version).encode('UTF-8')
    name_array      = [name_string]
    name_array.extend([b' '*(80-len(name_string))])
    name            = b''.join(name_array)

    hist_string     = ('CHIMERAX %s' % version).encode('UTF-8')
    hist_array      = [hist_string]
    hist_array.extend([b' '*(228-len(hist_string))])
    history         = b''.join(hist_array)

    strings = [

        # Image location number (1,2,3,...) - IMN
        binary_string(imn, int32),

        # Number of all 1D/2D images/sections following (0,1...) - IFOL
        binary_string(ifol, int32),

        # Error code for this image during IMAGIC run
        binary_string(0, int32),

        # Number of header blocks
        # each block containing 256 REAL/INTEGER values - NBLOCKS
        binary_string(1, int32),

        # Creation date
        binary_string(d.month, int32),
        binary_string(d.day, int32),
        binary_string(d.year, int32),
        binary_string(d.hour, int32),
        binary_string(d.minute, int32),
        binary_string(d.second, int32),

        # Ignored * 2
        binary_string([0]*2, int32),

        # Number of lines per image (for 1D data IXLP=1) - IXLP
        binary_string(ixlp, int32),

        # Number of pixels per line - IYLP
        binary_string(iylp, int32),

        # 4 coded characters determining the image type - TYPE
        type,

        # Ignored * 2
        binary_string([0]*2, int32),

        # Average density - AVDENS
        binary_string(avdens, float32),

        # Sigma - SIGMA
        binary_string(sigma, float32),

        # Ignored * 2
        binary_string([0]*2, int32),

        # Maximal and minimal density - DENSMAX, DENSMIN
        binary_string(densmax, float32),
        binary_string(densmin, float32),

        # Ignored * 6
        binary_string([0]*6, int32),

        # Coded NAME/TITLE of the image (80 characters) - NAME
        name,

        # Ignored * 11
        binary_string([0]*11, int32),

        # Number of 2D planes in 3D data - IZLP
        binary_string(izlp, int32),

        # Number of "objects" in file - I4LP
        binary_string(1, int32),

        # Ignored * 5
        binary_string([0]*5, int32),

        # IMAGIC version, which created the file (yyyymmdd) - IMAVERS
        binary_string(imavers, int32),

        # Floating point type / machine stamp - REALTYPE
        binary_string(endian, int32),

        # Ignored * 53
        binary_string([0]*53, int32),

        # Pixel/Voxel size - PIXSIZE
        binary_string(pixsize, float32),

        # Ignored * 76
        binary_string([0]*76, int32),

        # Coded history of image (228 characters) - HISTORY
        history,
        ]

    header = b''.join(strings)
    return header
    
#
# -----------------------------------------------------------------------------
#
# From array to string
#
def binary_string(values, type):

    from numpy import array
    return array(values, type).tostring()

#
# -----------------------------------------------------------------------------
#
# Retrieve densities data type
#
def volume_data_type(type):

    from numpy import float32, float64
    from numpy import int8, int16, int32, int64, character
    from numpy import uint, uint8, uint16, uint32, uint64

    if type in (float32, float64, float, int32, int64, int, uint, uint16, uint32, uint64):
        ctype = float32
    elif type in (int16, uint8):
        ctype = int16
    elif type in (int8, character):
        ctype = int8
    else:
        raise TypeError('Volume data has unrecognized type %s' % type)

    return ctype
