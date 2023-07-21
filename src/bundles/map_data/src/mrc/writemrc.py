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
# Write an MRC 2000 format file.
#
# Header contains four byte integer or float values:
#
# 1	NX	number of columns (fastest changing in map)	
# 2	NY	number of rows					
# 3	NZ	number of sections (slowest changing in map)	
# 4	MODE	data type :					
# 		0	image : signed 8-bit bytes range -128 to 127
# 		1	image : 16-bit halfwords		
# 		2	image : 32-bit reals			
# 		3	transform : complex 16-bit integers	
# 		4	transform : complex 32-bit reals	
# 5	NXSTART	number of first column in map			
# 6	NYSTART	number of first row in map			
# 7	NZSTART	number of first section in map			
# 8	MX	number of intervals along X			
# 9	MY	number of intervals along Y			
# 10	MZ	number of intervals along Z			
# 11-13	CELLA	cell dimensions in angstroms			
# 14-16	CELLB	cell angles in degrees				
# 17	MAP# axis corresp to cols (1,2,3 for X,Y,Z)		
# 18	MAPR	axis corresp to rows (1,2,3 for X,Y,Z)		
# 19	MAPS	axis corresp to sections (1,2,3 for X,Y,Z)	
# 20	DMIN	minimum density value				
# 21	DMAX	maximum density value				
# 22	DMEAN	mean density value				
# 23	ISPG	space group number 0 or 1 (default=0)		
# 24	NSYMBT	number of bytes used for symmetry data (0 or 80)
# 25-49   EXTRA	extra space used for anything			
# 50-52	ORIGIN  origin in X,Y,Z used for transforms		
# 53	MAP	character string 'MAP ' to identify file type	
# 54	MACHST	machine stamp					
# 55	RMS	rms deviation of map from mean density		
# 56	NLABL	number of labels being used			
# 57-256 LABEL(20,10) 10 80-character text labels		
#

# -----------------------------------------------------------------------------
#
def write_mrc2000_grid_data(grid_data, path, options = {}, progress = None):

    mtype = grid_data.value_type.type
    if options.get('value_type'):
        type = mrc_value_type_from_name(options.get('value_type'))
    else:
        type = closest_mrc2000_type(mtype)

    ccp4_format = options.get('ccp4_format')
    header = mrc2000_header(grid_data, type, ccp4_format = ccp4_format)

    f = open(path, 'wb')
    if progress:
        progress.close_on_cancel(f)

    f.write(header)

    stats = Matrix_Statistics()
    isz, jsz, ksz = grid_data.size
    for k in range(ksz):
        matrix = grid_data.matrix((0,0,k), (isz,jsz,1))
        if type != mtype:
            matrix = matrix.astype(type)
        f.write(matrix.tostring())
        stats.plane(matrix)
        if progress:
            progress.plane(k)

    # Put matrix statistics in header
    header = mrc2000_header(grid_data, type, stats, ccp4_format = ccp4_format)
    f.seek(0)
    f.write(header)

    f.close()

# -----------------------------------------------------------------------------
#
def mrc2000_header(grid_data, value_type, stats = None, ccp4_format = False):

    size = grid_data.size

    from numpy import float16, float32, int16, int8, int32, uint16
    if value_type == float32:         mode = 2
    elif value_type == int16:         mode = 1
    elif value_type == int8:          mode = 0
    elif value_type == uint16:        mode = 6
    elif value_type == float16:       mode = 12

    if ccp4_format:
        origin = (0,0,0)
        fstart = -grid_data.xyz_to_ijk((0,0,0))
        start = tuple(int(round(x)) for x in fstart)
        if max([abs(s-fs) for s,fs in zip(start, fstart)]) > 0.01:
            msg = ('Did not save CCP4 file, format requires an integer grid origin, ' +
                   'got non-integer %.5g,%.5g,%.5g' % tuple(fstart))
            from chimerax.core.errors import UserError
            raise UserError(msg)
    else:
        origin = grid_data.origin
        start = (0,0,0)

    cell_size = tuple(a*b for a,b in zip(grid_data.step, size))

    if stats:
        dmin, dmax = stats.min, stats.max
        dmean, rms = stats.mean_and_rms(size)
    else:
        dmin = dmax = dmean = rms = 0

    from numpy import little_endian
    if little_endian:
        machst = 0x00004144
    else:
        machst = 0x11110000

#    from chimerax.core import version
# TODO: Get ChimeraX version, currently not available.
    version = '0.1'
    from time import asctime
    ver_stamp = 'ChimeraX %s %s' % (version, asctime())
    labels = [ver_stamp[:80]]

    if grid_data.rotation != ((1,0,0),(0,1,0),(0,0,1)):
        from chimerax.geometry import matrix
        axis, angle = matrix.rotation_axis_angle(grid_data.rotation)
        r = 'Chimera rotation: %12.8f %12.8f %12.8f %12.8f' % (tuple(axis) + (angle,))
        labels.append(r)

    nlabl = len(labels)
    # Make ten 80 character labels.
    labels.extend(['']*(10-len(labels)))
    labels = [l + (80-len(l))*'\0' for l in labels]
    labelstr = ''.join(labels)

    strings = [
        binary_string(size, int32),  # nx, ny, nz
        binary_string(mode, int32),  # mode
        binary_string(start, int32), # nxstart, nystart, nzstart
        binary_string(size, int32),  # mx, my, mz
        binary_string(cell_size, float32), # cella
        binary_string(grid_data.cell_angles, float32), # cellb
        binary_string((1,2,3), int32), # mapc, mapr, maps
        binary_string((dmin, dmax, dmean), float32), # dmin, dmax, dmean
        binary_string(0, int32), # ispg
        binary_string(0, int32), # nsymbt
        binary_string([0]*25, int32), # extra
        binary_string(origin, float32), # origin
        'MAP '.encode('utf-8'), # map
        binary_string(machst, int32), # machst
        binary_string(rms, float32), # rms
        binary_string(nlabl, int32), # nlabl
        labelstr.encode('utf-8'),
        ]

    header = b''.join(strings)
    return header

# -----------------------------------------------------------------------------
#
def write_ccp4_grid_data(grid_data, path, options = {}, progress = None):
    ccp4_opts = options.copy()
    ccp4_opts['ccp4_format'] = True
    write_mrc2000_grid_data(grid_data, path, options = ccp4_opts, progress = progress)
    
# -----------------------------------------------------------------------------
#
class Matrix_Statistics:

    def __init__(self):

        self.min = None
        self.max = None
        self.sum = 0.0
        self.sum2 = 0.0

    def plane(self, matrix):

        from numpy import ravel, minimum, maximum, add, multiply, array, float32
        matrix_1d = matrix.ravel()
        dmin = minimum.reduce(matrix_1d)
        if self.min is None or dmin < self.min:
            self.min = dmin
        dmax = maximum.reduce(matrix_1d)
        if self.max is None or dmax > self.max:
            self.max = dmax
        self.sum += add.reduce(matrix_1d)
        # TODO: Don't copy array to get standard deviation.
        # Avoid overflow when squaring integral types
        m2 = array(matrix_1d, float32)
        multiply(m2, m2, m2)
        self.sum2 += add.reduce(m2)

    def mean_and_rms(self, size):

        vol = float(size[0])*float(size[1])*float(size[2])
        mean = self.sum / vol
        sdiff = self.sum2 - self.sum*self.sum
        if sdiff > 0:
            from math import sqrt
            rms = sqrt(sdiff) / vol
        else:
            rms = 0
        return mean, rms

# -----------------------------------------------------------------------------
#
def binary_string(values, type):

    from numpy import array
    return array(values, type).tostring()

# -----------------------------------------------------------------------------
#
def closest_mrc2000_type(type):

    from numpy import float16, float32, float64
    from numpy import int8, int16, int32, int64, character
    from numpy import uint, uint8, uint16, uint32, uint64
    if type in (float32, float64, float, int32, int64, int, uint, uint32, uint64):
        ctype = float32
    elif type in (int16, uint8):
        ctype = int16
    elif type in (int8, character):
        ctype = int8
    elif type in (uint16,):
        ctype = uint16
    elif type in (float16,):
        ctype = float16
    else:
        raise TypeError('Volume data has unrecognized type %s' % type)

    return ctype

# -----------------------------------------------------------------------------
#
def mrc_value_type_from_name(type_name):
    from numpy import float16, float32, int8,int16, uint16
    t = {'float16':float16, 'float32':float32, 'int8': int8, 'int16':int16, 'uint16':uint16}
    if type_name not in t:
        raise TypeError(f'Cannot write MRC type "{type_name}", must be int8, int16, uint16, float16 or float32')
    return t[type_name]
