# -----------------------------------------------------------------------------
# Read MRC or CCP4 map file format electron microscope data.
# Byte swapping will be done if needed.
#

# -----------------------------------------------------------------------------
# file_type can be 'mrc' or 'ccp4' or 'imod'.
#
class MRC_Data:

  def __init__(self, path, file_type):

    self.path = path

    import os.path
    self.name = os.path.basename(path)
    
    file = open(path, 'rb')

    file.seek(0,2)                              # go to end of file
    file_size = file.tell()
    file.seek(0,0)                              # go to beginning of file

    # Infer file byte order from column axis size nc.  Requires nc < 2**16
    # Was using mode value but 0 is allowed and does not determine byte order.
    self.swap_bytes = 0
    from numpy import int32
    nc = self.read_values(file, int32, 1)
    self.swap_bytes = not (nc > 0 and nc < 65536)
    file.seek(0,0)

    v = self.read_header_values(file, file_size, file_type)

    if v.get('imodStamp') == 1146047817:
      unsigned_8_bit = (v['imodFlags'] & 0x1 == 0)
    else:
      unsigned_8_bit = (file_type == 'imod' or v['type'] == 'mrc')
    self.element_type = self.value_type(v['mode'], unsigned_8_bit)

    self.check_header_values(v, file_size, file)
    self.header = v             # For dumpmrc.py standalone program.
    
    self.data_offset = file.tell()
    file.close()
    
    # Axes permutation.
    # Names c,r,s refer to fast, medium, slow file matrix axes.
    # Names i,j,k refer to x,y,z spatial axes.
    mapc, mapr, maps = v['mapc'], v['mapr'], v['maps']
    if (1 in (mapc, mapr, maps) and
        2 in (mapc, mapr, maps) and
        3 in (mapc, mapr, maps)):
      crs_to_ijk = (mapc-1,mapr-1,maps-1)
      ijk_to_crs = [None,None,None]
      for a in range(3):
        ijk_to_crs[crs_to_ijk[a]] = a
    else:
      crs_to_ijk = ijk_to_crs = (0, 1, 2)
    self.crs_to_ijk = crs_to_ijk
    self.ijk_to_crs = ijk_to_crs

    crs_size = v['nc'], v['nr'], v['ns']
    self.matrix_size = [int(s) for s in crs_size]
    self.data_size = [int(crs_size[a]) for a in ijk_to_crs]

    self.unit_cell_size = mx, my, mz = v['mx'], v['my'], v['mz']
    xlen, ylen, zlen = v['xlen'], v['ylen'], v['zlen']
    if mx > 0 and my > 0 and mz > 0 and xlen > 0 and ylen > 0 and zlen > 0:
      self.data_step = (xlen/mx, ylen/my, zlen/mz)
    else:
      self.data_step = (1.0, 1.0, 1.0)

    alpha, beta, gamma = (v['alpha'], v['beta'], v['gamma'])
    if not valid_cell_angles(alpha, beta, gamma, path):
      alpha = beta = gamma = 90
    self.cell_angles = (alpha, beta, gamma)

    from math import isnan
    if (v['type'] == 'mrc2000' and
        (v['zorigin'] != 0 or v['xorigin'] != 0 or v['yorigin'] != 0) and
        not (isnan(v['xorigin']) or isnan(v['yorigin']) or isnan(v['zorigin']))):
      #
      # This is a new MRC 2000 format file.  The xyz origin header parameters
      # are used instead of using ncstart, nrstart nsstart for new style files,
      # provided the xyz origin specified is not zero.  It turns out the
      # xorigin, yorigin, zorigin values are zero in alot of new files while
      # the ncstart, nrstart, nsstart give the correct (non-zero) origin. So in
      # cases where the xyz origin parameters and older nrstart, ncstart,
      # nsstart parameters specify different origins the one that is non-zero
      # is preferred.  And if both are non-zero, the newer xorigin, yorigin,
      # zorigin are used.
      #
      self.data_origin = (v['xorigin'], v['yorigin'], v['zorigin'])
    else:
      crs_start = v['ncstart'], v['nrstart'], v['nsstart']
      ijk_start = [crs_start[a] for a in ijk_to_crs]
      # Check if ijk_start values appear to be uninitialized.
      limit = 10*max(max(mx,my,mz), max(self.data_size))
      if [s for s in ijk_start if abs(s) > limit]:
        self.data_origin = (0., 0., 0.)
      else:
        from ..griddata import scale_and_skew
        self.data_origin = scale_and_skew(ijk_start, self.data_step,
                                          self.cell_angles)

    r = ((1,0,0),(0,1,0),(0,0,1))
    for lbl in v['labels']:
      if lbl.startswith(b'Chimera rotation: '):
        ax,ay,az,angle = [float(x) for x in lbl.rstrip('\0').split()[2:]]
        import matrix
        r = matrix.rotation_from_axis_angle((ax,ay,az), angle)
    self.rotation = r
    
    self.min_intensity = v['amin']
    self.max_intensity = v['amax']

  # ---------------------------------------------------------------------------
  # Format derived from C header file mrc.h.
  #
  def read_header_values(self, file, file_size, file_type):

    MRC_USER = 29
    CCP4_USER = 15
    MRC_NUM_LABELS = 10
    MRC_LABEL_SIZE = 80
    MRC_HEADER_LENGTH = 1024

    from numpy import int32, float32
    i32 = int32
    f32 = float32
    
    v = {}
    v['nc'], v['nr'], v['ns'] = self.read_values(file, i32, 3)
    v['mode'] = self.read_values(file, i32, 1)
    v['ncstart'], v['nrstart'], v['nsstart'] = self.read_values(file, i32, 3)
    v['mx'], v['my'], v['mz'] = self.read_values(file, i32, 3)
    v['xlen'], v['ylen'], v['zlen'] = self.read_values(file, f32, 3)
    v['alpha'], v['beta'], v['gamma'] = self.read_values(file, f32, 3)
    v['mapc'], v['mapr'], v['maps'] = self.read_values(file, i32, 3)
    v['amin'], v['amax'], v['amean'] = self.read_values(file, f32, 3)
    v['ispg'], v['nsymbt'] = self.read_values(file, i32, 2)
    if file_type == 'ccp4':
      v['lskflg'] = self.read_values(file, i32, 1)
      v['skwmat'] = self.read_values(file, f32, 9)
      v['skwtrn'] = self.read_values(file, f32, 3)
      v['user'] = self.read_values(file, i32, CCP4_USER)
      v['map'] = file.read(4)   # Should be 'MAP '.
      v['machst'] = self.read_values(file, i32, 1)
      v['rms'] = self.read_values(file, f32, 1)
      v['type'] = 'ccp4'
    else:
      # MRC file
      user = file.read(4*MRC_USER)
      if user[-4:] == 'MAP ':
        # New style MRC 2000 format file with xyz origin
        v['user'] = self.read_values_from_string(user, i32, MRC_USER)[:-4]
        xyz_origin = self.read_values_from_string(user[-16:-4], f32, 3)
        v['xorigin'], v['yorigin'], v['zorigin'] = xyz_origin
        v['imodStamp'] = self.read_values_from_string(user[56:60], i32, 1)
        v['imodFlags'] = self.read_values_from_string(user[60:64], i32, 1)
        v['machst'] = self.read_values(file, i32, 1)
        v['rms'] = self.read_values(file, f32, 1)
        v['type'] = 'mrc2000'
      else:
        # Old style MRC has xy origin instead of machst and rms.
        v['user'] = self.read_values_from_string(user, i32, MRC_USER)
        v['xorigin'], v['yorigin'] = self.read_values(file, f32, 2)
        v['type'] = 'mrc'

    v['nlabl'] = self.read_values(file, i32, 1)
    labels = []
    for i in range(MRC_NUM_LABELS):
      labels.append(file.read(MRC_LABEL_SIZE))
    v['labels'] = labels

    # Catch incorrect nsymbt value.
    if v['nsymbt'] < 0 or v['nsymbt'] + MRC_HEADER_LENGTH > file_size:
      raise SyntaxError(('MRC header value nsymbt (%d) is invalid'
                         % v['nsymbt']))
    v['symop'] = file.read(v['nsymbt'])

    return v

  # ---------------------------------------------------------------------------
  #
  def value_type(self, mode, unsigned_8_bit):

    MODE_char   = 0
    MODE_short  = 1
    MODE_float  = 2
    MODE_ushort  = 6            # Non-standard
    
    from numpy import uint8, int8, int16, uint16, float32, dtype
    if mode == MODE_char:
      if unsigned_8_bit:
        t = dtype(uint8)
      else:
        t = dtype(int8)        # CCP4 or MRC2000
    elif mode == MODE_short:
      t = dtype(int16)
    elif mode == MODE_ushort:
      t = dtype(uint16)
    elif mode == MODE_float:
      t = dtype(float32)
    else:
      raise SyntaxError(('MRC data value type (%d) ' % mode +
                         'is not 8 or 16 bit integers or 32 bit floats'))

    return t

  # ---------------------------------------------------------------------------
  #
  def check_header_values(self, v, file_size, file):

    if v['nc'] <= 0 or v['nr'] <= 0 or v['ns'] <= 0:
      raise SyntaxError(('Bad MRC grid size (%d,%d,%d)'
                         % (v['nc'],v['nr'],v['ns'])))

    esize = self.element_type.itemsize
    data_size = int(v['nc']) * int(v['nr']) * int(v['ns']) * esize
    header_end = file.tell()
    if header_end + data_size > file_size:
      if v['nsymbt'] and (header_end - v['nsymbt']) + data_size == file_size:
        # Sometimes header indicates symmetry operators are present but
        # they are not.  This error occurs in macromolecular structure database
        # entries emd_1042.map, emd_1048.map, emd_1089.map, ....
        # This work around code allows the incorrect files to be read.
        file.seek(-v['nsymbt'], 1)
        v['symop'] = ''
      else:
        msg = ('File size %d too small for grid size (%d,%d,%d)'
               % (file_size, v['nc'],v['nr'],v['ns']))
        if v['nsymbt']:
          msg += ' and %d bytes of symmetry operators' % (v['nsymbt'],)
        raise SyntaxError(msg)

  # ---------------------------------------------------------------------------
  #
  def read_values(self, file, etype, count):

    from numpy import array
    esize = array((), etype).itemsize
    string = file.read(esize * count)
    if len(string) < esize * count:
      raise SyntaxError(('MRC file is truncated.  Failed reading %d values, type %s'
                         % (count, etype.__name__)))
    values = self.read_values_from_string(string, etype, count)
    return values

  # ---------------------------------------------------------------------------
  #
  def read_values_from_string(self, string, etype, count):
  
    from numpy import fromstring
    values = fromstring(string, etype)
    if self.swap_bytes:
      values = values.byteswap()
    if count == 1:
      return values[0]
    return values

  # ---------------------------------------------------------------------------
  # Reads a submatrix from a the file.
  # Returns 3d numpy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    # ijk correspond to xyz.  crs refers to fast,medium,slow matrix file axes.
    crs_origin = [ijk_origin[a] for a in self.crs_to_ijk]
    crs_size = [ijk_size[a] for a in self.crs_to_ijk]
    crs_step = [ijk_step[a] for a in self.crs_to_ijk]

    from ..readarray import read_array
    matrix = read_array(self.path, self.data_offset,
                        crs_origin, crs_size, crs_step,
                        self.matrix_size, self.element_type, self.swap_bytes,
                        progress)
    if not matrix is None:
      matrix = self.permute_matrix_to_xyz_axis_order(matrix)
    
    return matrix

  # ---------------------------------------------------------------------------
  #
  def permute_matrix_to_xyz_axis_order(self, matrix):
    
    if self.ijk_to_crs == (0,1,2):
      return matrix

    kji_to_src = [2-self.ijk_to_crs[2-a] for a in (0,1,2)]
    m = matrix.transpose(kji_to_src)

    return m

  # ---------------------------------------------------------------------------
  #
  def symmetry_matrices(self):

    h = self.header
    for name in ('symop', 'xlen', 'ylen', 'zlen', 'alpha', 'beta', 'gamma'):
      if not name in h:
        return []

    # Read space group symmetries in fractional coordinates
    s = h['symop']
    nsym = len(s)/80
    from Crystal.space_groups import parse_symop
    try:
      usyms = [parse_symop(s[80*i:80*(i+1)].replace(' ', ''))
               for i in range(nsym)]
    except:
      try:
        msg = 'Unable to parse symmetry operators of %s\n%s\n' % (self.name, s)
      except:
        # Garbage in sym operator data can't be interpreted as text.
        msg = 'Unable to parse symmetry operators of %s\n' % (self.name,)
      print(msg)
      return []

    a, b, c = float(h['xlen']), float(h['ylen']), float(h['zlen'])
    if a == 0 or b == 0 or c == 0:
      return []
    from math import pi
    alpha, beta, gamma = [d*pi/180 for d in self.cell_angles]

    # Convert symmetries to unit cell coordinates
    import Crystal
    u2r = Crystal.unit_cell_to_xyz_matrix(a, b, c, alpha, beta, gamma)

    from matrix import invert_matrix, multiply_matrices
    r2u = invert_matrix(u2r)
    syms = [multiply_matrices(u2r, u2u, r2u) for u2u in usyms]
  
    return syms

# -----------------------------------------------------------------------------
#
def valid_cell_angles(alpha, beta, gamma, path):

  err = None
  
  for a in (alpha, beta, gamma):
    if a <= 0 or a >= 180:
      err = 'must be between 0 and 180'

  if alpha + beta + gamma >= 360 and err is None:
    err = 'sum must be less than 360'

  if max((alpha, beta, gamma)) >= 0.5 * (alpha + beta + gamma) and err is None:
    err = 'largest angle must be less than sum of other two'

  if err:
    from sys import stderr
    stderr.write('%s: invalid cell angles %.5g,%.5g,%.5g %s.\n'
                 % (path, alpha, beta, gamma, err))
    return False

  return True
