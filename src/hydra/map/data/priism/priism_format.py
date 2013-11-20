# -----------------------------------------------------------------------------
# Read Delta Vision file format for microscope images.
# Byte swapping is done if needed.
# This format is read by the Priism program developped by
# John Sedat and Dave Agard at UC San Francisco.
#

# -----------------------------------------------------------------------------
#
class Priism_Data:

  def __init__(self, path):

    self.path = path
    import os.path
    self.name = os.path.basename(path)
    
    file = open(path, 'rb')

    # Infer file byte order from column axis size nc.  Requires nc < 2**16
    # Was using mode value but 0 is allowed and does not determine byte order.
    self.swap_bytes = 0
    from numpy import fromstring, int32
    nc = fromstring(file.read(4), int32)
    self.swap_bytes = not (nc > 0 and nc < 65536)
    file.seek(0,0)

    v = self.read_header_values(file)

    self.data_size = (v['NumCol'],
                      v['NumRow'],
                      v['NumSections'] / (v['NumWaves'] * v['NumTimes']))
    xlen, ylen, zlen = v['dx'], v['dy'], v['dz']
    if xlen > 0 and ylen > 0 and zlen > 0:
      self.data_step = (xlen, ylen, zlen)
    else:
      self.data_step = (1.0, 1.0, 1.0)
    self.data_origin = (v['mxst'], v['myst'], v['mzst'])

    self.wavelength_data = self.read_wavelength_data(file, v)
    file.close()
    
    self.num_times = v['NumTimes']
    self.num_waves = v['NumWaves']
    self.axis_order = v['ImgSequence']

  # ---------------------------------------------------------------------------
  #
  def read_header_values(self, file):

    header_string = file.read(header_size)
    v = self.read_fields(header_fields, header_string)

    for attrib in ('NumCol', 'NumRow', 'NumSections', 'NumWaves', 'NumTimes'):
      if v[attrib] <= 0:
        if isinstance(v[attrib], int):
          msg = '%s = %d is <= 0' % (attrib, v[attrib])
        else:
          msg = '%s = %.5g is <= 0' % (attrib, v[attrib])
        raise SyntaxError, msg

    self.check_data_type(v)
      
    return v
    
  # ---------------------------------------------------------------------------
  #
  def read_wavelength_data(self, file, v):

    self.skip_extended_header(file, v)
    self.data_offset = file.tell()

    ptype = v['PixelType']
    from numpy import uint8, int16, float32, uint16, int32
    if ptype == 0:
      element_type = uint8
    elif ptype == 1 or ptype == 5:
      element_type = int16
    elif ptype == 2:
      element_type = float32
    elif ptype == 6:
      element_type = uint16
    elif ptype == 7:
      element_type = int32
      
    wlist = []
    for w in range(v['NumWaves']):
      wavelength = v['wave%d' % (w+1)]
      min_intensity = v['min%d' % (w+1)]
      max_intensity = v['max%d' % (w+1)]
      wd = Wavelength_Data(self, wavelength, w,
                           min_intensity, max_intensity,
                           element_type)
      wlist.append(wd)

    return wlist
    
  # ---------------------------------------------------------------------------
  #
  def print_header_values(self, v, output):

    for name in self.field_names():
      output.write(name + '\t\t=\t' + str(v[name]) + '\n')
  
  # ---------------------------------------------------------------------------
  #
  def read_fields(self, fields, string):

    format = self.header_format(fields)
    names = self.field_names()
    import struct
    values = struct.unpack(format, string)

    d = {}
    for i in range(len(names)):
      d[names[i]] = values[i]

    return d

  # ---------------------------------------------------------------------------
  #
  def field_names(self):

    names = []
    for bstart, bend, type, fname, descrip in header_fields:
      names.append(fname)
    return names
  
  # ---------------------------------------------------------------------------
  # Return format string suitable for Python struct module.
  #
  def header_format(self, fields):

    type_sizes = {'i':4, 'f':4, 'n':2, 'c30':30, 'c80':80}
    type_map = {'i':'i', 'f':'f', 'n':'h', 'c30':'30s', 'c80':'80s'}
    from numpy import little_endian
    big_endian_machine = not little_endian
    big_endian_file = ((big_endian_machine and not self.swap_bytes) or
		       (not big_endian_machine and self.swap_bytes))
    if big_endian_file:
      format = '>'
    else:
      format = '<'
    prev_bend = 0
    for bstart, bend, type, fname, descrip in fields:
      if type_sizes[type] != bend - bstart + 1:
        print 'Size error: ', bstart, bend, type, fname, descrip
      if bstart != prev_bend + 1:
        print 'Gap error: ', bstart, bend, type, fname, descrip
      prev_bend = bend
      format = format + type_map[type]
    return format
    
  # ---------------------------------------------------------------------------
  #
  def skip_extended_header(self, file, v):

    size = v['next']
    file.read(size)
    
  # ---------------------------------------------------------------------------
  #
  def check_data_type(self, v):
    
    itype = v['ImageType']
    if itype < 0 or itype > 3:
      raise SyntaxError, 'Priism image type %d is not image data (0-3)' % itype

    ptype = v['PixelType']
    if not ptype in (0,1,2,5,6,7):
      raise SyntaxError, ('Priism data value type (%d) ' % ptype +
                          'is not unsigned 8 bit integers, ' +
                          'signed or unsigned 16 bit integers, ' +
                          'signed 32 bit integers, or 32 bit floats')

    times = v['NumTimes']
    if times < 1:
      raise SyntaxError, 'Priism data time points < 1 (%d)' % (times,)
    waves = v['NumWaves']
    if waves < 1:
      raise SyntaxError, 'Priism data number of wavelengths < 1 (%d)' % (waves,)
    iseq = v['ImgSequence']
    if iseq < 0 or iseq > 2:
      raise SyntaxError, 'Priism data undefined axis order (%d)' % (iseq,)
    
# -----------------------------------------------------------------------------
#
class Wavelength_Data:

  def __init__(self, priism_data, wavelength, wave_index,
               min_intensity, max_intensity, element_type):

    self.priism_data = priism_data
    self.wavelength = wavelength
    self.wave_index = wave_index
    self.min_intensity = min_intensity
    self.max_intensity = max_intensity
    self.element_type = element_type    # NumPy type of elements

  # ---------------------------------------------------------------------------
  # Reads a submatrix from a potentially very large file.
  # Returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step,
                  time = 0, progress = None):

    from numpy import dtype
    element_size = dtype(self.element_type).itemsize
    d = self.priism_data
    xsize, ysize, zsize = d.data_size
    bytes = xsize * ysize * zsize * element_size
    w = self.wave_index
    nw = d.num_waves
    ao = d.axis_order

    if ao == 1: # XYWZT
      orig, size, step, fsize = [(vi,vj,nw*vk) for (vi,vj,vk) in
                                 (ijk_origin, ijk_size, ijk_step, d.data_size)]
      orig = (orig[0], orig[1], orig[2] + w)
      base = d.data_offset + time * bytes * nw
    else:
      orig, size, step, fsize = ijk_origin, ijk_size, ijk_step, d.data_size
      if ao == 0: # XYZTW
        vnum = w * d.num_times + time
      elif ao == 2: # XYZWT
        vnum = w + nw * time
      base = d.data_offset + vnum * bytes

    from VolumeData.readarray import read_array
    matrix = read_array(d.path, base, orig, size, step,
                        fsize, self.element_type, d.swap_bytes, progress)
    return matrix
    
# -----------------------------------------------------------------------------
# Format documentation taken from Priism file HTML/IM_ref.html.
#

#
# Important fields:
#
#   next        size of extended header (follows header) in bytes
#
#   NumWaves    number of wavelengths
#   NumTimes    number of times
#
#   NumCol, NumRow - xy size of data
#
#   NumSections - (z size) * (number of wavelengths) * (number of times)
#
#   PixelType   0 = unsigned byte
#               1 = signed 16 bit
#               2 = 4 byte float
#               3 = 4 byte complex consisting of two signed 16 bit
#               4 = 8 byte complex consisting of two floats
#               5 = signed 16-bit integer, EM.  Same as type 1.
#               6 = unsigned 16-bit integer
#               7 = signed 32-bit integer
#
#   ImageType   0 = image data
#               1 = tilt series
#               2 = stereo tilt series
#               3 = averaged images
#               3 = averaged stereo pairs
#
#   ImgSequence Order of 2D data sets in file
#               0=ZTW
#               1=WZT
#               2=ZWT
#

#
# Variable Types:
#                  
#     Variable Type                Code 
#     2-byte, signed integer       n
#     4-byte, signed integer       i
#     4-byte, floating-point       f
#     n-byte, character string     cn 
#

header_size = 1024

#
# Byte Numbers, Variable Type, Variable Name, Contents
#
header_fields = (
(1, 4, 'i', 'NumCol', 'Number of columns. Typically, NumCol represents the number of image elements along the X axis.'),
(5, 8, 'i', 'NumRow', 'Number of rows. Typically, NumRow represents the number of image elements along the Y axis.'),
(9, 12, 'i', 'NumSections', 'Total number of sections. (NumZSec*NumWave*NumTimes) '),
(13, 16, 'i', 'PixelType', 'Pixel data type: 0=integer, one byte; 1=integer, two byte signed integer; 2=floating-point, four byte; 3=complex, two byte signed integer, 4=complex, four byte floating-point '),
(17, 20, 'i', 'mxst', 'Starting point along the X axis of sub-image (pixel number). Default is 0. '),
(21, 24, 'i', 'myst', 'Starting point along the Y axis of sub-image (pixel number). Default is 0. '),
(25, 28, 'i', 'mzst', 'Starting point along the Z axis of sub-image (pixel number). Default is 0. '),
(29, 32, 'i', 'mx', 'Number of columns to read; X sub-image size. Default is NumCol. '),
(33, 36, 'i', 'my', 'Number of rows to read; Y sub-image size. Default is NumRow. '),
(37, 40, 'i', 'mz', 'Number of sections to read; Z sub-image size. Default is NumZSec. '),
(41, 44, 'f', 'dx', 'X element length. Width of an element along the X axis in um. '),
(45, 48, 'f', 'dy', 'Y element length. Width of an element along the Y axis in um. '),
(49, 52, 'f', 'dz', 'Z element length. Width of an element along the Z axis in um. '),
(53, 56, 'f', 'alpha', 'Rotation about the X axis in degrees. Default=0. '),
(57, 60, 'f', 'beta', 'Rotation about the Y axis in degrees. Default=0. '),
(61, 64, 'f', 'gamma', 'Rotation about the Z axis in degrees. Default=0. '),
(65, 68, 'i', 'colaxis', 'Column axis. Valid values are 1,2, or 3. Default is 1. '),
(69, 72, 'i', 'rowaxis', 'Row axis. Valid values are 1,2, or 3. Default is 2. '),
(73, 76, 'i', 'zaxis', 'Section axis. Valid values are 1,2, or 3. Default is 3. '),
(77, 80, 'f', 'min1', 'Minimum intensity of the 1st wavelength image. '),
(81, 84, 'f', 'max1', 'Maximum intensity of the 1st wavelength image. '),
(85, 88, 'f', 'mean', 'Mean intensity of the first wavelength image. '),
(89, 90, 'n', 'type', 'Image type. 0=optical sections. '),
(91, 92, 'n', 'nspg', 'Space group number. Applies to crystallography data. '),
(93, 96, 'i', 'next', 'Extended header size, in bytes. '),
(97, 98, 'n', 'dvid', 'DeltaVision ID value. (-16224) '),
(99, 128, 'c30', 'blank', 'Blank section. 30 bytes.'),
(129, 130, 'n', 'NumIntegers', 'Number of 4 byte integers stored in the extended header per section. '),
(131, 132, 'n', 'NumFloats', 'Number of 4 byte floating-point numbers stored in the extended header per section. '),
(133, 134, 'n', 'sub', 'Number of sub-resolution data sets stored within the image. Typically, this equals 1. '),
(135, 136, 'n', 'zfac', 'Reduction quotient for the z axis of the sub-resolution images. '),
(137, 140, 'f', 'min2', 'Minimum intensity of the 2nd wavelength image. '),
(141, 144, 'f', 'max2', 'Maximum intensity of the 2nd wavelength image. '),
(145, 148, 'f', 'min3', 'Minimum intensity of the 3rd wavelength image. '),
(149, 152, 'f', 'max3', 'Maximum intensity of the 3rd wavelength image. '),
(153, 156, 'f', 'min4', 'Minimum intensity of the 4th wavelength image. '),
(157, 160, 'f', 'max4', 'Maximum intensity of the 4th wavelength image. '),
(161, 162, 'n', 'ImageType', 'Image type. See Image Type table below. '),
(163, 164, 'n', 'LensNum', 'Lens identification number.'),
(165, 166, 'n', 'n1', 'Depends on the image type.'),
(167, 168, 'n', 'n2', 'Depends on the image type.'),
(169, 170, 'n', 'v1', 'Depends on the image type. '),
(171, 172, 'n', 'v2', 'Depends on the image type. '),
(173, 176, 'f', 'min5', 'Minimum intensity of the 5th wavelength image. '),
(177, 180, 'f', 'max5', 'Maximum intensity of the 5th wavelength image. '),
(181, 182, 'n', 'NumTimes', 'Number of time points.'),
(183, 184, 'n', 'ImgSequence', 'Image sequence. 0=ZTW, 1=WZT, 2=ZWT. '),
(185, 188, 'f', 'xtilt', 'X axis tilt angle (degrees). '),
(189, 192, 'f', 'ytilt', 'Y axis tilt angle (degrees). '),
(193, 196, 'f', 'ztilt', 'Z axis tilt angle (degrees). '),
(197, 198, 'n', 'NumWaves', 'Number of wavelengths.'),
(199, 200, 'n', 'wave1', 'Wavelength 1, in nm.'),
(201, 202, 'n', 'wave2', 'Wavelength 2, in nm.'),
(203, 204, 'n', 'wave3', 'Wavelength 3, in nm.'),
(205, 206, 'n', 'wave4', 'Wavelength 4, in nm.'),
(207, 208, 'n', 'wave5', 'Wavelength 5, in nm.'),
(209, 212, 'f', 'x0', 'X origin, in um.'),
(213, 216, 'f', 'y0', 'Y origin, in um.'),
(217, 220, 'f', 'z0', 'Z origin, in um.'),
(221, 224, 'i', 'NumTitles', 'Number of titles. Valid numbers are between 0 and 10. '),
(225, 304, 'c80', 'title1', 'Title 1. 80 characters long. '),
(305, 384, 'c80', 'title2', 'Title 2. 80 characters long. '),
(385, 464, 'c80', 'title3', 'Title 3. 80 characters long. '),
(465, 544, 'c80', 'title4', 'Title 4. 80 characters long. '),
(545, 624, 'c80', 'title5', 'Title 5. 80 characters long. '),
(625, 704, 'c80', 'title6', 'Title 6. 80 characters long. '),
(705, 784, 'c80', 'title7', 'Title 7. 80 characters long. '),
(785, 864, 'c80', 'title8', 'Title 8. 80 characters long. '),
(865, 944, 'c80', 'title9', 'Title 9. 80 characters long. '),
(945, 1024, 'c80', 'title10', 'Title 10. 80 characters long. '),
)
