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

# DeltaVision file reader ported from BioFormats java code DeltavisionReader.java from November 14, 2017.
# This format is based on MRC with extra header info.
class DeltaVision_Data:

  DV_MAGIC_BYTES_1 = 0xa0c0
  DV_MAGIC_BYTES_2 = 0xc0a0
  HEADER_LENGTH = 1024;

  def __init__(self, path):

      self.path = path
      
      file = open(path, 'rb')
      self.read_header(file)
      file.close()
      
  def read_header(self, file):

      from numpy import uint16, int32, int16, float32, dtype
      from numpy import little_endian as native_little_endian

      self.swap_bytes = False
      file.seek(96);
      magic = self.read_values(file, uint16)
      if magic != self.DV_MAGIC_BYTES_1 and magic != self.DV_MAGIC_BYTES_2:
          raise SyntaxError('DeltaVision file, wrong magic number at byte 97, got %x, expected %x or %x'
                            % (magic, self.DV_MAGIC_BYTES_1, self.DV_MAGIC_BYTES_2))
      little_endian = (magic == self.DV_MAGIC_BYTES_2)
      self.swap_bytes = (native_little_endian != little_endian)
      
      file.seek(0);
      sizeX, sizeY, imageCount = self.read_values(file, int32, 3)
      filePixelType = self.read_values(file, int32)

      file.seek(180);
      rawSizeT = self.read_values(file, uint16)
      self.ntimes = sizeT = 1 if rawSizeT == 0 else rawSizeT

      sequence = self.read_values(file, int16)

      file.seek(92);
      self.extSize = self.read_values(file, int32) # Size of extended header.
      self.data_offset = self.HEADER_LENGTH + self.extSize

      file.seek(196);
      rawSizeC = self.read_values(file, int16)
      self.nchannels = sizeC = 1 if rawSizeC == 0 else rawSizeC

      # --- compute some secondary values ---

      sizeZ = imageCount // (sizeC * sizeT)
      self.data_size = (sizeX, sizeY, sizeZ)
      self.matrix_size = (sizeX, sizeY, sizeZ)

      self.element_type = pixelType = self.numpy_value_type(filePixelType)
      self.plane_bytes = sizeX * sizeY * dtype(self.element_type).itemsize

      # --- parse extended header ---
      file.seek(128);
      # The number of ints in each extended header section. These fields appear
      # to be all blank but need to be skipped to get to the floats afterwards
      self.numIntsPerSection = self.read_values(file, int16)
      self.numFloatsPerSection = self.read_values(file, int16)

      self.setOffsetInfo(sequence, sizeZ, sizeC, sizeT)

      # Extra metadata
      file.seek(16)

      subImageStartX, subImageStartY, subImageStartZ = self.read_values(file, int32, 3)
      pixelSamplingX, pixelSamplingY, pixelSamplingZ = self.read_values(file, int32, 3)
      self.data_step = pixX, pixY, pixZ = self.read_values(file, float32, 3)
      self.cell_angles = xAxisAngle, yAxisAngle, zAxisAngle = self.read_values(file, float32, 3)
      xAxisSeq, yAxisSeq, zAxisSeq = self.read_values(file, int32, 3)

      minWave = [None]*5
      maxWave = [None]*5

      minWave[0] = self.read_values(file, float32)
      maxWave[0] = self.read_values(file, float32)

      meanIntensity = self.read_values(file, float32)
      spaceGroupNumber = self.read_values(file, int32)

      file.seek(132)

      numSubResSets = self.read_values(file, int16)
      zAxisReductionQuotient = self.read_values(file, int16)

      for i in (1,2,3):
          minWave[i] = self.read_values(file, float32)
          maxWave[i] = self.read_values(file, float32)

      type = self.read_values(file, int16)
      lensID = self.read_values(file, int16)

      file.seek(172)

      minWave[4] = self.read_values(file, float32)
      maxWave[4] = self.read_values(file, float32)

      file.seek(184)

      xTiltAngle, yTiltAngle, zTiltAngle = self.read_values(file, float32, 3)

      self.read_values(file, int16) # Skip 2 bytes

      waves = self.read_values(file, int16, 5)

      self.data_origin = xOrigin, yOrigin, zOrigin = self.read_values(file, float32, 3)

      self.read_values(file, int32) # Skip 4 bytes

      title = [file.read(80) for i in range(10)]

  def numpy_value_type(self, filePixelType):
      from numpy import uint8, int16, float32, uint16
      dtypes = {0: uint8,
                1: int16,
                2: float32,
                3: int16,
                4: float32,
                6: uint16}
      return dtypes.get(filePixelType, uint8)

  # This method calculates the size of a w, t, z section depending on which
  # sequence is being used (either ZTW, WZT, or ZWT)
  def setOffsetInfo(self, imgSequence, numZSections, numWaves, numTimes):
      plane_header_size = (self.numIntsPerSection + self.numFloatsPerSection) * 4
      if imgSequence == 0:
          # ZTW sequence
          zs = self.plane_bytes
          ts = zs * numZSections
          ws = ts * numTimes
          hzs = plane_header_size
          hts = hzs * numZSections
          hws = hts * numTimes
      elif imgSequence == 1 or imgSequence == 65536:
          # WZT sequence
          ws = self.plane_bytes
          zs = ws * numWaves
          ts = zs * numZSections
          hws = plane_header_size
          hzs = hws * numWaves
          hts = hzs * numZSections
      elif imgSequence == 2:
          # ZWT sequence
          zs = self.plane_bytes
          ws = zs * numZSections
          ts = ws * numWaves
          hzs = plane_header_size
          hws = hzs * numZSections
          hts = hws * numWaves
      else:
          raise SyntaxError('DeltaVision file unknown axis order %d, require 0,1,2 or 65536' % imgSequence)
      
      self.z_stride, self.time_stride, self.channel_stride = zs, ts, ws
      self.header_z_stride, self.header_time_stride, self.header_channel_stride = hzs, hts, hws

  # ---------------------------------------------------------------------------
  #
  def read_values(self, file, etype, count = 1):

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
  def read_values_from_string(self, string, etype, count = 1):
  
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
  def read_matrix(self, ijk_origin, ijk_size, ijk_step,
                  channel=0, time=0, progress=None):
    offset = self.data_offset + channel * self.channel_stride + time * self.time_stride
    if self.z_stride > self.plane_bytes:
        # Channels are interleaved between z planes.
        ijk_origin, ijk_size, ijk_step = list(ijk_origin), list(ijk_size), list(ijk_step)
        zs = self.z_stride // self.plane_bytes
        ijk_origin[2] *= zs
        ijk_size[2] *= zs
        ijk_step[2] *= zs
    from ..readarray import read_array
    matrix = read_array(self.path, offset,
                        ijk_origin, ijk_size, ijk_step,
                        self.matrix_size, self.element_type, self.swap_bytes,
                        progress)
    return matrix
