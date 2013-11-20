# -----------------------------------------------------------------------------
# Read a SITUS ascii density map file.
#

# -----------------------------------------------------------------------------
#
class SITUS_Density_Map:

  def __init__(self, path):

    self.path = path

    f = open(path, 'r')
    params = f.readline()
    self.data_offset = f.tell()
    f.close()

    fields = params.split()
    if len(fields) < 7:
      raise SyntaxError('First line of SITUS map must have 7 parameters: voxel size, origin xyz, grid size xyz, got\n' + params)

    try:
      self.voxel_size = float(fields[0])
      self.origin = map(float, fields[1:4])
      self.grid_size = map(int, fields[4:7])
    except:
      raise SyntaxError('Error parsing first line of SITUS map: voxel size, origin xyz, grid size xyz, got\n' + params)

    if self.voxel_size == 0:
      raise SyntaxError('Error parsing first line of SITUS map: first value is voxel size and must be non-zero, got line\n' + params)

  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    from VolumeData.readarray import read_text_floats
    data = read_text_floats(self.path, self.data_offset, self.grid_size,
                              progress = progress)
    return data
