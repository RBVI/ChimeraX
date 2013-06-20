#
# Read a plt map into a Python array.
#

class Plt_map:

	def __init__(self, path):
		self.path = path

		f = open(path, 'rb')
		import struct
		endianness = '<'
		magicFmt = endianness + 'i'
		msz = struct.calcsize(magicFmt)
		mstr = f.read(msz)
		if len(mstr) < msz:
			raise SyntaxError("gOpenMol file is empty")
		(magic,) = struct.unpack(magicFmt, mstr)
		if magic != 3:
			endianness = '>'
			magicFmt = endianness + 'i'
			f.seek(0)
			(magic,) = struct.unpack(magicFmt, f.read(msz))
			if magic != 3:
				raise SyntaxError("Not a gOpenMol file "
							"(magic number != 3)")
		self.endianness = endianness
		headerFmt = endianness + '4i6f'
		hsz = struct.calcsize(headerFmt)
		surfType, nz, ny, nx, minz, maxz, miny, maxy, minx, maxx = \
					struct.unpack(headerFmt, f.read(hsz))
		self.origin = (minx, miny, minz)
		self.extent = (nx, ny, nz)
		self.grid = ((maxx-minx)/nx, (maxy-miny)/ny, (maxz-minz)/nz)
		self.data_offset = f.tell()

		f.close()

	def matrix(self, ijk_origin, ijk_size, ijk_step, progress):

		from numpy import little_endian, float32
		file_little_endian = (self.endianness == '<')
		swap = (little_endian != file_little_endian)

		from VolumeData.readarray import read_array
		data = read_array(self.path, self.data_offset,
				  ijk_origin, ijk_size, ijk_step,
				  self.extent, float32, swap, progress)
		return data
