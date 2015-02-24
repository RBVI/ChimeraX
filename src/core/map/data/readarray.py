# vi: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
# Read part of a matrix from a binary file making at most one copy of array
# in memory.
#
# The code array.fromstring(file.read()) creates two copies in memory.
# The numpy.fromfile() routine can't read into an existing array.
#
def read_array(path, byte_offset, ijk_origin, ijk_size, ijk_step,
               full_size, type, byte_swap, progress = None):

    if (tuple(ijk_origin) == (0,0,0) and
        tuple(ijk_size) == tuple(full_size) and
        tuple(ijk_step) == (1,1,1)):
        m = read_full_array(path, byte_offset, full_size,
                            type, byte_swap, progress)
        return m

    matrix = allocate_array(ijk_size, type, ijk_step, progress)

    file = open(path, 'rb')

    if progress:
        progress.close_on_cancel(file)
        
    # Seek in file to read needed 1d slices.
    io, jo, ko = ijk_origin
    isize, jsize, ksize = ijk_size
    istep, jstep, kstep = ijk_step
    element_size = matrix.itemsize
    jbytes = full_size[0] * element_size
    kbytes = full_size[1] * jbytes
    ibytes = isize * element_size
    ioffset = io * element_size
    from numpy import fromstring
    for k in range(ko, ko+ksize, kstep):
      if progress:
        progress.plane((k-ko)/kstep)
      kbase = byte_offset + k * kbytes
      for j in range(jo, jo+jsize, jstep):
        offset = kbase + j * jbytes + ioffset
        file.seek(offset)
        data = file.read(ibytes)
        slice = fromstring(data, type)
        matrix[(k-ko)/kstep,(j-jo)/jstep,:] = slice[::istep]

    file.close()

    if byte_swap:
      matrix.byteswap(True)

    return matrix

# -----------------------------------------------------------------------------
# Read an array from a binary file making at most one copy of array in memory.
#
def read_full_array(path, byte_offset, size, type, byte_swap,
                    progress = None, block_size = 2**20):

    a = allocate_array(size, type)
    
    file = open(path, 'rb')
    file.seek(byte_offset)

    if progress:
        progress.close_on_cancel(file)
        a_1d = a.ravel()
        n = len(a_1d)
        nf = float(n)
        for s in range(0,n,block_size):
            b = a_1d[s:s+block_size]
            file.readinto(b)
            progress.fraction(s/nf)
        progress.done()
    else:
        file.readinto(a)
        
    file.close()

    if byte_swap:
        a.byteswap(True)

    return a

# -----------------------------------------------------------------------------
# Read ascii float values on as many lines as needed to get count values.
#
def read_text_floats(path, byte_offset, size, array = None,
                     transpose = False, line_format = None, progress = None):

    if array is None:
        shape = list(size)
        if not transpose:
            shape.reverse()
        from numpy import zeros, float32
        array = zeros(shape, float32)

    f = open(path, 'rb')

    if progress:
        f.seek(0,2)     # End of file
        file_size = f.tell()
        progress.text_file_size(file_size)
        progress.close_on_cancel(f)

    f.seek(byte_offset)

    try:
        read_float_lines(f, array, line_format, progress)
    except SyntaxError as msg:
        f.close()
        raise

    f.close()

    if transpose:
        array = array.transpose()
    
    if progress:
        progress.done()

    return array

# -----------------------------------------------------------------------------
#
def read_float_lines(f, array, line_format, progress = None):

    a_1d = array.ravel()
    count = len(a_1d)

    c = 0
    while c < count:
        line = f.readline()
        if line == '':
            msg = ('Too few data values in %s, found %d, expecting %d'
                   % (f.name, c, count))
            raise SyntaxError(msg)
        if line[0] == '#':
            continue                  # Comment line
        if line_format is None:
            fields = line.split()
        else:
            fields = split_fields(line, *line_format)
        if c + len(fields) > count:
            fields = fields[:count-c]
        try:
            values = map(float, fields)
        except:
            msg = 'Bad number format in %s, line\n%s' % (f.name, line)
            raise SyntaxError(msg)
        for v in values:
            a_1d[c] = v
            c += 1
        if progress:
            progress.fraction(float(c)/(count-1))
  
# -----------------------------------------------------------------------------
#
def split_fields(line, field_size, max_fields):

  fields = []
  for k in range(0, len(line), field_size):
    f = line[k:k+field_size].strip()
    if f:
      fields.append(f)
    else:
      break
  return fields[:max_fields]
  
# -----------------------------------------------------------------------------
#
from numpy import float32
def allocate_array(size, value_type = float32, step = None, progress = None,
                   reverse_indices = True, zero_fill = False):

    if step is None:
        msize = size
    else:
        msize = [1+(sz-1)/st for sz,st in zip(size, step)]

    shape = list(msize)
    if reverse_indices:
        shape.reverse()

    if zero_fill:
        from numpy import zeros as alloc
    else:
        from numpy import empty as alloc

    try:
        m = alloc(shape, value_type)
    except ValueError:
        # numpy 1.0.3 sometimes gives ValueError, sometimes MemoryError
        report_memory_error(msize, value_type)
    except MemoryError:
        report_memory_error(msize, value_type)

    if progress:
        progress.array_size(msize, m.itemsize)

    return m
  
# -----------------------------------------------------------------------------
#
def report_memory_error(size, value_type):

    from numpy import dtype, product, float
    vtype = dtype(value_type)
    tsize = vtype.itemsize
    bytes = product(size, dtype=float)*float(tsize)
    mbytes = bytes / 2**20
    sz = ','.join(['%d' % s for s in size])
    e = ('Could not allocate %.0f Mbyte array of size %s and type %s.\n'
         % (mbytes, sz, vtype.name))
    from chimera import replyobj, CancelOperation
    replyobj.error(e)
    raise CancelOperation(e)
