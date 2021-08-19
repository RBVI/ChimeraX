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

def ome_image_grids(path, found_paths = None, log=None):

    images = parse_ome_tiff_header(path, found_paths, log=log)
    grids = []
    gid = 1
    for i in images:
#        print (i.description())
        for c in range(i.nchannels):
            for t in range(i.ntimes):
                g = OMEImageGrid(i, c, t, gid)
                if i.ntimes > 1:
                    g.series_index = t
                grids.append(g)
                gid += 1
        if found_paths is not None:
            found_paths.update(i.files())
    return grids

# -----------------------------------------------------------------------------
#
from .. import GridData, FileFormatError
class OMEImageGrid(GridData):

  def __init__(self, ome_pixels, channel, time, grid_id):

    self.ome_pixels = d = ome_pixels
    self.initial_style = 'image'

    name = d.name
    if channel in d.channel_names:
        cname = d.channel_names[channel]
        name = '%s %s' % (cname, name) if channel == 0 else cname
    elif d.nchannels > 1:
        name += ' ch%d' % channel
    if d.ntimes > 1:
        from math import log10
        tformat = ' t%%0%dd' % int(log10(d.ntimes) + 1)
        name += tformat % time

    origin = (0,0,0)    # TODO: Is there an OME XML origin parameter?
    GridData.__init__(self, d.grid_size, d.value_type,
                      origin, d.grid_spacing,
                      name = name, path = d.path,
                      file_type = 'imagestack', grid_id = grid_id,
                      channel = channel, time = time)

    if channel in d.channel_colors:
        self.rgba = d.channel_colors[channel]
    else:
        from . import default_channel_colors
        self.rgba = default_channel_colors[channel % len(default_channel_colors)]

  # ---------------------------------------------------------------------------
  # Reading multiple planes at a time is twice as fast as one plane at a time
  # using tifffile.py so use read_matrix() instead of read_xy_plane().
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    klist = range(k0, k0+ksz, kstep)
    a = self.ome_pixels.planes_data(self.channel, self.time, klist)
    array = a[:, j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array

  # ---------------------------------------------------------------------------
  #
  def files(self):
      return self.ome_pixels.files()
  
def parse_ome_tiff_header(path, found_paths = None, log = None):

    from tifffile import TiffFile
    with TiffFile(path) as tif:
        tags = tif.pages[0].tags
        desc = tags['ImageDescription'].value if 'ImageDescription' in tags else None

    if desc is None:
        from os.path import basename
        raise FileFormatError('OME TIFF image %s does not have an image description tag'
                        % basename(path))
    elif not desc.startswith('<?xml'):
        from os.path import basename
        raise FileFormatError('OME TIFF image %s does not have an image description tag'
                        ' starting with "<?xml" as required by the OME TIFF specification,'
                        ' got description tags "%s"' % (basename(path), desc))

    try:
        images = ome_pixels_from_xml(path, desc, found_paths = found_paths, log = log)
    except FileFormatError:
        if log:
            log.warning('OME file "%s" xml header\n%s' % (path, desc))
        raise

    return images

def ome_pixels_from_xml(path, xml_header, found_paths = None, log = None):
    
    from xml.etree import ElementTree as ET
    r = ET.fromstring(xml_header)

    # OME
    #  Image (Name)
    #   Pixels (DimensionOrder, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ,
    #           SizeC, SizeT, SizeX, SizeY, SizeZ, Type)
    #    Channel (Color, Name)
    #    TiffData (FirstC, FirstT, FirstZ, IFD, PlaneCount)
    #     UUID (FileName)
    images = []
    for im in r:
        if tag_name(im) != 'Image':
            continue
        name = im.attrib['Name']
        for p in im:
            if tag_name(p) != 'Pixels':
                continue
            pa = p.attrib
            dorder = pa['DimensionOrder']
            sx, sy, sz = [(float(pa[sa]) if sa in pa else 1.0)
                          for sa in ('PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ')]
            nc, nt, nx, ny, nz = [int(pa[i]) for i in ('SizeC', 'SizeT', 'SizeX', 'SizeY', 'SizeZ')]
            value_type = pa.get('Type')
            if value_type is None:
                # This is non-standard, but OME TIFF from Cell Image Library entry 10523 has this.
                value_type = pa.get('PixelType')
            import numpy
            if not hasattr(numpy, value_type):
                raise FileFormatError('OME TIFF value type not a numpy type, got %s' % value_type)
            value_type = getattr(numpy, value_type)
            channels = [ch for ch in p if tag_name(ch) == 'Channel']
            cnames = channel_names(channels)
            ccolor = channel_colors(channels)
            tdata = [td for td in p if tag_name(td) == 'TiffData']
            ptable = plane_table(dorder, nz, nt, nc, path, found_paths, tdata, log=log)
            if not is_plane_table_complete(ptable, nc, nt, nz, log=log):
                ptable, nc, nt, nz = fixed_plane_table(path, ptable, nc, nt, nz, dorder, log=log)
                
            pi = OME_Pixels(path, name, dorder, (sx,sy,sz), (nx,ny,nz), nt, nc, value_type, ptable, cnames, ccolor)
            images.append(pi)

    return images

# Strip namespace prefix
def tag_name(e):
    t = e.tag
    return t.split('}',1)[1] if t[0] == '{' else t

def channel_names(channels):
    cnames = {}
    for ch in channels:
        ca = ch.attrib
        if 'ID' in ca and 'Name' in ca:
            cnum = int(ca['ID'].split(':')[-1])
            cnames[cnum] = ca['Name']
    return cnames

def channel_colors(channels):
    colors = {}
    for ch in channels:
        ca = ch.attrib
        if 'ID' in ca and 'Color' in ca:
            cnum = int(ca['ID'].split(':')[-1])
            cint32 = int(ca['Color'])
            rgba8 = [cint32 & 0xff, (cint32 & 0xff00) >> 8, (cint32 & 0xff0000) >> 16, (cint32 & 0xff000000) >> 24]
            if rgba8[3] == 0:
                rgba8[3] = 255
            rgba = tuple(r/255.0 for r in rgba8)
            colors[cnum] = rgba
    return colors

class OME_Pixels:
    def __init__(self, path, name, dimension_order, grid_spacing, grid_size,
                 ntimes, nchannels, value_type, plane_table,
                 channel_names, channel_colors):
        self.path = path
        self.name = name
        self.dimension_order = dimension_order
        self.grid_spacing = grid_spacing
        self.grid_size = grid_size
        self.ntimes = ntimes
        self.nchannels = nchannels
        self.value_type = value_type            # numpy type
        self.plane_table = plane_table		# Map (channel, time, z) to tiff file and image number
        self.channel_names = channel_names
        self.channel_colors = channel_colors
        self.image = None
        self._last_plane = 0

    def planes_data(self, channel, time, klist):
        '''
        Use tifffile.py to read multiple TIFF image pages.
        tifffile.py reads about 5 times faster than Pillow 5.4.1.
        '''
        # Find set of planes needed from each file.
        fplanes = []
        last_fname = None
        for k in klist:
            fname, plane = self.plane_table[(channel,time,k)]
            if fname != last_fname:
                last_fname = fname
                planes = []
                fplanes.append((fname, planes))
            planes.append(plane)

        # Use tifffile to read planes.
        arrays = []
        from tifffile import TiffFile
        from os.path import dirname, join
        dir = dirname(self.path)
        for fname, fp in fplanes:
            with TiffFile(join(dir,fname)) as tif:
                try:
                    a = tif.asarray(key = fp)
                except IndexError:
                    print('Error reading TIFF file', fname, 'planes', fp)
                    raise
                if a.ndim == 2:
                    a = a.reshape((1,) + tuple(a.shape))	# Make single-plane 3d
                arrays.append(a)

        if len(arrays) == 1:
            array = arrays[0]
        else:
            from numpy import concatenate
            array = concatenate(arrays)
            
        return array

    def plane_data(self, channel, time, k, pixel_values):
        '''Read single TIFF image plane using Pillow.'''
        fname, plane = self.plane_table[(channel,time,k)]
        im = self.image_plane(fname, plane)
        from numpy import array
        pixel_values[:] = array(im).ravel()

    def image_plane(self, filename, plane):
        '''Get Pillow image for single TIFF plane.'''
        im = self.image
        opened = False
        if im is None or filename != im.filename:
            # Switch image files for multi-file OME TIFF data.
            from os.path import dirname, join
            dpath = dirname(self.path)
            from PIL import Image
            self.image = im = Image.open(join(dpath,filename))
            im.filename = filename
            opened = True
        self._last_plane = plane
        try:
            im.seek(plane)
        except ValueError:
            if im.fp.closed and not opened:
                # PIL TIFF reader seems to close file after reading last image of compressed stack.
                # So reopen it.
                self.image = None
                im = self.image_plane(filename, plane)
            else:
                raise

        return im

    def files(self):
        fnames = set(fname for fname, plane in self.plane_table.values())
        from os.path import dirname, join
        dir = dirname(self.path)
        paths = set(join(dir, fname) for fname in fnames)
        return paths
        
    def description(self):
        from numpy import dtype
        d = ', '.join(['image name %s' % self.name,
                       'dimension order %s' % self.dimension_order,
                       'grid spacing %.4g, %.4g, %.4g' % self.grid_spacing,
                       'grid size %d, %d, %d' % self.grid_size,
                       'times %d' % self.ntimes,
                       'channels %d' % self.nchannels,
                       'value type %s' % dtype(self.value_type).name])
        return d

# TIFF file plane number corresoponding to each channel, time and z.
def plane_table(dimension_order, nz, nt, nc, path, found_paths, tdata, log=None):
    if dimension_order[:2] != 'XY':
        raise FileFormatError('OME TIFF dimension order does not start with XY, got %s'
                        % dimension_order)

    from os.path import basename, dirname, join, isfile
    bpath = basename(path)
    dir = dirname(path)
    
    ptable = {}

    # Build plane table using TiffData tags
    file_found = {}
    for td in tdata:
        # Handle images split across multiple files.
        fname = tuple(uuid.attrib['FileName'] for uuid in td if tag_name(uuid) == 'UUID')
        if len(fname) == 0:
            fname = bpath
        elif len(fname) == 1:
            fname = fname[0]
            if fname not in file_found:
                file_found[fname] = isfile(join(dir, fname))
            if not file_found[fname]:
                continue
        else:
            msg = 'OME TIFF %s has more than one UUID tag inside a TiffData tag, got %d' % (path, len(fname))
            raise FileFormatError(msg)

        a = td.attrib
        attrs = (('FirstC',0), ('FirstT',0), ('FirstZ',0), ('IFD',0), ('PlaneCount',1))
        has_attrs = [attr for attr, default_value in attrs if attr in a]
        if has_attrs:
            fc, ft, fz, ifd, pc = [int(a.get(attr,default_value)) for attr, default_value in attrs]
            set_plane_table_entries(fc, ft, fz, fname, ifd, pc, nc, nt, nz, dimension_order, ptable)

    # Report missing files.
    missing = [fname for fname, found in file_found.items() if not found]
    if missing:
        fnames = ', '.join(fname for fname in missing[:3])
        if len(missing) > 3:
            fnames += ' ...'
        msg = 'OME TIFF references %d files that were not found: %s' % (len(missing), fnames)
        if log:
            log.warning(msg)
        else:
            raise FileFormatError(msg)

    # Report files that were found.
    if ptable is not None and found_paths is not None:
        for fname, found in file_found.items():
            if found:
                found_paths.add(join(dir,fname))

    return ptable

def set_plane_table_entries(fc, ft, fz, fname, ifd, plane_count, nc, nt, nz, dimension_order, ptable):
    if plane_count == 1:
        ptable[(fc,ft,fz)] = (fname, ifd)
        return

    # Fill in IFD indices for multiple C,T,Z planes.
    sizes = {'Z':nz, 'T':nt, 'C':nc}
    ctz = {'C':fc, 'T':ft, 'Z':fz}
    for p in range(plane_count):
        pctz = (ctz['C'],ctz['T'],ctz['Z'])
        ptable[pctz] = (fname, ifd + p)
        # Increment c,t,z index to next plane
        for a in dimension_order[2:]:
            ctz[a] += 1
            if ctz[a] >= sizes[a]:
                ctz[a] = 0
            else:
                break

def is_plane_table_complete(ptable, nc, nt, nz, log=None):
    missing = missing_planes(ptable, nc, nt, nz)
    if len(missing) == 0:
        return True

    msg = 'Error in OME TIFF file header. It specifies the location of some 2d images but not all (%d of %d), missing CTZ=%s...' % (len(ptable), nc*nt*nz, ' '.join('(%d,%d,%d)' % ctz for ctz in missing[:5]))
    if log:
        log.warning(msg)
    else:
        raise FileFormatError(msg)

    return False

def missing_planes(ptable, nc, nt, nz):
    missing = []
    for c in range(nc):
        for t in range(nt):
            for z in range(nz):
                if (c,t,z) not in ptable:
                    missing.append((c,t,z))
    missing.sort()
    return missing

def fixed_plane_table(path, ptable, nc, nt, nz, dorder, log=None):
    if len(ptable) == 0:
        # TODO: Need to check that path contains nc*nt*nz planes.
        #  More likely it has just nz planes, and other files
        #  containing other times and channels are missing.
        #  Then we would like to still open this file.
        ptable = default_plane_table(path, nc, nt, nz, dorder)
        if log:
            log.warning('OME TIFF file did not include z-plane lookup table, using default')

    if len(ptable) != nc*nt*nz:
        maxc = maxt = maxz = 0
        for c,t,z in ptable.keys():
            maxc = max(maxc,c)
            maxt = max(maxt,t)
            maxz = max(maxz,z)
        if log and (nc, nt, nz) != (maxc+1, maxt+1, maxz+1):
            msg = ('OME TIFF header says there are %d channels, %d times, %d z-planes, but only found %d channesl, %d times, %d z-planes' % (nc, nt, nz, maxc+1, maxt+1, maxz+1))
            log.warning(msg)
        nc, nt, nz = maxc + 1, maxt + 1, maxz + 1
        
    if len(ptable) != nc*nt*nz:
        missing = missing_planes(ptable, nc, nt, nz)
        mp = ' '.join('(%d,%d,%d)' % ctz for ctz in missing[:3])
        if len(missing) > 3:
            mp += ' ...'
        raise FileFormatError('OME TIFF file cannot locate all z-planes.\n'
                              'Header reports %d channels, %d times, %d z-planes, got %d of %d planes, missing (c,t,z) = %s'
                              % (nc, nt, nz, len(ptable), nc*nt*nz, mp))
    return ptable, nc, nt, nz

def default_plane_table(path, nc, nt, nz, dimension_order):
    from os.path import basename
    fname = basename(path)
    sizes = {'Z':nz, 'T':nt, 'C':nc}
    axes = {'Z':0, 'T':1, 'C':2}
    axes_strides = [None, None, None]
    s = 1
    for a in dimension_order[2:]:
        if not a in axes:
            raise FileFormatError('OME TIFF dimension order requires Z, C, T as last 3 characters, got %s'
                            % dimension_order)
        axes_strides[axes[a]] = s
        s *= sizes[a]
    zstride, tstride, cstride = axes_strides

    ptable = {}
    for c in range(nc):
        for t in range(nt):
            for z in range(nz):
                ptable[(c,t,z)] = (fname, cstride*c + tstride*t + zstride*z)
    return ptable
