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

def ome_image_grids(path, found_paths = None):

    images = parse_ome_tiff_header(path)
    grids = []
    gid = 1
    for i in images:
        print (i.description())
        for c in range(i.nchannels):
            for t in range(i.ntimes):
                g = OME_Image_Grid(i, c, t, gid)
                if i.ntimes > 1:
                    g.series_index = t
                grids.append(g)
                gid += 1
        if found_paths is not None:
            found_paths.update(i.files())
    return grids

# -----------------------------------------------------------------------------
#
from .. import Grid_Data
class OME_Image_Grid(Grid_Data):

  def __init__(self, ome_pixels, channel, time, grid_id):

    self.ome_pixels = d = ome_pixels
    self.initial_style = 'solid'

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
    Grid_Data.__init__(self, d.grid_size, d.value_type,
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
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    array = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    dsize = self.size
    from numpy import zeros
    ia = zeros((dsize[1],dsize[0]), self.value_type)
    ia_1d = ia.ravel()
    c, t = self.channel, self.time
    op = self.ome_pixels
    for k in range(k0, k0+ksz, kstep):
      if progress:
        progress.plane((k-k0)//kstep)
      op.plane_data(c, t, k, ia_1d)
      array[(k-k0)//kstep,:,:] = ia[j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array
        
  # ---------------------------------------------------------------------------
  #
  def files(self):
      return self.ome_pixels.files()
  
def parse_ome_tiff_header(path):

    from PIL import Image
    i = Image.open(path)

    descs = [d for d in i.tag[270] if d.startswith('<?xml')]
    if descs:
        desc = descs[0]
    else:
        from os.path import basename
        raise TypeError('OME TIFF image %s does not have an image description tag'
                        ' starting with "<?xml" as required by the OME TIFF specification,'
                        ' got description tags "%s"' % (basename(path), str(i.tag[270])))
#    print ('ImageDescription for %s\n%s' % (path, desc))

    from xml.etree import ElementTree as ET
    r = ET.fromstring(desc)
    # OME
    #  Image (Name)
    #   Pixels (DimensionOrder, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ, SizeC, SizeT, SizeX, SizeY, SizeZ, Type)
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
            nc, nt, nx, ny, nz = [int(i) for i in (pa['SizeC'], pa['SizeT'], pa['SizeX'], pa['SizeY'], pa['SizeZ'])]
            value_type = pa['Type']
            import numpy
            if not hasattr(numpy, value_type):
                raise TypeError('OME TIFF value type not a numpy type, got %s' % value_type)
            value_type = getattr(numpy, value_type)
            channels = [ch for ch in p if tag_name(ch) == 'Channel']
            cnames = channel_names(channels)
            ccolor = channel_colors(channels)
            tdata = [td for td in p if tag_name(td) == 'TiffData']
            ptable = plane_table(dorder, nz, nt, nc, path, tdata)
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

    def plane_data(self, channel, time, k, pixel_values):
        fname, plane = self.plane_table[(channel,time,k)]
        im = self.image_plane(fname, plane)
        from numpy import array
        pixel_values[:] = array(im).ravel()

    def image_plane(self, filename, plane):
        im = self.image
        opened = False
        if im is None or filename != im.filename or plane < self._last_plane:
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

        # Use tifffile.py instead of PIL
        # if im is None or fname != im._filename:
        #     # Switch image files for multi-file OME TIFF data.
        #     from tifffile import TiffFile
        #     self.image = im = TiffFile(join(dpath,fname))
        #     im._filename = fname
        #     print ('opened', fname, plane)
        #     # PIL bug causes a ValueError: buffer is not large enough in mmap code on 4 Gbyte tiff file
        #     # after reading a few images.  Overwriting PIL's filename attribute avoids using this mmap code.
        # ia[:] = im.asarray(key = plane)

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
def plane_table(dimension_order, nz, nt, nc, path, tdata):
    if dimension_order[:2] != 'XY':
        raise TypeError('OME TIFF dimension order does not start with XY, got %s'
                        % dimension_order)
    sizes = {'Z':nz, 'T':nt, 'C':nc}
    axes = {'Z':0, 'T':1, 'C':2}
    axes_strides = [None, None, None]
    s = 1
    for a in dimension_order[2:]:
        if not a in axes:
            raise TypeError('OME TIFF dimension order requires Z, C, T as last 3 characters, got %s'
                            % dimension_order)
        axes_strides[axes[a]] = s
        s *= sizes[a]
    zstride, tstride, cstride = axes_strides

    from os.path import basename, dirname, join, isfile
    bpath = basename(path)
    ptable = {}
    for c in range(nc):
        for t in range(nt):
            for z in range(nz):
                ptable[(c,t,z)] = (bpath, cstride*c + tstride*t + zstride*z)

    # Revise plane table using TiffData tags
    file_found = {}
    for td in tdata:
        # TODO: Handle images split across multiple files.
        fname = tuple(uuid.attrib['FileName'] for uuid in td if tag_name(uuid) == 'UUID')
        if len(fname) == 0:
            fname = bpath
        elif len(fname) == 1:
            fname = fname[0]
            if fname not in file_found:
                fpath = join(dirname(path), fname)
                file_found[fname] = found = isfile(fpath)
                if not found and len(file_found) > 1:
                    # Multiple files specified in header and this one is missing.
                    raise TypeError('OME TIFF has UUID tag with FileName "%s" and file %s does not exist'
                                    % (fname, fpath))
            if not file_found[fname]:
                # File specified in OME header does not exist.
                # Maybe this file was renamed but header not updated.
                fname = bpath
        else:
            raise TypeError('OME TIFF more than one UUID tag inside a TiffData tag, got %d' % len(fname))
        a = td.attrib
        fc, ft, fz, ifd, pc = [int(i) for i in (a['FirstC'], a['FirstT'], a['FirstZ'], a['IFD'], a['PlaneCount'])]
        if pc != 1:
            raise TypeError('OME TIFF PlaneCount != 1 not supported, got %d' % pc)
        ptable[(fc,ft,fz)] = (fname, ifd)

    return ptable
