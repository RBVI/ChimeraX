# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Read IMARIS file format for 5D light microscopy 3d.  Based on HDF5.
#
# Format described here:
#
#   http://open.bitplane.com/Default.aspx?tabid=268
#
# HDF hierarchy:
#
# DataSet
#   ResolutionLevel 0
#     TimePoint 0
#       Channel 0
#         ImageSizeX = 285
#         ImageSizeY = 218
#         ImageSizeZ = 64
#         Data - 3d image array
#         HistogramMin = 0
#         HistogramMax = 255
#         Histogram
#   ResolutionLevel 1
#     TimePoint 0
#       Channel 0
#         Data
#         Histogram
#   ResolutionLevel 2
#
# DataSetInfo
#  ImarisDataSet		This section holds information about the data organization in the file. (always present)
#   Creator = Imaris	
#   NumberOfImages = 1	At the moment there is only one multi-channel-time image per file
#   Version = 5.5	The version of Imaris which created this file
#  Imaris		Information about the thumbnail and Imaris. (always present)
#   Filename = retina.ims	The name of the original file
#   ManufactorString = Generic Imaris 3.x	Manufactor information
#   ManufactorType = LSM	Manufactor type information
#   ThumbnailMode = thumbnailMIP	The type of data representation of the thumbnail. Valid values are "thumbnailNone", "thumbnailMiddleSection", "thumbnailMIP" or "thumbnailNone".
#   Version = 5.5	The version of Imaris which created this file
#  Image		Information about the DataSet. (always present)
#   Description = nucleus	Detailed description of the image in plain text (can be multiple lines)
#   ExtMax0 = 46.7464	Data max. extension X (in given units: um, nm, …)
#   ExtMax1 = 35.7182	Data max. extension Y
#   ExtMax2 = 12.6	Data max. extension Z
#   ExtMin0 = -10.3	Data origin X
#   ExtMin1 = -1.5	Data origin Y
#   ExtMin2 = 3.4	Data origin Z
#   LensPower = 63x	Deconvolution parameter
#   Name = m1193.pic	Short description of the image (some characters)
#   Noc = 2	Number of channels
#   RecordingDate = 1991-10-01 16:45:45	“YYYY-MM-DD HH:MM:SS”
#   Unit = um	"m“, "mm“, "um“ or "nm“
#   X = 285	Image Size X (in voxels)
#   Y = 218	Image Size Y (in voxels)
#   Z = 64	Image Size Z (in voxels)
#  Channel X		Information about channel X (there is one section per channel) (always present)
#   Color = 1 0 0	The base color (r,g,b float values from 0.0 to 1.0)
#   ColorMode = BaseColor	“BaseColor”, “TableColor” see example below for the color table mode.
#   ColorOpacity = 0.168	The opacity with which the volume rendering displays the channel (float value from 0.0 to 1.0)
#   ColorRange = 0 194.921	The display “contrast”
#   Description = Cy5	Detailed description of the channel
#   Gain = 0	Deconvolution parameter
#   LSMEmissionWavelength =	Emission Wavelength
#   LSMExcitationWavelength =	Excitation Wavelength
#   LSMPhotons =	Deconvolution parameter
#   LSMPinhole =	Pinhole diameter
#   Max = 255	The data maximum value of the channel image
#   MicroscopeMode =	Deconvolution parameter
#   Min = 0	The data minimum value of the channel image
#   Name = CollagenIV (TxRed)	Short description of the channel (some chars)
#   NumericalAperture =	Numerical Aperture
#   Offset = 0	Deconvolution parameter
#   Pinhole = 0	Deconvolution parameter
#   RefractionIndexEmbedding =	Deconvolution parameter
#   RefractionIndexImmersion =	Deconvolution parameter
#  TimeInfo		Information about time for all channels. (always present)
#   DataSetTimePoints = 1	The number of time points in the DataSet
#   FileTimePoints = 1	The number of time points in the file (currently the same as DataSetTimePoints)
#   TimePoint1 = 1991-10-01 16:45:45.000	Time for time point 1. Time must be strictly increasing between time points.
#  Log		Any information about the history of the image (especially the data manipulation).
#   Entries = 3	Number of entries (number of image data modifications). Can be zero.
#   Entry0 =	First modification
#   Entry1 =	Second modification
#   Entry2 =	Third modification# 
#
# Thumbnail
#
class IMS_Data:

    def __init__(self, path, keep_open = True, hdf5_cache_size = 2**30):

        self.path = path

        import os.path
        self.name = os.path.basename(path)
    
        import tables
        f = tables.open_file(path, chunk_cache_size = hdf5_cache_size)

        # Imaris image header values are numpy arrays of characters. Ugh.
        dsi = f.root.DataSetInfo
        im = dsi.Image._v_attrs
        extent = [(float(im['ExtMin%d' % a].tostring()), float(im['ExtMax%d' % a].tostring())) for a in (0,1,2)]
        size = [int(im[axis].tostring()) for axis in ('X','Y','Z')]
        origin = [e1 for e1,e2 in extent]
        step = [(e2-e1)/s for (e1,e2),s in zip(extent,size)]

        # Colors
        ccolors = self.channel_colors(dsi)
        default_colors = ((1,0,0,1),(0,1,0,1),(0,0,1,0),(0,1,1,1),(1,1,0,1),(1,0,1,1))

        agroups = self.find_arrays(f.root)
        if len(agroups) == 0:
            raise SyntaxError('Imaris file %s contains no 3d arrays' % path)

        # Sort data arrays by channel, time and resolution level.
        ch_arrays = {}
        for g,a in agroups:
            if len(a) != 1:
                # TODO: Warn, should never find multiple 3d arrays.
                continue
            gn = g._v_name
            if gn.startswith('Channel'):
                channel = int(gn.split()[1])
                gp = g._v_parent
                gpn = gp._v_name
                if gpn.startswith('TimePoint'):
                    time = int(gpn.split()[1])
                    gpp = gp._v_parent
                    gppn = gpp._v_name
                    if gppn.startswith('ResolutionLevel'):
                        level = int(gppn.split()[1])
                        if channel not in ch_arrays:
                            ch_arrays[channel] = {}
                        if time not in ch_arrays[channel]:
                            ch_arrays[channel][time] = []
                        ch_arrays[channel][time].append((level, g, a[0]))

        cimages = {}
        for c, times in ch_arrays.items():
            rgba = ccolors[c] if c < len(ccolors) else default_colors[c%len(default_colors)]
            cimages[c] = cim = []
            for t, levels in times.items():
                for lev,g,a in sorted(levels):
                    if lev == 0:
                        i = IMS_Image(t,c,g,a,size,origin,step,rgba)
                        cim.append(i)
                    else:
                        ss = 2**lev
                        cell_size = (ss,ss,ss)
                        ssize = tuple(s//ss for s in size)
                        i.subsamples.append((cell_size, ssize, a._v_pathname))
            cim.sort(key = lambda i: i.time)

        self.channel_images = cimages

        self.keep_open = keep_open
        self.hdf_file = f if keep_open else None
        if not keep_open:
            f.close()

    # --------------------------------------------------------------------------
    #
    def __del__(self):
        self.close_file()

    # --------------------------------------------------------------------------
    #
    def close_file(self):        
        f = self.hdf_file
        if f is not None:
            self.hdf_file = None
            f.close()
            
    # --------------------------------------------------------------------------
    # Return list of grouped arrays.  Each element is a tuple containing a
    # group and a list of 3-d arrays that are children of the group.
    #
    def find_arrays(self, parent, anodes = None):

        if anodes is None:
            anodes = []

        garrays = []
        groups = []
        from tables.array import Array
        from tables.group import Group
        for node in parent._f_iter_nodes():
            if isinstance(node, Array):
                dims = len(node.shape)
                if dims == 3:
                    garrays.append(node)
            elif isinstance(node, Group):
                groups.append(node)

        if garrays:
            anodes.append((parent, garrays))

        for g in groups:
            self.find_arrays(g, anodes)

        return anodes

    # --------------------------------------------------------------------------
    # Reads a submatrix returning 3D NumPy matrix with zyx index order.
    # array_path can be a HDF5 path to a 3d array or a list of paths to
    # a stack of 2d arrays.
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step,
                    array_path, array, progress):

#        import traceback
#        traceback.print_stack()
        
        import tables
        f = self.hdf_file if self.keep_open else tables.open_file(self.path)
        if progress:
            progress.close_on_cancel(f)
#        array_path = choose_chunk_size(f, array_paths, ijk_size)
        a = f.get_node(array_path)
        from ..cmap import copy_hdf5_array
        copy_hdf5_array(a, ijk_origin, ijk_size, ijk_step, array, progress)
        if progress:
            progress.done()

        if not self.keep_open:
            f.close()
        return array

    # --------------------------------------------------------------------------
    # Read channel colors from HDF5, returning list of rgba 0-1 float values.
    #
    def channel_colors(self, dataset_info):
        ccolors = []
        while True:
            cnum = len(ccolors)
            cd = getattr(dataset_info, 'Channel %d' % cnum, None)
            if cd is None:
                break
            a = cd._v_attrs
            color = a.Color.tostring()
            rgba = tuple(float(r) for r in color.split()) + (1.0,)
            ccolors.append(rgba)
        return ccolors

# -----------------------------------------------------------------------------
#
class IMS_Image:

    def __init__(self, time, channel, group, array, size, origin, step, color):

        self.time = time
        self.channel = channel
        
        self.array_path = array._v_pathname

        # The array size in the HDF5 is actually padded out to maybe multiples of 32 by Imaris. Ick.
        self.size = size
        # self.size = tuple(reversed(array.shape))
        # self.size = self.check_array_sizes(parrays)
        self.name = '%d' % time
        # self.name = self.find_name(group)
        self.step = step
        # self.step = self.find_plane_spacing(group)
        self.origin = origin
        # self.origin = self.find_origin(group)
        self.value_type = array.atom.dtype
        # self.value_type = self.check_array_types(parrays + sarrays)
        self.cell_angles = (90.0, 90.0, 90.0)
        # self.cell_angles = self.find_cell_angles(group)
        self.rotation = ((1,0,0),(0,1,0),(0,0,1))
        # self.rotation = self.find_rotation(group)
        self.symmetries = None
        # self.symmetries = self.find_symmetries(group)
        self.default_color = color

        self.subsamples = []
        # subsamples = []
        # stable = {}
        # for a in sarrays:
        #     step = tuple(a._v_attrs.subsample_spacing)
        #     size = tuple(reversed(a.shape))
        #     if not (step,size) in stable:
        #         array_paths = [a._v_pathname]
        #         subsamples.append((step, size, array_paths))
        #         stable[(step,size)] = array_paths
        #     else:
        #         stable[(step,size)].append(a._v_pathname)
        # self.subsamples = subsamples

    # --------------------------------------------------------------------------
    #
    def find_name(self, group):

        va = group._v_attrs
        if 'name' in va:
            name = va.name
            if isinstance(name, bytes):
                # This was needed in Python 2
                name = name.decode('utf-8')
            return name
        return ''

    # --------------------------------------------------------------------------
    #
    def find_plane_spacing(self, group):

        va = group._v_attrs
        if 'step' in va:
            step = tuple(float(s) for s in va.step)
        else:
            step = (1.0, 1.0, 1.0)
        return step

    # --------------------------------------------------------------------------
    #
    def find_origin(self, group):

        va = group._v_attrs
        if 'origin' in va:
            origin = tuple(float(o) for o in va.origin)
        else:
            origin = (0.0, 0.0, 0.0)
        return origin

    # --------------------------------------------------------------------------
    #
    def find_cell_angles(self, group):

        va = group._v_attrs
        if 'cell_angles' in va:
            cell_angles = tuple(float(a) for a in va.cell_angles)
        else:
            cell_angles = (90.0, 90.0, 90.0)
        return cell_angles

    # --------------------------------------------------------------------------
    #
    def find_rotation(self, group):

        va = group._v_attrs
        if 'rotation_axis' in va and 'rotation_angle' in va:
            axis = va.rotation_axis
            angle = va.rotation_angle
            from chimerax.geometry import matrix
            r = matrix.rotation_from_axis_angle(axis, angle)
        else:
            r = ((1,0,0),(0,1,0),(0,0,1))
        return r

    # --------------------------------------------------------------------------
    #
    def find_symmetries(self, group):

        va = group._v_attrs
        if 'symmetries' in va:
            from chimerax.geometry import Places
            sym = Places(place_array = va.symmetries)
        else:
            sym = None
        return sym

    # --------------------------------------------------------------------------
    #
    def check_array_sizes(self, arrays):

        shape = arrays[0].shape
        for a in arrays[1:]:
            if a.shape != shape:
                self.mismatched_arrays(arrays)
        size = list(reversed(shape))
        if len(size) == 2:
            size += [1]
        size = tuple(size)
        return size

    # --------------------------------------------------------------------------
    #
    def check_array_types(self, arrays):

        dtype = arrays[0].atom.dtype
        for a in arrays[1:]:
            if a.atom.dtype != dtype:
                self.mismatched_arrays(arrays)
        return dtype

    # --------------------------------------------------------------------------
    #
    def mismatched_arrays(self, arrays):

        a0 = arrays[0]
        file_path = a0._v_file.filename
        group_path = a0._v_parent._v_pathname
        message = 'Imaris file %s has a group %s containing arrays of different sizes or value types\n' % (file_path, group_path)
        sizes = '\n'.join(['  %s  (%d,%d,%d)  %s' %
                           ((a._v_name,) + tuple(a.shape) + (a.atom.dtype.name,))
                           for a in arrays])
        message += sizes
        raise SyntaxError(message)

# -----------------------------------------------------------------------------
#
def choose_chunk_size(f, array_paths, ijk_size):

    if len(array_paths) == 1:
        return array_paths[0]

    alist = []
    shape = tuple(reversed(ijk_size))
    for p in array_paths:
        a = f.get_node(p)
        cshape = a._v_chunkshape
        pad = [(cshape[i] - (shape[i]%cshape[i]))%cshape[i] for i in (0,1,2)]
        extra = sum([pad[i] * shape[(i+1)%3] * shape[(i+2)%3] for i in (0,1,2)])
        alist.append((extra, p))
    alist.sort()
    array_path = alist[0][1]
    return array_path
