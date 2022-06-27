# -----------------------------------------------------------------------------
#
import numpy
char = numpy.int8
uchar = numpy.uint8
short = numpy.int16
int = numpy.int32
uint = numpy.uint32
float = numpy.float32

# -----------------------------------------------------------------------------
#
model_format = [
((char,4),   'imodver'),      # String V1.2
((char,128),   'name'),         # Name of model file.
(int,    'xmax'),         # Maximum values for x,y and z.
(int,    'ymax'),         #  These are usually set to the
(int,    'zmax'),         #  image dimensions.
(int,    'objsize'),      # Number of objects in model.'
(uint,   'flags'),        # Model flags  (IMODF_FLAG...)
                 #  Bit 12 on : multiple clip planes are possible
                 #  Bit 13 on : mat1 and mat3 are stored as bytes
                 #  Bit 14 on : otrans has image origin values
                 #  Bit 15 on : current tilt angles are stored correctly
                 #  Bit 16 on : model last viewed on Y/Z flipped image
(int,   'drawmode'),      # 1 to draw model, -1 not to
(int,   'mousemode'),     # Mouse editing mode. (0=movie,1=model)
(int,   'blacklevel'),    # Contrast adjustment. (0-256) Default = 0.
(int,   'whitelevel'),    # Contrast adjustment. (0-256) Default = 255.
(float, 'xoffset'),       # Offsets used for display &amp; conversion, 
(float, 'yoffset'),       #  should be set to 0.                        
(float, 'zoffset'),       #  (unused, superceded by MINX information)
(float, 'xscale'),   # Scaling for pixel size. xscale and yscale should
(float, 'yscale'),   #  be 1.0 and zscale should be adjusted to account
(float, 'zscale'),   #  for the image thickness.
(int,   'object'),   # Current object, contour and point, used for
(int,   'contour'),  #  model editing.
(int,   'point'),
(int,   'res'),    # Minimum number of pixels between points when adding
                   #  points by dragging mouse with middle button down.
(int,   'thresh'),   # Threshold level for auto contour generator.
(float, 'pixsize'),  # Size of one pixel, in the units given next
(int,   'units'),    # 0 = pixels, 3 = km, 1 = m, -2 = cm, -3 = mm, 
                     # -6 = microns, -9 = nm, -10 = Angstroms, -12 = pm
(int,   'csum'),     # Checksum storage. Used for autosave only.
(float, 'alpha'),    # Angles used for orientation of the model with
(float, 'beta'),     #   image. Should be set to 0.0
(float, 'gamma'),    #   (unused, superceded by MINX information)
]

# -----------------------------------------------------------------------------
#
object_format = [
((char,64), 'name'),    # Name of Object. Null terminated string.
((uint,16), 'extra'),   # Extra data reserved for future use.
(int,  'contsize'),# Number of Contours in object.
(uint, 'flags'),  # bit flags for object (IMOD_OBJFLAG...).
                  # Bit 1 on : Off - turns off display
                  # Bit 3 on : Open - contours are not closed
                  # Bit 4 on : Wild - contours not constrained in Z
                  # Bit 5 on : Inside-out  - light inside surface
                  # Bit 6 on : Use fill color for spheres
                 # Bit 8 on : Fill - draw filled polygons
                 # Bit 9 on : Scattered - contours have scattered points
                 # Bit 10 on : Mesh - draw mesh data in 3D
                 # Bit 11 on : Lines - do not draw contours in 3D
                 # Bit 14 on : Fill color - use instead of regular color
                 # Bit 15 on : Anti-aliasing on
                 # Bit 16 on : Normals have magnitudes
                 # Bit 17 on : Draw normal maginitudes with false color
                 # Bit 18 on : Time - contours have time indexes
                 # Bit 19 on : Light both sides of surface
                 # Bit 31 reserved for temporary use
                 # In 3dmodv, Draw Data Type - Contour = Mesh flag off,
                 #                             Mesh    = Mesh flag on
                 # Drawing Style - Points = Lines flag on, Fill flag off
                 #                 Lines  = Lines flag off, Fill flag off
                 #                 Fill   = Lines flag on, Fill flag on
                 #          Fill Outline  = Lines flag off, Fill flag on

(int,   'axis'),      # Z = 0, X = 1, Y = 2.
(int,   'drawmode'),  # Tells type of scattered points to draw (unused)
(float, 'red'),       # Color values, range is (0.0 - 1.0)
(float, 'green'),      
(float, 'blue'),       
(int,   'pdrawsize'), # Default radius in pixels of scattered points in 3D.

(uchar, 'symbol'),     # Point Draw symbol in 2D, default is 1.
                       # 0 = circle, 1 = none, 2 = square, 3 = triangle,
                       # 4 = star.
(uchar, 'symsize'),    # Size of 2D symbol; radius in pixels.
(uchar, 'linewidth2'), # Linewidth in 2-D view.
(uchar, 'linewidth'),  # Linewidth in 3-D view.
(uchar, 'linesty'),    # Line draw style default is 0 = solid. 1 = dashed.
(uchar, 'symflags'),  # Bit 0 on : Fill the symbol.
                      # Bit 1 on : Draw beginning and end symbols.
(uchar, 'sympad'),    # Set to 0, for future use.
(uchar, 'trans'),     # Transparency, range is (0 to 100), maps to
                      # (1.0 to 0.0) in call to glColor4f or glMaterialfv
(int,   'meshsize'),  # Number of meshes in object.
(int,   'surfsize'),  # Max surfaces in object.
]

# -----------------------------------------------------------------------------
#
contour_format = [
(int,  'psize'), # Number of points in contour.
(uint, 'flags'), # Bit 3 on : open, do not connect endpoints
                 # Bit 4 on : wild, not in one Z plane
                 # Bit 15 on : type variable contains time index
(int,  'type'),  # The time index used by this contour.
(int,  'surf'),  # The surface index used by this contour.
((float,'psize',3), 'pt'),    # Array of point triplets.
]

# -----------------------------------------------------------------------------
#
mesh_format = [
(int,    'vsize'), # Size of vertex array (# of triple floats).
(int,    'lsize'), # Size of index array.
(uint,   'flag'),  # Bit 16 on  : normals have magnitudes.
                   # Bits 20-23 are resolution : 0 for high, 
                   #  1 for lower resolution, etc.
(short,  'type'),  # Contains a time index or 0
(short,  'pad'),   # Contains a surface number or 0
                             #  type and pad are stored as ints in the
                             #  internal data structure but shorts in the file
((float,'vsize',3), 'vert'),  # Array of points.
((int,'lsize'),   'index'), # Array of ints.
]

# -----------------------------------------------------------------------------
#
minx_format = [
(int, 'chunksize'),    # record size
((float,3), 'oscale'), # Old scale values (unused in file) 
((float,3), 'otrans'), # Old translations (file stores image origin values)
((float,3), 'orot'), # Old rotations around X, Y, Z axes (unused in file)
((float,3), 'cscale'), # New scale values in X, Y, Z
((float,3), 'ctrans'), # New translations in X, Y, Z
((float,3), 'crot'), # New rotations around X, Y, Z axes
]

# -----------------------------------------------------------------------------
#
ilabel = [
(int, 'index'), # point index.
(int, 'size'), # size of label string.
((uchar,'size',1,4), 'label'), # point label string padded to 4 byte chunks.
]

# -----------------------------------------------------------------------------
#
labl_format = [
(int, 'chunksize'),    # record size
(int, 'nlbl'), # Number of label items for points in contour.
(int, 'clsize'), # size of contour label
((uchar,'clsize',1,4), 'clabel'), # contour label string padded to 4 byte chunks.
((ilabel, 'nlbl'), 'plabels'), # label for each point
]

# -----------------------------------------------------------------------------
#
olbl_format = [
(int, 'chunksize'),    # record size
(int, 'nlbl'), # number of label items for surfaces in object
(int, 'size'), # size of top-level label = 0 (unused)
((uchar,'size',1,4), 'label'), # unused
((ilabel, 'nlbl'), 'slabels'), # surface labels
]

# -----------------------------------------------------------------------------
#
clip_format = [
(int, 'chunksize'),    # record size
(uchar, 'count'), # Number of clipping planes, can be 0 in which case there
                  #  is actually one normal and point.
(uchar, 'flags'), # Which clipping planes are on; for object planes,
                  #   bit 7 means ignore global planes
(uchar, 'trans'), # Transparency for clipped area. (future)
(uchar, 'plane'), # Current clipping plane.
((float,'count',3), 'normal'), # Normals to clipping planes
((float,'count',3), 'point'), # Point values of clipping planes
]

# -----------------------------------------------------------------------------
#
mclp_format = clip_format

# -----------------------------------------------------------------------------
#
imat_format = [
(int, 'chunksize'),    # record size
(uchar, 'ambient'), # Ambient property.  Range 0-255, scaled to 0-1, multiplied
                    #  by red, green and blue to set the GL_AMBIENT property.
(uchar, 'diffuse'), # Diffuse property.  Range 0-255, scaled to 0-1, multiplied
                    #  by red, green and blue to set the GL_DIFFUSE property.
(uchar, 'specular'), # Specular property.  Range 0-255, scaled to 0-1, added
                     #  to red, green and blue to set the GL_SPECULAR property.
(uchar, 'shininess'), # Shininess exponent in specular term.  Range 0-255, scaled
                      #  to 6.1 to 1.1 to set the GL_SHININESS property.

(uchar, 'fillred'),    # Fill color red.
(uchar, 'fillgreen'),  # Fill color green.
(uchar, 'fillblue'),   # Fill color blue.
(uchar, 'quality'),    # Sphere quality.
(uint,  'mat2'),       # Set to 0, use as flags.  Unused.
(uchar, 'valblack'),   # Black level for showing stored values.
(uchar, 'valwhite'),   # White level for showing stored values.
(uchar, 'matflags2'),  # Flags: bit 0 on: skip low end data in value draw 
                       #  bit 1 on: skip high end data in value draw
                       #  bit 2 on: keep color constant, not varied with value
(uchar, 'mat3b3'),     # Unused.
]

# -----------------------------------------------------------------------------
# TODO: psize comes from preceding CONT chunk.  Ugh.
#
size_format = [
(int, 'chunksize'),    # record size
#((float,'psize'), 'sizes'), # Size value for each point
((uchar, 'chunksize'), 'data'),
]

# -----------------------------------------------------------------------------
#
vinfo = [
(uint,      'flags'), #        bit flags IMOD_OBJFLAG... (see above)
(float,     'red'), #          Red (0 - 1.0)                 
(float,     'green'), #        Green (0 - 1.0)               
(float,     'blue'), #         Blue (0 - 1.0)                
(int,       'pdrawsize'), #    size to draw scattered objs   
(uchar,     'linewidth'), #    linewidth in 3-D              
(uchar,     'linesty'), #      line draw style               
(uchar,     'trans'), #        transparency                 
(uchar,     'clips.count'), #  number of additional clip planes. 
(uchar,     'clips.flags'), #  Which clip planes are on.         
(uchar,     'clips.trans'), #  Transparency for clipped area    
(uchar,     'clips.plane'), #  Current clip plane
((float,3), 'clips.normal'), # Normal for first clipping plane
((float,3), 'clips.point'), #  Point value for first clipping plane
(uchar, 	'ambient'), #      Ambient property.  (see above)
(uchar, 	'diffuse'), #      Diffuse property.  (see above)
(uchar, 	'specular'), #     Specular property.  (see above)
(uchar, 	'shininess'), #    Shininess exponent in specular term.  (see above)
(uchar,     'fillred'), #      Fill color red.
(uchar,     'fillgreen'), #    Fill color green.
(uchar,     'fillblue'), #     Fill color blue.
(uchar,     'quality'), #      Sphere quality.
(uint,      'mat2'), #         Set to 0, use as flags.  Unused.
(uchar,     'valblack'), #     Black level for showing stored values.
(uchar,     'valwhite'), #     White level for showing stored values.
(uchar,     'mat3b2'), #       Flags: bit 0 on: skip low end data in value draw
                         #     bit 1 on: skip high end data in value draw
(uchar,     'mat3b3'), #       Unused.
((float,5,3), 'clips.normal'), # Normals for clipping planes 2-6
((float,5,3), 'clips.point'), #  Point values for clipping planes 2-6
]

# -----------------------------------------------------------------------------
#
view_format = [
(int, 'chunksize'),    # record size
(float,     'fovy'), #    field of view of camera, perspective in degrees.  
(float,     'rad'), #     viewing radius of sphere encloseing bounding box. 
(float,     'aspect'), #  aspect ratio 
(float,     'cnear'), #   clip near: range 0.0 to 1.0, default 0.0. 
(float,     'cfar'), #    clip far: range 0.0 to 1.0 
((float,3),  'rot'), #     Model transformation values for model view: rotation
((float,3),  'trans'), #   translation
((float,3),  'scale'), #   scale
((float,16), 'mat'), #    World OpenGL transformation matrix
(int,       'world'), #   flags for lighting and transformation properties
                 # Bit 1 on : Light - draw with lighting
                 # Bit 2 on : Depth cue - draw with depth cue
                 # Bit 3 on : Wireframe - draw wireframes, not surfaces
                 # Bit 7 on : Draw lower resolution meshes
                 # Bits 8,9,10 : Global point quality
                 # Bit 11 on : Set Z clipping planes farther out
                 # Bit 12 on : Adjust all clipping planes together
((uchar,32),  'label'), #   Name for the view
(float,     'dcstart'), # Fog starting distance
(float,     'dcend'), #   Fog ending distance
(float,     'lightx'), #  X coordinate of light
(float,     'lighty'), #  Y coordinate of light
(float,     'plax'), #    Parallax angle for stereo
(int,       'objvsize'), #  Number of Iobjview structures following
(int,       'bytesObjv'), # Bytes per Iobjview structure
((vinfo, 'objvsize'), 'viewinfo'),  # Data per object view
]

# -----------------------------------------------------------------------------
#
most_format = [
(int, 'chunksize'),    # record size
(short, 'type'), # Type of information.  Currently defined types are:
                 #         1  Color change
                 #         2  Fill color change
                 #         3  Transparency change
                 #         4  Do not connect to next point or do not display
                 #         5  A connection number for meshing
                 #         6  3D line width change
                 #         7  2D line width change
                 #         8  Symbol type
                 #         9  Symbol size
                 #         10 General floating point value
                 #         11 Min and max of general values
(short, 'flags'), # 16 bits of flags.  Currently defined flags are:
                  #        Bits 0 and 1 indicate type of "index" and 2 and 3
                  #          indicate type of "value", with 0 for int, 1 for
                  #          float, 2 for short and 3 for byte
                  #        Bit 4 on : "index" is not really an index
                  #        Bit 5 on : Revert to default value
                  #        Bit 6 on : Index is surface number, not contour
                  #        Bit 7 on : The change applies to only one point
(int, 'index'), # Can contain one int or float, 2 shorts, or 4 bytes,
                #   generally has index of point or contour, or surface #
                #   has minimum for the min/max type
(int, 'value'), # Can contain one int or float, 2 shorts, or 4 bytes
                #   For colors, 3 bytes hold red, green, and blue
                #   Transparency is stored as an int from 0 to 255
                #   Symbol type is 0, 2, 3 for open circles, squares, or
                #     triangles, -1, -3, -4 for closed circles, squares,
                #     or triangles
                #   For min/max type, this holds the max
]

# -----------------------------------------------------------------------------
#
obst_format = most_format
cost_format = most_format
mest_format = most_format

# -----------------------------------------------------------------------------
#
slan_format = [
(int, 'chunksize'),    # record size
(int, 'time'), # Time value of image file to which angles apply
((float,3), 'angles'), # Rotation angles around X, Y, and Z axes
((float,3), 'center'), # Center coordinate of volume in slicer display
((uchar,32), 'label'), # Text label
]

# -----------------------------------------------------------------------------
#
optional_chunk_format = [
(int, 'chunk_size'),
((uchar,'chunk_size'), 'data'),
]


# -----------------------------------------------------------------------------
#
chunk_formats = {
    b'IMOD': model_format,
    b'OBJT': object_format,
    b'CONT': contour_format,
    b'MESH': mesh_format,
    b'MINX': minx_format,
    b'LABL': labl_format,
    b'OLBL': olbl_format,
#    b'CLIP': clip_format,       # count field not used consistently.
#    b'MCLP': mclp_format,
    b'IMAT': imat_format,
    b'SIZE': size_format,
#    b'VIEW': view_format,       # record fields vary with imod version
#    b'MOST': most_format,     # variable size record not supported
#    b'OBST': obst_format,     # variable size record not supported
#    b'COST': cost_format,     # variable size record not supported
#    b'MEST': mest_format,     # variable size record not supported
    b'SLAN': slan_format,
}
eof_id = b'IEOF'
