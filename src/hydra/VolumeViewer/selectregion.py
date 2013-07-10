# -----------------------------------------------------------------------------
# Mouse mode for selecting a volume subregion by dragging a 3D box.
#

# -----------------------------------------------------------------------------
#
class Select_Volume_Subregion:

  def __init__(self, box_changed_cb, box_name=None, save_in_session = True):

    self.box_model = Box_Model(box_name, save_in_session)
    self.box_changed_cb = box_changed_cb
    self.ijk_in = None                          # Box coordinates
    self.ijk_out = None
    self.mode_name = 'volume subregion'         # For mouse modes panel
    self.bound_button = None
    self.drag_mode = None
    self.rotation_handler = None
    
    callbacks = (self.mouse_down_cb, self.mouse_drag_cb, self.mouse_up_cb)

    icon_file = 'selectregion.gif'
    if icon_file:
      import os.path
      icon_path = os.path.join(os.path.dirname(__file__), icon_file)
      from PIL import Image
      image = Image.open(icon_path)
      from chimera import chimage
      from chimera import tkgui
      icon = chimage.get(image, tkgui.app)
    else:
      icon = None
      
    from chimera import mousemodes
    mousemodes.addFunction(self.mode_name, callbacks, icon)

  # ---------------------------------------------------------------------------
  # Bind mouse button to do region selection
  # , or restore it to the default binding.
  #
  def bind_mouse_button(self, button, modifiers):

    self.unbind_mouse_button()
    from chimera import mousemodes
    mousemodes.setButtonFunction(button, modifiers, self.mode_name)
    self.bound_button = (button, modifiers)
    
  # ---------------------------------------------------------------------------
  # Restore mouse button binding to the default binding.
  #
  def unbind_mouse_button(self):

    if self.bound_button:
      button, modifiers = self.bound_button
      from chimera import mousemodes
      def_mode = mousemodes.getDefault(button, modifiers)
      if def_mode:
        mousemodes.setButtonFunction(button, modifiers, def_mode)
      self.bound_button = None
      
  # ---------------------------------------------------------------------------
  #
  def mouse_down_cb(self, viewer, event):

    self.drag_occurred = False
    if self.box_model.box_shown():
      bm = self.box_model
      faces = bm.pierced_faces(event.x, event.y)
      if len(faces) == 0:
        self.drag_mode = 'move box'
      else:
        self.drag_mode = 'move face'
        shift_mask = 1
        shift = (event.state & shift_mask)
        if shift and len(faces) >= 2:           face = faces[1]
        else:                                   face = faces[0]
        self.drag_face = face
        nx, ny, nz = bm.face_normal(face[0], 1)
        from ..Matrix import sign
        if abs(nx) > abs(ny):
          self.face_direction = (sign(nx), 0)
        else:
          self.face_direction = (0, sign(ny))
      self.last_xy = (event.x, event.y)
    else:
      self.delete_box()      # box model may exist but be undisplayed
      self.drag_mode = 'create'
      self.ijk_in = None
      self.ijk_out = None
      self.sweep_out_box(event.x, event.y)
    
  # ---------------------------------------------------------------------------
  #
  def mouse_drag_cb(self, viewer, event):

    self.drag_occurred = True
    if self.drag_mode == 'move box':
      shift_mask = 1
      shift = (event.state & shift_mask)
      self.move_box(event.x, event.y, shift)
    if self.drag_mode == 'move face':
      self.move_face(event.x, event.y)
    elif self.drag_mode == 'create':
      self.sweep_out_box(event.x, event.y)
      
  # ---------------------------------------------------------------------------
  #
  def mouse_up_cb(self, viewer, event):

    m = self.drag_mode
    if not self.drag_occurred:
      if (m == 'create' and not self.box_model.box_shown()) or m == 'move box':
        self.toggle_box()

    self.drag_mode = None
    if m == 'create':
      self.box_changed()
      
  # ---------------------------------------------------------------------------
  #
  def toggle_box(self):

    if self.box_model.box_shown():
      self.delete_box()
    else:
      self.create_box()
    self.box_changed()

  # ---------------------------------------------------------------------------
  #
  def create_box(self, box = None):

    fbox, transform, xform = self.box_transform_and_xform(self.volume())
    if box is None:
      box = fbox
    if box and xform:
      self.box_model.reshape_box(box, transform, xform)
      if self.rotation_handler:
        self.activate_models(False)

  # ---------------------------------------------------------------------------
  #
  def delete_box(self):

    self.box_model.delete_box()
    if self.rotation_handler:
      self.activate_models(True)

  # ---------------------------------------------------------------------------
  #
  def rotate_box(self, rotate):

    self.activate_models(not rotate)

    from chimera import triggers as t
    rt = self.rotation_handler
    if rotate:
      if rt is None:
        self.rotation_handler = t.addHandler('OpenState', self.rotation_cb, None)
    elif rt:
      t.deleteHandler('OpenState', rt)
      self.rotation_handler = None

  # ---------------------------------------------------------------------------
  #
  def activate_models(self, active):

    from chimera import openModels
    for m in openModels.list():
      m.openState.active = active

    m = self.box_model.model()
    if m:
      m.openState.active = True

  # ---------------------------------------------------------------------------
  #
  def rotation_cb(self, tname, tdata, changes):

    if self.box_changed_cb:
      if 'transformation change' in changes.reasons:
        m = self.box_model.model()
        if m and m.openState in changes.modified:
          self.box_changed()

  # ---------------------------------------------------------------------------
  #
  def box_changed(self):

    if self.box_changed_cb:
      self.box_changed_cb(initial_box = (self.drag_mode == 'create'))

  # ---------------------------------------------------------------------------
  #
  def volume(self):

    from .volumedialog import active_volume
    v = active_volume()
    return v

  # ---------------------------------------------------------------------------
  #
  def sweep_out_box(self, pointer_x, pointer_y):

    box, tf, xform = self.box_transform_and_xform(self.volume())
    if box == None:
      return
    
    transform = box_to_eye_transform(tf, xform)
    
    from . import slice
    ijk_in, ijk_out = slice.box_intercepts(pointer_x, pointer_y,
                                           transform, box)
    if ijk_in == None or ijk_out == None:
      return
    
    if self.ijk_in == None:
      self.ijk_in = ijk_in
    if self.ijk_out == None:
      self.ijk_out = ijk_out
      
    drag_box = bounding_box((ijk_in, ijk_out, self.ijk_in, self.ijk_out))
    self.create_box(drag_box)
    
  # ---------------------------------------------------------------------------
  #
  def move_box(self, pointer_x, pointer_y, shift):

    if self.last_xy is None:
      return

    last_x, last_y = self.last_xy
    dx = pointer_x - last_x
    dy = pointer_y - last_y
    self.last_xy = (pointer_x, pointer_y)

    bm = self.box_model
    d = bm.view_distance()
    psize = pixel_size(d)
    
    if shift:
      if abs(dx) > abs(dy):     dz = dx
      else:                     dz = dy
      delta_xyz = (0, 0, dz * psize)
    else:
      delta_xyz = (dx * psize, -dy * psize, 0)
    
    bm.move_box(delta_xyz)
    self.box_changed()
    
  # ---------------------------------------------------------------------------
  #
  def move_face(self, pointer_x, pointer_y):

    if self.last_xy is None:
      return

    last_x, last_y = self.last_xy
    dx = pointer_x - last_x
    dy = pointer_y - last_y
    self.last_xy = (pointer_x, pointer_y)

    bm = self.box_model
    d = bm.view_distance()
    psize = pixel_size(d)
    fd = self.face_direction
    dist = (dx * fd[0] + -dy * fd[1]) * psize

    axis, side = self.drag_face
    bm.move_face(axis, side, dist)
    self.box_changed()
    
  # ---------------------------------------------------------------------------
  #
  def ijk_box_bounds(self, xform, ijk_to_xyz_transform = None):

    bm = self.box_model
    if bm.box is None:
      return None, None
    if ijk_to_xyz_transform is None:
      ijk_to_xyz_transform = bm.box_transform
    corners = box_corners(bm.box)
    from ..Matrix import multiply_matrices, apply_matrix
    tf = multiply_matrices(eye_to_box_transform(ijk_to_xyz_transform, xform),
                           box_to_eye_transform(bm.box_transform, bm.xform()))
                           
    ijk_box = bounding_box([apply_matrix(tf, c) for c in corners])
    return ijk_box
    
  # ---------------------------------------------------------------------------
  # Create a new grid data object with zero values within the selection box.
  #
  def subregion_grid(self, voxel_size, xform, name):

    bm = self.box_model
    if bm is None or bm.box is None or bm.model() is None:
      return None

    # Deterime array size.  Use half voxel padding on all sides.
    elength = bm.edge_lengths()
    from math import ceil
    size = [max(1,int(ceil((elength[a]-voxel_size[a])/voxel_size[a])))
            for a in (0,1,2)]

    # Allocate array.
    from ..VolumeData import allocate_array
    array = allocate_array(size, zero_fill = True)

    # Determine origin, rotation, and cell angles.
    b2vxf = bm.xform()
    b2vxf.premultiply(xform.inverse())
    from ..Matrix import xform_matrix, apply_matrix, apply_matrix_without_translation, cell_angles_and_rotation
    b2v = xform_matrix(b2vxf)
    origin = apply_matrix(b2v, bm.origin())
    vaxes = [apply_matrix_without_translation(b2v, v) for v in bm.axes()]
    cell_angles, rotation = cell_angles_and_rotation(vaxes)

    # Create grid.
    from ..VolumeData import Array_Grid_Data
    g = Array_Grid_Data(array, origin, voxel_size, cell_angles, rotation,
                        name = name)

    # Add half voxel padding.
    g.set_origin(g.ijk_to_xyz((0.5,0.5,0.5)))

    return g
    
  # ---------------------------------------------------------------------------
  #
  def box_transform_and_xform(self, model):

    return box_transform_and_xform(model)
  
# -----------------------------------------------------------------------------
#
def box_transform_and_xform(v):

  if v is None:
    return None, None, None

  xform = v.model_transform()
  if xform == None:
    return None, None, None

  tf = v.data.ijk_to_xyz_transform
  box = v.ijk_bounds()

  return box, tf, xform

# -----------------------------------------------------------------------------
# Combine transform of ijk_box to object coordinates and model xform mapping
# object coordinates to eye coordinates
#
def box_to_eye_transform(box_transform, model_transform):

  from ..Matrix import invert_matrix, multiply_matrices
  # TODO: requires viewer
  m2c = invert_matrix(viewer.camera_view)
  transform = multiply_matrices(m2c, model_transform, box_transform)
  return transform

# -----------------------------------------------------------------------------
# Eye to box coordinate transform.
#
def eye_to_box_transform(box_transform, xform):

  tf = box_to_eye_transform(box_transform, xform)
  from ..Matrix import invert_matrix
  inv_tf = invert_matrix(tf)
  return inv_tf

# -----------------------------------------------------------------------------
#
class Box_Model:

  def __init__(self, name=None, save_in_session=True):

    if name is None:
      self.name = "Subregion Selection Box"
    else:
      self.name = name
    self.save_in_session = save_in_session
    self.corner_atoms = None
    self.box = None             # ijk_min, ijk_max
    self.box_transform = None   # box ijk to local molecule xyz coordinates

  # ---------------------------------------------------------------------------
  # Box corner postion in local coordinates.
  #
  def origin(self):

    c0 = transform_box_corners(self.box, self.box_transform)[0]
    p = tuple(c0)
    return p

  # ---------------------------------------------------------------------------
  #
  def create_box(self):

    m = chimera.Molecule()
    m.name = self.name
    m.color = chimera.MaterialColor(0,1,0,1)
    m.isRealMolecule = False
    if not self.save_in_session:
      import SimpleSession
      SimpleSession.noAutoRestore(m)
    chimera.openModels.add([m], noprefs = True)
    chimera.addModelClosedCallback(m, self.model_closed_cb)

    rid = chimera.MolResId(1)
    r = m.newResidue('markers', rid)

    corners = []
    for name in ('a000', 'a001', 'a010', 'a011',
                 'a100', 'a101', 'a110', 'a111'):
      a = m.newAtom(name, chimera.elements.H)
      r.addAtom(a)
      corners.append(a)

    for a1, a2 in ((0,1), (2,3), (4,5), (6,7),
                   (0,2), (1,3), (4,6), (5,7),
                   (0,4), (1,5), (2,6), (3,7)):
      b = m.newBond(corners[a1], corners[a2])
      b.drawMode = chimera.Bond.Wire

    return corners
    
  # ---------------------------------------------------------------------------
  #
  def model(self):
    
    if self.corner_atoms:
      return self.corner_atoms[0].molecule
    return None
    
  # ---------------------------------------------------------------------------
  #
  def xform(self):
    
    m = self.model()
    if m == None:
      return None

    xf = m.openState.xform
    return xf
    
  # ---------------------------------------------------------------------------
  #
  def display_box(self, show):

    m = self.model()
    if m:
      m.display = show
    
  # ---------------------------------------------------------------------------
  #
  def box_shown(self):

    m = self.model()
    return m != None and m.display
    
  # ---------------------------------------------------------------------------
  #
  def delete_box(self):

    if self.corner_atoms:
      m = self.model()
      chimera.openModels.close([m])
      self.corner_atoms = None
    
  # ---------------------------------------------------------------------------
  #
  def reshape_box(self, box, box_transform, xform = None):

    if self.corner_atoms == None:
      self.corner_atoms = self.create_box()

    if xform:
      # Make sure box transform is same as volume model
      m = self.model()
      m.openState.xform = xform

    corners = transform_box_corners(box, box_transform)
    c = chimera.Coord()
    for k in range(8):
      c.x, c.y, c.z = corners[k]
      self.corner_atoms[k].setCoord(c)

    self.display_box(1)

    self.box = box
    self.box_transform = box_transform
    
  # ---------------------------------------------------------------------------
  # Move atom positions.  Don't change xform.
  #
  def move_box(self, delta_xyz):

    xf = self.xform()
    if xf == None or self.box == None:
      return
    tf = eye_to_box_transform(self.box_transform, xf)
    from ..Matrix import apply_matrix_without_translation
    shift = apply_matrix_without_translation(tf, delta_xyz)
    box = translate_box(self.box, shift)
    self.reshape_box(box, self.box_transform)
    
  # ---------------------------------------------------------------------------
  # Move atom positions for one face.  Delta is in eye distance units.
  #
  def move_face(self, axis, side, delta_eye):

    # Figure out delta in box coordinates
    axis_vector = [0,0,0]
    axis_vector[axis] = 1
    from ..Matrix import apply_matrix_without_translation, length
    scale = length(apply_matrix_without_translation(self.box_transform,
                                                    axis_vector))
    delta_box = delta_eye / scale
    
    box = map(list, self.box)
    box[side][axis] += delta_box

    if box[0][axis] > box[1][axis]:
      min, max = box[1][axis], box[0][axis]
      box[0][axis], box[1][axis] = min, max

    self.reshape_box(box, self.box_transform)
      
  # ---------------------------------------------------------------------------
  # Return z distance from eye to center of box or near clip plane distance,
  # whichever is larger.
  #
  def view_distance(self):

    xf = self.xform()
    if xf == None or self.box == None:
      d = near_clip_plane_distance()
    else:
      tf = box_to_eye_transform(self.box_transform, xf)
      from ..Matrix import apply_matrix
      center = apply_matrix(tf, box_center(self.box))
      z_box = center[2]
      eye_number = 0
      z_eye = chimera.viewer.camera.eyePos(eye_number)[2]
      z_range = z_eye - z_box
      d = max(z_range, near_clip_plane_distance())

    return d
  
  # ---------------------------------------------------------------------------
  # Outward normal in eye coordinates.
  #
  def face_normal(self, axis, side):
    
    xform = self.xform()
    if xform == None:
      return None

    from ..Matrix import invert_matrix, xform_matrix
    from ..Matrix import apply_matrix_without_translation, normalize_vector
    inv_s = invert_matrix(self.box_transform)
    n = inv_s[axis,:3]
    if side == 0:
      n = -n
    model_transform = xform_matrix(xform)
    ne = apply_matrix_without_translation(model_transform, n)
    ne = normalize_vector(ne)
    
    return ne
    
  # ---------------------------------------------------------------------------
  #
  def pierced_faces(self, screen_x, screen_y):

    xf = self.xform()
    if xf == None or self.box == None:
      return []

    transform = box_to_eye_transform(self.box_transform, xf)

    from . import slice
    ijk_in, ijk_out = slice.box_intercepts(screen_x, screen_y,
                                           transform, self.box)

    faces = []
    if ijk_in:
      faces.append(self.closest_face(ijk_in, self.box))
    if ijk_out:
      faces.append(self.closest_face(ijk_out, self.box))

    return faces
    
  # ---------------------------------------------------------------------------
  #
  def closest_face(self, ijk, box):

    closest_dist = None
    for axis in range(3):
      for side in range(2):
        d = abs(ijk[axis] - box[side][axis])
        if closest_dist == None or d < closest_dist:
          closest_dist = d
          closest = (axis, side)
    return closest
      
  # ---------------------------------------------------------------------------
  #
  def edge_lengths(self):

    ijk_min, ijk_max = self.box
    s = [ijk_max[a] - ijk_min[a] + 1 for a in (0,1,2)]
    from ..Matrix import transpose, length
    ev = transpose(self.box_transform)[:3]              # Edge vectors
    el = tuple([length(ev[a])*s[a] for a in (0,1,2)])
    return el
      
  # ---------------------------------------------------------------------------
  #
  def axes(self):

    from ..Matrix import transpose
    a = transpose(self.box_transform)[:3]
    return a
  
  # ---------------------------------------------------------------------------
  #
  def model_closed_cb(self, model):

    self.corner_atoms = None
    self.box = None
    self.box_transform = None

# -----------------------------------------------------------------------------
#
def transform_box_corners(box, transform):

  corners = box_corners(box)
  from ..Matrix import apply_matrix
  tcorners = map(lambda p: apply_matrix(transform, p), corners)
  return tcorners

# -----------------------------------------------------------------------------
#
def translate_box(box, offset):

  box_min, box_max = box
  shifted_box = (map(lambda a,b: a+b, box_min, offset),
                 map(lambda a,b: a+b, box_max, offset))
  return shifted_box

# -----------------------------------------------------------------------------
#
def box_center(box):

  ijk_min, ijk_max = box
  c = map(lambda a,b: .5*(a+b), ijk_min, ijk_max)
  return c

# -----------------------------------------------------------------------------
#
def box_corners(box):
  
  corners = []
  for i0,i1,i2 in ((0,0,0), (0,0,1), (0,1,0), (0,1,1),
                   (1,0,0), (1,0,1), (1,1,0), (1,1,1)):
    c = (box[i0][0], box[i1][1], box[i2][2])
    corners.append(c)
  return corners
  
# -----------------------------------------------------------------------------
#
def bounding_box(points):

  xyz_min = [None, None, None]
  xyz_max = [None, None, None]
  for p in points:
    for a in range(3):
      if xyz_min[a] == None or p[a] < xyz_min[a]:
        xyz_min[a] = p[a]
      if xyz_max[a] == None or p[a] > xyz_max[a]:
        xyz_max[a] = p[a]

  return xyz_min, xyz_max
  
# -----------------------------------------------------------------------------
#
def near_clip_plane_distance():

  c = chimera.viewer.camera
  eye_number = 0
  left, right, bottom, top, znear, zfar, f = c.window(eye_number)
  return znear
  
# -----------------------------------------------------------------------------
#
def pixel_size(view_distance):

  v = chimera.viewer
  c = v.camera
  eye_number = 0
  llx, lly, width, height = c.viewport(eye_number)
  left, right, bottom, top, znear, zfar, f = c.window(eye_number)
  pscale = v.pixelScale

  psize = float(right - left) * pscale / width
  if not c.ortho:
    zratio = view_distance / znear
    psize = psize * zratio

  return psize
