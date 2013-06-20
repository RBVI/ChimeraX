# -----------------------------------------------------------------------------
# Classes to hold the state of some Chimera objects.  These are for use with
# SessionUtil/objecttree.py and the SimpleSession module for saving and
# restoring the state of extensions.
#

# -----------------------------------------------------------------------------
#
class Model_State:

  version = 4

  state_attributes = ('name', 'id', 'subid', 'osl_identifier',
		      'display', 'xform', 'active',
                      'use_clip_plane', 'use_clip_thickness', 'clip_thickness',
                      'clip_plane_origin', 'clip_plane_normal',
                      'version')
  
  # ---------------------------------------------------------------------------
  #
  def state_from_model(self, model):

    self.name = model.name
    self.id = model.id
    self.subid = model.subid
    self.osl_identifier = model.oslIdent()
    self.display = model.display
    self.xform = Xform_State()
    self.xform.state_from_xform(model.openState.xform)
    self.active = model.openState.active
    self.use_clip_plane = model.useClipPlane
    self.use_clip_thickness = model.useClipThickness
    self.clip_thickness = model.clipThickness
    p = model.clipPlane
    self.clip_plane_origin = p.origin.data()
    self.clip_plane_normal = p.normal.data()
      
  # ---------------------------------------------------------------------------
  #
  def restore_state(self, model):

    model.name = self.name
    model.display = self.display
    model.openState.xform = self.xform.create_object()
    model.openState.active = self.active

    #
    # Record how model id number has been remapped.
    #
    if self.version >= 2:
      import SimpleSession
      if hasattr(SimpleSession, 'modelMap'):
        mid = (self.id, self.subid)
        SimpleSession.modelMap.setdefault(mid, []).append(model)
    else:
      from SimpleSession import updateOSLmap
      updateOSLmap(self.osl_identifier, model.oslIdent())

    if self.version >= 3:
      p = model.clipPlane
      import chimera
      p.origin = chimera.Point(*self.clip_plane_origin)
      n = chimera.Vector(*self.clip_plane_normal)
      if n.length == 0:
        n = chimera.Vector(0,0,-1)
      p.normal = n
      model.clipPlane = p
      model.clipThickness = self.clip_thickness
      model.useClipPlane = self.use_clip_plane
      if self.version >= 4:
        model.useClipThickness = self.use_clip_thickness

# -----------------------------------------------------------------------------
#
class Xform_State:

  version = 1

  state_attributes = ('translation', 'rotation_axis', 'rotation_angle',
		      'version')
  
  # ---------------------------------------------------------------------------
  #
  def state_from_xform(self, xform):

    t = xform.getTranslation()
    self.translation = (t.x, t.y, t.z)
    axis, angle = xform.getRotation()
    self.rotation_axis = (axis.x, axis.y, axis.z)
    self.rotation_angle = angle
    
  # ---------------------------------------------------------------------------
  #
  def create_object(self):

    import chimera
    xf = chimera.Xform()
    trans = apply(chimera.Vector, self.translation)
    xf.translate(trans)
    axis = apply(chimera.Vector, self.rotation_axis)
    xf.rotate(axis, self.rotation_angle)
    return xf


# -----------------------------------------------------------------------------
# Use integer identifiers for numeric array so they can appear more than once
# in a session without duplicating the array data.
#
class Arrays_State:

  version = 1

  state_attributes = ('id_to_encoded_array', 'version')

  def __init__(self):

    self.id_to_encoded_array = {}
    self.array_to_id = {}
    self.id_to_array = {}

  def id(self, array, atype):

    if array is None:
      return None
    
    key = (id(array), atype)
    a2id = self.array_to_id
    if key in a2id:
      aid = a2id[key]
    else:
      a2id[key] = aid = len(a2id)
      from SessionUtil import encoded_array
      self.id_to_encoded_array[aid] = encoded_array(array, atype)
    return aid

  def array(self, id):

    if id is None:
      return None

    a = self.id_to_array.get(id)
    if a is None:
      ea = self.id_to_encoded_array[id]
      from SessionUtil import decoded_array
      a = decoded_array(ea)
      self.id_to_array[id] = a
    return a
    
