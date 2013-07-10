# -----------------------------------------------------------------------------
# Save and restore volume viewer state.
#
  
# -----------------------------------------------------------------------------
# Saves volume dialog state, but not data regions.
#
def save_volume_dialog_state(volume_dialog, file):

  s = Volume_Dialog_State()
  s.state_from_dialog(volume_dialog)
  
  from ..SessionUtil import objecttree
  t = objecttree.instance_tree_to_basic_tree(s)

  file.write('\n')
  file.write('def restore_volume_dialog():\n')
  file.write(' volume_dialog_state = \\\n')
  objecttree.write_basic_tree(t, file, indent = '  ')
  file.write('\n')
  file.write(' from VolumeViewer import session\n')
  file.write(' session.restore_volume_dialog_state(volume_dialog_state)\n')
  file.write('\n')
  file.write('try:\n')
  file.write('  restore_volume_dialog()\n')
  file.write('except:\n')
  file.write("  reportRestoreError('Error restoring volume dialog')\n")
  file.write('\n')
  
# -----------------------------------------------------------------------------
# Saves volume data sets, but not dialog state.  Can be used in nogui mode.
#
def save_volume_data_state(volume_manager, file):

  s = Volume_Manager_State()
  s.state_from_manager(volume_manager)

  from os.path import dirname
  directory = dirname(file.fileName)
  if directory:
    s.use_relative_paths(directory)
  
  from ..SessionUtil import objecttree
  t = objecttree.instance_tree_to_basic_tree(s)

  file.write('\n')
  file.write('def restore_volume_data():\n')
  file.write(' volume_data_state = \\\n')
  objecttree.write_basic_tree(t, file, indent = '  ')
  file.write('\n')
  file.write(' from VolumeViewer import session\n')
  file.write(' session.restore_volume_data_state(volume_data_state)\n')
  file.write('\n')
  file.write('try:\n')
  file.write('  restore_volume_data()\n')
  file.write('except:\n')
  file.write("  reportRestoreError('Error restoring volume data')\n")
  file.write('\n')
  
# -----------------------------------------------------------------------------
# This routine name was used in older session files.
#
def restore_volume_state(volume_dialog_basic_state):

  restore_volume_dialog_state(volume_dialog_basic_state)
  
# -----------------------------------------------------------------------------
#
def restore_volume_dialog_state(volume_dialog_basic_state):

  from chimera import nogui
  if nogui:
    return

  vds = volume_dialog_state_from_basic_tree(volume_dialog_basic_state)

  from .volumedialog import volume_dialog
  d = volume_dialog(create = True)

  vds.restore_state(d)

# -----------------------------------------------------------------------------
#
def save_scene(volume_manager, scene):

  vms = Volume_Manager_State()
  vms.state_from_manager(volume_manager, include_unsaved_volumes=True)
  from ..SessionUtil import objecttree
  s = objecttree.instance_tree_to_basic_tree(vms)
  scene.tool_settings['VolumeViewer'] = s

# -----------------------------------------------------------------------------
#
def restore_scene(volume_manager, scene):

  s = scene.tool_settings.get('VolumeViewer')
  if s:
    vms = volume_manager_state_from_basic_tree(s)
    vms.set_attributes()

# -----------------------------------------------------------------------------
#
def volume_dialog_state_from_basic_tree(volume_dialog_basic_state):
  
  from ..SessionUtil.stateclasses import Model_State, Xform_State

  classes = (
    Volume_Dialog_State,
    Data_Set_State,
    Data_State,
    Volume_State,
    Region_List_State,
    Rendering_Options_State,
    Component_Display_Parameters_State,
    Model_State,
    Xform_State,
    )
  name_to_class = {}
  for c in classes:
    name_to_class[c.__name__] = c
  name_to_class['Data_Region_State'] = Volume_State     # Historical name

  from ..SessionUtil import objecttree
  vds = objecttree.basic_tree_to_instance_tree(volume_dialog_basic_state,
                                               name_to_class)
  return vds
  
# -----------------------------------------------------------------------------
#
def restore_volume_data_state(volume_data_basic_state):

  vms = volume_manager_state_from_basic_tree(volume_data_basic_state)
  from .volume import volume_manager
  vms.restore_state(volume_manager)

# -----------------------------------------------------------------------------
#
def volume_manager_state_from_basic_tree(volume_data_basic_state):
  
  from ..SessionUtil.stateclasses import Model_State, Xform_State

  classes = (
    Volume_Manager_State,
    Data_State,
    Volume_State,
    Region_List_State,
    Rendering_Options_State,
    Model_State,
    Xform_State,
    )
  name_to_class = {}
  for c in classes:
    name_to_class[c.__name__] = c
  name_to_class['Data_Region_State'] = Volume_State     # Historical name

  from ..SessionUtil import objecttree
  vms = objecttree.basic_tree_to_instance_tree(volume_data_basic_state,
                                               name_to_class)
  return vms

# -----------------------------------------------------------------------------
#
class Volume_Dialog_State:

  version = 12
  
  state_attributes = ('is_visible',
                      'geometry',
                      'shown_panels',
                      'data_cache_size',
                      'representation',
                      'max_histograms',
                      'use_initial_colors',
                      'initial_colors',
                      'immediate_update',
                      'show_on_open',
                      'voxel_limit_for_open',
                      'show_plane',
                      'voxel_limit_for_plane',
                      'selectable_subregions',
                      'subregion_button',
                      'auto_show_subregion',
                      'adjust_camera',
                      'box_padding',
                      'zone_radius',
                      'focus_volume',
                      'histogram_volumes',
                      'histogram_active_order',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_dialog(self, volume_dialog):

    d = volume_dialog
    self.is_visible = d.isVisible()
    self.geometry = d.toplevel_widget.wm_geometry()
    self.shown_panels = [p.name for p in d.shown_panels()]

    dsp = d.display_style_panel
    self.representation = dsp.representation.get()

    srp = d.subregion_panel
    self.selectable_subregions = srp.selectable_subregions.get()
    self.subregion_button = srp.subregion_button.get()
    self.auto_show_subregion = srp.auto_show_subregion.get()

    abp = d.atom_box_panel
    self.box_padding = abp.box_padding.get()

    zp = d.zone_panel
    self.zone_radius = zp.zone_radius.value()

    dop = d.display_options_panel
    self.max_histograms = dop.max_histograms.get()
    self.use_initial_colors = dop.use_initial_colors.get()
    self.initial_colors = tuple([dop.initial_colors[c].rgba for c in range(10)])
    self.data_cache_size = dop.data_cache_size.get()
    self.immediate_update = dop.immediate_update.get()
    self.show_on_open = dop.show_on_open.get()
    self.voxel_limit_for_open = dop.voxel_limit_for_open.get()
    self.show_plane = dop.show_plane.get()
    self.voxel_limit_for_plane = dop.voxel_limit_for_plane.get()
    self.adjust_camera = dop.adjust_camera.get()

    if d.focus_region:
      self.focus_volume = session_volume_id(d.focus_region)
    else:
      self.focus_volume = None

    tp = d.thresholds_panel
    hpanes = [hp for hp in tp.histogram_panes
              if hp.data_region and hp.data_region.data.path]
    self.histogram_volumes = [session_volume_id(hp.data_region)
                              for hp in hpanes]
    self.histogram_active_order = [hpanes.index(hp) for hp in tp.active_order
                                   if hp in hpanes]

    # Data sets and data regions are saved by volume manager.
    
  # ---------------------------------------------------------------------------
  #
  def restore_state(self, volume_dialog):

    d = volume_dialog
    if self.is_visible:
      d.enter()

    from ..SessionUtil import set_window_position
    set_window_position(d.toplevel_widget, self.geometry)

    if self.version >= 7:
      d.show_panels(self.shown_panels)

    if self.version >= 3:
      dop = d.display_options_panel
      dop.data_cache_size.set(self.data_cache_size, invoke_callbacks = 0)
      dop.cache_size_cb(None)
      if self.version >= 9:
        dop.max_histograms.set(self.max_histograms, invoke_callbacks = 0)
        if self.version >= 11:
          dop.set_gui_state({'use_initial_colors': self.use_initial_colors,
                             'initial_colors': self.initial_colors})
          
    dsp = d.display_style_panel
    dsp.representation.set(self.representation, invoke_callbacks = 0)

    srp = d.subregion_panel
    srp.selectable_subregions.set(self.selectable_subregions,
                                  invoke_callbacks = 0)
    # Handle button names from older Chimera versions.
    b2b = {'button 1':'left', 'ctrl button 1':'ctrl left',
           'button 2':'middle', 'ctrl button 2':'ctrl middle',
           'button 3':'right', 'ctrl button 3':'ctrl right',}
    if self.subregion_button in b2b:
      self.subregion_button = b2b[self.subregion_button]
    srp.subregion_button.set(self.subregion_button, invoke_callbacks = 0)
    srp.auto_show_subregion.set(self.auto_show_subregion, invoke_callbacks = 0)

    abp = d.atom_box_panel
    abp.box_padding.set(self.box_padding, invoke_callbacks = 0)
    
    if self.version >= 6:
      zp = d.zone_panel
      zp.zone_radius.set_value(self.zone_radius, invoke_callbacks = 0)

    dop = d.display_options_panel
    dop.immediate_update.set(self.immediate_update, invoke_callbacks = 0)
    dop.show_on_open.set(self.show_on_open, invoke_callbacks = 0)
    dop.voxel_limit_for_open.set(self.voxel_limit_for_open,
                                 invoke_callbacks = 0)
    if self.version >= 8:
      dop.show_plane.set(self.show_plane, invoke_callbacks = 0)
      dop.voxel_limit_for_plane.set(self.voxel_limit_for_plane,
                                    invoke_callbacks = 0)
    dop.adjust_camera.set(self.adjust_camera, invoke_callbacks = 0)

    # Cache of Grid_Data objects improves load speed when multiple regions
    # are using same data file.  Especially important for files that contain
    # many data arrays.
    gdcache = {}        # (path, grid_id) -> Grid_Data object
    if self.version <= 4:
      for ds in self.data_set_states:
        ds.create_object(gdcache)
    elif self.version <= 9:
      for ds, drslist in self.data_and_regions_state:
        data = ds.create_object(gdcache)
        if data:        # Can be None if user does not replace missing file.
          for drs in drslist:
            drs.create_object(data)

    # Restore order of data histograms.
    if self.version >= 10:
      tp = d.thresholds_panel
      if self.version < 12:
        self.histogram_volumes = self.histogram_region_names
      drlist = [find_volume_by_session_id(id) for id in self.histogram_volumes]
      cur_drlist = [hp.data_region for hp in tp.histogram_panes]
      if drlist != cur_drlist:
        for hp in tuple(tp.histogram_panes):
          tp.close_histogram_pane(hp)
        for v in drlist:
          tp.update_panel_widgets(v)
      hpanes = tp.histogram_panes
      tp.active_order = [hpanes[i] for i in self.histogram_active_order
                         if i >= 0 and i < len(hpanes)]

    # Set focus region.
    if self.version < 12:
      self.focus_volume = self.focus_region_name
    v = find_volume_by_session_id(self.focus_volume)
    if v:
      d.display_region_info(v)

# -----------------------------------------------------------------------------
#
def session_volume_id(v):

    # Generate a unique volume id as a random string of characters.
    if not hasattr(v, 'session_volume_id'):
      import random, string
      sid = ''.join(random.choice(string.printable) for i in range(32))
      v.session_volume_id = sid
    return v.session_volume_id

# -----------------------------------------------------------------------------
#
def find_volume_by_session_id(id):

  from .volume import volume_list
  for v in volume_list():
    if hasattr(v, 'session_volume_id') and v.session_volume_id == id:
      return v
  return None

# -----------------------------------------------------------------------------
#
class Volume_Manager_State:

  version = 2
  
  state_attributes = ('data_and_regions_state',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_manager(self, volume_manager, include_unsaved_volumes = False):

    dvlist = []
    unsaved_data = []
    for data, volumes in volume_manager.data_to_regions.items():
      if data.path == '' and not include_unsaved_volumes:
        unsaved_data.append(data)
        continue                # Do not save data sets with no path
      ds = Data_State()
      ds.state_from_data(data)
      vslist = []
      for v in volumes:
        vs = Volume_State()
        vs.state_from_data_region(v)
        vslist.append(vs)
      dvlist.append((ds, vslist))
    self.data_and_regions_state = dvlist

    if unsaved_data:
      import SimpleSession
      if getattr(SimpleSession, 'temporarySession', True):
        return
      names = ', '.join([d.name for d in unsaved_data])
      from chimera.replyobj import warning
      warning('Volume data sets\n\n' +
              '\t%s\n\n' % names +
              'were not saved in the session.  To have them included\n' +
              'in the session they must be saved in separate volume\n' +
              'files before the session is saved.  The session file only\n' +
              'records file system paths to the volume data.')

  # ---------------------------------------------------------------------------
  #
  def use_relative_paths(self, directory):

    for ds, vslist in self.data_and_regions_state:
      p = ds.path
      if p:
        ds.path = relative_path(p, directory)

  # ---------------------------------------------------------------------------
  #
  def restore_state(self, volume_manager):

    # Cache of Grid_Data objects improves load speed when multiple regions
    # are using same data file.  Especially important for files that contain
    # many data arrays.
    gdcache = {}        # (path, grid_id) -> Grid_Data object
    for ds, vslist in self.data_and_regions_state:
      data = ds.create_object(gdcache)
      if data:        # Can be None if user does not replace missing file.
        for vs in vslist:
          for v in vs.create_object(data):
            volume_manager.add_volume(v)

  # ---------------------------------------------------------------------------
  # Used for scene restore using already existing volume models.
  #
  def set_attributes(self):

    for ds, vslist in self.data_and_regions_state:
      volumes = [find_volume_by_session_id(vs.session_volume_id)
                 for vs in vslist]
      dset = set(v.data for v in volumes if not v is None)
      for data in dset:
        ds.set_attributes(data)
      for vs, volume in zip(vslist, volumes):
        if volume:
          vs.set_attributes(volume)

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def relative_path(path, dir):

  if isinstance(path, (tuple, list)):
    return tuple([relative_path(p, dir) for p in path])

  if not isinstance(path, str):
    return path

  from os.path import join
  d = join(dir, '')       # Make directory end with "/".
  if not path.startswith(d):
    return path

  rpath = path[len(d):]
  return rpath

# -----------------------------------------------------------------------------
# Path can be a tuple of paths.
#
def absolute_path(path):

  from os.path import abspath
  if isinstance(path, (tuple, list)):
    apath = tuple([abspath(p) for p in path])
  elif isinstance(path, str):
    apath = abspath(path)
  else:
    apath = path
  return apath

# -----------------------------------------------------------------------------
# Used only for restoring old session files.
# No longer used for saving sessions.
#
class Data_Set_State:

  version = 1

  state_attributes = ('name',
                      'data_state',
                      'data_region_state',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_data_set(self, data_set):
    
    ds = data_set
  
    self.name = ds.name

    s = Data_State()
    s.state_from_data(ds.data)
    self.data_state = s

    self.data_region_state = []
    for dr in ds.regions:
      rs = Volume_State()
      rs.state_from_data_region(dr)
      self.data_region_state.append(rs)

  # ---------------------------------------------------------------------------
  #
  def create_object(self, gdcache):

    data = self.data_state.create_object(gdcache)
    if data == None:
      from chimera import replyobj
      replyobj.warning('Could not restore %s\n' % self.name)
      return None

    for rs in self.data_region_state:
      drlist = rs.create_object(data)

    # Data_Set class is obsolete -- only Volume objects created.

    return None

# -----------------------------------------------------------------------------
#
class Data_State:

  version = 6

  state_attributes = ('path',           # Can be a tuple of paths
                      'file_type',
                      'name',
                      'grid_id',
                      'xyz_step',
                      'xyz_origin',
                      'cell_angles',
                      'rotation',
                      'symmetries',
                      'available_subsamplings',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_data(self, data):
    
    dt = data
    self.path = dt.path
    self.file_type = dt.file_type
    self.name = dt.name
    self.grid_id = dt.grid_id

    if dt.step != dt.original_step:
      self.xyz_step = dt.step
    else:
      self.xyz_step = None              # use value from data file

    if dt.origin != dt.original_origin:
      self.xyz_origin = dt.origin
    else:
      self.xyz_origin = None            # use value from data file

    self.cell_angles = data.cell_angles
    self.rotation = data.rotation
    self.symmetries = data.symmetries

    self.available_subsamplings = {}
    from ..VolumeData import Subsampled_Grid
    if isinstance(dt, Subsampled_Grid):
      for csize, ssdata in dt.available_subsamplings.items():
        if ssdata.path != dt.path:
          s = Data_State()
          s.state_from_data(ssdata)
          self.available_subsamplings[csize] = s

  # ---------------------------------------------------------------------------
  #
  def create_object(self, gdcache):

    path = absolute_path(self.path)
    if self.version >= 2 and (path, self.grid_id) in gdcache:
      # Caution: If data objects for the same file array can have different
      #          coordinates then cannot use this cached object.
      dlist = [gdcache[(path, self.grid_id)]]
    else:
      from ..VolumeData import opendialog
      paths_and_types = [(path, self.file_type)]
      grids, error_message = opendialog.open_grid_files(paths_and_types,
                                                        stack_images = False)
      if error_message:
        print ('Error opening map', error_message)
        msg = error_message + '\nPlease select replacement file.'
        from chimera import tkgui
        grids = opendialog.select_grids(tkgui.app, 'Replace File', msg)
        if grids is None:
          grids = []
        if self.version >= 2:
          # grid_id added in version 2.
          for data in grids:
            gdcache[(path, self.grid_id)] = data # Cache using old path.

      for data in grids:
        gdcache[(data.path, data.grid_id)] = data

      if self.version >= 2:
        # In version 2 a Grid_Data object can only have one array.
        id = (path, self.grid_id)
        if not id in gdcache:
          return []
        dlist = [gdcache[id]]
      else:
        # In version 1 a Grid_Data object could hold multiple arrays.
        dlist = grids

      if self.version >= 6:
        for data in dlist:
          data.name = self.name

      if self.version <= 4 and self.file_type == 'dsn6':
        # DSN6 format did not scale data value prior to version 5 sessions.
        for data in grids:
          data.use_value_scaling(False)

    if self.xyz_step:
      for data in dlist:
        data.set_step(self.xyz_step)

    if self.xyz_origin:
      for data in dlist:
        data.set_origin(self.xyz_origin)

    if self.version >= 3:
      for data in dlist:
        data.set_cell_angles(self.cell_angles)
        data.set_rotation(self.rotation)

    if self.version >= 4 and self.symmetries:
      for data in dlist:
        data.symmetries = self.symmetries
      
    if self.available_subsamplings:
      # Subsamples may be from separate files or the same file.
      from ..VolumeData import Subsampled_Grid
      dslist = []
      for data in dlist:
        if not isinstance(data, Subsampled_Grid):
          data = Subsampled_Grid(data)
        dslist.append(data)
      dlist = dslist
      for cell_size, dstate in self.available_subsamplings.items():
        if absolute_path(dstate.path) != path:
          ssdlist = dstate.create_object(gdcache)
          for i,ssdata in enumerate(ssdlist):
            dlist[i].add_subsamples(ssdata, cell_size)

    return dlist

  # ---------------------------------------------------------------------------
  # Used for scene restore on existing grid data object.
  #
  def set_attributes(self, data):

    data.set_step(self.xyz_step if self.xyz_step else data.original_step)
    data.set_origin(self.xyz_origin if self.xyz_origin else data.original_origin)
    data.set_cell_angles(self.cell_angles)
    data.set_rotation(self.rotation)

# -----------------------------------------------------------------------------
#
class Volume_State:

  version = 6

  state_attributes = ('region',
                      'representation',
                      'rendering_options',
                      'surface_model',
                      'solid_model',
                      'region_list',
                      'surface_levels',
                      'surface_colors',
                      'surface_brightness_factor',
                      'transparency_factor',
                      'solid_levels',
                      'solid_colors',
                      'solid_brightness_factor',
                      'transparency_depth',
                      'default_rgba',
                      'session_volume_id',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_data_region(self, volume):

    v = volume

    for attr in ('region', 'representation',
                 'surface_levels', 'surface_colors',
                 'surface_brightness_factor', 'transparency_factor',
                 'solid_levels', 'solid_colors', 'solid_brightness_factor',
                 'transparency_depth', 'default_rgba'):
      setattr(self, attr, getattr(v, attr))

    s = Rendering_Options_State()
    s.state_from_rendering_options(v.rendering_options)
    self.rendering_options = s
    
    #
    # Assume the surface and Solid models correspond to current
    # volume settings.  So only whether the model exists, is visible,
    # and its transform are saved.
    #
    self.surface_model = None
    m = v.surface_model()
#    if m:
    if False:
      from ..SessionUtil.stateclasses import Model_State
      s = Model_State()
      s.state_from_model(m)
      self.surface_model = s

    self.solid_model = None
#    if v.solid:
    if False:
      m = v.solid.model()
      if m:
        from ..SessionUtil.stateclasses import Model_State
        s = Model_State()
        s.state_from_model(m)
        self.solid_model = s

    rls = Region_List_State()
    rls.state_from_region_list(v.region_list)
    self.region_list = rls

    self.session_volume_id = session_volume_id(v)

  # ---------------------------------------------------------------------------
  #
  def create_object(self, data):

    ro = self.rendering_options.create_object()
    sm = self.surface_model
    if sm:
      from SimpleSession import modelOffset
      model_id = (sm.id + modelOffset, sm.subid)
    else:
      model_id = None
    vlist = []
    from .volume import Volume
    if self.version >= 3:
      v = Volume(data[0], self.region, ro, model_id)
      vlist.append(v)
      if self.version >= 6:
        sid = self.session_volume_id
      else:
        if self.version == 5:
          suffix = self.region_name
        else:
          suffix = self.name
        if suffix:
          sid = '%s %s' % (v.name, suffix)
        else:
          sid = v.name
        if sm.name.endswith(' ' + suffix):
          sm.name = sm.name[:-len(suffix)-1]
      v.session_volume_id = sid
      for attr in ('representation', 'surface_levels', 'surface_colors',
                   'surface_brightness_factor', 'transparency_factor',
                   'solid_levels', 'solid_colors', 'solid_brightness_factor',
                   'transparency_depth', 'default_rgba'):
        setattr(v, attr, getattr(self, attr))
    else:
      for i,d in enumerate(data):
        v = Volume(d, self.region, ro, model_id)
        v.representation = self.representation
        self.component_display_parameters[i].restore_state(v)
        vlist.append(v)
        if self.name:
          sid = '%s %s' % (v.name, self.name)
        else:
          sid = v.name
        v.session_volume_id = sid
    if isinstance(v.data.path, str):
      v.openedAs = (v.data.path, v.data.file_type, None, False)
        
    if self.version <= 3:
      # Changed definition of transparency depth from length to fraction
      # in version 4.
      dsize = [a*b for a,b in zip(v.data.step, v.data.size)]
      v.transparency_depth /= min(dsize)

    som = self.solid_model
    for v in vlist:
#      if sm or som:
      v.show()
      if sm:
        sm.restore_state(v.surface_model())
      if som and v.solid:
        som.restore_state(v.solid.model())        

    if self.version >= 2:
      for v in vlist:
        self.region_list.restore_state(v.region_list)

    return vlist

  # ---------------------------------------------------------------------------
  # Used for scene restore on existing volumes.
  #
  def set_attributes(self, volume):

    v = volume
    v.rendering_options = self.rendering_options.create_object()
    for attr in ('representation', 'surface_levels', 'surface_colors',
                 'surface_brightness_factor', 'transparency_factor',
                 'solid_levels', 'solid_colors', 'solid_brightness_factor',
                 'transparency_depth', 'default_rgba'):
      setattr(v, attr, getattr(self, attr))
    v.new_region(*self.region, show = False, adjust_step = False)
    self.region_list.restore_state(v.region_list)

    sm = self.surface_model
    som = self.solid_model
    if sm or som:
      v.show()
    if sm:
      sm.restore_state(v.surface_model())
    if som and v.solid:
      som.restore_state(v.solid.model())

    v.call_change_callbacks(('representation changed',
                             'region changed',
                             'thresholds changed',
                             'displayed',
                             'colors changed',
                             'rendering options changed',
                             'coordinates changed'))

# -----------------------------------------------------------------------------
#
class Region_List_State:

  version = 1

  state_attributes = ('region_list',
                      'current_index',
                      'named_regions',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_region_list(self, region_list):

    for attr in self.state_attributes:
      if attr != 'version':
        setattr(self, attr, getattr(region_list, attr))

  # ---------------------------------------------------------------------------
  #
  def restore_state(self, region_list):

    for attr in self.state_attributes:
      if hasattr(self, attr) and attr != 'version':
        setattr(region_list, attr, getattr(self, attr))

# -----------------------------------------------------------------------------
#
class Rendering_Options_State:

  version = 1

  state_attributes = ('show_outline_box',
                      'outline_box_rgb',
                      'outline_box_linewidth',
                      'limit_voxel_count',
                      'voxel_limit',
                      'color_mode',
                      'projection_mode',
                      'dim_transparent_voxels',
                      'bt_correction',
                      'minimal_texture_memory',
                      'maximum_intensity_projection',
                      'linear_interpolation',
                      'dim_transparency',
                      'line_thickness',
                      'smooth_lines',
                      'mesh_lighting',
                      'two_sided_lighting',
                      'flip_normals',
                      'subdivide_surface',
                      'subdivision_levels',
                      'surface_smoothing',
                      'smoothing_factor',
                      'smoothing_iterations',
                      'square_mesh',
                      'cap_faces',
                      'box_faces',
                      'orthoplanes_shown',
                      'orthoplane_positions',
                      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_rendering_options(self, rendering_options):

    for attr in self.state_attributes:
      if attr != 'version':
        setattr(self, attr, getattr(rendering_options, attr))

  # ---------------------------------------------------------------------------
  #
  def create_object(self):

    from .volume import Rendering_Options
    ro = Rendering_Options()
    for attr in self.state_attributes:
      if hasattr(self, attr):
        setattr(ro, attr, getattr(self, attr))

    if hasattr(self, 'use_2d_textures'):
      # Older rendering option superceded by projection_mode.
      if self.use_2d_textures:        ro.projection_mode = '2d-xyz'
      else:                           ro.projection_mode = '3d'

    return ro

# -----------------------------------------------------------------------------
# Used only for restoring old session files (Volume_State version <= 2).
# No longer used for saving sessions.
#
class Component_Display_Parameters_State:

  version = 1

  state_attributes = ('surface_levels',
                      'surface_colors',
                      'surface_brightness_factor',
                      'transparency_factor',
                      'solid_levels',
                      'solid_colors',
                      'solid_brightness_factor',
                      'transparency_depth',
                      'default_rgba',
                      'version',
                      )
  
  # ---------------------------------------------------------------------------
  #
  def state_from_component_display_parameters(self, dr):

    for attr in self.state_attributes:
      if attr != 'version':
        setattr(self, attr, getattr(dr, attr))

  # ---------------------------------------------------------------------------
  #
  def restore_state(self, dr):

    for attr in self.state_attributes:
      if hasattr(self, attr) and attr != 'version':
        setattr(dr, attr, getattr(self, attr))
