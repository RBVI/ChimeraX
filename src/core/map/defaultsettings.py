# vi: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
# Save and restore volume default settings.
#
# This should probably be unified with session saving, but is currently
# completely separate.
#
    
# ---------------------------------------------------------------------------
#
class Volume_Default_Settings:

  def __init__(self):

    options = self.factory_defaults()
#    from chimera import preferences
#    self.saved_prefs = preferences.addCategory('Volume Viewer',
#                                               preferences.HiddenCategory,
#                                               optDict = options)
    self.current_prefs = options.copy()
    self.change_callbacks = []

    #
    # Fix values from old preferences files (<= 1.2154) which were written
    # as strings instead of floats.
    #
    cp = self.current_prefs
    for name in ('box_padding', 'data_cache_size', 'voxel_limit',
                 'voxel_limit_for_open'):
      if name in cp and type(cp[name]) is str:
        cp[name] = float(cp[name])
    
  # ---------------------------------------------------------------------------
  #
  def factory_defaults(self):

    defaults = {
        'max_histograms': 3,
        'use_initial_colors': True,
        'initial_colors': ((.7,.7,.7,1),
                           (1,1,.7,1),
                           (.7,1,1,1),
                           (.7,.7,1,1),
                           (1,.7,1,1),
                           (1,.7,.7,1),
                           (.7,1,.7,1),
                           (.9,.75,.6,1),
                           (.6,.75,.9,1),
                           (.8,.8,.6,1)),
	'data_cache_size': 512.0,                # Mbytes
        'selectable_subregions': False,
        'subregion_button': 'middle',
        'box_padding': 0.0,
        'zone_radius': 2.0,
        'immediate_update': True,
        'show_on_open': True,
        'voxel_limit_for_open': 256.0,          # Mvoxels
        'show_plane': True,
        'voxel_limit_for_plane': 256.0,         # Mvoxels
        'limit_voxel_count': True,
        'voxel_limit': 1.0,                     # Mvoxels
        'auto_show_subregion': False,
        'adjust_camera': False,
        'shown_panels': ('Threshold and Color', 'Display style'),
        'show_outline_box': False,
        'outline_box_rgb': (1.0,1.0,1.0),
        'outline_box_linewidth': 1.0,
        'color_mode': 'auto8',
        'projection_mode': 'auto',
        'bt_correction': False,
        'minimal_texture_memory': False,
        'maximum_intensity_projection': False,
        'linear_interpolation': True,
        'dim_transparency': True,
        'dim_transparent_voxels': True,
        'line_thickness': 1.0,
        'smooth_lines': False,
        'mesh_lighting': True,
        'two_sided_lighting': True,
        'flip_normals': False,
        'subdivide_surface': False,
        'subdivision_levels': 1,
        'surface_smoothing': False,
        'smoothing_factor': 0.3,
        'smoothing_iterations': 2,
        'square_mesh': True,
        'cap_faces': True,
        'box_faces': False,
        'orthoplanes_shown': (False, False, False),
        'orthoplane_positions': (0,0,0),
    }

    try:
      from .. import mac_os_cpp
      msize = mac_os_cpp.memory_size()
    except:
      msize = 2**32
    csize = msize//2
    csize_mb = csize/(2**20)
    defaults['data_cache_size'] = csize_mb

    return defaults

  # ---------------------------------------------------------------------------
  #
  def rendering_option_names(self):

    return ('show_outline_box',
            'outline_box_rgb',
            'outline_box_linewidth',
            'limit_voxel_count',
            'voxel_limit',
            'color_mode',
            'projection_mode',
            'bt_correction',
            'minimal_texture_memory',
            'maximum_intensity_projection',
            'linear_interpolation',
            'dim_transparency',
            'dim_transparent_voxels',
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
            )

  # ---------------------------------------------------------------------------
  #
  def __getitem__(self, key):

    return self.current_prefs[key]

  # ---------------------------------------------------------------------------
  #
  def set_gui_to_defaults(self, dialog, data_settings = True,
                          global_settings = True, panel_settings = True):

    d = dialog
    p = self.current_prefs

    if global_settings:
      srp = d.subregion_panel
      srp.selectable_subregions.set(p['selectable_subregions'],
                                    invoke_callbacks = 0)
      srp.subregion_button.set(p['subregion_button'],
                               invoke_callbacks = 0)
      srp.auto_show_subregion.set(p['auto_show_subregion'],
                                  invoke_callbacks = 0)
      
      zp = d.zone_panel
      zp.zone_radius.set_value(number_string(p['zone_radius']),
                               invoke_callbacks = 0)

      abp = d.atom_box_panel
      abp.box_padding.set(number_string(p['box_padding']), invoke_callbacks = 0)

      dop = d.display_options_panel
      dop.set_gui_state(p)

    if data_settings:
      ro_defaults = self.rendering_option_defaults()
      dop = d.display_options_panel
      dop.set_gui_from_rendering_options(ro_defaults)

      slop = d.solid_options_panel
      slop.set_gui_from_rendering_options(ro_defaults)

      sop = d.surface_options_panel
      sop.set_gui_from_rendering_options(ro_defaults)

    if panel_settings:
      d.update_default_panels(p['shown_panels'])
      d.show_panels(p['shown_panels'])

  # ---------------------------------------------------------------------------
  #
  def rendering_option_defaults(self):
    
    from .volume import Rendering_Options
    ro = Rendering_Options()
    p = self.current_prefs
    
    for attr in self.rendering_option_names():
      setattr(ro, attr, p[attr])

    return ro

  # ---------------------------------------------------------------------------
  #
  def set_defaults_from_gui(self, dialog, data_settings = True,
                            global_settings = True, panel_settings = True):

    d = dialog
    p = self.current_prefs
    s = {}

    if data_settings:
      ro = d.rendering_options_from_gui()
      for attr in self.rendering_option_names():
        s[attr] = getattr(ro, attr)

    if global_settings:
      srp = d.subregion_panel
      s['selectable_subregions'] = srp.selectable_subregions.get()
      s['subregion_button'] = srp.subregion_button.get()
      s['auto_show_subregion'] = srp.auto_show_subregion.get()

      zp = d.zone_panel
      s['zone_radius'] = float_value(zp.zone_radius.value(), p['zone_radius'])

      abp = d.atom_box_panel
      s['box_padding'] = float_value(abp.box_padding.get(), p['box_padding'])

      dop = d.display_options_panel
      dop.get_gui_state(s)

    if panel_settings:
      s['shown_panels'] = [p.name for p in dialog.shown_panels()]

    self.update(s)
    
  # ---------------------------------------------------------------------------
  #
  def set(self, key, value):

    self.current_prefs[key] = value

  # ---------------------------------------------------------------------------
  #
  def update(self, dict):

    for key, value in dict.items():
      self.set(key, value)

    if dict:
      for cb in self.change_callbacks:
        cb(self, dict)

  # ---------------------------------------------------------------------------
  #
  def add_change_callback(self, cb):

    self.change_callbacks.append(cb)
    
  # ---------------------------------------------------------------------------
  #
  def remove_change_callback(self, cb):

    self.change_callbacks.remove(cb)

  # ---------------------------------------------------------------------------
  #
  def save_to_preferences_file(self, data_settings = True,
                               global_settings = True, panel_settings = True):

    keys = []
    if data_settings:
      keys.extend(self.rendering_option_names())
    if global_settings:
      keys.extend(['selectable_subregions', 'subregion_button', 'zone_radius',
                   'box_padding', 'max_histograms',
                   'use_initial_colors', 'initial_colors',
                   'immediate_update', 'show_on_open', 'voxel_limit_for_open',
                   'show_plane', 'voxel_limit_for_plane',
                   'voxel_limit_for_plane', 'limit_voxel_count', 'voxel_limit',
                   'data_cache_size', 'data_cache_size',
                   'auto_show_subregion', 'adjust_camera'])
    if panel_settings:
      keys.extend(['shown_panels'])

    s = self.saved_prefs
    p = self.current_prefs
    for key in keys:
      s.set(key, p[key], saveToFile = False)
    s.saveToFile()

  # ---------------------------------------------------------------------------
  #
  def restore_factory_defaults(self, dialog):

    options = self.factory_defaults()
    self.current_prefs = options.copy()
    self.saved_prefs.load(options.copy())
    self.set_gui_to_defaults(dialog)
    
# ---------------------------------------------------------------------------
# Represent float values that are integers without a decimal point.
#
def number_string(x):

  if type(x) == float and x == int(x):
    return str(int(x))
  return str(x)

# ---------------------------------------------------------------------------
#
def float_value(s, default = None):

  try:
    x = float(s)
  except ValueError:
    x = default
  return x
  
