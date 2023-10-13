# vim: set expandtab shiftwidth=4 softtabstop=4:

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
# Save and restore volume default settings.
#
# This should probably be unified with session saving, but is currently
# completely separate.
#

# -----------------------------------------------------------------------------
#
from chimerax.core.settings import Settings

class _VolumeSettings(Settings):
    EXPLICIT_SAVE = {
        'max_histograms': 1000,
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
        'pickable': True,
        'show_on_open': True,
        'voxel_limit_for_open': 256.0,          # Mvoxels
        'show_plane': True,
        'voxel_limit_for_plane': 256.0,         # Mvoxels
        'limit_voxel_count': True,
        'voxel_limit': 16.0,                    # Mvoxels
        'auto_show_subregion': False,
        'adjust_camera': False,
        'shown_panels': ('Threshold and Color', 'Display style'),
        'show_outline_box': False,
        'outline_box_rgb': (1.0,1.0,1.0),
        'outline_box_linewidth': 1.0,
        'color_mode': 'auto8',
        'colormap_on_gpu': False,
        'colormap_size': 2048,
        'colormap_extend_left': False,
        'colormap_extend_right': True,
        'blend_on_gpu': False,
        'projection_mode': 'auto',
        'plane_spacing': 'min',
        'full_region_on_gpu': False,
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
        'orthoplanes_shown': (False, False, False),
        'orthoplane_positions': (0,0,0),
        'tilted_slab_axis': (0,0,1),
        'tilted_slab_offset': 0,
        'tilted_slab_spacing': 1,
        'tilted_slab_plane_count': 1,
        'image_mode': 'full region',
        'backing_color': None,
    }
    
# ---------------------------------------------------------------------------
#
class VolumeDefaultSettings:

  def __init__(self, session):

    self._settings = _VolumeSettings(session, 'volume')
    self.change_callbacks = []

  # ---------------------------------------------------------------------------
  #
  def __getitem__(self, key):

    if key == 'data_cache_size':
      return self.data_cache_size()	# Compute default value if none is saved.
    return getattr(self._settings, key)

  # ---------------------------------------------------------------------------
  #
  def data_cache_size(self):
    csize_mb = self._settings.data_cache_size
    if csize_mb is None:
        try:
          import psutil
          m = psutil.virtual_memory()
          msize = m.total
        except Exception:
          msize = 2**32
        csize = msize//2
        csize_mb = csize/(2**20)
    return csize_mb

  # ---------------------------------------------------------------------------
  #
  def rendering_option_names(self):

    return ('show_outline_box',
            'outline_box_rgb',
            'outline_box_linewidth',
            'limit_voxel_count',
            'voxel_limit',
            'color_mode',
            'colormap_on_gpu',
            'colormap_size',
            'blend_on_gpu',
            'projection_mode',
            'plane_spacing',
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
            'orthoplanes_shown',
            'orthoplane_positions',
            'tilted_slab_axis',
            'tilted_slab_offset',
            'tilted_slab_spacing',
            'tilted_slab_plane_count',
            'image_mode',
            'backing_color',
            )

  # ---------------------------------------------------------------------------
  #
  def set_gui_to_defaults(self, dialog, data_settings = True,
                          global_settings = True, panel_settings = True):

    d = dialog
    s = self._settings

    global_settings = False
    if global_settings:
      srp = d.subregion_panel
      srp.selectable_subregions.set(s.selectable_subregions, invoke_callbacks = 0)
      srp.subregion_button.set(s.subregion_button, invoke_callbacks = 0)
      srp.auto_show_subregion.set(s.auto_show_subregion, invoke_callbacks = 0)
      
      zp = d.zone_panel
      zp.zone_radius.set_value(number_string(s.zone_radius), invoke_callbacks = 0)

      abp = d.atom_box_panel
      abp.box_padding.set(number_string(s.box_padding), invoke_callbacks = 0)

      dop = d.display_options_panel
      dop.set_gui_state(p)

    data_settings = False
    if data_settings:
      ro_defaults = self.rendering_option_defaults()
      dop = d.display_options_panel
      dop.set_gui_from_rendering_options(ro_defaults)

      imop = d.image_options_panel
      imop.set_gui_from_rendering_options(ro_defaults)

      sop = d.surface_options_panel
      sop.set_gui_from_rendering_options(ro_defaults)

    if panel_settings:
#      d.update_default_panels(s.shown_panels)
      d.show_panels(s.shown_panels)

  # ---------------------------------------------------------------------------
  #
  def rendering_option_defaults(self):
    
    from .volume import RenderingOptions
    ro = RenderingOptions()
    s = self._settings
    
    for attr in self.rendering_option_names():
      setattr(ro, attr, getattr(s, attr))

    return ro

  # ---------------------------------------------------------------------------
  #
  def set_defaults_from_gui(self, dialog, data_settings = True,
                            global_settings = True, panel_settings = True):

    d = dialog
    p = self._settings
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
      s['zone_radius'] = float_value(zp.zone_radius.value(), p.zone_radius)

      abp = d.atom_box_panel
      s['box_padding'] = float_value(abp.box_padding.get(), p.box_padding)

      dop = d.display_options_panel
      dop.get_gui_state(s)

    if panel_settings:
      s['shown_panels'] = [panel.name for panel in dialog.shown_panels()]

    self.update(s)
    
  # ---------------------------------------------------------------------------
  #
  def set(self, key, value):

    setattr(self._settings, key, value)

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
  def save_to_preferences_file(self):

    self._settings.save()

  # ---------------------------------------------------------------------------
  #
  def restore_factory_defaults(self, dialog = None):

    for key, value in self._settings.EXPLICIT_SAVE.items():
      self.set(key, value)
    if dialog is not None:
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
  
