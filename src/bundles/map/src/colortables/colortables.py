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
# Handle Horos medical imaging 3d preset .plist files and color lookup table
# .plist files for setting 3d image rendering colors, brightness and transparency.
#

_appearances = {
    'chest':
    {
        'image_levels':((-683, 0.0), (-634, 0.985), (-479, 1.0), (-426, 0.0),
                        (34, 0.0), (38, 1.0), (86, 0.985), (97, 0.0),
                        (321, 0.0), (343, 1.0), (3012, 1.0)),

        'image_colors':((0.0, 0.605, 0.705, 1), (0.0, 0.605, 0.705, 1), (0.0, 0.605, 0.705, 1), (0.0, 0.605, 0.705, 1),
                        (1.0, 0.764, 0.964, 1.0), (1.0, 0.764, 0.964, 1.0), (1.0, 0.764, 0.964, 1.0), (1.0, 0.764, 0.964, 1.0),
                        (0.909, 1.0, 0.647, 1.0), (0.909, 1.0, 0.647, 1.0), (0.909, 1.0, 0.647, 1.0)),

        'dim_transparent_voxels': True,
    },
    'airways':
    {
        'image_levels':((-700, 0), (-500, 0.98), (-300, 0)),
        'image_colors':((0.0, 0.60, 0.70, 1), (0.0, 0.60, 0.70, 1), (0.0, 0.60, 0.70, 1)),
        'dim_transparent_voxels': True,
        'transparency_depth': 0.1,
        'image_brightness_factor': 1,
    },
    'brain':
    {
        'image_levels':((-100, 0), (-50, 1), (-1, 0),
                        (0, 0), (60, 1), (300,1), (301,0),
                        (500, 0), (1500, 1)),
        'image_colors':((1.0, 1.0, 0.5, 1), (1.0, 1.0, 0.5, 1), (1.0, 1.0, 0.5, 1),
                        (1.0, 1.0, 1.0, 1), (1.0, 1.0, 1.0, 1), (1.0, 1.0, 1.0, 1), (1.0, 1.0, 1.0, 1),
                        (0.8, 1.0, 1.0, 1), (0.8, 1.0, 1.0, 1)),
        'dim_transparent_voxels': True,
        'transparency_depth': 0.5,
        'image_brightness_factor': 1,
    },
 }

# -----------------------------------------------------------------------------
#
def appearance_names(session):
    nset = set(_appearances.keys())
    from os import listdir
    nset.update([filename[:-6] for filename in listdir(preset_directory())
                 if filename.endswith('.plist')])
    nset.update([filename[:-5] for filename in listdir(clut_directory())
                 if filename.endswith('.clut')])
    ap = _custom_appearance_settings(session)
    nset.update(ap.appearances.keys())
    nset.add('initial')
    names = list(nset)
    names.sort()
    return tuple(names)

# -----------------------------------------------------------------------------
#
def preset_directory():
   from os.path import dirname, join
   dir = join(dirname(__file__), 'presets')
   return dir
     
# -----------------------------------------------------------------------------
#
def clut_directory():
   from os.path import dirname, join
   dir = join(dirname(__file__), 'cluts')
   return dir

# -----------------------------------------------------------------------------
#
def add_appearance(name, v):
    ap = _custom_appearance_settings(v.session)
    # Must copy dictionary value or Settings object decides
    # the value has not changed and does not save it.
    appearances = dict(ap.appearances)
    appearances[name] = _volume_appearance(v)
    ap.appearances = appearances
    ap.save()

# -----------------------------------------------------------------------------
#
def _volume_appearance(v):
    return {
        'image_levels':v.image_levels,
        'image_colors':v.image_colors,
        'dim_transparent_voxels': v.rendering_options.dim_transparent_voxels,
        'transparency_depth': v.transparency_depth,
        'image_brightness_factor': v.image_brightness_factor,
    }

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import DynamicEnum
class AppearanceArg(DynamicEnum):
    def __init__(self, session):
        values = lambda s=session: appearance_names(s)
        DynamicEnum.__init__(self, values, case_sensitive = True)

# -----------------------------------------------------------------------------
#
def delete_appearance(name, session):
    ap = _custom_appearance_settings(session)
    if name in ap.appearances:
        # Must copy dictionary value or Settings object decides
        # the value has not changed and does not save it.
        appearances = dict(ap.appearances)
        del appearances[name]
        ap.appearances = appearances
        ap.save()

# -----------------------------------------------------------------------------
#
def _custom_appearance_settings(session):
    ap = getattr(session, '_volume_appearances', None)
    if ap is None:
        from chimerax.core.settings import Settings
        class _VolumeAppearanceSettings(Settings):
            EXPLICIT_SAVE = {'appearances': {}}
        session._volume_appearances = ap = _VolumeAppearanceSettings(session, 'volume_appearance')
    return ap

# -----------------------------------------------------------------------------
#
def appearance_settings(name, v):
    if name == 'initial':
        return initial_settings(v)
    cap = _custom_appearance_settings(v.session)
    if name in cap.appearances:
        return cap.appearances[name]
    if name in _appearances:
        return _appearances[name]
    from os.path import join
    hpath = join(preset_directory(), name + '.plist')
    from os.path import isfile
    if isfile(hpath):
        kw = read_horos_3d_preset(hpath)
    else:
        mpath = join(clut_directory(), name + '.clut')
        if isfile(mpath):
            kw = read_mricrogl_clut(mpath)
        else:
            raise ValueError('Color lookup table for %s not found as %s or %s'
                             % (name, hpath, mpath))

    return kw

def read_horos_3d_preset(path):
    f = open(path, 'rb')
    import plistlib
    fields = plistlib.load(f)
    f.close()
#    print (fields)
    colors = fields.get('16bitClutColors')
    curves = fields.get('16bitClutCurves')
    kw = {}
    if isinstance(colors, list) and isinstance(curves, list) and len(colors) == len(curves):
        kw['image_colors'] = scolors = []
        kw['image_levels'] = slevels = []
        kw['dim_transparent_voxels'] = False
        for color_seg, curve_seg in zip(colors, curves):
            seg_colors = [(c['red'], c['green'], c['blue']) for c in color_seg]
            seg_levels = [(xy['x'], xy['y']) for xy in curve_seg]
            # When two segments are given make sure control points at y = 0
            # are included so interval between two curve segments has zero brightness.
            xs,ys = seg_levels[0]
            xe,ye = seg_levels[-1]
            if ys != 0:
                pad = 0.001 * (xe-xs)
                seg_levels.insert(0, (xs-pad, 0))
                seg_colors.insert(0, seg_colors[0])
            if ye != 0:
                pad = 0.001 * (xe-xs)
                seg_levels.append((xe+pad, 0))
                seg_colors.append(seg_colors[-1])
            scolors.extend(seg_colors)
            slevels.extend(seg_levels)
        if 'opacity' in fields:
            otable = fields['opacity']
            if otable == 'Logratithmic Inverse Table':
                # TODO: Set color alpha?
                pass
    elif 'CLUT' in fields and fields['CLUT'] != 'No CLUT' and 'wl' in fields and 'ww' in fields:
        center = fields['wl']
        width = fields['ww']
        height = 0.5
        clut_name = fields['CLUT']
        from os.path import join
        cpath = join(clut_directory(), clut_name + '.plist')
        cf = open(cpath, 'rb')
        cfields = plistlib.load(cf)
        if 'Red' in cfields and 'Green' in cfields and 'Blue' in cfields:
            # 256 colors
            colors = [(r/255,g/255,b/255) for r,g,b in zip(cfields['Red'], cfields['Green'], cfields['Blue'])]
            colors = colors[::16]
            nc = len(colors)
            if nc >= 2:
                kw['image_colors'] = colors
                s = center - 0.5*width
                step = width / (nc-1)
                kw['image_levels'] = [(s + i*step, height) for i in range(nc)]
                kw['dim_transparent_voxels'] = False
            
    return kw

mricrogl_clut_value_types = {
    '[FLT]': float,
    '[INT]': int,
    '[BYT]': int,
    '[RGBA255]': lambda rgba: tuple(int(r) for r in rgba.split('|')),
    }
    
def read_mricrogl_clut(path):
    values = read_mricrogl_values(path)
    for key in ('numnodes', 'min', 'max'):
        if key not in values:
            raise ValueError('No "%s" key in MRIcroGL clut file %s' % (key, path))
    n = values['numnodes']
    min, max = values['min'], values['max']
    step = (max - min) / n
    levels = []
    colors = []
    for i in range(n):
        ki = 'nodeintensity%d' % i
        if ki not in values:
            raise ValueError('No "%s" key in MRIcroGL clut file %s' % (ki, path))
        kc = 'nodergba%d' % i
        if kc not in values:
            raise ValueError('No "%s" key in MRIcroGL clut file %s' % (kc, path))
        x = values[ki]/255
        level = min * (1-x) + max * x
        rgba = tuple(r/255 for r in values[kc])
#        rgb1 = rgba[:3] + (1,)
#        levels.append((level, rgba[3]))
#        colors.append(rgb1)
        levels.append((level, 1))
        colors.append(rgba)
    kw = {'image_colors': colors,
          'image_levels': levels,
          'transparency_depth': 0.05,
          'dim_transparent_voxels': True,
#          'dim_transparent_voxels': False,
          }
    return kw
    
def read_mricrogl_values(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    values = {}
    for line in lines:
        sline = line.strip()
        if sline in mricrogl_clut_value_types:
            vtype = mricrogl_clut_value_types[sline]
        else:
            kv = sline.split('=')
            if len(kv) == 2:
                key,value = kv
                values[key] = vtype(value)
    return values

def initial_settings(v):
    levels, colors = v.initial_image_levels()
    kw = {'image_levels': levels,
          'image_colors': colors,
          'dim_transparent_voxels': True,
          }
    return kw
