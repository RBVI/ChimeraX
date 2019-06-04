# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# -----------------------------------------------------------------------------
# Handle Horos medical imaging 3d preset .plist files and color lookup table
# .plist files for setting 3d image rendering colors, brightness and transparency.
#
    
# -----------------------------------------------------------------------------
#
def appearance_names():
    from os import listdir
    names = [filename[:-6] for filename in listdir(preset_directory()) if filename.endswith('.plist')]
    names.extend([filename[:-5] for filename in listdir(clut_directory()) if filename.endswith('.clut')])
    names.append('initial')
    return names
     
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
def appearance_settings(name, v):
    if name == 'initial':
        return initial_settings(v)
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
