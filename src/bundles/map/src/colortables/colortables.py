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
    dir = preset_directory()
    from os import listdir
    names = [filename[:-6] for filename in listdir(dir) if filename.endswith('.plist')]
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
def appearance_settings(name):
    dir = preset_directory()
    from os.path import join
    path = join(dir, name + '.plist')
    f = open(path, 'rb')
    import plistlib
    fields = plistlib.load(f)
    f.close()
#    print (fields)
    colors = fields.get('16bitClutColors')
    curves = fields.get('16bitClutCurves')
    kw = {}
    if isinstance(colors, list) and isinstance(curves, list) and len(colors) == len(curves):
        kw['solid_colors'] = scolors = []
        kw['solid_levels'] = slevels = []
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
    elif 'CLUT' in fields and 'wl' in fields and 'ww' in fields:
        center = fields['wl']
        width = fields['ww']
        height = 0.5
        clut_name = fields['CLUT']
        cpath = join(clut_directory(), clut_name + '.plist')
        cf = open(cpath, 'rb')
        cfields = plistlib.load(cf)
        if 'Red' in cfields and 'Green' in cfields and 'Blue' in cfields:
            # 256 colors
            colors = [(r/255,g/255,b/255) for r,g,b in zip(cfields['Red'], cfields['Green'], cfields['Blue'])]
            colors = colors[::16]
            nc = len(colors)
            if nc >= 2:
                kw['solid_colors'] = colors
                s = center - 0.5*width
                step = width / (nc-1)
                kw['solid_levels'] = [(s + i*step, height) for i in range(nc)]
                kw['dim_transparent_voxels'] = False
            
    return kw

            
        
