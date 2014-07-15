# -----------------------------------------------------------------------------
#
class CommandError(Exception):
    pass

# -----------------------------------------------------------------------------
#
def register_commands(commands):
    '''
    Registers the standard commands.
    '''
    add = commands.add_command
    from ..file_io.opensave import open_command, save_command, close_command
    add('open', open_command)
    add('save', save_command)
    add('close', close_command)
    from ..file_io import fetch_emdb, fetch_eds
    from ..molecule import fetch_pdb
    s = commands.session
    fetch_pdb.register_pdb_fetch(s)
    fetch_emdb.register_emdb_fetch(s)
    fetch_eds.register_eds_fetch(s)
    from ..map import molmap
    add('molmap', molmap.molmap_command)
    from ..map import volumecommand
    add('volume', volumecommand.volume_command)
    from ..map.fit import fitcmd
    add('fitmap', fitcmd.fitmap_command)
    from ..map.filter import vopcommand
    add('vop', vopcommand.vop_command)
    from ..molecule import align, showcmd, colorcmd
    add('align', align.align_command)
    add('show', showcmd.show_command)
    add('hide', showcmd.hide_command)
    add('color', colorcmd.color_command)
    from ..surface import gridsurf
    add('surface', gridsurf.surface_command)
    from .. import scenes
    add('scene', scenes.scene_command)
    from . import cameracmd
    add('camera', cameracmd.camera_command)
    from . import device
    add('device', device.device_command)
    from . import lightcmd
    add('lighting', lightcmd.lighting_command)
    from . import materialcmd
    add('material', materialcmd.material_command)
    from . import gui
    add('windowsize', gui.window_size_command)
    from ..molecule import blastpdb
    add('blast', blastpdb.blast_command)
    from ..molecule import ambient
    add('ambient', ambient.ambient_occlusion_command)

# -----------------------------------------------------------------------------
#
class Commands:
    '''Keep the list of commands and run them.'''
    def __init__(self, session):
        self.session = session
        self.commands = {}
        self.cmdabbrev = None
        self.history = Command_History(session)

    def add_command(self, name, function):
        '''
        Register a command with a given name and function to call.
        '''
        self.commands[name] = function
        self.cmdabbrev = None

    def run_command(self, text):
        '''
        Invoke a command.  The command and arguments are a string that will be
        parsed by a registered command function.
        '''
        ses = self.session
        ses.show_info('> %s' % text, color = '#008000')
        fields = text.split(maxsplit = 1)
        if len(fields) == 0:
            return
        self.history.add_to_command_history(text)
        cmd = fields[0]
        cab = self.cmdabbrev
        if cab is None:
            self.cmdabbrev = cab = abbreviation_table(self.commands.keys())
        if cmd in cab:
            cmd = cab[cmd]
            f = self.commands[cmd]
            args = fields[1] if len(fields) >= 2 else ''
            failed = False
            try:
                f(cmd, args, ses)
            except CommandError as e:
                ses.show_status(str(e))
                failed = True
            if not failed:
                ses.log.insert_graphics_image()
        else:
            ses.show_status('Unknown command %s' % cmd)

class Command_History:
    def __init__(self, session):
        self.session = session
        self.commands = None
        self.file_lines = None

    def command_list(self):
        if self.commands is None:
            self.read_command_history()
            self.session.at_quit(self.save_command_history)
        return self.commands
        
    def add_to_command_history(self, text):
        self.command_list().append(text)

    def read_command_history(self, filename = 'commands'):
        from ..file_io import history
        path = history.user_settings_path(filename)
        import os.path
        if os.path.exists(path):
            f = open(path, 'r')
            h = [line.rstrip() for line in f.readlines()]
            f.close()
        else:
            h = []
        self.commands = remove_repeats(h)
        self.file_lines = len(self.commands)

    def save_command_history(self, filename = 'commands'):
        h = self.commands
        if h is None:
            return
        if len(h) == self.file_lines:
            return      # No new commands
        from ..file_io import history
        path = history.user_settings_path(filename)
        f = open(path, 'a')
        for cmd in h[self.file_lines:]:
            f.write(cmd.strip() + '\n')
        f.close()

    def show_command_history(self, filename = 'commands'):
        '''
        Show a text window showing the invoked commands and their arguments for
        this session in the order they were executed.  Clicking on a command causes
        it to be run again.
        '''
        mw = self.session.main_window
        if mw.showing_text() and mw.text_id == 'command history':
            mw.show_graphics()
            return

        cmds = self.unique_commands()
        html = self.history_html(cmds)

        mw.show_text(html, html=True, id = 'command history',
                     anchor_callback = lambda url, s=self, c=cmds: s.insert_clicked_command(url,c))

    def unique_commands(self):
        lines = self.command_list()

        # Get unique lines, order most recent first.
        cmds = []
        found = set()
        for line in lines[::-1]:
            cmd = line.rstrip()
            if not cmd in found:
                cmds.append(cmd)
                found.add(cmd)
        return cmds

    def history_html(self, cmds):
        hlines = ['<html>', '<head>', '<style>',
                  'a { text-decoration: none; }',   # No underlining links
                  '</style>', '</head>', '<body>']
        hlines.extend(['<a href="%d"><p>%s</p></a>' % (i,line) for i,line in enumerate(cmds)])
        hlines.extend(['</body>','</html>'])
        html = '\n'.join(hlines)
        return html

    def insert_clicked_command(self, url, shown_commands):
        surl = url.toString(url.PreferLocalFile)
        # Work around Qt bug where it prepends a directory path to the anchor
        # even when QtTextBrowser search path and source are cleared.
        cnum = surl.split('/')[-1]
        c = int(cnum)         # command number
        if c < len(shown_commands):
            cmd = shown_commands[c]
            self.show_command(cmd)
            self.session.commands.run_command(cmd)

    def show_command(self, cmd):
        cline = self.session.main_window.command_line
        cline.clear()
        cline.insert(cmd)

    def show_previous_command(self, step = -1):
        cl = self.command_list()
        n = len(cl)
        if n == 0:
            return
        p = getattr(self, 'prev_command', n)
        p += step
        if p >= 0 and p < n:
            self.prev_command = p
            self.show_command(cl[p])

    def show_next_command(self):
        self.show_previous_command(step = 1)

# -----------------------------------------------------------------------------
#
def remove_repeats(strings):
    us = []
    sprev = None
    for s in strings:
        if s != sprev:
            us.append(s)
            sprev = s
    return us

# -----------------------------------------------------------------------------
#
def parse_arguments(cmd_name, arg_string, session,
                    required_args = (), optional_args = (), keyword_args = ()):

    fields = split_fields(arg_string)
    n = len(fields)
    akw = {}
    a = 0
    spec = None
    args = []

    # Make keyword abbreviation table.
    kw_args = tuple(optional_args) + tuple(keyword_args)
    kwnames = [arg_specifier_name(s) for s in kw_args]
    kwt = abbreviation_table(kwnames)

    # Parse keyword arguments.
    keyword_spec = dict([(arg_specifier_name(s),s) for s in kw_args])
    while a < n:
        f = fields[a]
        fl = f.lower()
        if not fl in kwt:
            args.append(f)
            a += 1
            continue
        spec = keyword_spec[kwt[fl]]
        a += 1 + parse_arg(fields[a+1:], spec, cmd_name, session, akw)

    # Parse required arguments.
    a = 0
    na = len(args)
    for spec in required_args:
        if a >= na:
            raise CommandError('%s: Missing required argument "%s"'
                               % (cmd_name, arg_specifier_name(spec)))
        a += parse_arg(args[a:], spec, cmd_name, session, akw)

    # Parse optional arguments.
    for spec in optional_args:
        if a >= na:
            break
        a += parse_arg(args[a:], spec, cmd_name, session, akw)

    # Allow last positional argument to have multiple values.
    if spec:
        multiple = arg_spec(spec)[3]
        if multiple:
            while a < na:
                a += parse_arg(args[a:], spec, cmd_name, session, akw)

    if a < na:
        raise CommandError('%s: Extra argument "%s"' % (cmd_name, args[a]))

    return akw

# -----------------------------------------------------------------------------
#
def parse_arg(fields, spec, cmd_name, session, akw):

    name, parse, pkw, multiple = arg_spec(spec)

    n = len(parse) if isinstance(parse, tuple) else 1
    if len(fields) < n:
        raise CommandError('%s: Missing argument for keyword "%s"'
                           % (cmd_name, name))
    
    try:
        if isinstance(parse, tuple):
            value = True if n == 0 else tuple(p(a, session, **pkw)
                                              for p,a in zip(parse,fields[:n]))
        else:
            value = parse(fields[0], session, **pkw)
    except CommandError as e:
        args = ' '.join(f for f in fields[:n])
        raise CommandError('%s invalid %s argument "%s": %s'
                           % (cmd_name, name, args, str(e)))
    except:
        args = ' '.join(f for f in fields[:n])
        raise
        raise CommandError('%s invalid %s argument "%s"'
                           % (cmd_name, name, args))
    if multiple:
        if name in akw:
            akw[name].append(value)
        else:
            akw[name] = [value]
    else:
        akw[name] = value

    return n

# -----------------------------------------------------------------------------
# Return argument name, parsing function, parsing function keyword dictionary.
# Original specifiers can be just a name or tuple of name and parse func.
#
def arg_spec(s):

    if isinstance(s, str):
        s = [s]
    arg_name = s[0]
    multiple = (len(s) > 1 and s[-1] == 'multiple')
    if multiple:
        s = s[:-1]
    parse = s[1] if len(s) >= 2 else string_arg
    kw = s[2] if len(s) >= 3 else {}
    return (arg_name, parse, kw, multiple)

# -----------------------------------------------------------------------------
#
def arg_specifier_name(s):

    if isinstance(s, str):
        return s
    return s[0]

# -----------------------------------------------------------------------------
#
no_arg = ()

# -----------------------------------------------------------------------------
#
def string_arg(s, session):

    return s

# -----------------------------------------------------------------------------
#
def path_arg(s, session):

    from os.path import expanduser
    return expanduser(s)

# -----------------------------------------------------------------------------
#
def bool_arg(s, session):

    return s.lower() not in ('false', 'f', '0', 'no', 'n', 'off')

# -----------------------------------------------------------------------------
#
def bool3_arg(s, session):

    b = [bool_arg(x) for x in s.split(',')]
    if len(b) != 3:
        raise CommandError('Require 3 comma-separated values, got %d' % len(b))
    return b

# -----------------------------------------------------------------------------
#
def enum_arg(s, session, values, multiple = False):

    if multiple:
        e = s.split(',')
        for v in e:
            if not v in values:
                raise CommandError('Values must be in %s, got "%s"'
                                   % (', '.join(values), s))
    elif s in values:
        e = s
    else:
        raise CommandError('Value must be one of %s, got "%s"'
                           % (', '.join(values), s))
    return e

# -----------------------------------------------------------------------------
#
def float_arg(s, session, min = None, max = None):

    x = float(s)
    if not min is None and x < min:
        raise CommandError('Value must be >= %g, got %g' % (min, x))
    if not max is None and x > max:
        raise CommandError('Value must be <= %g, got %g' % (max, x))
    return x

# -----------------------------------------------------------------------------
#
def float3_arg(s, session):

    fl = [float(x) for x in s.split(',')]
    if len(fl) != 3:
        raise CommandError('Require 3 comma-separated values, got %d' % len(fl))
    return fl

# -----------------------------------------------------------------------------
#
def float1or3_arg(s, session):

    fl = [float(x) for x in s.split(',')]
    if len(fl) == 1:
        f = fl[0]
        fl = [f,f,f]
    elif len(fl) != 3:
        raise CommandError('Require float value or 3 comma-separated values')
    return fl

# -----------------------------------------------------------------------------
#
def floats_arg(s, session, allowed_counts = None):

    fl = [float(x) for x in s.split(',')]
    if not allowed_counts is None and not len(fl) in allowed_counts:
        allowed = ','.join('%d' % c for c in allowed_counts)
        raise CommandError('Wrong number of values, require %s, got %d'
                           % (allowed, len(fl)))
    return fl

# -----------------------------------------------------------------------------
#
def int_arg(s, session):

    return int(s)

# -----------------------------------------------------------------------------
#
def int3_arg(s, session):

    il = [int(x) for x in s.split(',')]
    if len(il) != 3:
        raise CommandError('Require 3 comma-separated values, got %d' % len(il))
    return il

# -----------------------------------------------------------------------------
#
def int1or3_arg(s, session):

    il = [int(x) for x in s.split(',')]
    if len(il) == 1:
        i = il[0]
        il = [i,i,i]
    elif len(il) != 3:
        raise CommandError('Require integer or 3 comma-separated values')
    return il

# -----------------------------------------------------------------------------
#
def ints_arg(s, session, allowed_counts = None):

    il = [int(x) for x in s.split(',')]
    if not allowed_counts is None and not len(il) in allowed_counts:
        allowed = ','.join('%d' % c for c in allowed_counts)
        raise CommandError('Wrong number of values, require %s, got %d'
                           % (allowed, len(il)))
    return il

# -----------------------------------------------------------------------------
#
def value_type_arg(s, session):

    if s is None:
        return None

    types = ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
             'float32', 'float64')
    if s in types:
        import numpy
        vt = getattr(numpy, s)
    else:
        raise CommandError('Unknown data value type "%s", use %s'
                           % (s, ', '.join(types.keys())))
    return vt

# -----------------------------------------------------------------------------
#
def molecule_arg(s, session):

    sel = parse_specifier(s, session)
    mlist = sel.molecules()
    if len(mlist) == 0:
        raise CommandError('No molecule specified')
    elif len(mlist) > 1:
        raise CommandError('Multiple molecules specified')
    return mlist[0]

# -----------------------------------------------------------------------------
#
def molecules_arg(s, session, min = 0):

    sel = parse_specifier(s, session)
    mlist = sel.molecules()
    if len(mlist) < min:
        if len(mlist) == 0:
            raise CommandError('No molecule specified, require %d' % min)
        else:
            raise CommandError('%d molecules specified, require at least %d' % (len(mlist), min))
    return mlist

# -----------------------------------------------------------------------------
#
def chain_arg(s, session):

    sel = parse_specifier(s, session)
    clist = sel.chains()
    if len(clist) == 0:
        raise CommandError('No chains specified')
    elif len(clist) > 1:
        raise CommandError('Multiple chains specified')
    return clist[0]

# -----------------------------------------------------------------------------
#
def atoms_arg(s, session):

    sel = parse_specifier(s, session)
    aset = sel.atom_set()
    return aset

# -----------------------------------------------------------------------------
#
def model_arg(s, session):

    sel = parse_specifier(s, session)
    mlist = sel.models()
    if len(mlist) == 0:
        raise CommandError('No models specified')
    elif len(mlist) > 1:
        raise CommandError('Multiple models specified')
    return mlist[0]

# -----------------------------------------------------------------------------
#
def models_arg(s, session):

    sel = parse_specifier(s, session)
    mlist = sel.models()
    return mlist

# -----------------------------------------------------------------------------
#
def model_id_arg(s, session):

    return parse_model_id(s)

# -----------------------------------------------------------------------------
#
def specifier_arg(s, session):

    sel = parse_specifier(s, session)
    return sel

# -----------------------------------------------------------------------------
#
def openstate_arg(s, session):

    sel = parse_specifier(s, session)
    oslist = set([m.openState for m in sel.models()])
    if len(oslist) == 0:
        raise CommandError('No models specified')
    elif len(oslist) > 1:
        raise CommandError('Multiple coordinate systems specified')
    return oslist.pop()

# -----------------------------------------------------------------------------
#
def volumes_from_specifier(spec, session):

    try:
        sel = parse_specifier(spec, session)
    except CommandError:
        return []

    from ..map import Volume
    vlist = [m for m in sel.models() if isinstance(m, Volume)]

    return vlist

# -----------------------------------------------------------------------------
#
def volume_arg(v, session):

    vlist =  volumes_arg(v, session)
    if len(vlist) > 1:
        raise CommandError('Multiple volumes specified')
    return vlist[0]

# -----------------------------------------------------------------------------
#
def volumes_arg(v, session):

    sel = parse_specifier(v, session)
    vlist = sel.maps()
    if len(vlist) == 0:
        raise CommandError('No volumes specified')
    return vlist

# -----------------------------------------------------------------------------
#
def grid_data_arg(v, session):

    return [v.data for v in volumes_arg(v,session)]

# -----------------------------------------------------------------------------
#
def filter_volumes(models, keyword = ''):

    if keyword:
        keyword += ' '
        
    if isinstance(models, str):
        raise CommandError('No %svolumes specified by "%s"' % (keyword, models))
    
    from ..map import Volume
    from _volume import Volume_Model
    vids = set([v.id for v in models if isinstance(v, (Volume, Volume_Model))])
    for v in models:
        if not isinstance(v, (Volume, Volume_Model)) and not v.id in vids:
            raise CommandError('Model %s is not a volume' % v.name)
    volumes = [v for v in models if isinstance(v, Volume)]
    if len(volumes) == 0:
        raise CommandError('No %svolumes specified' % keyword)
    return volumes

# -----------------------------------------------------------------------------
#
def parse_floats(value, name, count, default = None):

    return parse_values(value, float, name, count, default)

# -----------------------------------------------------------------------------
#
def parse_ints(value, name, count, default = None):

    return parse_values(value, int, name, count, default)

# -----------------------------------------------------------------------------
#
def parse_values(value, vtype, name, count, default = None):

    if isinstance(value, (tuple, list)):
        vlist = value
    elif isinstance(value, str):
        vlist = value.split(',')
    elif value is None:
        return default
    else:
        vlist = []
    try:
        v = [vtype(c) for c in vlist]
    except:
        v = []
    if not len(v) == count:
        raise CommandError('%s value must be %d comma-separated numbers, got %s'
                           % (name, count, str(value)))
    return v

# -----------------------------------------------------------------------------
# Case insenstive and converts unique prefix to full string.
#
def parse_enumeration(string, strings, default = None):

    ast = abbreviation_table(strings)
    s = ast.get(string.lower(), default)
    return s

# -----------------------------------------------------------------------------
# Return table mapping unique abbreviations for names to the full name.
#
def abbreviation_table(names, lowercase = True):

    a = {}
    for name in names:
        for i in range(1,len(name)+1):
            aname = name[:i]
            if lowercase:
                aname = aname.lower()
            if aname in a:
                a[aname].append(name)
            else:
                a[aname] = [name]
    # Delete non-unique abbreviations except in case where shortest name
    # is a prefix of all names that have that abbreviation.
    for n,v in tuple(a.items()):
        shortest = min(v, key = lambda s: len(s))
        for fn in v:
            if not fn.startswith(shortest):
                del a[n]
                break
        if n in a:
            a[n] = shortest
    return a

# -----------------------------------------------------------------------------
#
def parse_model_id(modelId):

    from chimera import openModels as om
    if modelId is None:
        return (om.Default, om.Default)
    mid = None
    if isinstance(modelId, str):
        if modelId and modelId[0] == '#':
            modelId = modelId[1:]
        try:
            mid = tuple([int(i) for i in modelId.split('.')])
        except ValueError:
            mid = None
        if len(mid) == 1:
            mid = (mid[0], om.Default)
        elif len(mid) != 2:
            mid = None
    elif isinstance(modelId, int):
        mid = (modelId, om.Default)
    if mid is None:
        raise CommandError('modelId must be integer, got "%s"' % str(modelId))
    return mid

# -----------------------------------------------------------------------------
#
def parse_step(value, name = 'step', require_3_tuple = False):

    if isinstance(value, int):
        step = value
    else:
        try:
            step = parse_values(value, int, name, 3, None)
        except CommandError:
            step = parse_values(value, int, name, 1, None)[0]
    if require_3_tuple and isinstance(step, int):
        step = (step, step, step)
    return step

# -----------------------------------------------------------------------------
#
def parse_subregion(value, name = 'subregion'):

    if value.count(',') == 0:
        r = value       # Named region.
    else:
        s6 = parse_values(value, int, name, 6, None)
        r = (s6[:3], s6[3:])
    return r

volume_region_arg = parse_subregion

# -----------------------------------------------------------------------------
#
def parse_rgba(color):

    from chimera import MaterialColor
    if isinstance(color, MaterialColor):
        rgba = color.rgba()
    elif isinstance(color, (tuple,list)):
        rgba = color
    else:
        raise CommandError('Unknown color "%s"' % str(color))
    return rgba

# -----------------------------------------------------------------------------
#
def check_number(value, name, type = (float,int), allow_none = False,
                 positive = False, nonnegative = False):

    if allow_none and value is None:
        return
    if not isinstance(value, type):
        raise CommandError('%s must be number, got "%s"' % (name, str(value)))
    if positive and value <= 0:
        raise CommandError('%s must be > 0' % name)
    if nonnegative and value < 0:
        raise CommandError('%s must be >= 0' % name)

# -----------------------------------------------------------------------------
#
def check_in_place(inPlace, volumes):

    if not inPlace:
        return
    nwv = [v for v in volumes if not v.data.writable]
    if nwv:
        names = ', '.join([v.name for v in nwv])
        raise CommandError("Can't modify volume in place: %s" % names)

# -----------------------------------------------------------------------------
#
def check_matching_sizes(v1, v2, step, subregion, operation):

    if v1.data.size != v2.data.size:
        raise CommandError('Cannot %s grids of different size' % operation)
    if step or subregion:
        if not v1.same_region(v1.region, v2.region):
            raise CommandError('Cannot %s grids with different subregions' % operation)

# -----------------------------------------------------------------------------
#
def surfaces_arg(s, session):

    sel = parse_specifier(s, session)
    surfs = filter_surfaces(sel.models())
    return surfs

# -----------------------------------------------------------------------------
#
def filter_surfaces(surfaces):
    
    from ..molecule import Molecule
    surfs = set([s for s in surfaces if not isinstance(s, Molecule)])
    if len(surfs) == 0:
        raise CommandError('No surfaces specified')
    return surfs

# -----------------------------------------------------------------------------
#
def surface_pieces_arg(spec, session):

    sel = parse_specifier(spec, session)
    import Surface
    plist = Surface.selected_surface_pieces(sel)
    return plist

# -----------------------------------------------------------------------------
#
def single_volume(mlist):

    if mlist is None:
        return mlist
    vlist = filter_volumes(mlist)
    if len(vlist) != 1:
        raise CommandError('Must specify only one volume')
    return vlist[0]

# -----------------------------------------------------------------------------
#
def surface_center_axis(surf, center, axis, csys):

    if center is None:
        have_box, box = surf.bbox()
        if have_box:
            c = box.center()
        else:
            from chimera import Point
            c = Point(0,0,0)
    elif csys:
        sixf = surf.openState.xform.inverse()
        c = sixf.apply(csys.xform.apply(center))
    else:
        c = center

    if axis is None:
        from chimera import Vector
        a = Vector(0,0,1)
    elif csys:
        sixf = surf.openState.xform.inverse()
        a = sixf.apply(csys.xform.apply(axis))
    else:
        a = axis

    return c.data(), a.data()

# -----------------------------------------------------------------------------
#
def parse_vector(varg, cmdname):

    from Commands import parse_axis
    v, axis_point, csys = parse_axis(varg, cmdname)
    if csys:
        v = csys.xform.apply(v)
    vc = v.data()
    return vc

# -----------------------------------------------------------------------------
# The returned center and axis are in csys coordinates.
#
def parse_center_axis(center, axis, csys, cmdname):

    from Commands import parseCenterArg, parse_axis

    if isinstance(center, (tuple, list)):
        from chimera import Point
        center = Point(*center)
        ccs = csys
    elif center:
        center, ccs = parseCenterArg(center, cmdname)
    else:
        ccs = None

    if isinstance(axis, (tuple, list)):
        from chimera import Vector
        axis = Vector(*axis)
        axis_point = None
        acs = csys
    elif axis:
        axis, axis_point, acs = parse_axis(axis, cmdname)
    else:
        axis_point = None
        acs = None

    if not center and axis_point:
        # Use axis point if no center specified.
        center = axis_point
        ccs = acs

    # If no coordinate system specified use axis or center coord system.
    cs = (ccs or acs)
    if csys is None and cs:
        csys = cs
        xf = cs.xform.inverse()
        if center and not ccs:
            center = xf.apply(center)
        if axis and not acs:
            axis = xf.apply(axis)

    # Convert axis and center to requested coordinate system.
    if csys:
        xf = csys.xform.inverse()
        if center and ccs:
            center = xf.apply(ccs.xform.apply(center))
        if axis and acs:
            axis = xf.apply(acs.xform.apply(axis))

    return center, axis, csys

# -----------------------------------------------------------------------------
#
def parse_colormap(cmap, cmapRange, reverseColors):

    if not isinstance(cmap, str):
        raise CommandError('Invalid colormap specification: "%s"' % repr(cmap))

    pname = {'redblue': 'red-white-blue',
             'rainbow': 'rainbow',
             'gray': 'grayscale',
             'cyanmaroon': 'cyan-white-maroon'}
    if cmap.lower() in pname:
        from SurfaceColor import standard_color_palettes as scp
        rgba = scp[pname[cmap.lower()]]
        n = len(rgba)
        cmap = [(c/float(n-1),rgba[c]) for c in range(n)]
        if not cmapRange:
            cmapRange = 'full'
    else:
        vclist = cmap.split(':')
        if len(vclist) < 2:
            raise CommandError('Invalid colormap specification: "%s"' % cmap)
        cmap = [parse_value_color(vc) for vc in vclist]

    if cmapRange:
        cmrerr = 'cmapRange must be "full" or two numbers separated by a comma'
        if not isinstance(cmapRange, str):
            raise CommandError(cmrerr)

        if cmapRange.lower() == 'full':
            cmapRange = cmapRange.lower()
        else:
            try:
                cmapRange = [float(x) for x in cmapRange.split(',')]
            except ValueError:
                raise CommandError(cmrerr)
            if len(cmapRange) != 2:
                raise CommandError(cmrerr)

    if reverseColors:
        n = len(cmap)
        cmap = [(cmap[c][0],cmap[n-1-c][1]) for c in range(n)]

    return cmap, cmapRange

# -----------------------------------------------------------------------------
#
def parse_value_color(vc):

    err = 'Colormap entry must be value,color: got "%s"' % vc
    svc = vc.split(',',1)
    if len(svc) != 2:
        raise CommandError(err)
    try:
        v = float(svc[0])
    except ValueError:
        raise CommandError(err)
    from Commands import convertColor
    try:
        c = convertColor(svc[1]).rgba()
    except:
        raise CommandError(err)
    return v, c

# -----------------------------------------------------------------------------
#
def parse_color(color, session):

    if isinstance(color, (tuple, list)):
        if len(color) == 4:
            return tuple(color)
        elif len(color) == 3:
            return tuple(color) + (1,)
    elif isinstance(color, str):
        fields = color.split(',')
        if len(fields) == 1:
            from .qt import QtGui
            qc = QtGui.QColor(color)
            if qc.isValid():
                rgba = (qc.redF(), qc.greenF(), qc.blueF(), qc.alphaF())
                return rgba
        else:
            try:
                rgba = tuple(float(c) for c in fields)
            except:
                raise CommandError('Unrecognized color: "%s"' % repr(color))
            if len(rgba) == 3:
                rgba = rgba + (1.0,)
            if len(rgba) == 4:
                return rgba
                
    raise CommandError('Unrecognized color: "%s"' % repr(color))

color_arg = parse_color

# -----------------------------------------------------------------------------
#
def perform_operation(cmdname, args, ops, session):

    abbr = abbreviation_table(ops.keys())

    alist = args.split()
    a0 = alist[0].lower() if alist else None
    if not a0 in abbr:
        opnames = list(ops.keys())
        opnames.sort()
        raise CommandError('First argument must be one of %s'
                           % ', '.join(opnames))

    f, req_args, opt_args, kw_args = ops[abbr[a0]]
    kw = parse_arguments(cmdname, args[len(a0):], session, req_args, opt_args, kw_args)
    import inspect
    if 'session' in inspect.getargspec(f)[0] and not 'session' in kw:
        kw['session'] = session
    f(**kw)

# -----------------------------------------------------------------------------
#
def multiscale_surface_pieces_arg(spec, session):

    from Commands import CommandError

    mspec = spec[:spec.find(':')] if ':' in spec else spec
    from _surface import SurfaceModel
    slist = [m for m in parse_specifier(mspec, session).models() if isinstance(m, SurfaceModel)]
    if len(slist) == 0:
        raise CommandError('No surface models specified by "%s"' % spec)

    plist = sum([list(s.surfacePieces) for s in slist], [])

    if len(mspec) < len(spec):
        pspec = spec[len(mspec)+1:]
        try:
            pspecs = piece_specs(pspec)
        except:
            raise CommandError('Invalid surface piece specifier "%s"' % pspec)
        plist = [p for p in plist if matching_piece_spec(p, pspecs)]

    return plist

# -----------------------------------------------------------------------------
#
def piece_specs(pspec):

    pspecs = pspec.split(',')
    prlist = []
    for s in pspecs:
        nc = s.split('.')
        if len(nc) == 1:
            nc.append('')
        n,c = nc
        if len(n) == 0:
            nr = (None, None)
        else:
            nr = [int(i) for i in n.split('-')]
            if len(nr) == 1:
                nr = nr*2
        if len(c) == 0:
            cr = (None, None)
        else:
            cr = c.split('-')
            if len(cr) == 1:
                cr = cr*2
        prlist.append((nr,cr))
    return prlist

# -----------------------------------------------------------------------------
#
def matching_piece_spec(p, pspecs):

    nc = p.oslName.split('.')
    if len(nc) != 2:
        return False
    n,c = nc
    try:
        n = int(n)
    except ValueError:
        return False
    for (nmin,nmax), (cmin,cmax) in pspecs:
        if ((nmin is None or n >= nmin) and (nmax is None or n <= nmax) and
            (cmin is None or c >= cmin) and (cmax is None or c <= cmax)):
            return True
    return False

# -----------------------------------------------------------------------------
# Handle quoted sub-strings.
#
def split_fields(s):

    if s.find('"') == -1 and s.find("'") == -1:
        fields = s.split()
    else:
        fields = []
        rest = s
        while rest:
            f, rest = split_first(rest)
            if f:
                fields.append(f)
    return fields

# -----------------------------------------------------------------------------
# Parse first white space delimited word handling quotes if present.
# Quotes are removed.
#
def split_first(s):

    t = s.lstrip()
    if t and t[0] in ('"', "'"):
        e = t[1:].find(t[0] + " ")
        if e >= 0:
            return t[1:e+1], t[e+2:]
        elif t[-1] == t[0]:
            return t[1:-1], ''
    return (t.split(None, 1) + ['', ''])[:2]

# -----------------------------------------------------------------------------
# Examples:
#        #1.A:7-52@CA
#        #2:HOH
#
def parse_specifier(spec, session):

    parts = specifier_parts(spec)
    mids = cid = rrange = rname = aname = None
    invert = False
    for p in parts:
        if p.startswith('#'):
            mids = integer_set(p[1:])
            if len(mids) == 0:
                mids = set(m.id for m in session.model_list())
        elif p.startswith('.'):
            cids = p[1:].split(',')
        elif p.startswith(':'):
            rrange = integer_range(p[1:])
            if rrange is None:
                rname = p[1:]
        elif p.startswith('@'):
            aname = p[1:]
        elif p.startswith('!'):
            invert = True
    
    if mids is None:
        if cids is None and rrange is None and rname is None and aname is None:
            if spec == 'all':
                mlist = session.model_list()
            else:
                mlist = []
        else:
            mlist = session.molecules()
    else:
        mlist = [m for m in session.model_list() if m.id in mids]
    if len(mlist) == 0:
        raise CommandError('No models specified by "%s"' % spec)

    s = Selection()
    from ..molecule import Molecule
    smodels = []
    rnums = None
    for m in mlist:
        if isinstance(m, Molecule):
            atoms = m.atom_subset(aname, cids, rrange, rnums, rname, invert)
            if atoms.count() > 0:
                s.add_atoms(atoms)
        else:
            smodels.append(m)
    if invert:
        sm = set(smodels)
        smodels = [m for m in session.model_list()
                   if not isinstance(m, Molecule) and not m in sm]
        if not mids is None:
            smodels.extend(m for m in session.molecules() if not m.id in mids)
    s.add_models(smodels)

    return s

# -----------------------------------------------------------------------------
# Example #1.A:7-52@CA
#
def specifier_parts(spec):

    parts = []
    p = ''
    for c in spec:
        if c in ('!', '#', '.', ':', '@') and p:
            parts.append(p)
            p = ''
        p += c
    if p:
        parts.append(p)
    return parts

# -----------------------------------------------------------------------------
#
def integer_range(rstring):
    r = rstring.split('-')
    try:
        r1 = int(r[0])
        r2 = int(r[1]) if len(r) == 2 else r1
    except ValueError:
        return None
    return (r1,r2)

# -----------------------------------------------------------------------------
#
def integer_set(rstring):
    s = set()
    for rint in rstring.split(','):
        r = rint.split('-')
        if r[0]:
            r1 = int(r[0])
            r2 = int(r[1]) if len(r) == 2 else r1
            s |= set(range(r1,r2+1))
    return s

# -----------------------------------------------------------------------------
#
class Selection:
    def __init__(self):
        self._models = []              # Non-molecule models
        from ..molecule import Atoms
        self.aset = Atoms()
    def add_models(self, models):
        from ..molecule import Molecule
        mols = [m for m in models if isinstance(m, Molecule)]
        self.aset.add_molecules(mols)
        other = [m for m in models if not isinstance(m, Molecule)]
        self._models.extend(other)
    def add_atoms(self, atoms):
        self.aset.add_atoms(atoms)
    def models(self):
        return self._models + list(self.molecules())
    def molecules(self):
        return self.aset.molecules()
    def chains(self):
        return self.aset.chains()
    def atom_set(self):
        return self.aset
    def maps(self):
        from ..map import Volume
        mlist = [m for m in self.models() if isinstance(m,Volume)]
        return mlist

# -----------------------------------------------------------------------------
#
def points_arg(a, session):
    s = parse_specifier(a, session)
    points = s.atom_coordinates()
    return points
