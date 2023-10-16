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

def read_pseudobond_file(session, stream, file_name, *args,
                         radius = 0.5, color = (255,255,0,255),
                         halfbond_coloring = False, dashes = None,
                         **kw):
    lines = stream.readlines()
    stream.close()

    g = session.pb_manager.get_group(file_name)
    if g.id is None:
        ret_models = [g]
    else:
        ret_models = []
        g.clear()

    from chimerax.atomic import AtomsArg
    num_preexisting = g.num_pseudobonds
    for i, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue

        if line.lstrip().startswith(';'):
            opt = _global_options(line, session)
            if 'radius' in opt:
                radius = opt['radius']
            if 'color' in opt:
                color = opt['color']
            if 'halfbond' in opt:
                halfbond_coloring = opt['halfbond']
            if 'dashes' in opt:
                dashes = opt['dashes']
            continue

        fields = line.split()
        aspec1, aspec2 = fields[:2]
        a1, used, rest = AtomsArg.parse(aspec1, session)
        a2, used, rest = AtomsArg.parse(aspec2, session)
        for a, aspec in ((a1,aspec1), (a2,aspec2)):
            if len(a) != 1:
                from chimerax.core.errors import UserError
                raise UserError('Line %d "%s", got %d atoms for spec "%s", require exactly 1'
                                % (i, line, len(a), aspec))
        b = g.new_pseudobond(a1[0], a2[0])
        if len(fields) >= 3:
            b.color = _parse_color(fields[2], session)
        else:
            b.color = color
        b.radius = radius
        b.halfbond = halfbond_coloring

    if dashes is not None:
        g.dashes = dashes
        
    return ret_models, 'Opened Pseudobonds %s, %d bonds' % (file_name, g.num_pseudobonds - num_preexisting)

def _global_options(line, session):
    line = line.lstrip()
    if not line.startswith(';'):
        return {}

    opt = {}
    keyval = line[1:].split('=')
    if len(keyval) == 2:
        k,v = keyval
        name = k.strip()
        if name == 'radius':
            opt = {'radius': float(v)}
        elif name == 'color':
            opt = {'color': _parse_color(v, session)}
        elif name == 'halfbond':
            opt = {'halfbond': _parse_bool(v, session)}
        elif name == 'dashes':
            opt = {'dashes': int(v)}

    return opt

def _parse_color(string, session):
    from chimerax.core.commands import ColorArg
    c, used, rest = ColorArg.parse(string.strip(), session)
    return c.uint8x4()

def _parse_bool(string, session):
    from chimerax.core.commands import BoolArg
    c, used, rest = BoolArg.parse(string.strip(), session)
    return c

def write_pseudobond_file(session, path, models=None, selected_only=False):
    if models is None:
        from chimerax import atomic
        models = atomic.all_pseudobond_groups(session)

    lines = []
    radius = None
    bcount = 0
    from chimerax.atomic import PseudobondGroup
    for pbg in models:
        if isinstance(pbg, PseudobondGroup):
            for pb in pbg.pseudobonds:
                if selected_only and not pb.selected:
                    continue
                if pb.radius != radius:
                    lines.append('; radius = %.5g' % pb.radius)
                    radius = pb.radius
                a1, a2 = pb.atoms
                color = '#%02x%02x%02x%02x' % tuple(pb.color)
                lines.append('%s\t%s\t%s' % (a1.atomspec, a2.atomspec, color))
                bcount += 1

    f = open(path, 'w')
    f.write('\n'.join(lines))
    f.close()

    session.logger.info('Saved %d pseudobonds to file %s' % (bcount, path))
