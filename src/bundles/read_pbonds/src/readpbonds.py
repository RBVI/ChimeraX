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

def read_pseudobond_file(session, stream, file_name, *args,
                         radius = 0.5, color = (255,255,0,255), **kw):
    lines = stream.readlines()
    stream.close()

    g = session.pb_manager.get_group(file_name)
    if g.id is None:
        ret_models = [g]
    else:
        ret_models = []
        g.clear()

    from chimerax.atomic import AtomsArg
    for i, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        
        if line.lstrip().startswith(';'):
            keyval = line.lstrip()[1:].split('=')
            if len(keyval) == 2:
                k,v = keyval
                if k.strip() == 'radius':
                    radius = float(v)
            continue

        fields = line.split()
        aspec1, aspec2 = fields[:2]
        a1, used, rest = AtomsArg.parse(aspec1, session)
        a2, used, rest = AtomsArg.parse(aspec2, session)
        for a, aspec in ((a1,aspec1), (a2,aspec2)):
            if len(a) != 1:
                raise SyntaxError('Line %d, got %d atoms for spec "%s", require exactly 1'
                                  % (i, len(a), aspec))
        b = g.new_pseudobond(a1[0], a2[0])
        if len(fields) >= 3:
            from chimerax.core.commands import ColorArg
            c, used, rest = ColorArg.parse(fields[2], session)
            b.color = c.uint8x4()
        else:
            b.color = color
        b.radius = radius
        b.halfbond = False

    return ret_models, 'Opened Pseudobonds %s, %d bonds' % (file_name, len(lines))

def write_pseudobond_file(session, path, models=None, selected_only=False):
    if models is None:
        from chimerax import atomic
        models = atomic.all_pseudobond_groups(session)

    lines = []
    radius = None
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

    f = open(path, 'w')
    f.write('\n'.join(lines))
    f.close()

    session.logger.info('Saved %d pseudobonds to file %s' % (len(lines), path))
