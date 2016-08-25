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

def read_pseudobond_file(session, file, name, radius = 0.5, color = (255,255,0,255), as_ = None):
    lines = file.readlines()
    file.close()

    g = session.pb_manager.get_group(name)

    from ..commands import AtomsArg
    for i, line in enumerate(lines):
        aspec1, aspec2 = line.decode('utf-8').split()[:2]
        a1, used, rest = AtomsArg.parse(aspec1, session)
        a2, used, rest = AtomsArg.parse(aspec2, session)
        for a, aspec in ((a1,aspec1), (a2,aspec2)):
            if len(a) != 1:
                raise SyntaxError('Line %d, got %d atoms for spec "%s", require exactly 1'
                                  % (i, len(a), aspec))
        b = g.new_pseudobond(a1[0], a2[0])
        b.color = color
        b.radius = radius
        b.halfbond = False

    return [g], 'Opened Pseudobonds %s, %d bonds' % (name, len(lines))

def register_pbonds_format():
    from .. import io, toolshed
    io.register_format("Pseudobonds", toolshed.GENERIC3D, (".pb",),
                       ("pseudobonds",), open_func = read_pseudobond_file)
