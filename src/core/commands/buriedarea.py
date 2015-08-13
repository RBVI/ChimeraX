# vi: set expandtab shiftwidth=4 softtabstop=4:

def buriedarea_command(session, atoms1, with_atoms2 = None, probe_radius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    if with_atoms2 is None:
        from . import AnnotationError
        raise AnnotationError('Require "with" keyword: buriedarea #1 with #2')
    atoms2 = with_atoms2

    ni = len(atoms1.intersect(atoms2))
    if ni > 0:
        from . import AnnotationError
        raise AnnotationError('Two sets of atoms must be disjoint, got %d atoms in %s and %s'
                              % (ni, atoms1.spec, atoms2.spec))

    from ..molsurf import buried_area
    ba, a1a, a2a, a12a = buried_area(atoms1, atoms2, probe_radius)

    # Report result
    msg = 'Buried area between %s and %s = %.5g' % (atoms1.spec, atoms2.spec, ba)
    log = session.logger
    log.status(msg)
    msg += ('\n  area %s = %.5g, area %s = %.5g, area both = %.5g'
            % (atoms1.spec, a1a, atoms2.spec, a2a, a12a))
    log.info(msg)

def register_buriedarea_command():
    from . import CmdDesc, register, AtomsArg, FloatArg
    _buriedarea_desc = CmdDesc(
        required = [('atoms1', AtomsArg)],
        keyword = [('with_atoms2', AtomsArg),
                   ('probe_radius', FloatArg),],
        synopsis = 'compute buried area')
    register('buriedarea', _buriedarea_desc, buriedarea_command)
