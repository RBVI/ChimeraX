# vi: set expandtab shiftwidth=4 softtabstop=4:

def buriedarea(session, atoms1, with_atoms2 = None, probe_radius = 1.4):
    '''
    Compute buried solvent accessible surface (SAS) area between two sets of atoms.
    This is the sum of the SAS area of each set of atoms minus the SAS area of the
    combined sets of atoms, that difference divided by two since each set of atoms
    has surface at the interface, so the interface area is defined as half the buried
    area.  Atoms not specified in either atom set are ignored when considering where
    the probe sphere can reach.

    Parameters
    ----------
    atoms1 : Atoms
      First set of atoms.
    with_atoms2 : Atoms
      Second set of atoms -- must be disjoint from first set.
    probe_radius : float
      Radius of the probe sphere.
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

def register_command(session):
    from . import CmdDesc, register, AtomsArg, FloatArg
    _buriedarea_desc = CmdDesc(
        required = [('atoms1', AtomsArg)],
        keyword = [('with_atoms2', AtomsArg),
                   ('probe_radius', FloatArg),],
        synopsis = 'compute buried area')
    register('buriedarea', _buriedarea_desc, buriedarea)
