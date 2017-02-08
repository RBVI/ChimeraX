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

def measure_buriedarea(session, atoms1, with_atoms2 = None, probe_radius = 1.4,
                       list_residues = False, cutoff_area = 1, color = None, select = False):
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
    list_residues : bool
      Whether to report a list of contacting residues for each set of atoms.
    cutoff_area : float
      Per-residue minimum area for listing residues.
    color : Color or None
      Color contacting residues.
    select : bool
      Whether to select contacting residues.
    '''
    atoms2 = with_atoms2
    ni = len(atoms1.intersect(atoms2))
    if ni > 0:
        from . import AnnotationError
        raise AnnotationError('Two sets of atoms must be disjoint, got %d atoms in %s and %s'
                              % (ni, atoms1.spec, atoms2.spec))

    from ..atomic import buried_area
    ba, a1a, a2a, a12a = buried_area(atoms1, atoms2, probe_radius)

    # Report result
    msg = 'Buried area between %s and %s = %.5g' % (atoms1.spec, atoms2.spec, ba)
    log = session.logger
    log.status(msg)
    msg += ('\n  area %s = %.5g, area %s = %.5g, area both = %.5g'
            % (atoms1.spec, a1a.sum(), atoms2.spec, a2a.sum(), a12a.sum()))
    log.info(msg)

    if list_residues or color or select:
        n1 = len(a1a)
        a1b = a1a - a12a[:n1]
        a2b = a2a - a12a[n1:]
        res1, rba1 = atoms1.residue_sums(a1b)
        res2, rba2 = atoms2.residue_sums(a2b)
        r1mask = (rba1 > cutoff_area)
        res1, rba1 = res1.filter(r1mask), rba1[r1mask]
        r2mask = (rba2 > cutoff_area)
        res2, rba2 = res2.filter(r2mask), rba2[r2mask]
        if color:
            c8 = color.uint8x4()
            res1.atoms.colors = res2.atoms.colors = c8
            res1.ribbon_colors = res2.ribbon_colors = c8
        if select:
            session.selection.clear()
            res1.atoms.selected = True
            res2.atoms.selected = True
        if list_residues:
            lines = ['%15s %.5g' % (str(r), a) for r, a in tuple(zip(res1, rba1)) + tuple(zip(res2, rba2))]
            log.info('%d contacting residues\n%s' % (len(res1) + len(res2), '\n'.join(lines)))
            
def register_command(session):
    from . import CmdDesc, register, AtomsArg, FloatArg, BoolArg, ColorArg
    desc = CmdDesc(
        required = [('atoms1', AtomsArg)],
        keyword = [('with_atoms2', AtomsArg),
                   ('probe_radius', FloatArg),
                   ('list_residues', BoolArg),
                   ('cutoff_area', FloatArg),
                   ('color', ColorArg),
                   ('select', BoolArg),],
        required_arguments = ['with_atoms2'],
        synopsis = 'compute buried area')
    register('measure buriedarea', desc, measure_buriedarea)
