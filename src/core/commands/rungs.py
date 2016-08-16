# vim: set expandtab shiftwidth=4 softtabstop=4:

def rungs(session, atoms = None, color = None, radius = None, halfbond = None,
          hide = False, hide_atoms = True, show_ribbon = True, hide_hbonds = True):
    '''
    Make a cylinder for each base for nucleic acid residues.
    The cylinders are pseudobonds between C3' and N1 (A,G) or N3 (C,T).

    Parameters
    ----------
    atoms : Atoms or None
        Make rungs for these atoms or all atoms of None specified.
    color : Color
        Color of cylinders.
    radius : float
        Radius of cylinders.
    halfbond : bool
        Whether to color bond to match end atom colors.
    hide : bool
        Hide already shown rungs.
    hide_atoms : bool
        Whether to hide atoms of residues shown as cylinders.
    show_ribbon : bool
        Whether to show ribbon for residues shown as cylinders.
    hide_hbonds : bool
        Whether to hide hydrogen bonds read from mmCIF file.
    '''
    from ..atomic import all_atoms
    if atoms is None:
        atoms = all_atoms(session)

    ribose_base = "C3'"		# Base of rung
    purine_atom = 'C8'		# Distinguishes purines from pyrimidines
    purine_tip = 'N1'		# Tip of rung
    pyrimidine_tip = 'N3'	# Tip of rung

    rungs = []
    ratoms = atoms.unique_residues.atoms
    abase = ratoms.filter(ratoms.names == ribose_base)
    for m, ab in abase.by_structure:
        g = m.pseudobond_group('rungs')
        pbonds = g.pseudobonds		# Recolor existing pseudobonds
        if len(pbonds) == 0:
            g.dashes = 0
        ca1, ca2 = pbonds.atoms
        cur_pb = {(a1,a2):pb for a1,a2,pb in zip(ca1,ca2,pbonds)}
        for a1 in ab:
            r = a1.residue
            ratoms = r.atoms
            is_purine = len(ratoms.filter(ratoms.names == purine_atom)) > 0
            tip_atom_name = purine_tip if is_purine else pyrimidine_tip
            a2list = ratoms.filter(ratoms.names == tip_atom_name)
            if len(a2list) == 1:
                a2 = a2list[0]
                b = cur_pb.get((a1,a2))	# See if pseudobond already exists.
                if b is None and not hide:
                    b = g.new_pseudobond(a1, a2)
                    b.color = r.ribbon_color if color is None else color.uint8x4()
                    b.radius = 0.5 if radius is None else radius
                    b.halfbond = False
                rungs.append(b)

    if hide:
        for b in rungs:
            if b.display:
                a1, a2 = b.atoms
                b.display = a1.display = a2.display = False
    else:
        for b in rungs:
            b.display = True
            if color is not None:
                b.color = color.uint8x4()
            if radius is not None:
                b.radius = radius
            if halfbond is not None:
                b.halfbond = halfbond
            a1, a2 = b.atoms
            a1.draw_mode = a2.draw_mode = a1.STICK_STYLE
            if hide_atoms:
                a1.residue.atoms.displays = False
                a1.display = a2.display = True
            if show_ribbon:
                a1.residue.ribbon_display = True
                
        if hide_hbonds:
            for m in set(b.atoms[0].structure for b in rungs):
                pbg_hbonds = m.pbg_map.get('hydrogen bonds')
                if pbg_hbonds:
                    pbg_hbonds.pseudobonds.displays = False

def register_command(session):
    from . import register, CmdDesc, AtomsArg, EmptyArg, ColorArg, Or, FloatArg, BoolArg, NoArg
    desc = CmdDesc(optional = [('atoms', AtomsArg),
                               ('color', ColorArg),
                               ('radius', FloatArg),
                               ('halfbond', BoolArg),
                               ('hide', NoArg),
                               ('hide_atoms', BoolArg),
                               ('show_ribbon', BoolArg),
                               ('hide_hbonds', BoolArg)],
                   synopsis='depict nucleic acid residues as cylinders')
    register('rungs', desc, rungs)
