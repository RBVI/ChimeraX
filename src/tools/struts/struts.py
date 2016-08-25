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

#
# Add bonds between atoms of a molecule to strengthen it for 3-d printing.
#
# Algorithm is to consider every pair of CA atoms within straight line distance D of each other
# and connect them if there is currently no path between them through bonds shorter than P.
#
def struts(session, atoms, length = 7.0, loop = 30.0, radius = 0.6, color = None,
           fatten_ribbon = True, replace = True, name = None, model_id = None):

    if len(atoms) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No atoms specified')

    pbg = strut_model(session, atoms, replace, name, model_id)

    log = session.logger
    log.status('Computing struts, %d atoms' % (len(atoms),))
    brace(atoms, length, loop, pbg, log)

    c = (178,178,178,255) if color is None else color.uint8x4()
        
    for b in pbg.pseudobonds:
        b.radius = radius
        b.color = c

    msg = 'Created %d struts for %s, max length %.3g, max loop length %.3g' % (
        pbg.num_pseudobonds, atoms.spec, length, loop)
    log.status(msg)
    log.info(msg + '\n')

    if fatten_ribbon:
        thick_ribbon(atoms)

    return pbg

def register_struts_command():

    from chimerax.core.commands import CmdDesc, register, AtomsArg, FloatArg, ColorArg, BoolArg, StringArg, ModelIdArg

    desc = CmdDesc(
        required = [('atoms', AtomsArg)],
        keyword = [('length', FloatArg),
                   ('loop', FloatArg),
                   ('radius', FloatArg),
                   ('color', ColorArg),
                   ('fatten_ribbon', BoolArg),
                   ('replace', BoolArg),
                   ('name', StringArg),
                   ('model_id', ModelIdArg)]
    )
    register('struts', desc, struts)

    desc = CmdDesc(optional = [('atoms', AtomsArg)])
    register('struts delete', desc, struts_delete)

def brace(atoms, max_length, max_loop_length, model, log):

    # Find all atom pairs within distance d of each other.
    apairs = []
    xyz = atoms.scene_coords
    na = len(atoms)
    for i1 in range(na):
        if log and i1 > 0 and i1 % 1000 == 0:
            log.status('Close atom pairs %d of %d atoms' % (i1, na))
        dxyz = xyz[i1+1:] - xyz[i1]
        d2 = (dxyz*dxyz).sum(axis=1)
        close = (d2 <= (max_length*max_length))
        for i2 in close.nonzero()[0]:
            apairs.append((d2[i2],i1,i1+1+i2))
    apairs.sort(key = lambda ap: ap[0])

    # Map atom index to list of connected (atom index, distance)
    sc = {}
    ai = {a:i for i,a in enumerate(atoms)}	# Map atom to index
    con = connections(ai, max_loop_length, log)
    for i1,i2,d in con:
        sc.setdefault(i1,[]).append((i2,d))
        
    # Add preexisting pseudobond connections
    for pb in model.pseudobonds:
        a1,a2 = pb.atoms
        i1,i2 = ai[a1], ai[a2]
        d = pb.length
        sc.setdefault(i1,[]).append((i2,d))
        sc.setdefault(i2,[]).append((i1,d))

    # Add connections between close atom pairs which are distantly connected.
    struts = []
    from chimerax.core.geometry import distance
    for c, (d12, i1, i2) in enumerate(apairs):
        if log and c > 0 and c % 1000 == 0:
            log.status('Evaluating struts %d of %d' % (c, len(apairs)))
        if not short_connection(i1, i2, max_loop_length, sc):
            struts.append((i1,i2))
            d = distance(xyz[i1], xyz[i2])
            sc.setdefault(i1,[]).append((i2,d))
            sc.setdefault(i2,[]).append((i1,d))

    # Create pseudobonds for struts
    for i1,i2 in struts:
        a1, a2 = atoms[i1], atoms[i2]
        b = model.new_pseudobond(a1, a2)
        for a in (a1, a2):
            a.display = True
            r = a.residue
            if r.ribbon_display:
                r.ribbon_hide_backbone = False
                a.hide = 0
                a.draw_mode = a1.BALL_STYLE

def connections(ai, dmax, log):

    con = []
    for i,a in enumerate(ai.keys()):
        if log and i % 500 == 0 and i > 0:
            log.status('Finding connections %d of %d atoms' % (i, len(ai)))
        adist = {a:0}
        bndry = set([a])
        acon = []
        while bndry:
            a2 = bndry.pop()
            d = adist[a2]
            for b in a2.bonds:
                an = b.other_atom(a2) 
                dn = d + b.length
                if dn <= dmax and (an not in adist or dn < adist[an]):
                    adist[an] = dn
                    if an in ai:
                        acon.append(an)
                    else:
                        bndry.add(an)
        for a2 in acon:
            con.append((ai[a], ai[a2], adist[a2]))
    return con

def short_connection(i1, i2, dmax, strut_connections):

    adist = {i1:0}
    bndry = set((i1,))
    sc = strut_connections
    while bndry:
        i = bndry.pop()
        d = adist[i]
        acon = sc.get(i,[])
        for n,dc in acon:
            dn = d + dc
            if dn <= dmax:
                if n == i2:
                    return True
                if not n in adist or dn < adist[n]:
                    adist[n] = dn
                    bndry.add(n)
    return False

def thick_ribbon(atoms):
    tw,th = .75,.75
    hw,hh = 1.5,.75
    sw,sh = 1.5,.75
    abw,abh,atw,ath = 2.5,.75,.75,.75
    nw,nh = 2.5,1.5
    dw,dh = .75,.75
    mols = atoms.unique_structures
    for m in mols:
        xsm = m.ribbon_xs_mgr
        xsm.set_helix_scale(hw,hh)
        xsm.set_helix_arrow_scale(abw,abh,atw,ath)
        xsm.set_coil_scale(tw,th)
        xsm.set_sheet_scale(sw,sh)
        xsm.set_sheet_arrow_scale(abw,abh,atw,ath)
        xsm.set_nucleic_scale(nw, nh)

def struts_delete(session, atoms = None):

    slist = strut_models(session)
    if atoms is None:
        session.models.close(slist)
    else:
        sclose = []
        for s in slist:
            remove = set()
            pbonds = s.pseudobonds
            a1, a2 = pbonds.atoms
            pbrem = a1.mask(atoms) | a2.mask()
            if pbrem.sum() == len(pbonds):
                sclose.append(s)
            else:
                for b in pbonds.filter(pbrem):
                    s.delete_pseudobond(b)
                if s.num_pseudobonds == 0:
                    sclose.append(s)
        if sclose:
            session.models.close(sclose)

def strut_models(session, model_id = None):
    from chimerax.core import atomic 
    pbglist = atomic.all_pseudobond_groups(session.models)
    slist = [s for s in pbglist if hasattr(s, 'strut_atoms')]
    if not model_id is None:
        slist = [s for s in slist if s.id == model_id]
    return slist
    
def strut_model(session, atoms, replace, name, model_id):

    if replace:
        mlist = [m for m in strut_models(session, model_id) if m.strut_atoms == atoms]
        if mlist:
            session.models.close(mlist)

    pbg = session.pb_manager.get_group(unique_struts_name(session))
    pbg.dashes = 0
    pbg.strut_atoms = atoms
    pbg.model_id = model_id
    session.models.add([pbg])

    if name is None:
        name = ('struts %s' % ', '.join(m.name for m in atoms.unique_structures)) if atoms else 'struts'
    pbg.name = name

    return pbg

def unique_struts_name(session):
    # PseudobondGroup category name must be unique.
    from chimerax.core import atomic 
    pbglist = atomic.all_pseudobond_groups(session.models)
    cats = set(pbg.category for pbg in pbglist)
    for i in range(len(cats)+1):
        name = 'struts %d' % (i+1)
        if not name in cats:
            break
    return name
