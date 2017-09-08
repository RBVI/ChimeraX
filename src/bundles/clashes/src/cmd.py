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

from .settings import defaults

def cmd_clashes(session, test_atoms, *,
        group_name="clashes",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["clash_threshold"],
        **kw):
    return _cmd(session, test_atoms, group_name, hbond_allowance, overlap_cutoff, **kw)

def cmd_contacts(session, test_atoms, *,
        group_name="contacts",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["contact_threshold"],
        **kw):
    return _cmd(session, test_atoms, group_name, hbond_allowance, overlap_cutoff, **kw)

def _cmd(session, test_atoms, group_name, hbond_allowance, overlap_cutoff,
        atom_color=defaults["atom_color"],
        attr_name=defaults["attr_name"],
        bond_separation=defaults["bond_separation"],
        color_atoms=defaults["action_color"],
        continuous=False,
        coordset=None,
        inter_model=True,
        inter_submodel=False,
        intra_mol=defaults["intra_mol"],
        intra_res=defaults["intra_res"],
        log=defaults["action_log"],
        make_pseudobonds=defaults["action_pseudobonds"],
        naming_style=None,
        nonatom_color=defaults["nonatom_color"],
        pb_color=defaults["pb_color"],
        pb_radius=defaults["pb_radius"],
        reveal=False,
        save_file=None,
        set_attrs=defaults["action_attr"],
        select_clashes=defaults["action_select"],
        summary=True,
        test="cross"):
    from chimerax.core.colors import Color
    if isinstance(atom_color, Color):
        atom_color = atom_color.uint8x4()
    if isinstance(nonatom_color, Color):
        nonatom_color = nonatom_color.uint8x4()
    if isinstance(pb_color, Color):
        pb_color = pb_color.uint8x4()
    from chimerax.core.errors import UserError
    continuous_attr = "_clashes_continuous_id"
    from chimerax.core.atomic import get_triggers
    if continuous:
        if set_attrs or save_file != None or log:
            raise UserError("log/setAttrs/saveFile not allowed with continuous detection")
        if getattr(session, continuous_attr, None) == None:
            from inspect import getargvalues, currentframe
            arg_names, fArgs, fKw, frame_dict = getargvalues(currentframe())
            call_data = [frame_dict[an] for an in arg_names]
            def changes_cb(trig_name, changes, session=session, call_data=call_data):
                if 'position change' in changes.atomic_structure_reasons():
                    if not call_data[0]:
                        # all atoms gone
                        delattr(session, continuous_attr)
                        from chimerax.core.triggerset import DEREGISTER
                        return DEREGISTER
                    _cmd(*tuple(call_data))
                    return _motionCB(myData)
            setattr(session, continuous_attr, get_triggers(session).add_handler(
                        'changes', changes_cb))
    elif getattr(session, continuous_attr, None) != None:
        get_triggers(session).remove_handler(getattr(session, continuous_attr))
        delattr(session, continous_attr)
    from .clashes import find_clashes
    #TODO: 'inter_model' keyword is new (no "model" option in 'test' now)
    clashes = find_clashes(test_atoms, attr_name=attr_name, bond_separation=bond_separation,
        clash_threshold=overlap_cutoff, coordset=coordset, group_name=group_name,
        hbond_allowance=hbond_allowance, inter_model=inter_model, inter_submodel=inter_submodel,
        intra_res=intra_res, intra_mol=intra_mol, test=test)
    if select_clashes:
        chimera.selectionOperation(clashes.keys())
    if test == "self":
        outputGrouping = set()
    else:
        outputGrouping = test_atoms
    info = (overlap_cutoff, hbond_allowance, bond_separation, intra_res, intra_mol,
                            clashes, outputGrouping)
    if log:
        import sys
        # put a separator in the Reply Log
        print>>sys.stdout, ""
        _fileOutput(sys.stdout, info, naming_style=naming_style)
    if save_file == '-':
        from FindHBond.MolInfoDialog import SaveMolInfoDialog
        SaveMolInfoDialog(info, _fileOutput, initialfile="overlaps",
                title="Choose Overlap Info Save File",
                historyID="Overlap info")
    elif save_file is not None:
        _fileOutput(save_file, info, naming_style=naming_style)
    if summary == True:
        def _summary(msg):
            from chimera import replyobj
            replyobj.status(msg)
            replyobj.info(msg + '\n')
        summary = _summary
    if summary:
        if clashes:
            total = 0
            for clashList in clashes.values():
                total += len(clashList)
            summary("%d contacts" % (total/2))
        else:
            summary("No contacts")
    if not (set_attrs or color_atoms or make_pseudobonds or reveal):
        nukeGroup()
        return clashes
    if test in ("cross", "model"):
        atoms = [a for m in chimera.openModels.list(
            modelTypes=[chimera.Molecule]) for a in m.atoms]
    else:
        atoms = test_atoms
    if set_attrs:
        # delete the attribute in _all_ atoms...
        for m in chimera.openModels.list(modelTypes=[chimera.Molecule]):
            for a in m.atoms:
                if hasattr(a, attrName):
                    delattr(a, attrName)
        for a in atoms:
            if a in clashes:
                clashVals = clashes[a].values()
                clashVals.sort()
                setattr(a, attrName, clashVals[-1])
    if color_atoms:
        for a in atoms:
            a.surfaceColor = None
            if a in clashes:
                if atom_color is not None:
                    a.color = atom_color
            elif nonatom_color is not None:
                a.color = nonatom_color
    if reveal:
        needShow = set([a.residue for a in clashes.keys() if not a.display])
        for ns in needShow:
            for a in ns.oslChildren():
                a.display = True
    if make_pseudobonds:
        from chimera.misc import getPseudoBondGroup
        pbg = getPseudoBondGroup(groupName)
        pbg.deleteAll()
        #TODO
        #pbg.pb_radius = pb_radius
        if pb_color is not None:
            pbg.color = pb_color
        seen = set()
        for a in atoms:
            if a not in clashes:
                continue
            seen.add(a)
            for clasher in clashes[a].keys():
                if clasher in seen:
                    continue
                pbg.newPseudoBond(a, clasher)
    else:
        nukeGroup()
    global _sceneHandlersAdded
    if not _sceneHandlersAdded:
        from chimera import triggers, SCENE_TOOL_SAVE, SCENE_TOOL_RESTORE
        triggers.addHandler(SCENE_TOOL_SAVE, _sceneSave, None)
        triggers.addHandler(SCENE_TOOL_RESTORE, _sceneRestore, None)
        _sceneHandlersAdded = True
    return clashes

def register_command(command_name, logger):
    from chimerax.core.commands \
        import CmdDesc, register, BoolArg, FloatArg, ColorArg, Or, EnumOf, AtomsArg, \
            StructuresArg, SaveFileNameArg, NonNegativeIntArg, StringArg, EmptyArg
    if command_name == "hbonds":
        desc = CmdDesc(required=[('atoms', Or(AtomsArg,EmptyArg))],
            keyword = [('make_pseudobonds', BoolArg), ('radius', FloatArg), ('color', ColorArg),
                ('show_dist', BoolArg),
                ('restrict', Or(EnumOf(('cross', 'both', 'any')), AtomsArg)),
                ('inter_submodel', BoolArg), ('inter_model', BoolArg),
                ('intra_model', BoolArg), ('intra_mol', BoolArg), ('intra_res', BoolArg),
                ('cache_DA', FloatArg), ('relax', BoolArg), ('dist_slop', FloatArg),
                ('angle_slop', FloatArg), ('two_colors', BoolArg), ('slop_color', ColorArg),
                ('reveal', BoolArg), ('retain_current', BoolArg), ('save_file', SaveFileNameArg),
                ('log', BoolArg), ('naming_style', EnumOf(('simple', 'command', 'serial'))),
                ('batch', BoolArg), ('dashes', NonNegativeIntArg), ('salt_only', BoolArg),
                ('name', StringArg)],
            synopsis = 'Find hydrogen bonds'
        )
        register('hbonds', desc, cmd_hbonds, logger=logger)
    else:
        desc = CmdDesc(keyword = [('name', StringArg)], synopsis = 'Clear hydrogen bonds')
        register('~hbonds', desc, cmd_xhbonds, logger=logger)
