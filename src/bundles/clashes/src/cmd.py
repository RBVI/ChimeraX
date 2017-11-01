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
    return _cmd(session, test_atoms, group_name, hbond_allowance, overlap_cutoff, "clashes", **kw)

def cmd_contacts(session, test_atoms, *,
        group_name="contacts",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["contact_threshold"],
        **kw):
    return _cmd(session, test_atoms, group_name, hbond_allowance, overlap_cutoff, "contacts", **kw)

_continuous_attr = "_clashes_continuous_id"
def _cmd(session, test_atoms, group_name, hbond_allowance, overlap_cutoff, test_type,
        atom_color=defaults["atom_color"],
        attr_name=defaults["attr_name"],
        bond_separation=defaults["bond_separation"],
        color_atoms=defaults["action_color"],
        continuous=False,
        distance_only=None,
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
        select=defaults["action_select"],
        summary=True,
        test="other"):
    from chimerax.core.colors import Color
    if atom_color is not None and not isinstance(atom_color, Color):
        atom_color = Color(rgba=atom_color)
    if nonatom_color is not None and not isinstance(nonatom_color, Color):
        nonatom_color = Color(rgba=nonatom_color)
    if pb_color is not None and not isinstance(pb_color, Color):
        pb_color = Color(rgba=pb_color)
    from chimerax.core.errors import UserError
    from chimerax.core.atomic import get_triggers
    if continuous:
        if set_attrs or save_file != None or log:
            raise UserError("log/setAttrs/saveFile not allowed with continuous detection")
        if getattr(session, _continuous_attr, None) == None:
            from inspect import getargvalues, currentframe
            arg_names, fArgs, fKw, frame_dict = getargvalues(currentframe())
            call_data = [frame_dict[an] for an in arg_names]
            def changes_cb(trig_name, changes, session=session, call_data=call_data):
                if 'position change' in changes.atomic_structure_reasons():
                    if not call_data[0]:
                        # all atoms gone
                        delattr(session, _continuous_attr)
                        from chimerax.core.triggerset import DEREGISTER
                        return DEREGISTER
                    _cmd(*tuple(call_data))
                    return _motionCB(myData)
            setattr(session, _continuous_attr, get_triggers(session).add_handler(
                        'changes', changes_cb))
    elif getattr(session, _continuous_attr, None) != None:
        get_triggers(session).remove_handler(getattr(session, _continuous_attr))
        delattr(session, continous_attr)
    from .clashes import find_clashes
    clashes = find_clashes(session, test_atoms, attr_name=attr_name,
        bond_separation=bond_separation, clash_threshold=overlap_cutoff,
        distance_only=distance_only, group_name=group_name, hbond_allowance=hbond_allowance,
        inter_model=inter_model, inter_submodel=inter_submodel, intra_res=intra_res,
        intra_mol=intra_mol, test=test)
    if select:
        session.selection.clear()
        for a in clashes.keys():
            a.selected = True
    if test == "self":
        output_grouping = set()
    else:
        output_grouping = test_atoms
    test_type = "distances" if distance_only else test_type
    info = (overlap_cutoff, hbond_allowance, bond_separation, intra_res, intra_mol,
                        clashes, output_grouping, test_type)
    if log:
        import io
        buffer = io.StringIO()
        buffer.write("<pre>")
        _file_output(buffer, info, naming_style)
        buffer.write("</pre>")
        session.logger.info(buffer.getvalue(), is_html=True)
    if save_file is not None:
        _file_output(save_file, info, naming_style)
    if summary:
        if clashes:
            total = 0
            for clash_list in clashes.values():
                total += len(clash_list)
            session.logger.status("%d %s" % (total/2, test_type), log=True)
        else:
            session.logger.status("No %s" % test_type, log=True)
    if not (set_attrs or color_atoms or make_pseudobonds or reveal):
        _xcmd(session, group_name)
        return clashes
    from chimerax.core.atomic import all_atoms
    if test == "self":
        attr_atoms = test_atoms
    elif test == "others" and inter_model:
        attr_atoms = all_atoms(session)
    else:
        attr_atoms = test_atoms.unique_structures.atoms
    from chimerax.core.atomic import Atoms
    clash_atoms = Atoms([a for a in attr_atoms if a in clashes])
    if set_attrs:
        # delete the attribute in _all_ atoms...
        for a in all_atoms(session):
            if hasattr(a, attr_name):
                delattr(a, attr_name)
        for a in clash_atoms:
            clash_vals = clashes[a].values()
            clash_vals.sort()
            setattr(a, attr_name, clash_vals[-1])
    if color_atoms:
        from chimerax.core.commands.scolor import scolor
        if atom_color is not None:
            clash_atoms.colors = atom_color.uint8x4()
            scolor(session, clash_atoms, atom_color)
        if nonatom_color is not None:
            noncolor_atoms = Atoms([a for a in attr_atoms if a not in clashes])
            noncolor_atoms.colors = nonatom_color.uint8x4()
            scolor(session, noncolor_atoms, nonatom_color)
    if reveal:
        # display sidechain or backbone as appropriate for undisplayed atoms
        reveal_atoms = clash_atoms.filter(clash_atoms.displays == False)
        sc_rv_atoms = reveal_atoms.filter(reveal_atoms.is_side_chains == True)
        if sc_rv_atoms:
            sc_res_atoms = sc_rv_atoms.unique_residues.atoms
            sc_res_atoms.filter(sc_res_atoms.is_side_chains == True).displays = True
        bb_rv_atoms = reveal_atoms.filter(reveal_atoms.is_backbones == True)
        if bb_rv_atoms:
            bb_res_atoms = bb_rv_atoms.unique_residues.atoms
            bb_res_atoms.filter(bb_res_atoms.is_backbones == True).displays = True
        # also reveal non-polymeric atoms
        reveal_atoms.displays = True
    if make_pseudobonds:
        if len(clash_atoms.unique_structures) > 1:
            pbg = session.pb_manager.get_group(group_name)
        else:
            pbg = clash_atoms[0].structure.pseudobond_group(group_name)
        pbg.clear()
        pbg.radius = pb_radius
        if pb_color is not None:
            pbg.color = pb_color.uint8x4()
        seen = set()
        for a in clash_atoms:
            seen.add(a)
            for clasher in clashes[a].keys():
                if clasher in seen:
                    continue
                pbg.new_pseudobond(a, clasher)
    else:
        _xcmd(session, group_name)
    return clashes

def _file_output(file_name, info, naming_style):
    overlap_cutoff, hbond_allowance, bond_separation, intra_res, intra_mol, \
                        clashes, output_grouping, test_type = info
    from chimerax.core.io import open_filename
    out_file = open_filename(file_name, 'w')
    if test_type != "distances":
        print("Allowed overlap: %g" % overlap_cutoff, file=out_file)
        print("H-bond overlap reduction: %g" % hbond_allowance, file=out_file)
    print("Ignore %s between atoms separated by %d bonds or less" % (test_type, bond_separation),
        file=out_file)
    print("Detect intra-residue %s:" % test_type, intra_res, file=out_file)
    print("Detect intra-molecule %s:" % test_type, intra_mol, file=out_file)
    seen = set()
    data = []
    from chimerax.core.geometry import distance
    for a, aclashes in clashes.items():
        for c, val in aclashes.items():
            if (c, a) in seen:
                continue
            seen.add((a, c))
            if a in output_grouping:
                out1, out2 = a, c
            else:
                out1, out2 = c, a
            l1, l2 = out1.__str__(style=naming_style), out2.__str__(style=naming_style)
            data.append((val, l1, l2, distance(out1.scene_coord, out2.scene_coord)))
    data.sort()
    data.reverse()
    print("\n%d %s" % (len(data), test_type), file=out_file)
    field_width1 = max([len(l1) for v, l1, l2, d in data] + [5])
    field_width2 = max([len(l2) for v, l1, l2, d in data] + [5])
    #print("%*s  %*s  overlap  distance" % (0-field_width1, "atom1", 0-field_width2, "atom2"),
    print(f"{'atom1':^{field_width1}}  {'atom2':^{field_width2}}  overlap  distance",
        file=out_file)
    for v, l1, l2, d in data:
        print(f"%*s  %*s   %5.3f    %5.3f" % (0-field_width1, l1, 0-field_width2, l2, v, d),
            file=out_file)
    if file_name != out_file:
        # only close file if we opened it...
        out_file.close()

def cmd_xclashes(session, group_name="clashes"):
    _xcmd(session, group_name)

def cmd_xcontacts(session, group_name="contacts"):
    _xcmd(session, group_name)

def _xcmd(session, group_name):
    if getattr(session, _continuous_attr, None) != None:
        get_triggers(session).remove_handler(getattr(session, _continuous_attr))
    pbg = session.pb_manager.get_group(group_name, create=False)
    pbgs = [pbg] if pbg else []
    from chimerax.core.atomic import AtomicStructure
    for s in [m for m in session.models if isinstance(m, AtomicStructure)]:
        pbg = s.pseudobond_group(group_name, create_type=None)
        if pbg:
            pbgs.append(pbg)
    if pbgs:
        session.models.close(pbgs)

def register_command(command_name, logger):
    from chimerax.core.commands \
        import CmdDesc, register, BoolArg, FloatArg, ColorArg, Or, EnumOf, AtomsArg, NoneArg, \
            SaveFileNameArg, NonNegativeIntArg, StringArg, AttrNameArg
    if command_name in ["clashes", "contactz"]:
        kw = { 'required': [('test_atoms', AtomsArg)],
            'keyword': [('group_name', StringArg), ('hbond_allowance', FloatArg),
                ('overlap_cutoff', FloatArg), ('atom_color', Or(NoneArg,ColorArg)),
                ('attr_name', AttrNameArg), ('bond_separation', NonNegativeIntArg),
                ('color_atoms', BoolArg), ('continuous', BoolArg), ('distance_only', FloatArg),
                ('inter_model', BoolArg), ('inter_submodel', BoolArg), ('intra_mol', BoolArg),
                ('intra_res', BoolArg), ('log', BoolArg), ('make_pseudobonds', BoolArg),
                ('naming_style', EnumOf(('simple', 'command', 'serial'))),
                ('nonatom_color', Or(NoneArg,ColorArg)), ('pb_color', Or(NoneArg,ColorArg)),
                ('pb_radius', FloatArg), ('reveal', BoolArg), ('save_file', SaveFileNameArg),
                ('set_attrs', BoolArg), ('select', BoolArg), ('summary', BoolArg),
                ('test', Or(EnumOf(('others', 'self')), AtomsArg))], }
        register('clashes', CmdDesc(**kw, synopsis="Find clashes"), cmd_clashes, logger=logger)
        register('contactz', CmdDesc(**kw, synopsis="Find contacts"), cmd_contacts, logger=logger)
    else:
        kw = { 'keyword': [('group_name', StringArg)] }
        desc = CmdDesc(keyword = [('group_name', StringArg)], synopsis = 'Clear hydrogen bonds')
        register('~clashes', CmdDesc(synopsis="Remove clash pseudobonds", **kw), cmd_xclashes,
            logger=logger)
        register('~contacts', CmdDesc(synopsis="Remove contact pseudobonds", **kw), cmd_xcontacts,
            logger=logger)
