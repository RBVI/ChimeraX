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
        name="clashes",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["clash_threshold"],
        **kw):
    if 'color' in kw:
        color = kw.pop('color')
    else:
        color = defaults['clash_pb_color']
    if 'radius' in kw:
        radius = kw.pop('radius')
    else:
        radius = defaults['clash_pb_radius']
    if 'dashes' not in kw:
        kw['dashes'] = 4
    return _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, "clashes", color, radius, **kw)

def cmd_contacts(session, test_atoms, *,
        name="contacts",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["contact_threshold"],
        **kw):
    if 'color' in kw:
        color = kw.pop('color')
    else:
        color = defaults['contact_pb_color']
    if 'radius' in kw:
        radius = kw.pop('radius')
    else:
        radius = defaults['contact_pb_radius']
    return _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, "contacts", color, radius, **kw)

_continuous_attr = "_clashes_continuous_id"
def _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, test_type, color, radius, *,
        atom_color=defaults["atom_color"],
        attr_name=defaults["attr_name"],
        bond_separation=defaults["bond_separation"],
        color_atoms=defaults["action_color"],
        continuous=False,
        dashes=None,
        distance_only=None,
        inter_model=True,
        inter_submodel=False,
        intra_mol=defaults["intra_mol"],
        intra_res=defaults["intra_res"],
        log=defaults["action_log"],
        make_pseudobonds=defaults["action_pseudobonds"],
        naming_style=None,
        other_atom_color=defaults["other_atom_color"],
        res_separation=None,
        reveal=False,
        save_file=None,
        set_attrs=defaults["action_attr"],
        select=defaults["action_select"],
        show_dist=False,
        summary=True,
        test="others"):
    from chimerax.core.errors import UserError
    if not test_atoms:
        raise UserError("No atoms in given atom specifier")
    from chimerax.core.colors import Color
    if atom_color is not None and not isinstance(atom_color, Color):
        atom_color = Color(rgba=atom_color)
    if other_atom_color is not None and not isinstance(other_atom_color, Color):
        other_atom_color = Color(rgba=other_atom_color)
    if color is not None and not isinstance(color, Color):
        color = Color(rgba=color)
    from chimerax.atomic import get_triggers
    ongoing = False
    if continuous:
        if set_attrs or save_file != None or log:
            raise UserError("log/setAttrs/saveFile not allowed with continuous detection")
        if getattr(session, _continuous_attr, None) == None:
            from inspect import getargvalues, currentframe
            arg_names, fArgs, fKw, frame_dict = getargvalues(currentframe())
            call_data = [frame_dict[an] for an in arg_names]
            def changes_cb(trig_name, changes, session=session, call_data=call_data):
                s_reasons = changes.atomic_structure_reasons()
                a_reasons = changes.atom_reasons()
                if 'position changed' in s_reasons \
                or 'active_coordset changed' in s_reasons \
                or 'coord changed' in a_reasons \
                or 'alt_loc changed' in a_reasons:
                    if not call_data[1]:
                        # all atoms gone
                        delattr(session, _continuous_attr)
                        from chimerax.core.triggerset import DEREGISTER
                        return DEREGISTER
                    _cmd(*tuple(call_data))
            setattr(session, _continuous_attr, get_triggers().add_handler(
                        'changes', changes_cb))
        else:
            ongoing = True
    elif getattr(session, _continuous_attr, None) != None:
        get_triggers().remove_handler(getattr(session, _continuous_attr))
        delattr(session, _continuous_attr)
    from .clashes import find_clashes
    clashes = find_clashes(session, test_atoms, attr_name=attr_name,
        bond_separation=bond_separation, clash_threshold=overlap_cutoff,
        distance_only=distance_only, hbond_allowance=hbond_allowance,
        inter_model=inter_model, inter_submodel=inter_submodel, intra_res=intra_res,
        intra_mol=intra_mol, res_separation=res_separation, test=test)
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
            session.logger.status("%d %s" % (total/2, test_type), log=not ongoing)
        else:
            session.logger.status("No %s" % test_type, log=not ongoing)
    if not (set_attrs or color_atoms or make_pseudobonds or reveal):
        _xcmd(session, name)
        return clashes
    from chimerax.atomic import all_atoms
    if test == "self":
        attr_atoms = test_atoms
    elif test == "others":
        if inter_model:
            attr_atoms = all_atoms(session)
        else:
            attr_atoms = test_atoms.unique_structures.atoms
    else:
        from chimerax.atomic import concatenate
        attr_atoms = concatenate([test_atoms, test], remove_duplicates=True)
    from chimerax.atomic import Atoms
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
        from chimerax.core.commands.color import color_surfaces_at_atoms
        if atom_color is not None:
            clash_atoms.colors = atom_color.uint8x4()
            color_surfaces_at_atoms(clash_atoms, atom_color)
        if other_atom_color is not None:
            other_color_atoms = Atoms([a for a in attr_atoms if a not in clashes])
            other_color_atoms.colors = other_atom_color.uint8x4()
            color_surfaces_at_atoms(other_color_atoms, other_atom_color)
    if reveal:
        # display sidechain or backbone as appropriate for undisplayed atoms
        reveal_atoms = clash_atoms.filter(clash_atoms.displays == False)
        reveal_residues = reveal_atoms.unique_residues
        sc_rv_atoms = reveal_atoms.filter(reveal_atoms.is_side_chains == True)
        if sc_rv_atoms:
            sc_residues = sc_rv_atoms.unique_residues
            sc_res_atoms = sc_residues.atoms
            sc_res_atoms.filter(sc_res_atoms.is_side_chains == True).displays = True
            reveal_residues = reveal_residues - sc_residues
        bb_rv_atoms = reveal_atoms.filter(reveal_atoms.is_backbones() == True)
        if bb_rv_atoms:
            bb_residues = bb_rv_atoms.unique_residues
            bb_res_atoms = bb_residues.atoms
            bb_res_atoms.filter(bb_res_atoms.is_backbones() == True).displays = True
            reveal_residues = reveal_residues - bb_residues
        # also reveal non-polymeric atoms
        reveal_residues.atoms.displays = True
    if make_pseudobonds:
        if len(attr_atoms.unique_structures) > 1:
            pbg = session.pb_manager.get_group(name)
        else:
            pbg = attr_atoms[0].structure.pseudobond_group(name)
        pbg.clear()
        pbg.radius = radius
        if color is not None:
            pbg.color = color.uint8x4()
        if dashes is not None:
            pbg.dashes = dashes
        seen = set()
        for a in clash_atoms:
            seen.add(a)
            for clasher in clashes[a].keys():
                if clasher in seen:
                    continue
                pbg.new_pseudobond(a, clasher)
        if show_dist:
            session.pb_dist_monitor.add_group(pbg)
        else:
            session.pb_dist_monitor.remove_group(pbg)
        if pbg.id is None:
            session.models.add([pbg])
    else:
        _xcmd(session, name)
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
            l1, l2 = out1.string(style=naming_style), out2.string(style=naming_style)
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

def cmd_xclashes(session, name="clashes"):
    _xcmd(session, name)

def cmd_xcontacts(session, name="contacts"):
    _xcmd(session, name)

def _xcmd(session, group_name):
    if getattr(session, _continuous_attr, None) != None:
        from chimerax.atomic import get_triggers
        get_triggers().remove_handler(getattr(session, _continuous_attr))
        delattr(session, _continuous_attr)
    pbg = session.pb_manager.get_group(group_name, create=False)
    pbgs = [pbg] if pbg else []
    from chimerax.atomic import AtomicStructure
    for s in [m for m in session.models if isinstance(m, AtomicStructure)]:
        pbg = s.pseudobond_group(group_name, create_type=None)
        if pbg:
            pbgs.append(pbg)
    if pbgs:
        session.models.close(pbgs)

def register_command(command_name, logger):
    from chimerax.core.commands \
        import CmdDesc, register, BoolArg, FloatArg, ColorArg, Or, EnumOf, NoneArg, \
            SaveFileNameArg, NonNegativeIntArg, StringArg, AttrNameArg, PositiveIntArg
    from chimerax.atomic import AtomsArg
    if command_name in ["clashes", "contacts"]:
        kw = { 'required': [('test_atoms', AtomsArg)],
            'keyword': [('name', StringArg), ('hbond_allowance', FloatArg),
                ('overlap_cutoff', FloatArg), ('atom_color', Or(NoneArg,ColorArg)),
                ('attr_name', AttrNameArg), ('bond_separation', NonNegativeIntArg),
                ('color_atoms', BoolArg), ('continuous', BoolArg), ('distance_only', FloatArg),
                ('inter_model', BoolArg), ('inter_submodel', BoolArg), ('intra_mol', BoolArg),
                ('intra_res', BoolArg), ('log', BoolArg), ('make_pseudobonds', BoolArg),
                ('naming_style', EnumOf(('simple', 'command', 'serial'))),
                ('other_atom_color', Or(NoneArg,ColorArg)), ('color', Or(NoneArg,ColorArg)),
                ('radius', FloatArg), ('res_separation', PositiveIntArg), ('reveal', BoolArg),
                ('save_file', SaveFileNameArg), ('set_attrs', BoolArg), ('select', BoolArg),
                ('show_dist', BoolArg), ('dashes', NonNegativeIntArg),
                ('summary', BoolArg), ('test', Or(EnumOf(('others', 'self')), AtomsArg))], }
        register('clashes', CmdDesc(**kw, synopsis="Find clashes"), cmd_clashes, logger=logger)
        register('contacts', CmdDesc(**kw, synopsis="Find contacts", url="help:user/commands/clashes.html"),
            cmd_contacts, logger=logger)
    else:
        kw = { 'keyword': [('name', StringArg)] }
        register('~clashes', CmdDesc(synopsis="Remove clash pseudobonds", **kw), cmd_xclashes, logger=logger)
        register('~contacts', CmdDesc(synopsis="Remove contact pseudobonds",
            url="help:user/commands/clashes.html", **kw), cmd_xcontacts, logger=logger)
