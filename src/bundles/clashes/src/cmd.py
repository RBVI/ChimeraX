# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .settings import defaults

def cmd_clashes(session, test_atoms, *,
        name="clashes",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["clash_threshold"],
        **kw):
    color, radius = handle_clash_kw(kw)
    return _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, "clashes", color, radius, **kw)

def handle_clash_kw(kw):
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
    return color, radius

def cmd_contacts(session, test_atoms, *,
        name="contacts",
        hbond_allowance=defaults["clash_hbond_allowance"],
        overlap_cutoff=defaults["contact_threshold"],
        **kw):
    color, radius = handle_contact_kw(kw)
    return _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, "contacts", color, radius, **kw)

def handle_contact_kw(kw):
    if 'color' in kw:
        color = kw.pop('color')
    else:
        color = defaults['contact_pb_color']
    if 'radius' in kw:
        radius = kw.pop('radius')
    else:
        radius = defaults['contact_pb_radius']
    if 'dashes' not in kw:
        kw['dashes'] = 6
    return color, radius

_continuous_attr = "_clashes_continuous_id"
def _cmd(session, test_atoms, name, hbond_allowance, overlap_cutoff, test_type, color, radius, *,
        attr_name=defaults["attr_name"],
        bond_separation=defaults["bond_separation"],
        continuous=False,
        dashes=None,
        distance_only=None,
        ignore_hidden_models=defaults["ignore_hidden_models"],
        inter_model=True,
        inter_submodel=False,
        intra_model=True,
        intra_mol=defaults["intra_mol"],
        intra_res=defaults["intra_res"],
        log=defaults["action_log"],
        make_pseudobonds=defaults["action_pseudobonds"],
        naming_style=None,
        res_separation=None,
        restrict="any",
        reveal=False,
        save_file=None,
        set_attrs=defaults["action_attr"],
        select=defaults["action_select"],
        show_dist=False,
        summary=True):
    from chimerax.core.errors import UserError
    if test_atoms is None:
        from chimerax.atomic import AtomicStructure, AtomicStructures
        test_atoms = AtomicStructures([s for s in session.models if isinstance(s, AtomicStructure)]).atoms
    if not test_atoms:
        raise UserError("No atoms match given atom specifier")
    from chimerax.core.colors import Color
    if color is not None and not isinstance(color, Color):
        color = Color(rgba=color)
    from chimerax.atomic import get_triggers
    ongoing = False
    if continuous:
        if set_attrs or save_file != None or log:
            raise UserError("log/setAttrs/saveFile not allowed with continuous detection")
        if getattr(session, _continuous_attr, None) == None:
            from inspect import getargvalues, currentframe, getfullargspec
            arg_names, fArgs, fKw, frame_dict = getargvalues(currentframe())
            arg_spec = getfullargspec(_cmd)
            args = [frame_dict[an] for an in arg_names[:len(arg_spec.args)]]
            kw = { k:frame_dict[k] for k in arg_names[len(arg_spec.args):] }
            call_data = (args, kw)
            def changes_cb(trig_name, changes, session=session, call_data=call_data):
                s_reasons = changes.atomic_structure_reasons()
                a_reasons = changes.atom_reasons()
                if 'position changed' in s_reasons \
                or 'active_coordset changed' in s_reasons \
                or 'coord changed' in a_reasons \
                or 'alt_loc changed' in a_reasons:
                    args, kw = call_data
                    if not args[1]:
                        # all atoms gone
                        delattr(session, _continuous_attr)
                        from chimerax.core.triggerset import DEREGISTER
                        return DEREGISTER
                    _cmd(*tuple(args), **kw)
            setattr(session, _continuous_attr, get_triggers().add_handler(
                        'changes', changes_cb))
        else:
            ongoing = True
    elif getattr(session, _continuous_attr, None) != None:
        get_triggers().remove_handler(getattr(session, _continuous_attr))
        delattr(session, _continuous_attr)
    from .clashes import find_clashes
    clashes = find_clashes(session, test_atoms, attr_name=attr_name, bond_separation=bond_separation,
        clash_threshold=overlap_cutoff, distance_only=distance_only, hbond_allowance=hbond_allowance,
        ignore_hidden_models=ignore_hidden_models, inter_model=inter_model, inter_submodel=inter_submodel,
        intra_model=intra_model, intra_res=intra_res, intra_mol=intra_mol, res_separation=res_separation,
        restrict=restrict)
    if select:
        session.selection.clear()
        for a in clashes.keys():
            a.selected = True
    # if relevant, put the test_atoms in the first column
    if restrict == "both":
        output_grouping = set()
    else:
        output_grouping = test_atoms
    test_type = "distances" if distance_only else test_type
    info = (overlap_cutoff, hbond_allowance, bond_separation, intra_res, intra_mol,
                        clashes, output_grouping, test_type, res_separation)
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
    if not (set_attrs or make_pseudobonds or reveal):
        _xcmd(session, name)
        return clashes
    from chimerax.atomic import all_atoms
    if restrict == "both":
        attr_atoms = test_atoms
    elif restrict in ("any", "cross"):
        if inter_model:
            attr_atoms = all_atoms(session)
        else:
            attr_atoms = test_atoms.unique_structures.atoms
    else:
        from chimerax.atomic import concatenate
        attr_atoms = concatenate([test_atoms, restrict], remove_duplicates=True)
    from chimerax.atomic import Atoms
    clash_atoms = Atoms([a for a in attr_atoms if a in clashes])
    if set_attrs:
        # delete the attribute in _all_ atoms...
        for a in all_atoms(session):
            if hasattr(a, attr_name):
                delattr(a, attr_name)
        for a in clash_atoms:
            clash_vals = list(clashes[a].values())
            clash_vals.sort()
            setattr(a, attr_name, clash_vals[-1])
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
                        clashes, output_grouping, test_type, res_separation = info
    from chimerax.io import open_output
    out_file = open_output(file_name, 'utf-8')
    if test_type == "distances":
        overlap_title = ""
        data_fmt = "%*s  %*s    %5.3f"
    else:
        overlap_title = "  overlap"
        data_fmt = "%*s  %*s   %5.3f    %5.3f"
        print("Allowed overlap: %g" % overlap_cutoff, file=out_file)
        print("H-bond overlap reduction: %g" % hbond_allowance, file=out_file)
    print("Ignore %s between atoms separated by %d bonds or less" % (test_type, bond_separation),
        file=out_file)
    if res_separation:
        print("Ignore %s between atoms in residues less than %d apart in sequence" % (test_type,
            res_separation), file=out_file)
    print("Detect intra-residue %s:" % test_type, intra_res, file=out_file)
    print("Detect intra-molecule %s:" % test_type, intra_mol, file=out_file)
    seen = set()
    data = []
    from chimerax.geometry import distance
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
    print(f"{'atom1':^{field_width1}}  {'atom2':^{field_width2}}{overlap_title}  distance",
        file=out_file)
    for v, l1, l2, d in data:
        if overlap_title:
            data = (0-field_width1, l1, 0-field_width2, l2, v, d)
        else:
            data = (0-field_width1, l1, 0-field_width2, l2, d)
        print(data_fmt % data, file=out_file)
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
        import CmdDesc, register, BoolArg, FloatArg, ColorArg, Or, EnumOf, NoneArg, EmptyArg, \
            SaveFileNameArg, NonNegativeIntArg, StringArg, AttrNameArg, PositiveIntArg
    from chimerax.atomic import AtomsArg
    del_kw = { 'keyword': [('name', StringArg)] }
    if command_name in ["clashes", "contacts"]:
        kw = { 'required': [('test_atoms', Or(AtomsArg,EmptyArg))],
            'keyword': [('name', StringArg), ('hbond_allowance', FloatArg), ('overlap_cutoff', FloatArg),
                ('attr_name', AttrNameArg), ('bond_separation', NonNegativeIntArg), ('continuous', BoolArg),
                ('distance_only', FloatArg), ('ignore_hidden_models', BoolArg), ('inter_model', BoolArg),
                ('inter_submodel', BoolArg), ('intra_model', BoolArg), ('intra_mol', BoolArg),
                ('intra_res', BoolArg), ('log', BoolArg), ('make_pseudobonds', BoolArg),
                ('naming_style', EnumOf(('simple', 'command', 'serial'))), ('color', Or(NoneArg,ColorArg)),
                ('radius', FloatArg), ('res_separation', PositiveIntArg),
                ('restrict', Or(EnumOf(('cross', 'both', 'any')), AtomsArg)), ('reveal', BoolArg),
                ('save_file', SaveFileNameArg), ('set_attrs', BoolArg), ('select', BoolArg),
                ('show_dist', BoolArg), ('dashes', NonNegativeIntArg), ('summary', BoolArg)], }
        register('clashes', CmdDesc(**kw, synopsis="Find clashes"), cmd_clashes, logger=logger)
        register('contacts', CmdDesc(**kw, synopsis="Find contacts", url="help:user/commands/clashes.html"),
            cmd_contacts, logger=logger)
        register('clashes delete',
            CmdDesc(synopsis="Remove clash pseudobonds", **del_kw), cmd_xclashes, logger=logger)
        register('contacts delete', CmdDesc(synopsis="Remove contact pseudobonds",
            url="help:user/commands/clashes.html", **del_kw), cmd_xcontacts, logger=logger)
    else:
        register('~clashes',
            CmdDesc(synopsis="Remove clash pseudobonds", **del_kw), cmd_xclashes, logger=logger)
        register('~contacts', CmdDesc(synopsis="Remove contact pseudobonds",
            url="help:user/commands/clashes.html", **del_kw), cmd_xcontacts, logger=logger)
