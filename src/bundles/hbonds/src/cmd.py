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

from .hbond import rec_dist_slop, rec_angle_slop, find_hbonds, find_coordset_hbonds

from chimerax.core.atomic import AtomicStructure, Atoms
from chimerax.core.colors import BuiltinColors

def cmd_hbonds(session, atoms, intra_model=True, inter_model=True, relax=True,
    dist_slop=rec_dist_slop, angle_slop=rec_angle_slop, two_colors=False,
    restrict="any", radius=AtomicStructure.default_hbond_radius, save_file=None, batch=False,
    inter_submodel=False, make_pseudobonds=True, retain_current=False,
    reveal=False, naming_style=None, log=False, cache_DA=None,
    color=AtomicStructure.default_hbond_color, slop_color=BuiltinColors["dark orange"],
    show_dist=False, intra_res=True, intra_mol=True, dashes=None,
    salt_only=False, name="hydrogen bonds", per_coordset=True):

    """Wrapper to be called by command line.

       Use hbonds.find_hbonds for other programming applications.
    """

    if atoms is None:
        from chimerax.core.atomic import concatenate
        structures_atoms = [m.atoms for m in session.models if isinstance(m, AtomicStructure)]
        if structures_atoms:
            atoms = concatenate(structures_atoms)
        else:
            atoms = Atoms()

    from chimerax.core.errors import UserError
    if not atoms and not batch:
        raise UserError("Atom specifier selects no atoms")

    bond_color = color

    if restrict == "both":
        donors = acceptors = atoms
        structures = atoms.unique_structures
    elif restrict in ["cross", "any"]:
        donors = acceptors = None
        if inter_model:
            structures = [m for m in session.models if isinstance(m, AtomicStructure)]
        else:
            structures = atoms.unique_structures
    else: # another Atom collection
        if not restrict and not batch:
            raise UserError("'restrict' atom specifier selects no atoms")
        combined = atoms | restrict
        donors = acceptors = combined
        structures = combined.unique_structures

    if not relax:
        dist_slop = angle_slop = 0.0

    base_kw = {
        'inter_model': inter_model,
        'intra_model': intra_model,
        'donors': donors,
        'acceptors': acceptors,
        'inter_submodel': inter_submodel,
        'cache_da': cache_DA
    }

    doing_per_coordset = per_coordset and len(structures) == 1 and structures[0].num_coordsets > 1
    if doing_per_coordset:
        hb_func = find_coordset_hbonds
        struct_info = structures[0]
    else:
        hb_func = find_hbonds
        struct_info = structures

    result = hb_func(session, struct_info, dist_slop=dist_slop, angle_slop=angle_slop, **base_kw)
    if doing_per_coordset:
        hb_lists = result
    else:
        hb_lists = [result]
    for hbonds in hb_lists:
        # filter on salt bridges first, since we need access to all H-bonds in order
        # to assess which histidines should be considered salt-bridge donors
        if salt_only:
            sb_donors, sb_acceptors = salt_preprocess(hbonds)
            hbonds = [hb for hb in hbonds
                if hb[0] in sb_donors and hb[1] in sb_acceptors]
        hbonds[:] = restrict_hbonds(hbonds, atoms, restrict)
        if not intra_mol:
            mol_num = 0
            mol_map = {}
            for s in structures:
                for m in s.molecules:
                    mol_num += 1
                    for a in m:
                        mol_map[a] = mol_num
            hbonds[:] = [hb for hb in hbonds if mol_map[hb[0]] != mol_map[hb[1]]]
        if not intra_res:
            hbonds[:] = [hb for hb in hbonds if hb[0].residue != hb[1].residue]


    if doing_per_coordset:
        cs_ids = structures[0].coordset_ids
        output_info = (inter_model, intra_model, relax, dist_slop, angle_slop,
                                structures, hb_lists, cs_ids)
    else:
        output_info = (inter_model, intra_model, relax, dist_slop, angle_slop,
                                structures, result, None)
    if log:
        import io
        buffer = io.StringIO()
        buffer.write("<pre>")
        _file_output(buffer, output_info, naming_style)
        buffer.write("</pre>")
        session.logger.info(buffer.getvalue(), is_html=True)
    if save_file is not None:
        _file_output(save_file, output_info, naming_style)

    if doing_per_coordset:
        session.logger.status("%d hydrogen bonds found in %d coordsets" % (sum([len(hbs)
            for hbs in hb_lists]), len(cs_ids)), log=True, blank_after=120)
    else:
        session.logger.status("%d hydrogen bonds found" % len(result), log=True, blank_after=120)
    if not make_pseudobonds:
        return

    if two_colors:
        # color relaxed constraints differently
        precise_result = hb_func(session, struct_info, **base_kw)
        if doing_per_coordset:
            precise_lists = precise_result
        else:
            precise_lists = [precise_result]
        for precise in precise_lists:
            precise[:] = restrict_hbonds(precise, atoms, restrict)
            if not intra_mol:
                precise[:] = [hb for hb in precise is mol_map[hb[0]] != mol_map[hb[1]]]
            if not intra_res:
                precise[:] = [hb for hb in precise if hb[0].residue != hb[1].residue]
            if salt_only:
                precise[:] = [hb for hb in precise
                    if hb[0] in sb_donors and hb[1] in sb_acceptors]
        # give another opportunity to read the result...
        if doing_per_coordset:
            session.logger.status("%d strict hydrogen bonds found in %d coordsets" % (sum([len(hbs)
                for hbs in precise_lists]), len(cs_ids)), log=True, blank_after=120)
        else:
            session.logger.status("%d strict hydrogen bonds found" % len(precise_result),
                log=True, blank_after=120)

    # a true inter-model computation should be placed in a global group, otherwise
    # into individual per-structure groups
    submodels = False
    m_ids = set()
    for s in structures:
        m_id = s.id[:-1] if len(s.id) > 1 else s.id
        if m_id in m_ids:
            submodels = True
        m_ids.add(m_id)
    global_comp = (inter_model and len(m_ids) > 1) or (submodels and inter_submodel)
    if global_comp:
        # global comp nukes per-structure groups it covers if intra-model also
        if intra_model and not retain_current:
            closures = []
            for s in structures:
                pbg = s.pseudobond_group(name, create_type=None)
                if pbg:
                    closures.append(pbg)
            if closures:
                session.models.close(closures)
        hb_info = [(result, session.pb_manager.get_group(name))]
    elif doing_per_coordset:
        hb_info = [(result, structures[0].pseudobond_group(name, create_type="coordset"))]
    else:
        per_structure = {s:[] for s in structures}
        for hb in result:
            per_structure[hb[0].structure].append(hb)
        hb_info = [(hbs, s.pseudobond_group(name, create_type="coordset"))
            for s, hbs in per_structure.items()]

    for grp_hbonds, pbg in hb_info:
        if not retain_current:
            pbg.clear()
            pbg.color = bond_color.uint8x4()
            pbg.radius = radius
            pbg.dashes = dashes if dashes is not None else AtomicStructure.default_hbond_dashes
        else:
            if dashes is not None:
                pbg.dashes = dashes
        if not doing_per_coordset:
            grp_hbonds = [grp_hbonds]

        for i, cs_hbonds in enumerate(grp_hbonds):
            pre_existing = {}
            if doing_per_coordset:
                cs_id = cs_ids[i]
                pbg_pseudobonds = pbg.get_pseudobonds(cs_id)
            else:
                pbg_pseudobonds = pbg.pseudobonds
            if two_colors:
                precise = set(precise_lists[i])
            if retain_current:
                for pb in pbg_pseudobonds:
                    pre_existing[pb.atoms] = pb

            from chimerax.core.geometry import distance_squared
            for don, acc in cs_hbonds:
                nearest = None
                heavy_don = don
                for h in [x for x in don.neighbors if x.element.number == 1]:
                    sqdist = distance_squared(h.scene_coord, acc.scene_coord)
                    if nearest is None or sqdist < nsqdist:
                        nearest = h
                        nsqdist = sqdist
                if nearest is not None:
                    don = nearest
                if (don,acc) in pre_existing:
                    pb = pre_existing[(don,acc)]
                else:
                    if doing_per_coordset:
                        pb = pbg.new_pseudobond(don, acc, cs_id)
                    else:
                        pb = pbg.new_pseudobond(don, acc)
                if two_colors:
                    if (heavy_don, acc) in precise:
                        color = bond_color
                    else:
                        color = slop_color
                else:
                    color = bond_color
                rgba = pb.color
                rgba[:3] = color.uint8x4()[:3] # preserve transparency
                pb.color = rgba
                pb.radius = radius
                if reveal:
                    for end in [don, acc]:
                        if end.display:
                            continue
                        for ea in end.residue.atoms:
                            ea.display = True
        if pbg.id is None:
            session.models.add([pbg])

        if show_dist:
            session.pb_dist_monitor.add_group(pbg)
        else:
            session.pb_dist_monitor.remove_group(pbg)

def restrict_hbonds(hbonds, atoms, restrict):
    filtered = []
    both = restrict == "both"
    cross = restrict == "cross"
    if not isinstance(restrict, str):
        custom_atoms = set(restrict)
    else:
        custom_atoms = None
    for d, a in hbonds:
        d_in = d in atoms
        a_in = a in atoms
        num = a_in + d_in
        if num == 0:
            continue
        if custom_atoms != None:
            if not ((d in custom_atoms and a_in)
                    or (a in custom_atoms and d_in)):
                continue
        else:
            if num == 1:
                if both:
                    continue
            elif cross:
                continue
        filtered.append((d, a))
    return filtered

def _file_output(file_name, output_info, naming_style):
    inter_model, intra_model, relax_constraints, \
            dist_slop, angle_slop, structures, hbond_info, cs_ids = output_info
    from chimerax.core.io import open_filename
    out_file = open_filename(file_name, 'w')
    if inter_model:
        out_file.write("Finding intermodel H-bonds\n")
    if intra_model:
        out_file.write("Finding intramodel H-bonds\n")
    if relax_constraints:
        out_file.write("Constraints relaxed by %g angstroms"
            " and %d degrees\n" % (dist_slop, angle_slop))
    else:
        out_file.write("Using precise constraint criteria\n")
    out_file.write("Models used:\n")
    for s in structures:
        out_file.write("\t%s %s\n" % (s.id_string(), s.name))
    if cs_ids is None:
        hbond_lists = [hbond_info]
    else:
        hbond_lists = hbond_info

    for i, hbonds in enumerate(hbond_lists):
        if cs_ids is None:
            cs_id = None
        else:
            cs_id = cs_ids[i]
            out_file.write("\nCoordinate set %d" % cs_id)
        out_file.write("\nH-bonds (donor, acceptor, hydrogen, D..A dist, D-H..A dist):\n")
        # want the bonds listed in some kind of consistent order...
        hbonds.sort()

        # figure out field widths to make things line up
        dwidth = awidth = hwidth = 0
        labels = {}
        from chimerax.core.geometry import distance
        for don, acc in hbonds:
            if cs_id is None:
                don_coord = don.scene_coord
                acc_coord = acc.scene_coord
            else:
                don_coord = don.get_coordset_coord(cs_id)
                acc_coord = acc.get_coordset_coord(cs_id)
            labels[don] = don.__str__(style=naming_style)
            labels[acc] = acc.__str__(style=naming_style)
            dwidth = max(dwidth, len(labels[don]))
            awidth = max(awidth, len(labels[acc]))
            da = distance(don_coord, acc_coord)
            dha = None
            for h in don.neighbors:
                if h.element.number != 1:
                    continue
                if cs_id is None:
                    h_coord = h.scene_coord
                else:
                    h_coord = h.get_coordset_coord(cs_id)
                d = distance(h_coord, acc_coord)
                if dha is None or d < dha:
                    dha = d
                    hyd = h
            if dha is None:
                dha_out = "N/A"
                hyd_out = "no hydrogen"
            else:
                dha_out = "%5.3f" % dha
                hyd_out = hyd.__str__(style=naming_style)
            hwidth = max(hwidth, len(hyd_out))
            labels[(don, acc)] = (hyd_out, da, dha_out)
        for don, acc in hbonds:
            hyd_out, da, dha_out = labels[(don, acc)]
            out_file.write("%*s  %*s  %*s  %5.3f  %s\n" % (
                0-dwidth, labels[don], 0-awidth, labels[acc],
                0-hwidth, hyd_out, da, dha_out))
    if out_file != file_name:
        # we opened it, so close it...
        out_file.close()

def cmd_xhbonds(session, name="hydrogen bonds"):
    pbg = session.pb_manager.get_group(name, create=False)
    pbgs = [pbg] if pbg else []
    for s in [m for m in session.models if isinstance(m, AtomicStructure)]:
        pbg = s.pseudobond_group(name, create_type=None)
        if pbg:
            pbgs.append(pbg)
    if pbgs:
        session.models.close(pbgs)

def salt_preprocess(hbonds):
    donors = set()
    acceptors = set()
    his_data = {}
    his_names = ("HIS", "HIP") # HID/HIE cannot form salt bridge, so ignore them
    histidines = set()
    structures = set()
    for d, a in hbonds:
        if d.idatm_type[-1] == '+':
            donors.add(d)
        if a.idatm_type[-1] == '-':
            acceptors.add(a)
        if d.residue.name in his_names and d.name in ("NE2", "ND1"):
            histidines.add(d.residue)
            structures.add(d.structure)
            d_hbs, a_hbs, = his_data.setdefault(d.residue, (set(), set()))
            d_hbs.add((d, a))
        if a.residue.name in his_names and a.name in ("NE2", "ND1"):
            histidines.add(a.residue)
            structures.add(a.structure)
            d_hbs, a_hbs, = his_data.setdefault(a.residue, (set(), set()))
            a_hbs.add((d, a))
    # histidines involved in metal coordination can't be salt-bridge donors
    # so identify those
    for s in structures:
        coord_group = s.pseudobond_group(s.PBG_METAL_COORDINATION, create_type=None)
        if not coord_group:
            continue
        for pb in coord_group.pseudobonds:
            for a in pb.atoms:
                if a.name in ("NE2", "ND1"):
                    histidines.discard(a.residue)
    for his in histidines:
        ne = his.find_atom("NE2")
        nd = his.find_atom("ND1")
        if not ne or not nd:
            continue
        if his.name == "HIP":
            donors.add(ne)
            donors.add(nd)
            continue
        import numpy
        ne_has_protons = numpy.any(ne.neighbors.elements.numbers == 1)
        nd_has_protons = numpy.any(nd.neighbors.elements.numbers == 1)
        if ne_has_protons and nd_has_protons:
            donors.add(ne)
            donors.add(nd)
            continue
        if ne_has_protons or nd_has_protons:
            continue
        # Okay, implicitly protonated HIS residue; assume it's a salt-bridge
        # donor unless one of the nitrogens is unambigously an acceptor...
        d_hbs, a_hbs = his_data[his]
        for d, a in a_hbs:
            if (a, d) not in d_hbs:
                break
        else:
            donors.add(ne)
            donors.add(nd)
    return donors, acceptors

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
                ('name', StringArg), ('per_coordset', BoolArg)],
            synopsis = 'Find hydrogen bonds'
        )
        register('hbonds', desc, cmd_hbonds, logger=logger)
    else:
        desc = CmdDesc(keyword = [('name', StringArg)], synopsis = 'Clear hydrogen bonds')
        register('~hbonds', desc, cmd_xhbonds, logger=logger)
