# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.atomic import all_atomic_structures, Residue, Structure, all_residues, concise_residue_spec, \
    Atoms, all_atoms

ball_and_stick = [
    "style ball",
    "~nuc",
    "~ribbon",
    "disp",
    "size H atomRadius 1.2",
    "size stickRadius 0.24",
    "size ballScale 0.3",
    "size pseudobondRadius 0.3",
    "size ions atomRadius +0.35"
]

base_setup = [
    "graphics bgcolor white",
    "color name marine 0,50,100",
    "color name forest 13.3,54.5,13.3",
    "color name tangerine 95.3,51.8,0",
    "color name grape 64.3,0,86.7",
    "color name nih_blue 12.5,33.3,54.1",
    "color name jmol_carbon 56.5,56.5,56.5",
    "color name bond_purple 57.6,43.9,85.9",
    "color name struts_grey 48,48,48",
    "color name carbon_grey 22.2,22.2,22.2",
    # remaining for undoing monkeyshines from previous presets
    "surface close",
    "preset 'initial styles' 'original look'",
]

base_macro_model = [
    "delete solvent",
    "delete H"
]

base_ribbon = [
    "preset 'initial styles' cartoon",
    "nucleotides ladder radius 1.2",
    "color white target abc", # in particular, not (metal-coordination) pseudobonds
    "color helix marine; color strand firebrick; color coil goldenrod; color nucleic-acid forest",
    "color :A:C:G:U grape",
    "color byatom",
    "select (C & ligand) | (C & ligand :< 5 & ~nucleic-acid) | (C & protein) | (C & disulfide)",
    "color sel carbon_grey atoms",
    "color ligand | protein & sideonly byhet atoms",
    "~select"
]

base_surface = [
    "delete solvent",
    "hide H|ligand|~(protein|nucleic-acid) atoms",
    "~nuc",
    "~ribbon",
    "~display",
]

color_by_het = [
    "color jmol_carbon",
    "color byhet"
]

print_ribbon = [
    # make missing-structure pseudobonds bigger relative to upcoming hbonds
    "size min-backbone pseudobondRadius 1.1",
    "size ions atomRadius +0.8",
    "cartoon style sides 16",
    "select backbone & protein | nucleic-acid & min-backbone | ions | ligand"
        " | ligand :< 5 & ~nucleic-acid",
    "hbonds sel color white restrict both",
    #"size hbonds pseudobondRadius 0.6",
    "size pseudobondRadius 0.6",
    # ribbons need to be up to date for struts to work right
    # also, these struts parameters need to be mirrored in AF_single_biggest()
    "wait 1; struts (@ca|ligand|P)&(@@display|::ribbon_display) length 8 loop 60 rad 0.75 color struts_grey",
    "~struts @PB,PG resetRibbon false",
    "~struts adenine|cytosine|guanine|thymine|uracil resetRibbon false",
    #"color struts_grey pseudobonds",
    "color hbonds white pseudobonds",
    "~select"
]

p3ms_pbg_name = "3D-printable missing structure"

undo_printable = [
    "~struts",
    "~hbonds",
    "close ##name='%s'" % p3ms_pbg_name,
    "show ##name='%s' models" % Structure.PBG_MISSING_STRUCTURE,
    "size atomRadius default stickRadius 0.2 pseudobondRadius 0.2",
    "style dashes 7",
    "graphics quality bondSides default",
    "graphics quality pseudobondSides 10",
    "cartoon style sides 12",
]

def addh_cmds(session):
    return [ "addh %s hb f" % s.atomspec for s in all_atomic_structures(session) if s.num_atoms < 25000 ]

def by_chain_cmds(session, rainbow=False, target_atoms=False):
    cmds = []
    for s in all_atomic_structures(session):
        if rainbow:
            cmds.append(rainbow_cmd(s, target_atoms=target_atoms))
        cmds.append("color zone %s near %s & main distance 20" % (s.atomspec, s.atomspec))
    return cmds

def check_AF(session, *, pae=False):
    from chimerax.core.errors import UserError
    structures = []
    if pae:
        for s in all_atomic_structures(session):
            if hasattr(s, 'alphafold_pae'):
                structures.append(s)
                continue
        missing = "PAE information"
    else:
        for s in all_atomic_structures(session):
            for r in s.residues:
                if hasattr(r, 'pLDDT_score'):
                    structures.append(s)
                    break
        missing = "pLDDT scores assigned"
    if not structures:
        raise UserError(f"No structures have {missing}!")
    return ''.join([s.atomspec for s in structures])

def color_by_hydrophobicity_cmds(session, target="rs"):
    kdh_info = {
        "asp": -3.5,
        "glu": -3.5,
        "asn": -3.5,
        "gln": -3.5,
        "lys": -3.9,
        "arg": -4.5,
        "his": -3.2,
        "gly": -0.4,
        "pro": -1.6,
        "ser": -0.8,
        "thr": -0.7,
        "cys": 2.5,
        "met": 1.9,
        "mse": 1.9,
        "ala": 1.8,
        "val": 4.2,
        "ile": 4.5,
        "leu": 3.8,
        "phe": 2.8,
        "trp": -0.9,
        "tyr": -1.3,
    }
    for s in all_atomic_structures(session):
        for r in s.residues:
            if r.name.lower() in kdh_info:
                r.kd_hydrophobicity = kdh_info[r.name.lower()]
    # need to register the attribute to make it known to 'color byattribute'
    Residue.register_attr(session, "kd_hydrophobicity", "NIH3D preset", attr_type=float)
    return [
        "color magenta",
        "color byattribute kd_hydrophobicity protein target %s palette 16,67,87:white:100,45,0"
            " novalue magenta" % target
    ]

def get_AF_surf_spec(session, printable):
    high = connected_high_AF_confidence(session, single_biggest=printable, surface=True)
    spec_lookup = {}
    for s in all_atomic_structures(session):
        s_residues = [r for r in high if r.structure == s]
        if s_residues:
            spec_lookup[s] = concise_residue_spec(session, s_residues)
        else:
            spec_lookup[s] = s.atomspec
    return spec_lookup

def hide_AF_low_confidence(session, printable):
    # When this is called, 'ribbon_display' may be different than when the commands this
    # generates are executed, so do not screen out residues to hide based on current ribbon_display
    high = connected_high_AF_confidence(session, single_biggest=printable)
    hide_residues = []
    for s in all_atomic_structures(session):
        for r in s.residues:
            if r not in high:
                hide_residues.append(r)
    if hide_residues:
        return ["~ribbon " + concise_residue_spec(session, hide_residues)]
    return []

def connected_high_AF_confidence(session, *, single_biggest=False, surface=False):
    high = set()
    if single_biggest:
        high.update(AF_single_biggest(session, surface))
    else:
        for s in all_atomic_structures(session):
            for chain in s.chains:
                connected = []
                for r in chain.residues:
                    if r and hasattr(r, 'pLDDT_score') and r.pLDDT_score > 50:
                        connected.append(r)
                    else:
                        if len(connected) > 3:
                            high.update(connected)
                        connected = []
                if len(connected) > 3:
                    high.update(connected)
    return high

def AF_single_biggest(session, surface):
    segments = []
    for s in all_atomic_structures(session):
        for chain in s.chains:
            segment = set()
            for r in chain.residues:
                if r and hasattr(r, 'pLDDT_score') and r.pLDDT_score > 50:
                    segment.add(r)
                else:
                    if len(segment) > 3:
                        segments.append(segment)
                    segment = set()
            if len(segment) > 3:
                segments.append(segment)
    if not segments:
        return set()
    class HashAtoms(Atoms):
        def __hash__(self):
            # Atoms are mutable, so no __hash__, but we know no atoms are going to be deleted during
            # the lifetime of this object, so...
            return id(self)
    segment_info = []
    for segment in segments:
        segment_info.append((len(segment), HashAtoms([a for r in segment for a in r.atoms])))
    segment_info.sort(key=lambda tup: -tup[0])
    biggest_size = 0
    if surface:
        from chimerax.clashes.clashes import find_clashes, defaults
    else:
        from chimerax.struts.struts import struts
        # struts (@ca|ligand|P)&(@@display|::ribbon_display) length 8 loop 60 rad 0.75 color struts_grey",
        plddt_atoms = HashAtoms()
        seg_lookup = {}
        seg_connect = {}
        for seg_len, seg_atoms in segment_info:
            seg_connect[seg_atoms] = set()
            for a  in seg_atoms:
                seg_lookup[a] = seg_atoms
            plddt_atoms += seg_atoms
        strutable_atoms = plddt_atoms.filter(
            (plddt_atoms.names == "CA") | (plddt_atoms.element_names == "P"))
        strutable_atoms.spec = "custom NIH preset"
        struts_pbg = struts(session, strutable_atoms, length=8, loop=60, radius=0.75, fatten_ribbon=False)
        for strut in struts_pbg.pseudobonds:
            a1, a2 = strut.atoms
            seg1, seg2 = seg_lookup[a1], seg_lookup[a2]
            if seg1 is seg2:
                continue
            if seg2 not in seg_connect[seg1]:
                print("Strut between", a1, "and", a2, "connects", concise_residue_spec(session, seg1.unique_residues), "with",
                    concise_residue_spec(session, seg2.unique_residues))
                seg_connect[seg1].add(seg2)
                seg_connect[seg2].add(seg1)
        session.models.close([struts_pbg])

    while sum([si[0] for si in segment_info]) > biggest_size:
        group_size, group_atoms = segment_info.pop(0)
        added_one = True
        while added_one:
            added_one = False
            for size, atoms in segment_info:
                if surface:
                    if find_clashes(session, group_atoms, clash_threshold=defaults["contact_threshold"],
                            restrict=atoms):
                        group_size += size
                        group_atoms += atoms
                        segment_info.remove((size, atoms))
                    else:
                        continue
                else:
                    # ribbon
                    if atoms in seg_connect[group_atoms]:
                        group_size += size
                        next_atoms = group_atoms + atoms
                        next_connect = set()
                        next_connect.update(seg_connect[group_atoms])
                        next_connect.update(seg_connect[atoms])
                        next_connect.discard(group_atoms)
                        next_connect.discard(atoms)
                        del seg_connect[atoms]
                        del seg_connect[group_atoms]
                        for seg, connected_segs in seg_connect.items():
                            if atoms in connected_segs:
                                connected_segs.discard(atoms)
                                connected_segs.add(next_atoms)
                            if group_atoms in connected_segs:
                                connected_segs.discard(group_atoms)
                                connected_segs.add(next_atoms)
                        seg_connect[next_atoms] = next_connect
                        group_atoms = next_atoms
                        segment_info.remove((size, atoms))
                    else:
                        continue
                added_one = True
                break
        if group_size > biggest_size:
            biggest_size = group_size
            biggest_group = group_atoms
    return set(biggest_group.unique_residues)

base_palette = ["marine", "goldenrod", "firebrick", "forest", "tangerine", "grape"]
def palette(num_chains):
    palette = base_palette[:]
    if num_chains < 2:
        return palette[0] + ':' + palette[0]
    while num_chains > len(palette):
        palette += base_palette
    return ':'.join(palette[:num_chains])

def print_prep(session=None, *, pb_radius=0.4, ion_size_increase=0.0, bond_sides="default", pb_sides=16):
    always_cmds = [
        "size stickRadius 0.8",
        "style dashes 0",
        "graphics quality bondSides %s" % bond_sides,
        "graphics quality pseudobondSides %s" % pb_sides,
    ]
    if session is None:
        # not a ribbon preset
        cmds = always_cmds
    else:
        # So that they connect to ribbon ends, have missing structure pseudobonds go from CA->CA [#8388]
        cmds = []
        from chimerax.atomic import all_atomic_structures, all_pseudobond_groups
        for pbg in all_pseudobond_groups(session)[:]:
            if pbg.name == p3ms_pbg_name:
                session.models.close([pbg])
        pb_file = None
        for s in all_atomic_structures(session):
            ms_pbg = s.pseudobond_group(s.PBG_MISSING_STRUCTURE, create_type=None)
            if ms_pbg is None:
                continue
            if pb_file is None:
                import tempfile
                pb_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pb', delete=False)
                import atexit, os
                atexit.register(os.unlink, pb_file.name)
                cmds += [
                    "open " + pb_file.name,
                    "rename ##name=%s '%s'" % (os.path.basename(pb_file.name), p3ms_pbg_name)
                ]
            cmds.append("hide %s models" % ms_pbg.atomspec)
            print("; halfbond = true", file=pb_file)
            for pb in ms_pbg.pseudobonds:
                a1, a2 = pb.atoms
                ca1 = a1.residue.find_atom("CA") or a1
                ca2 = a2.residue.find_atom("CA") or a2
                print("%s %s" % (ca1.string(style="command"), ca2.string(style="command")), file=pb_file)
        if pb_file is not None:
            pb_file.close()

        cmds += always_cmds
    if pb_radius is not None:
        cmds += [ "size pseudobondRadius %g" % pb_radius ]
    if ion_size_increase:
        cmds += ["size ions atomRadius %+g" % ion_size_increase]
    return cmds

def rainbow_cmd(structure, target_atoms=False):
    target_arg = "target rfs%s" % ("a" if target_atoms else "")
    from chimerax.mmcif import get_mmcif_tables_from_metadata
    if structure.num_chains > 0 and max([len(chain.chain_id) for chain in structure.chains]) > 1:
        remapping = get_mmcif_tables_from_metadata(structure, ['pdbe_chain_remapping'])[0]
        if remapping:
            by_asym_okay = True
            asym_to_sym = {}
            for asym_id, sym_id in remapping.fields(['orig_label_asym_id', 'new_label_asym_id']):
                if len(asym_id) > 1 or not sym_id.startswith(asym_id):
                    by_asym_okay = False
                asym_to_sym.setdefault(asym_id, []).append(sym_id)
            cmds = []
            for i, asym_id in enumerate(sorted(list(asym_to_sym.keys()))):
                if by_asym_okay:
                    chain_spec = '/' + asym_id + '*'
                else:
                    chain_spec = ''.join(['/' + cid for cid in asym_to_sym[asym_id]])
                cmds.append("color %s%s %s %s" % (structure.atomspec, chain_spec,
                    base_palette[i % len(base_palette)], target_arg))
            return ' ; '.join(cmds)
    color_arg = " chains palette " + palette(structure.num_chains)
    return "rainbow %s@ca,c4'%s %s" % (structure.atomspec, color_arg, target_arg)

def alphafold_ribbon_command(session, name, coloring_cmds):
    printable = "printable" in name
    if "high confidence" in name:
        confidence_cmds = hide_AF_low_confidence(session, printable)
    else:
        confidence_cmds = []
    if printable:
        # confidence_cmds needs to be before print_ribbon so that the correct struts get placed
        initial_cmds = base_setup + base_macro_model + base_ribbon + confidence_cmds + print_ribbon
        final_cmds = print_prep(session, pb_radius=None)
    else:
        initial_cmds = undo_printable + base_setup + base_macro_model + base_ribbon + confidence_cmds
        final_cmds = []
    return initial_cmds + coloring_cmds + final_cmds

def alphafold_surface_command(session, name, coloring_cmds, **kw):
    printable = "printable" in name
    if "AlphaFold" in name:
        check_AF(session, pae=("PAE" in name))
    if "high confidence" in name:
        spec_lookup = get_AF_surf_spec(session, printable)
    else:
        spec_lookup = None
    return undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session,
        printable, spec_lookup=spec_lookup, **kw) + coloring_cmds

def run_preset(session, name, mgr):
    if name.startswith("ribbon by secondary structure"):
        cmd = alphafold_ribbon_command(session, name, [])
    elif name == "ribbon by chain":
        cmd = undo_printable + base_setup + base_macro_model + base_ribbon + [
            rainbow_cmd(s) for s in all_atomic_structures(session)
        ]
    elif name == "ribbon by chain (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + [
            rainbow_cmd(s) for s in all_atomic_structures(session)
        ] + print_ribbon + print_prep(session, pb_radius=None)
    elif name.startswith("ribbon rainbow"):
        cmd = alphafold_ribbon_command(session, name, ["rainbow @ca,c4'"])
    elif name == "ribbon by polymer (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + [
            "color bypolymer"
        ] + print_prep(session, pb_radius=None)
    elif name.startswith("ribbon monochrome"):
        cmd = alphafold_ribbon_command(session, name, ["color nih_blue", "setattr p color nih_blue"])
    elif name == "ribbon monochrome (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + [
            "color nih_blue",
            "setattr p color nih_blue"
        ] + print_prep(session, pb_radius=None)
    elif name.startswith("ribbon AlphaFold/pLDDT"):
        struct_spec = check_AF(session)
        cmd = alphafold_ribbon_command(session, name,
            [f"color byattribute r:pLDDT_score {struct_spec} palette alphafold"])
    elif name.startswith("ribbon AlphaFold/PAE domains"):
        struct_spec = check_AF(session, pae=True)
        cmd = alphafold_ribbon_command(session, name, [f"alphafold pae {struct_spec} colorDomains true"])
    elif name.startswith("surface monochrome"):
        cmd = alphafold_surface_command(session, name, ["color nih_blue"])
    elif name.startswith("surface coulombic"):
        cmd = alphafold_surface_command(session, name,
            ["color white", "coulombic surfaces #* chargeMethod gasteiger"])
        from chimerax.atomic import AtomicStructures
        structures = AtomicStructures(all_atomic_structures(session))
        main_atoms = structures.atoms.filter(structures.atoms.structure_categories == "main")
        main_residues = main_atoms.unique_residues
        incomplete_residues = main_residues.filter(main_residues.is_missing_heavy_template_atoms)
        if len(incomplete_residues) > len(main_residues) / 10:
            session.logger.warning("More than 10% of residues are incomplete;"
                " electrostatics probably inaccurate")
        elif "HIS" in incomplete_residues.names:
            session.logger.warning("Incomplete HIS residue; coulombic will likely fail")
    elif name.startswith("surface hydrophobicity"):
        cmd = alphafold_surface_command(session, name, color_by_hydrophobicity_cmds(session), sharp=True)
    elif name.startswith("surface by chain"):
        printable = "printable" in name
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session,
            printable) + by_chain_cmds(session, rainbow=True, target_atoms=True)
    elif name.startswith("surface by polymer"):
        printable = "printable" in name
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session,
            printable) + [ "color bypolymer target ar" ] + by_chain_cmds(session)
    elif name == "surface blob by chain":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + [
                "surf %s%s resolution 18 grid 6; %s" % (s.atomspec,
                ("" if s.num_atoms < 250000 else " enclose %s" % s.atomspec),
                rainbow_cmd(s, target_atoms=True))
                    for s in all_atomic_structures(session)
            ]
    elif name == "surface blob by polymer":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + [
                "surf %s%s resolution 18 grid 6"
                % (s.atomspec, ("" if s.num_atoms < 250000 else " enclose %s" % s.atomspec))
                    for s in all_atomic_structures(session)
            ] + [ "color bypolymer target ar" ] + by_chain_cmds(session)
    elif name.startswith("surface AlphaFold/pLDDT"):
        struct_spec = check_AF(session)
        cmd = alphafold_surface_command(session, name,
            [f"color byattribute r:pLDDT_score {struct_spec} palette alphafold"], sharp=True)
    elif name.startswith("surface AlphaFold/PAE domains"):
        struct_spec = check_AF(session, pae=True)
        printable = "printable" in name
        if "high confidence" in name:
            spec_lookup = get_AF_surf_spec(session, printable)
        else:
            spec_lookup = None
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + \
            surface_cmds(session, True, sharp=True, spec_lookup=spec_lookup) + [
            f"alphafold pae {struct_spec} colorDomains true",
            f"color {struct_spec} & ~::pae_domain dark gray",
            f"color {struct_spec} fromAtoms" ]
    elif name == "sticks":
        cmd = undo_printable + base_setup + color_by_het + [
            "style stick",
            "~nuc",
            "~ribbon",
            "disp"
        ]
    elif name == "sticks (printable)":
        cmd = undo_printable + base_setup + color_by_het + [
            "style stick",
            "~nuc",
            "~ribbon",
            "disp"
        ] + print_prep(ion_size_increase=0.35, bond_sides=32)
    elif name == "sticks monochrome":
        cmd = undo_printable + base_setup + [
            "style stick",
            "~nuc",
            "~ribbon",
            "disp",
            "color nih_blue",
        ]
    elif name == "sticks monochrome (printable)":
        cmd = undo_printable + base_setup + [
            "style stick",
            "~nuc",
            "~ribbon",
            "disp",
            "color nih_blue",
        ] + print_prep(ion_size_increase=0.35, bond_sides=32)
    elif name == "CPK":
        cmd = undo_printable + base_setup + color_by_het + [
            "style sphere",
            "~nuc",
            "~ribbon",
            "disp",
            "size H atomRadius 1.1",  # rescale H atoms to get better-looking balls
            "size ions atomRadius +0.35"
        ] + print_prep(ion_size_increase=0.35)
    elif name == "CPK monochrome":
        cmd = undo_printable + base_setup + [
            "style sphere",
            "~nuc",
            "~ribbon",
            "disp",
            "color nih_blue",
            "size H atomRadius 1.1",  # rescale H atoms to get better-looking balls
            "size ions atomRadius +0.35"
        ] + print_prep(ion_size_increase=0.35)
    elif name == "ball and stick":
        cmd = undo_printable + base_setup + color_by_het + ball_and_stick + [ "color pbonds bond_purple" ]
    elif name == "ball and stick monochrome":
        cmd = undo_printable + base_setup + color_by_het + ball_and_stick + [ "color nih_blue" ]
    elif name.startswith("volume "):
        cmd = volume_preset_cmds(session, name[7:])
    else:
        from chimerax.core.errors import UserError
        raise UserError("Unknown NIH3D preset '%s'" % name)
    # need the leading space in case the previous command arg end with a quote
    cmd = " ; ".join(cmd)
    mgr.execute(cmd)

def surface_cmds(session, printable, *, sharp=False, spec_lookup=None):
    import math
    # the newline preceding the 'size' command means that the preset manager will run the commands
    # preceding 'size' in their own run() call, will will allow check_for_changes to happen.  Otherwise
    # surface may get recolored (to default coloring) by an unexpectedly late check_for_changes
    cmds = ["\nsize atomRadius default"]
    for s in all_atomic_structures(session):
        # AddH won't actually run until after this command is generated, so base the grid value
        # on the number of heavy atoms involved in the surface for consistency, but then multiply
        # by 2 to account for probable amount of hydrogens
        import numpy
        num_heavys = len(s.atoms.filter(
            numpy.logical_and(s.atoms.elements.numbers != 1, s.atoms.structure_categories == "main")))
        if printable:
            grid_size = min(2.0, max(0.3, math.log10(2*num_heavys) - 3.2))
        else:
            grid_size = min(2.5, max(0.5, math.log10(2*num_heavys) - 2.5))
        if spec_lookup is None:
            spec = s.atomspec
        else:
            spec = spec_lookup[s]
        cmds.append("surface %s enclose %s grid %g sharp %s"
            % (s.atomspec, spec, grid_size, "true" if sharp else "false"))
    return cmds

def volume_cleanup_cmds(session, contour_cmds=None):
    from chimerax.map import VolumeSurface
    from chimerax.surface.area import measure_volume
    from chimerax.core.commands import run
    if contour_cmds:
        for contour_cmd in contour_cmds:
            run(session, contour_cmd, log=False)
        run(session, "wait 1", log=False)
    cmds = []
    for surface in session.models.list(type=VolumeSurface):
        orig_enclosed = measure_volume(session, [surface])

        # MESH CLEANING
        run(session, "surface dust %s size 1 metric 'size rank' ; wait 1" % surface.atomspec, log=False)
        enclosed = measure_volume(session, [surface], include_masked=False)
        working_surface = surface
        if enclosed < 0.5 * orig_enclosed:
            session.logger.info("Contour level does not connect pieces; trying other levels")
            volume = surface.volume
            mtx = volume.matrix()
            # mtx.min/max() usually returns floats, but can return 16-bit signed integers, so that
            # min-max could be negative, so...
            vmin, vmax = float(mtx.min()), float(mtx.max())
            contour_step = (vmax - vmin) * 0.05
            contour = orig_contour = surface.level
            connecting_contour = None
            while contour - contour_step > vmin:
                contour -= contour_step
                session.models.close([working_surface])
                contour_cmd = "volume %s region all style surface level %g limitVoxelCount true" \
                    " voxelLimit 8; wait 1" % (volume.atomspec, contour)
                run(session, contour_cmd, log=False)
                working_surface = session.models[-1]
                run(session, "surface dust %s size 1 metric 'size rank' ; wait 1" % working_surface.atomspec,
                    log=False)
                enclosed = measure_volume(session, [working_surface], include_masked=False)
                if enclosed >= 0.5 * orig_enclosed:
                    contour_level = connecting_contour = contour
                    session.logger.info("Contour level %g connects pieces; using that" % connecting_contour)
                    cmds.append(contour_cmd)
                    break
            if connecting_contour is None:
                session.logger.info(
                    "No contour level found that connects pieces; reverting to original level")
                contour_level = orig_contour
                session.models.close([working_surface])
                run(session, "volume %s region all style surface level %g limitVoxelCount true voxelLimit 8"
                    % (volume.atomspec, contour_level), log=False)
                working_surface = session.models[-1]
        cmds.append("wait 1")
        cmds.append("surface dust %s size 1 metric 'size rank'" % working_surface.atomspec)
    return cmds

def volume_contour_cmds(session, requested_contour_level=None):
    from chimerax.map import Volume
    cmds = []
    for v in session.models.list(type=Volume):
        contour_level = requested_contour_level
        if contour_level is None and v.name.startswith("emdb ") and v.name[5:].isdigit():
            import requests
            emdb_id = v.name[5:]
            data_key = "EMD-" + emdb_id
            response = requests.get("https://www.ebi.ac.uk/pdbe/api/emdb/entry/map/%s" % data_key)
            if response.status_code == requests.codes.ok:
                data = response.json()
                try:
                    info = data[data_key]
                except KeyError:
                    session.logger.warning("Key %s not found in metadata for EMDB %s" % (data_key, emdb_id))
                else:
                    try:
                        contour_level = info[0]["map"]["contour_level"]["value"]
                    except:
                        session.logger.info("No suggested contour level found in metadata for %s" % emdb_id)
                    else:
                        session.logger.info("Using EMDB-recommended contour level of %g" % contour_level)
            else:
                session.logger.warning("Could not access metadata for EMDB entry %s" % emdb_id)
        if contour_level is None:
            cmds.append("volume %s region all style surface step 1 limitVoxelCount false" % v.atomspec)
        else:
            cmds.append("volume %s region all style surface level %g limitVoxelCount true voxelLimit 8"
                % (v.atomspec, contour_level))

    return cmds

def volume_setup_cmds(session, contour_level=None):
    cmds = undo_printable + base_setup
    contour_cmds = volume_contour_cmds(session, requested_contour_level=contour_level)
    cmds += contour_cmds
    cmds += volume_cleanup_cmds(session, contour_cmds=contour_cmds)
    return cmds

def volume_finishing_cmds(session, preset):
    cmds = []
    if preset == "white":
        cmds += [ "color white" ]
    elif preset == "radial":
        from chimerax.map import Volume
        for v in session.models.list(type=Volume):
            cmds += [ "color radial %s palette red:yellow:green:cyan:blue center %s"
                % (v.atomspec, v.atomspec) ]
    elif preset == "monochrome":
        cmds += [ "color nih_blue" ]
    else:
        from chimerax.core.errors import UserError
        raise UserError("Unknown NIH3D preset 'volume %s'" % name)
    return cmds

def volume_preset_cmds(session, preset, contour_level=None):
    # split into two calls so that the NIH 3D print exchange can call them directly
    cmds = volume_setup_cmds(session, contour_level)
    cmds += volume_finishing_cmds(session, preset)
    return cmds
