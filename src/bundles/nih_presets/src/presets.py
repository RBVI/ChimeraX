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

from chimerax.atomic import all_atomic_structures, Residue

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
    "select backbone & protein | nucleic-acid & min-backbone | ions | ligand"
        " | ligand :< 5 & ~nucleic-acid",
    "hbonds sel color white restrict both",
    #"size hbonds pseudobondRadius 0.6",
    "size pseudobondRadius 0.6",
    # ribbons need to be up to date for struts to work right
    "wait 1; struts @ca|ligand|P|##num_atoms<500 length 8 loop 60 rad 0.75 color struts_grey",
    "~struts @PB,PG resetRibbon false",
    "~struts adenine|cytosine|guanine|thymine|uracil resetRibbon false",
    #"color struts_grey pseudobonds",
    "color hbonds white pseudobonds",
    "~select"
]

undo_printable = [
    "~struts",
    "~hbonds",
    "size atomRadius default stickRadius 0.2 pseudobondRadius 0.2",
    "style dashes 7"
]

def addh_cmds(session):
    return [ "addh %s" % s.atomspec for s in all_atomic_structures(session) if s.num_atoms < 25000 ]

def by_chain_cmds(session, rainbow=False, target_atoms=False):
    cmds = []
    for s in all_atomic_structures(session):
        if rainbow:
            cmds.append(rainbow_cmd(s, target_atoms=target_atoms))
        cmds.append("color zone %s near %s distance 20" % (s.atomspec, s.atomspec))
    return cmds

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

base_palette = ["marine", "goldenrod", "firebrick", "forest", "tangerine", "grape"]
def palette(num_chains):
    palette = base_palette[:]
    if num_chains < 2:
        return palette[0] + ':' + palette[0]
    while num_chains > len(palette):
        palette += base_palette
    return ':'.join(palette[:num_chains])

def print_prep(*, pb_radius=0.4, ion_size_increase=0.0):
    cmds = [
        "size stickRadius 0.8",
        "style dashes 0"
    ]
    if pb_radius is not None:
        cmds += [ "size pseudobondRadius %g" % pb_radius ]
    if ion_size_increase:
        cmds += ["size ions atomRadius %+g" % ion_size_increase]
    return cmds

def rainbow_cmd(structure, target_atoms=False):
    color_arg = " chains palette " + palette(structure.num_chains)
    return "rainbow %s@ca,c4'%s target rs%s" % (structure.atomspec, color_arg, ("a" if target_atoms else ""))

def run_preset(session, name, mgr):
    if name == "ribbon by secondary structure":
        cmd = undo_printable + base_setup + base_macro_model + base_ribbon
    elif name == "ribbon by secondary structure (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + print_prep(pb_radius=None)
    elif name == "ribbon by chain":
        cmd = undo_printable + base_setup + base_macro_model + base_ribbon + [
            rainbow_cmd(s) for s in all_atomic_structures(session)
        ]
    elif name == "ribbon by chain (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + [
            rainbow_cmd(s) for s in all_atomic_structures(session)
        ] + print_ribbon + print_prep(pb_radius=None)
    elif name == "ribbon rainbow":
        cmd = undo_printable + base_setup + base_macro_model + base_ribbon + [ "rainbow @CA target r" ]
    elif name == "ribbon rainbow (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + [
            "rainbow @CA"
        ] + print_ribbon + print_prep(pb_radius=None)
    elif name == "ribbon by polymer (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + [
            "color bypolymer"
        ] + print_prep(pb_radius=None)
    elif name == "ribbon monochrome":
        cmd = undo_printable + base_setup + base_macro_model + base_ribbon + [
            "color nih_blue",
            "setattr p color nih_blue"
        ]
    elif name == "ribbon monochrome (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + [
            "color nih_blue",
            "setattr p color nih_blue"
        ] + print_prep(pb_radius=None)
    elif name == "surface monochrome":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session) \
            + [ "color nih_blue" ]
    elif name == "surface coulombic":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session) \
            + [ "color white", "coulombic surfaces #*" ]
        from chimerax.atomic import AtomicStructures
        structures = AtomicStructures(all_atomic_structures(session))
        main_atoms = structures.atoms.filter(structures.atoms.structure_categories == "main")
        main_residues = main_atoms.unique_residues
        incomplete_residues = main_residues.filter(main_residues.is_missing_heavy_template_atoms)
        if len(incomplete_residues) > len(main_residues) / 10:
            session.logger.warning("More than 10% or residues are incomplete;"
                " electrostatics probably inaccurate")
        elif "HIS" in incomplete_residues.names:
            session.logger.warning("Incomplete HIS residue; coulombic will likely fail")
    elif name == "surface hydrophobicity":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session) \
            + color_by_hydrophobicity_cmds(session)
    elif name == "surface by chain":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session) \
            + by_chain_cmds(session, rainbow=True, target_atoms=True)
    elif name == "surface by polymer":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + surface_cmds(session) \
            + [ "color bypolymer target ar" ] + by_chain_cmds(session)
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
        ] + print_prep(ion_size_increase=0.35)
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
        ] + print_prep(ion_size_increase=0.35)
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

def surface_cmds(session):
    import math
    cmds = []
    for s in all_atomic_structures(session):
        if s.num_atoms < 250000:
            grid_size = min(2.5, max(0.5, math.log10(s.num_atoms) - 2.5))
            cmds.append("surface %s enclose %s grid %g sharp true" % (s.atomspec, s.atomspec, grid_size))
        else:
            cmds.append("surface %s enclose %s resolution 18 grid 6" % (s.atomspec, s.atomspec))
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
            vmin, vmax = mtx.min(), mtx.max()
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
