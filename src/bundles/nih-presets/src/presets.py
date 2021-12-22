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

ADDH_CUTOFF = COULOMBIC_CUTOFF = 25000
HUGE_CUTOFF = 250000

from chimerax.atomic import all_atomic_structures, Residue

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
    "surface close"
]

base_macro_model = [
    "delete solvent",
    "delete H"
]

base_ribbon = [
    # 'struts' doesn't work right until ribbon data structures get updated, so wait 1 frame
    "preset 'initial styles' cartoon; wait 1",
    "nucleotides ladder radius 1.2",
    "color white",
    "color helix marine; color strand firebrick; color coil goldenrod; color nucleic-acid forest",
    "color :A:C:G:U grape",
    "color byatom",
    "select (C & ligand) | (C & ligand :< 5 & ~nucleic-acid) | (C & protein) | (C & disulfide)",
    "color sel carbon_grey atoms",
    "color ligand | protein & sideonly byhet atoms",
    "~select"
]

base_surface = [
    "delete H|ligand|~(protein|nucleic-acid)",
    "~nuc",
    "~ribbon",
    "~display",
]

print_ribbon = [
    # make missing-structure pseudobonds bigger relative to upcoming hbonds
    "size min-backbone pseudobondRadius 1.1",
    "select backbone & protein | nucleic-acid & min-backbone | ions | ligand"
        " | ligand :< 5 & ~nucleic-acid",
    "hbonds sel color white restrict both",
    "size hbonds pseudobondRadius 0.6",
    "struts @ca|ligand|P|##num_atoms<500 length 8 loop 60 rad 0.75 color struts_grey",
    "~struts @PB,PG",
    "~struts adenine|cytosine|guanine|thymine|uracil",
    "color struts_grey pseudobonds",
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

def by_chain_cmds(session, rainbow=False):
    cmds = []
    for s in all_atomic_structures(session):
        if rainbow:
            cmds.append(rainbow_cmd(s))
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

def rainbow_cmd(structure):
    color_arg = " chains palette " + palette(structure.num_chains)
    return "rainbow %s@ca,c4'%s" % (structure.atomspec, color_arg)

def run_preset(session, name, mgr):
    if name == "ribbon rainbow":
        cmd = undo_printable + base_setup + base_macro_model + base_ribbon
    elif name == "ribbon rainbow (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + print_prep(pb_radius=None)
    elif name == "ribbon by polymer (printable)":
        cmd = base_setup + base_macro_model + base_ribbon + print_ribbon + [
            "color bypolymer"
        ] + print_prep(pb_radius=None)
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
            + by_chain_cmds(session, rainbow=True)
    elif name == "surface blob by chain":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + [
                "surf %s resolution 18 grid 6; %s" % (s.atomspec, rainbow_cmd(s))
                    for s in all_atomic_structures(session)
            ]
    elif name == "surface blob by polymer":
        cmd = undo_printable + base_setup + base_surface + addh_cmds(session) + [
                "surf %s resolution 18 grid 6; %s" % (s.atomspec, rainbow_cmd(s))
                    for s in all_atomic_structures(session)
            ] + [ "color bypolymer target ar" ] + by_chain_cmds(session)
    else:
        from chimerax.core.errors import UserError
        raise UserError("Unknown NIH3D preset '%s'" % name)
    cmd = "; ".join(cmd)
    mgr.execute(cmd)

def surface_cmds(session):
    import math
    cmds = []
    for s in all_atomic_structures(session):
        grid_size = min(2.5, max(0.5, math.log10(s.num_atoms) - 2.5))
        cmds.append("surface %s enclose %s grid %g sharp true" % (s.atomspec, s.atomspec, grid_size))
    return cmds
