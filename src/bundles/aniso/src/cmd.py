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

builtin_presets = {
    "simple": {
        'axis_color': None,
        'axis_factor': None,
        'axis_thickness': 0.01,
        'color': None,
        'ellipse_color': None,
        'ellipse_factor': None,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': True,
        'smoothing': 3,
        'transparency': None,
    },
    "simple-axes": {
        'axis_color': None,
        'axis_factor': 1.0,
        'axis_thickness': 0.01,
        'color': None,
        'ellipse_color': None,
        'ellipse_factor': None,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': False,
        'smoothing': 3,
        'transparency': None,
    },
    "ellipses": {
        'axis_color': None,
        'axis_factor': None,
        'axis_thickness': 0.01,
        'color': None,
        'ellipse_color': None,
        'ellipse_factor': 1.0,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': False,
        'smoothing': 3,
        'transparency': None,
    },
    "axes": {
        'axis_color': None,
        'axis_factor': 1.5,
        'axis_thickness': 0.01,
        'color': None,
        'ellipse_color': None,
        'ellipse_factor': None,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': True,
        'smoothing': 3,
        'transparency': None,
    },
    "octant": {
        'axis_color': None,
        'axis_factor': None,
        'axis_thickness': 0.01,
        'color': None,
        'ellipse_color': (0,0,0,255),
        'ellipse_factor': 1.01,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': True,
        'smoothing': 3,
        'transparency': None,
    },
    "snowglobe-axes": {
        'axis_color': None,
        'axis_factor': 0.99,
        'axis_thickness': 0.01,
        'color': (255,255,255,255),
        'ellipse_color': None,
        'ellipse_factor': None,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': True,
        'smoothing': 3,
        'transparency': 50,
    },
    "snowglobe-ellipses": {
        'axis_color': None,
        'axis_factor': None,
        'axis_thickness': 0.01,
        'color': (255,255,255,255),
        'ellipse_color': None,
        'ellipse_factor': 0.99,
        'ellipse_thickness': 0.02,
        'scale': 1.0,
        'show_ellipsoid': True,
        'smoothing': 3,
        'transparency': 50,
    },
}

from chimerax.core.errors import UserError

def check_atoms(session, atoms, *, error_text="atoms"):
    from chimerax.atomic import all_atoms
    if atoms is None:
        atoms = all_atoms(session)

    if not atoms:
        raise UserError("No %s specified" % error_text)

    atoms = atoms.filter(atoms.has_aniso_u)
    if not atoms:
        raise UserError("None of the specified %s have anisotropic temperature factors" % error_text)

    return atoms

def aniso_show(session, atoms=None):
    ''' Command to display thermal ellipsoids '''

    atoms = check_atoms(session, atoms)

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAnisoManager(session, s)
        mgr_info[s].show(atoms=s_atoms)

def aniso_style(session, structures=None, **kw):
    ''' Command to display thermal ellipsoids '''

    if structures is None:
        from chimerax.atomic import all_atomic_structures
        atoms = all_atomic_structures(session).atoms
    else:
        atoms = structures.atoms
    atoms = check_atoms(session, atoms, error_text="structures")

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAnisoManager(session, s)
        if kw:
            mgr_info[s].style(**kw)
        else:
            mgr_info[s].report_style_settings()

def aniso_hide(session, atoms=None):
    ''' Command to hide thermal ellipsoids '''

    atoms = check_atoms(session, atoms)

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            continue
        mgr_info[s].hide(atoms=s_atoms)

def get_preset_name(presets, name):
    matches = []
    name = name.lower()
    for preset_name in presets.keys():
        pname = preset_name.lower()
        if name == pname:
            return [preset_name]
        if pname.startswith(name):
            matches.append(preset_name)
    return matches

def make_preset_info(session):
    from .settings import get_settings
    settings = get_settings(session)
    preset_info = { k: (True, v) for k,v in builtin_presets.items() }
    preset_info.update({ k: (False, v) for k,v in settings.custom_presets.items() })
    return preset_info

def aniso_preset(session, structures=None, name=None):
    if name is not None:
        name = name.strip()
    preset_info = make_preset_info(session)
    if not name:
        output_lines = ["Preset names (built-in names in <b>bold</b>):", '<ul style="list-style: none;">']
        names = sorted(list(preset_info.keys()))
        for name in names:
            builtin, style_settings = preset_info[name]
            output_lines.append('<li>%s</li>' % (('<b>%s</b>' if builtin else '%s') % name))
        output_lines.append("</ul>")
        session.logger.info("\n".join(output_lines), is_html=True)
        return

    matching_names = get_preset_name(preset_info, name)
    if not matching_names:
        raise UserError("No preset name matches or begins with '%s'" % name)
    if len(matching_names) > 1:
        raise UserError("Multiple preset names match or begin with '%s': %s"
            % (name, ', '.join(matching_names)))
    aniso_style(session, structures, **preset_info[matching_names[0]][1])

def aniso_preset_delete(session, name):
    name = name.strip()
    if not name:
        raise UserError("No preset name provided")
    preset_info = make_preset_info(session)
    matching_names = get_preset_name(preset_info, name)
    if not matching_names:
        raise UserError("No preset name matches or begins with '%s'" % name)
    if len(matching_names) > 1:
        matching_custom = []
        for matching_name in matching_names:
            if not preset_info[matching_name][0]:
                matching_custom.append(matching_name)
        if len(matching_custom) == 0:
            raise UserError("Cannot delete built-in presets; multiple built-in presets match or begin"
                " with '%s': %s" % (name, ', '.join(matching_names)))
        elif len(matching_custom) > 1:
            raise UserError("Multiple preset names match or begin with '%s': %s"
                % (name, ', '.join(matching_custom)))
        matching_names = matching_custom

    from .settings import get_settings
    settings = get_settings(session)
    # settings doesn't recognize when a dict has changed internally, so...
    presets = settings.custom_presets.copy()
    del presets[matching_names[0]]
    settings.custom_presets = presets
    settings.save()
    session.logger.info("Deleted custom preset '%s'" % matching_names[0])

def aniso_preset_save(session, structures=None, name=None):
    if name is not None:
        name = name.strip()
    if not name:
        raise UserError("No preset name provided")
    preset_info = make_preset_info(session)
    updating = False
    for preset, info in preset_info.items():
        if name == preset:
            builtin, style_settings = info
            if builtin:
                raise UserError("Cannot change built-in %s preset" % name)
            updating = True
            break

    if structures is None:
        from chimerax.atomic import all_atomic_structures
        atoms = all_atomic_structures(session).atoms
    else:
        atoms = structures.atoms
    atoms = check_atoms(session, atoms, error_text="structures")

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    preset_settings = None
    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAnisoManager(session, s)
        s_settings = mgr_info[s].drawing_params.copy()
        if preset_settings is None:
            preset_settings = s_settings
        else:
            if s_settings != preset_settings:
                raise UserError("The structures have different depiction settings. Specify the"
                    " structure you want to save settings from as the first argument of the command.")
    from .settings import get_settings
    settings = get_settings(session)
    presets = settings.custom_presets.copy()
    presets[name] = preset_settings
    settings.custom_presets = presets
    settings.save()
    session.logger.info("%s settings for preset '%s'" % (("Updated" if updating else "Saved"), name))

def register_command(logger, name):
    from chimerax.core.commands import register, CmdDesc
    from chimerax.core.commands import Or, EmptyArg, Color8TupleArg, NoneArg, PositiveFloatArg, BoolArg
    from chimerax.core.commands import PositiveIntArg, Bounded, FloatArg, RestOfLine
    from chimerax.atomic import AtomsArg, AtomicStructuresArg

    tilde_desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))], synopsis='hide thermal ellipsoids')
    if name == "aniso":
        show_desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))],
            synopsis='show depictions of thermal ellipsoids')
        style_desc = CmdDesc(required=[('structures', Or(AtomicStructuresArg, EmptyArg))],
            keyword=[
                ('axis_color', Or(Color8TupleArg, NoneArg)),
                ('axis_factor', Or(PositiveFloatArg, NoneArg)),
                ('axis_thickness', PositiveFloatArg),
                ('color', Or(Color8TupleArg, NoneArg)),
                ('ellipse_color', Or(Color8TupleArg, NoneArg)),
                ('ellipse_factor', Or(PositiveFloatArg, NoneArg)),
                ('ellipse_thickness', PositiveFloatArg),
                ('scale', PositiveFloatArg),
                ('show_ellipsoid', BoolArg),
                ('smoothing', PositiveIntArg),
                ('transparency', Bounded(FloatArg, min=0, max=100,
                    name="a percentage (number between 0 and 100)")),
            ],
            synopsis='change style of depictions of thermal ellipsoids')
        register('aniso', show_desc, aniso_show, logger=logger)
        register('aniso style', style_desc, aniso_style, logger=logger)
        register('aniso hide', tilde_desc, aniso_hide, logger=logger)
        preset_desc = CmdDesc(
            required=[('structures', Or(AtomicStructuresArg, EmptyArg)), ('name', RestOfLine)],
            synopsis='apply thermal-ellipsoid preset')
        register('aniso preset', preset_desc, aniso_preset, logger=logger)
        preset_save_desc = CmdDesc(
            required=[('structures', Or(AtomicStructuresArg, EmptyArg)), ('name', RestOfLine)],
            synopsis='save thermal-ellipsoid settings as preset')
        register('aniso preset save', preset_save_desc, aniso_preset_save, logger=logger)
        preset_delete_desc = CmdDesc(
            required=[('name', RestOfLine)],
            synopsis='delete custom thermal-ellipsoid preset')
        register('aniso preset delete', preset_delete_desc, aniso_preset_delete, logger=logger)
    else:
        register('~aniso', tilde_desc, aniso_hide, logger=logger)
