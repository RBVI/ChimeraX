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

from chimerax.core.commands import CmdDesc, EmptyArg, EnumOf, Or, StringArg, AtomSpecArg, ModelsArg, ListOf, BoolArg
from .util import report_models, report_chains, report_polymers, report_residues
from .util import report_residues, report_atoms, report_attr, report_distmat


def info(session, models=None, *, return_json=False):
    '''
    Report state of models, such as whether they are displayed, color, number of children,
    number of instances...

    Parameters
    ----------
    models : list of models

    If 'return_json' is True, the returned JSON will be a list of JSON objects, one per model.  Each object
    will have at a minimum the following name/value pairs:

        spec:  the atom specifier for this model
        name:  the name of the model
        shown:  whether the model-level display attribute is true
        num triangles:  if the model is a surface of some kind, how many triangles does it have; for
            non-surface models, this will be 0
        num instances:  how many graphical "instances" of the model are there, so at least 1
        num selected instances:  how many of the graphical instances are selected

    For Structure (or AtomicStructure) models, there will be these additional name/value pairs:

        num atoms:  the number of atoms
        num_bonds:  the number of bonds
        num residues:  the number of residues
        chains:  a list of chain IDs for polymeric chains in the structure
        num coordsets:  the number of coordinate sets in the structure
        pseudobond groups:  list of JSON, one per pseudobond subgroup; those objects will have the following
            name/value pairs:

            name: name of the pseudobond group
            num pseudobonds:  number of pseudobonds in the group

    For global PseudobondGroup models (i.e. not a submodel of a structure), there will be this additional
    name/value pair:

            num pseudobonds:  number of pseudobonds in the group

    For Volume models, there will be additional names: size, step, voxel size, surface levels, image levels,
        minimum value, maximum value, value type, and num symmetry operators.
    '''
    m = session.models
    if models is None:
        models = m.list()
    
    if return_json:
        model_infos = []
    else:
        lines = []
    from . import util
    for m in sorted(models, key = lambda m: m.id):
        if return_json:
            info = {}
            model_infos.append(info)
            util.model_info(m, info_dict=info)
            util.structure_info(m, info_dict=info)
            util.pseudobond_group_info(m, info_dict=info)
            util.volume_info(m, info_dict=info)
        else:
            line = (util.model_info(m) +
                    util.structure_info(m) +
                    util.pseudobond_group_info(m) +
                    util.volume_info(m))
            lines.append(line)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(model_infos), None)
    msg = '%d models\n' % len(models) + '\n'.join(lines)
    session.logger.info(msg)

info_desc = CmdDesc(optional=[('models', ModelsArg)],
                    synopsis='Report info about models')


def info_bounds(session, models=None, *, return_json=False):
    '''
    Report bounds of displayed parts of models in scene coordinates.
    If not models are given the bounds for the entire scene is reported.

    Parameters
    ----------
    models : list of models

    If 'return_json' is True, the JSON returned depends on if 'models' is None.  If models is None,
    then the bounds of the scene is returned (null if no bounds), as a list [min, max] where min and max
    are 3 numbers (xyz).  Otherwise, a JSON object is returned where names are model atom specifiers and
    values are bounds for that model (in the same form as the scene bounds).
    '''
    from .util import bounds_description
    if models is None:
        b = session.main_view.drawing_bounds()
        msg = 'Scene %s' % bounds_description(b)
        if return_json:
            bounds_info = None if b is None else (b.xyz_min, b.xyz_max)
    else:
        lines = ['#%s, %s, %s' % (m.id_string, m.name, bounds_description(m.bounds()))
                 for m in sorted(models, key = lambda m: m.id)]
        msg = '\n'.join(lines)
        if return_json:
            bounds_info = {}
            for m in models:
                b = m.bounds()
                bounds_info[m.atomspec] = None if b is None else (b.xyz_min, b.xyz_max)
    session.logger.info(msg)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(bounds_info), None)

info_bounds_desc = CmdDesc(optional=[('models', ModelsArg)],
                           synopsis='Report scene bounding boxes for models')


def info_models(session, atoms=None, type_=None, attribute="name", *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of JSON objects, one per model.  Each object
    will have the following name/value pairs:

        spec:  the atom specifier for this model
        class:  the Python class of the model
        attribute:  the attribute being tested for
        present:  whether the attribute is defined in the model instance
        value:  the value of the attribute.  If 'present' is false, this will be null, which could possibly
            also be the value for some instances where the attribute *is* present.
    '''
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    if type_ is not None:
        type_ = type_.lower()
    models = [m for m in results.models
              if type_ is None or type(m).__name__.lower() == type_]
    return report_models(session.logger, models, attribute, return_json=return_json)
info_models_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                           keyword=[("type_", StringArg),
                                    ("attribute", StringArg),],
                           synopsis="Report model information")


def info_chains(session, atoms=None, attribute="chain_id", *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of JSON objects, one per chain.  Each object
    will have the following name/value pairs:

        spec:  the atom specifier for this chain
        sequence: a string containing the chain sequence
        residues: a list of residue specifiers in the chain; for residues with no structure the "specifier"
            will be null
        attribute:  the attribute being tested for
        present:  whether the attribute is defined in the chain instance
        value:  the value of the attribute.  If 'present' is false, this will be null, which could possibly
            also be the value for some instances where the attribute *is* present.
    '''
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    chains = []
    for m in results.models:
        try:
            chains.extend(m.chains)
        except AttributeError:
            # No chains, no problem
            pass
    return report_chains(session.logger, chains, attribute, return_json=return_json)
info_chains_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                           keyword=[("attribute", StringArg),],
                           synopsis="Report chain information")


def info_polymers(session, atoms=None, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of lists, one per polymer.  Each list
    will contain the atom specs that compose the polymer, in polymer order
    '''
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    polymers = []
    residues = results.atoms.unique_residues
    from chimerax.atomic import AtomicStructure
    for m in results.atoms.unique_structures:
        try:
            for p, ptype in m.polymers(AtomicStructure.PMS_TRACE_CONNECTS, False):
                if p.intersects(residues):
                    polymers.append(p)
        except AttributeError:
            # No chains, no problem
            pass
    return report_polymers(session.logger, polymers, return_json=return_json)
info_polymers_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                             synopsis="Report polymer information")


def info_residues(session, atoms=None, attribute="name", *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of JSON objects, one per residue.  Each object
    will have the following name/value pairs:

        spec:  the atom specifier for this residue
        attribute:  the attribute being tested for
        present:  whether the attribute is defined in the residue instance
        value:  the value of the attribute.  If 'present' is false, this will be null, which could possibly
            also be the value for some instances where the attribute *is* present.
    '''
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    residues = results.atoms.unique_residues
    return report_residues(session.logger, residues, attribute, return_json=return_json)
info_residues_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                             keyword=[("attribute", StringArg),],
                             synopsis="Report residue information")


def info_atoms(session, atoms=None, attribute="idatm_type", *, return_json=False):
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    residues = results.atoms.unique_residues
    return report_atoms(session.logger, results.atoms, attribute, return_json=return_json)
info_atoms_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                          keyword=[("attribute", StringArg),],
                          synopsis="Report atom information")


def info_selection(session, level=None, attribute=None, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will correspond to the function appropriate for the 'level',
    namely, but restricted to selected items:

        atom/None   info_atoms
        residue     info_residues
        chain       info_chains
        structure   info_models (but only Structure/AtomicStructure models)
        models      info_models
    '''
    json_info = None
    if level is None or level == "atom":
        if attribute is None:
            attribute = "idatm_type"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            json_info = report_atoms(session.logger, concatenate(atoms), attribute, return_json=return_json)
    elif level == "residue":
        if attribute is None:
            attribute = "name"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            residues = concatenate([a.unique_residues for a in atoms])
            json_info = report_residues(session.logger, residues, attribute, return_json=return_json)
    elif level == "chain":
        if attribute is None:
            attribute = "chain_id"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            chains = concatenate([a.residues.unique_chains for a in atoms])
            json_info = report_chains(session.logger, chains, attribute, return_json=return_json)
    elif level == "structure":
        if attribute is None:
            attribute = "name"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            mols = concatenate([a.unique_structures for a in atoms])
            json_info = report_models(session.logger, mols, attribute, return_json=return_json)
    elif level == "model":
        if attribute is None:
            attribute = "name"
        json_info = report_models(session.logger, session.selection.models(), attribute,
            return_json=return_json)
    if return_json:
        return json_info
info_selection_desc = CmdDesc(keyword=[("level", EnumOf(["atom",
                                                         "residue",
                                                         "chain",
                                                         "structure",
                                                         "model"])),
                                       ("attribute", StringArg),],
                              synopsis="Report selection information")

def info_atomattr(session, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of atom attribute names.
    '''
    from chimerax.core.attributes import type_attrs
    from chimerax.atomic import Atom
    attrs = type_attrs(Atom)
    for a in attrs:
        report_attr(session.logger, "atom", a)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(attrs), None)
info_atomattr_desc = CmdDesc(synopsis="Report atom attribute information")

def info_bondattr(session, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of bond attribute names.
    '''
    from chimerax.core.attributes import type_attrs
    from chimerax.atomic import Bond
    attrs = type_attrs(Bond)
    for a in attrs:
        report_attr(session.logger, "bond", a)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(attrs), None)
info_bondattr_desc = CmdDesc(synopsis="Report bond attribute information")

def info_resattr(session, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a list of residue attribute names.
    '''
    from chimerax.core.attributes import type_attrs
    from chimerax.atomic import Residue
    attrs = type_attrs(Residue)
    for a in attrs:
        report_attr(session.logger, "res", a)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(attrs), None)
info_resattr_desc = CmdDesc(synopsis="Report residue attribute information")


def info_distmat(session, atoms, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a JSON object, with the following name/value pairs:

        atoms:  a list of the specifiers for the atoms used when computing the distance matrix
        distance matrix:  the "flattened" upper-right triangle of the distance matrix, as per
            http://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
    '''
    from scipy.spatial.distance import pdist
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    atoms = results.atoms
    coords = atoms.scene_coords
    distmat = pdist(coords, "euclidean")
    report_distmat(session.logger, atoms, distmat)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode({
            'atoms': [a.atomspec for a in atoms],
            'distance matrix': distmat
        }), None)
info_distmat_desc = CmdDesc(required=([("atoms", Or(AtomSpecArg, EmptyArg))]),
                            synopsis="Report distance matrix information")

from .util import Notifier
_WhatArg = EnumOf(Notifier.SupportedTypes)

def info_notify_start(session, what, client_id, prefix="", url=None):
    Notifier.Find(what, client_id, session, prefix, url).start()
info_notify_start_desc = CmdDesc(required=[("what", _WhatArg),
                                           ("client_id", StringArg),],
                                 keyword=[("prefix", StringArg),
                                          ("url", StringArg),],
                                 synopsis="Start notifications for events")


def info_notify_stop(session, what, client_id):
    Notifier.Find(what, client_id).stop()
info_notify_stop_desc = CmdDesc(required=[("what", _WhatArg),
                                          ("client_id", StringArg),],
                                synopsis="Stop notifications for events")


def info_notify_suspend(session, what, client_id):
    Notifier.Find(what, client_id).suspend()
info_notify_suspend_desc = CmdDesc(required=[("what", _WhatArg),
                                             ("client_id", StringArg),],
                                   synopsis="Suspend notifications")


def info_notify_resume(session, what, client_id):
    Notifier.Find(what, client_id).resume()
info_notify_resume_desc = CmdDesc(required=[("what", _WhatArg),
                                            ("client_id", StringArg),],
                                  synopsis="Resume notifications")


def info_path(session, which="all", version="all", what=None, *, return_json=False):
    '''
    If 'return_json' is True, the returned JSON will be a JSON object with one or two names (depending on
    the arguments given), namely "versioned" and/or "unversioned".  The value(s) will also be JSON objects
    with one or more names (again, depending on the args) from among: site_config_dir, site_data_dir,
    user_cache_dir, user_config_dir, user_data_dir, user_log_dir, and/or user_state_dir.  The values will
    be the appropriate directory name.
    '''
    logger = session.logger
    if return_json:
        info_dict = {}
        kw = { 'info_dict': info_dict }
    else:
        kw = {}
    if which in ["all", "system"]:
        if version in ["all", "versioned"]:
            _info_path_show(logger, "system", "versioned", what, **kw)
        if version in ["all", "unversioned"]:
            _info_path_show(logger, "system", "unversioned", what, **kw)
    if which == "all" or which == "user":
        if version in ["all", "versioned"]:
            _info_path_show(logger, "user", "versioned", what, **kw)
        if version in ["all", "unversioned"]:
            _info_path_show(logger, "user", "unversioned", what, **kw)
    if return_json:
        from chimerax.core.commands import JSONResult, ArrayJSONEncoder
        import json
        return JSONResult(ArrayJSONEncoder().encode(info_dict), None)
path_names = EnumOf(["config", "data", "log", "state", "cache"])
info_path_desc = CmdDesc(optional=[("which", EnumOf(["all",
                                                       "system",
                                                       "user"])),
                                     ("version", EnumOf(["all",
                                                         "versioned",
                                                         "unversioned"])),
                                     ("what", ListOf(path_names)),
                                    ],
                           synopsis="Report directory paths")


def _info_path_show(logger, which, version, what, *, info_dict=None):
    if what is None:
        names = path_names.values
    else:
        names = what
    if which == "system":
        which_prefix = "site_"
    else:
        which_prefix = "user_"
    if version == "versioned":
        from chimerax import app_dirs
    else:
        from chimerax import app_dirs_unversioned as app_dirs
    for n in names:
        attr_name = which_prefix + n + "_dir"
        try:
            attr_value = getattr(app_dirs, attr_name)
        except AttributeError:
            if what is not None:
                # User actually typed this
                logger.info("There is no %s %s %s directory" %
                            (which, version, n))
        else:
            logger.info("%s %s %s directory: %s" %
                        (which, version, n, attr_value))
            if info_dict is not None:
                info_dict.setdefault(version, {})[attr_name] = attr_value
