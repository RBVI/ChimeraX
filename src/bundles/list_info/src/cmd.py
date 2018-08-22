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


def info(session, models=None):
    '''
    Report state of models, such as whether they are displayed, color, number of children,
    number of instances...

    Parameters
    ----------
    models : list of models
    '''
    m = session.models
    if models is None:
        models = m.list()
    
    lines = []
    from . import util
    for m in sorted(models, key = lambda m: m.id):
        line = (util.model_info(m) +
                util.structure_info(m) +
                util.pseudobond_group_info(m) +
                util.volume_info(m))
        lines.append(line)
    msg = '%d models\n' % len(models) + '\n'.join(lines)
    session.logger.info(msg)

info_desc = CmdDesc(optional=[('models', ModelsArg)],
                    synopsis='Report info about models')


def info_bounds(session, models=None):
    '''
    Report bounds of displayed parts of models in scene coordinates.
    If not models are given the bounds for the entire scene is reported.

    Parameters
    ----------
    models : list of models
    '''
    from .util import bounds_description
    if models is None:
        b = session.main_view.drawing_bounds()
        msg = 'Scene %s' % bounds_description(b)
    else:
        lines = ['#%s, %s, %s' % (m.id_string, m.name, bounds_description(m.bounds()))
                 for m in sorted(models, key = lambda m: m.id)]
        msg = '\n'.join(lines)
    session.logger.info(msg)

info_bounds_desc = CmdDesc(optional=[('models', ModelsArg)],
                           synopsis='Report scene bounding boxes for models')


def info_models(session, atoms=None, type_=None, attribute="name"):
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    if type_ is not None:
        type_ = type_.lower()
    models = [m for m in results.models
              if type_ is None or type(m).__name__.lower() == type_]
    report_models(session.logger, models, attribute)
info_models_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                           keyword=[("type_", StringArg),
                                    ("attribute", StringArg),],
                           synopsis="Report model information")


def info_chains(session, atoms=None, attribute="chain_id"):
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
    report_chains(session.logger, chains, attribute)
info_chains_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                           keyword=[("attribute", StringArg),],
                           synopsis="Report chain information")


def info_polymers(session, atoms=None):
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
    report_polymers(session.logger, polymers)
info_polymers_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                             synopsis="Report polymer information")


def info_residues(session, atoms=None, attribute="name"):
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    residues = results.atoms.unique_residues
    report_residues(session.logger, residues, attribute)
info_residues_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                             keyword=[("attribute", StringArg),],
                             synopsis="Report residue information")


def info_atoms(session, atoms=None, attribute="idatm_type"):
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    residues = results.atoms.unique_residues
    report_atoms(session.logger, results.atoms, attribute)
info_atoms_desc = CmdDesc(required=[("atoms", Or(AtomSpecArg, EmptyArg))],
                          keyword=[("attribute", StringArg),],
                          synopsis="Report atom information")


def info_selection(session, level=None, attribute=None):
    if level is None or level == "atom":
        if attribute is None:
            attribute = "idatm_type"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            report_atoms(session.logger, concatenate(atoms), attribute)
    elif level == "residue":
        if attribute is None:
            attribute = "name"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            residues = concatenate([a.unique_residues for a in atoms])
            report_residues(session.logger, residues, attribute)
    elif level == "chain":
        if attribute is None:
            attribute = "chain_id"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            chains = concatenate([a.unique_chains for a in atoms])
            report_chains(session.logger, chains, attribute)
    elif level == "structure":
        if attribute is None:
            attribute = "name"
        atoms = session.selection.items("atoms")
        if atoms:
            from chimerax.atomic import concatenate
            mols = concatenate([a.unique_structures for a in atoms])
            report_models(session.logger, mols, attribute)
    elif level == "model":
        if attribute is None:
            attribute = "name"
        report_models(session.logger, session.selection.models(), attribute)
info_selection_desc = CmdDesc(keyword=[("level", EnumOf(["atom",
                                                         "residue",
                                                         "chain",
                                                         "structure",
                                                         "model"])),
                                       ("attribute", StringArg),],
                              synopsis="Report selection information")


def _type_attrs(t):
    from types import GetSetDescriptorType
    attrs = [name for name in dir(t)
        if name[0] != '_' and type(getattr(t, name)) in [property, GetSetDescriptorType]]
    attrs.extend(t._attr_registration.reg_attr_info.keys())
    attrs.sort()
    return attrs

def info_atomattr(session):
    from chimerax.atomic import Atom
    for a in _type_attrs(Atom):
        report_attr(session.logger, "atom", a)
info_atomattr_desc = CmdDesc(synopsis="Report atom attribute information")

def info_bondattr(session):
    from chimerax.atomic import Bond
    for a in _type_attrs(Bond):
        report_attr(session.logger, "bond", a)
info_bondattr_desc = CmdDesc(synopsis="Report bond attribute information")

def info_resattr(session):
    from chimerax.atomic import Residue
    for a in _type_attrs(Residue):
        report_attr(session.logger, "res", a)
info_resattr_desc = CmdDesc(synopsis="Report residue attribute information")


def info_distmat(session, atoms):
    from scipy.spatial.distance import pdist, squareform
    if atoms is None:
        from chimerax.core.commands import atomspec
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    atoms = results.atoms
    coords = atoms.scene_coords
    distmat = pdist(coords, "euclidean")
    report_distmat(session.logger, atoms, distmat)
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


def info_path(session, which="all", version="all", what=None):
    logger = session.logger
    if which in ["all", "system"]:
        if version in ["all", "versioned"]:
            _info_path_show(logger, "system", "versioned", what)
        if version in ["all", "unversioned"]:
            _info_path_show(logger, "system", "unversioned", what)
    if which == "all" or which == "user":
        if version in ["all", "versioned"]:
            _info_path_show(logger, "user", "versioned", what)
        if version in ["all", "unversioned"]:
            _info_path_show(logger, "user", "unversioned", what)
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


def _info_path_show(logger, which, version, what):
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
