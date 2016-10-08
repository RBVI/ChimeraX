# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc, EnumOf, StringArg, AtomSpecArg
from .util import report_models, report_chains, report_polymers, report_residues
from .util import report_residues, report_atoms, report_resattr, report_distance

def listinfo_models(session, spec=None, type_=None, attribute="name"):
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    if type_ is not None:
        type_ = type_.lower()
    models = [m for m in results.models
              if type_ is None or type(m).__name__ == type_]
    report_models(session.logger, models, attribute)
listinfo_models_desc = CmdDesc(keyword=[("spec", AtomSpecArg),
                                        ("type_", StringArg),
                                        ("attribute", StringArg),],
                               synopsis="Report model information")


def listinfo_chains(session, spec=None, attribute="chain_id"):
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    chains = []
    for m in results.models:
        try:
            chains.extend(m.chains)
        except AttributeError:
            # No chains, no problem
            pass
    report_chains(session.logger, chains, attribute)
listinfo_chains_desc = CmdDesc(keyword=[("spec", AtomSpecArg),
                                        ("attribute", StringArg),],
                               synopsis="Report chain information")


def listinfo_polymers(session, spec=None):
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    polymers = []
    for m in results.atoms.unique_structures:
        try:
            for p in m.polymers(False, False):
                if p.intersects(results.atoms):
                    polymers.append(p)
        except AttributeError:
            # No chains, no problem
            pass
    report_polymers(session.logger, polymers)
listinfo_polymers_desc = CmdDesc(keyword=([("spec", AtomSpecArg)]),
                                 synopsis="Report polymer information")


def listinfo_residues(session, spec=None, attribute="name"):
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    residues = results.atoms.unique_residues
    report_residues(session.logger, residues, attribute)
listinfo_residues_desc = CmdDesc(keyword=[("spec", AtomSpecArg),
                                          ("attribute", StringArg),],
                                 synopsis="Report residue information")


def listinfo_atoms(session, spec=None, attribute="idatm_type"):
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    residues = results.atoms.unique_residues
    report_atoms(session.logger, results.atoms, attribute)
listinfo_atoms_desc = CmdDesc(keyword=[("spec", AtomSpecArg),
                                       ("attribute", StringArg),],
                              synopsis="Report atom information")


def listinfo_selection(session, level=None, mode=None, attribute=None):
    if level is None or level == "atom":
        if attribute is None:
            attribute = "idatm_type"
        atoms = session.selection.items("atoms")
        if atoms:
            report_atoms(session.logger, atoms[0], attribute)
    elif level == "residue":
        if attribute is None:
            attribute = "name"
        atoms = session.selection.items("atoms")
        if atoms:
            report_residues(session.logger, atoms[0].unique_residues, attribute)
    elif level == "chain":
        if attribute is None:
            attribute = "chain_id"
        atoms = session.selection.items("atoms")
        if atoms:
            report_chains(session.logger, atoms[0].residues.unique_chains,
                          attribute)
    elif level == "molecule":
        if attribute is None:
            attribute = "name"
        atoms = session.selection.items("atoms")
        if atoms:
            report_models(session.logger, atoms[0].unique_structures, attribute)
    elif level == "model":
        if attribute is None:
            attribute = "name"
        report_models(session.logger, session.selection.all_models(), attribute)
listinfo_selection_desc = CmdDesc(keyword=[("level", EnumOf(["atom",
                                                             "residue",
                                                             "chain",
                                                             "molecule",
                                                             "model"])),
                                           ("mode", EnumOf(["any", "all"])),
                                           ("attribute", StringArg),],
                                  synopsis="Report selection information")


def listinfo_resattr(session):
    for a in _ResidueAttributes:
        report_resattr(session.logger, a)
listinfo_resattr_desc = CmdDesc(synopsis="Report residue attribute information")
_ResidueAttributes = [
    "chain_id",
    "description",
    "insertion_code",
    "is_helix",
    "is_strand",
    "polymer_type",
    "name",
    "num_atoms",
    "number",
    "ribbon_display",
    "ribbon_color",
    "ribbon_style",
    "ribbon_adjust",
    "ss_id",
]


def listinfo_distmat(session, spec):
    from scipy.spatial.distance import pdist, squareform
    if spec is None:
        from chimerax.core.commands import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    atoms = results.atoms
    coords = atoms.scene_coords
    distmat = squareform(pdist(coords, "euclidean"))
    for i in range(len(atoms)):
        for j in range(i+1,len(atoms)):
            report_distance(session.logger, atoms[i], atoms[j], distmat[i,j])
listinfo_distmat_desc = CmdDesc(required=([("spec", AtomSpecArg)]),
                                synopsis="Report distance matrix information")

from .util import Notifier
_WhatArg = EnumOf(Notifier.SupportedTypes)

def listinfo_notify_start(session, what, client_id, prefix="", url=None):
    Notifier.Find(what, client_id, session, prefix, url).start()
listinfo_notify_start_desc = CmdDesc(required=[("what", _WhatArg),
                                               ("client_id", StringArg),],
                                     keyword=[("prefix", StringArg),
                                              ("url", StringArg),],
                                     synopsis="Start notifications for events")


def listinfo_notify_stop(session, what, client_id):
    Notifier.Find(what, client_id).stop()
listinfo_notify_stop_desc = CmdDesc(required=[("what", _WhatArg),
                                              ("client_id", StringArg),],
                                    synopsis="Stop notifications for events")


def listinfo_notify_suspend(session, what, client_id):
    Notifier.Find(what, client_id).stop()
listinfo_notify_suspend_desc = CmdDesc(required=[("what", _WhatArg),
                                                 ("client_id", StringArg),],
                                       synopsis="Suspend notifications")


def listinfo_notify_resume(session, what, client_id):
    Notifier.Find(what, client_id).resume()
listinfo_notify_resume_desc = CmdDesc(required=[("what", _WhatArg),
                                                ("client_id", StringArg),],
                                      synopsis="Resume notifications")
