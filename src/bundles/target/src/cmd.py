# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc, StringArg, ObjectsArg, NoArg


def target(session, name="Targets", all=False):
    from .tool import TargetsTool
    tool = TargetsTool(session, name, all, log_errors=True)
    tool.setup()
    return tool
target_desc = CmdDesc(optional=[("name", StringArg),
                                ("all", NoArg)])


def target_define(session, name, objects):
    from chimerax.core.commands import register_selector
    register_selector(name, objects, session.logger)
target_define_desc = CmdDesc(required=[("name", StringArg),
                                       ("objects", ObjectsArg)])


def target_undefine(session, name):
    from chimerax.core.commands import deregister_selector
    deregister_selector(name, session.logger)
target_undefine_desc = CmdDesc(required=[("name", StringArg)])


def target_list(session, all=False):
    from chimerax.core.commands import list_selectors, get_selector
    from chimerax.core.objects import Objects
    targets = {}
    for name in sorted(list_selectors()):
        sel = get_selector(name)
        if callable(sel):
            if not all and sel.__module__.startswith("chimerax.core"):
                continue
            value = "Function"
        elif isinstance(sel, Objects):
            value = ("[%d atoms, %d bonds, %d pbonds, %d models]" %
                     (sel.num_atoms, sel.num_bonds, sel.num_pseudobonds,
                      len(sel.models)))
        else:
            value = str(sel)
        session.logger.info('\t'.join([name, value]))
        targets[name] = value
    if not targets:
        session.logger.info("There are no user-defined targets.")
    return targets
target_list_desc = CmdDesc(optional=[("all", NoArg)])
