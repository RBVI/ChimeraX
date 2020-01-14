# vim: set expandtab ts=4 sw=4:

from chimerax.core.errors import UserError
from chimerax.core.commands import CmdDesc, StringArg, AtomSpecArg, ObjectsArg
from chimerax.core.commands import RestOfLine, BoolArg, AnnotationError


def name(session, name, text=None, skip_check=False):
    if name == "all":
        raise UserError("\"all\" is reserved and cannot be shown or defined")
    if text is None:
        from chimerax.core.commands import get_selector_description
        try:
            desc = get_selector_description(name, session)
        except KeyError:
            raise UserError("\"%s\" is not defined" % name)
        else:
            if desc:
                session.logger.info('\t'.join([name, desc]))
    else:
        if _is_reserved(name):
            raise UserError("\"%s\" is reserved and cannot be redefined" % name)
        if not skip_check:
            try:
                ast, used, unused = AtomSpecArg.parse(text, session)
                if unused:
                    raise AnnotationError("contains extra trailing text")
            except AnnotationError as e:
                raise UserError("\"%s\": %s" % (text, str(e)))
        def selector(session, models, results, spec=text):
            objects, used, unused = ObjectsArg.parse(spec, session)
            results.combine(objects)
        from chimerax.core.commands import register_selector
        register_selector(name, selector, session.logger, user=True, desc=text)
        session.basic_actions.define(name, text)
name_desc = CmdDesc(required=[("name", StringArg)],
                    optional=[("text", RestOfLine)],
                    non_keyword=["text"])


def _is_reserved(name):
    from chimerax.core.commands import is_selector_user_defined
    try:
        return not is_selector_user_defined(name)
    except KeyError:
        return name == "all"


def name_frozen(session, name, objects):
    if _is_reserved(name):
        raise UserError("\"%s\" is reserved and cannot be redefined" % name)
    if objects.empty():
        raise UserError("nothing is selected by specifier")
    from chimerax.core.commands import register_selector
    register_selector(name, objects, session.logger, user=True)
    session.basic_actions.define(name, objects)
name_frozen_desc = CmdDesc(required=[("name", StringArg),
                                     ("objects", ObjectsArg)])


def name_delete(session, name):
    if name != "all":
        if _is_reserved(name):
            raise UserError("\"%s\" is reserved and cannot be deleted" % name)
        from chimerax.core.commands import deregister_selector
        deregister_selector(name, session.logger)
    else:
        from chimerax.core.commands import list_selectors, deregister_selector
        for name in list(list_selectors()):
            if not _is_reserved(name):
                deregister_selector(name, session.logger)
    session.basic_actions.remove(name)
name_delete_desc = CmdDesc(required=[("name", StringArg)])


def name_list(session, builtins=False, log=True):
    from chimerax.core.commands import list_selectors, is_selector_user_defined
    from chimerax.core.commands import get_selector_description
    names = {}
    for name in sorted(list_selectors()):
        if not builtins and not is_selector_user_defined(name):
            continue
        desc = get_selector_description(name, session)
        if desc:
            if log:
                session.logger.info('\t'.join([name, desc]))
            names[name] = desc
    if not names and log:
        session.logger.info("There are no user-defined specifier names.")
    return names
name_list_desc = CmdDesc(keyword=[("builtins", BoolArg)])
