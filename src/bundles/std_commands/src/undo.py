# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===


def undo(session):
    '''Undo last undoable action.
    '''
    try:
        session.undo.undo(silent=False)
    except IndexError:
        from chimerax.core.errors import UserError
        raise UserError("No undo action is available")


def undo_list(session, nested=False):
    '''List undoable actions
    '''
    msgs = _list_stack("undo", session.undo.undo_stack, nested)
    msgs.extend(_list_stack("redo", session.undo.redo_stack, nested))
    session.logger.info(''.join(msgs), is_html=True)


def _list_stack(label, stack, show_nested):
    from chimerax.core.undo import UndoAggregateAction
    msgs = ["<p>There are %d %s actions</p>" % (len(stack), label)]
    def show_items(stack):
        msgs.append("<ul>")
        for item in stack:
            msgs.append("<li>%s</li>" % item.name)
            if show_nested and isinstance(item, UndoAggregateAction):
                show_items(item.actions)
        msgs.append("</ul>")
    show_items(stack)
    return msgs


def undo_clear(session):
    '''Clear all undoable and redoable actions
    '''
    stack = session.undo.clear()


def undo_depth(session, depth=None):
    if depth is None:
        session.logger.info("Undo stack depth is %d" % session.undo.max_depth)
    else:
        if depth < 0:
            depth = 0
        session.undo.set_depth(depth)
        if session.undo.max_depth > 0:
            session.logger.info("Undo stack depth is set to %d" %
                                session.undo.max_depth)
        else:
            session.logger.info("Undo stack depth is unlimited")


def redo(session):
    '''Redo last undone action.
    '''
    try:
        session.undo.redo(silent=False)
    except IndexError:
        from chimerax.core.errors import UserError
        raise UserError("No redo action is available")


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, IntArg, NoArg
    desc = CmdDesc(synopsis='undo last action')
    register('undo', desc, undo, logger=logger)
    desc = CmdDesc(synopsis='list available undo actions',
            optional=[('nested', NoArg)], hidden=['nested'])
    register('undo list', desc, undo_list, logger=logger)
    desc = CmdDesc(synopsis='clear all undo and redo actions')
    register('undo clear', desc, undo_clear, logger=logger)
    desc = CmdDesc(optional=[('depth', IntArg)],
                   synopsis='set undo/redo stack depth')
    register('undo depth', desc, undo_depth, logger=logger)
    desc = CmdDesc(
        synopsis='redo last undone action',
        url='help:user/commands/undo.html#redo'
    )
    register('redo', desc, redo, logger=logger)
