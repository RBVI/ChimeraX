# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc


def hello_world(session):
    # All command functions are invoked with ``session`` as its
    # first argument.  Useful session attributes include:
    #   logger: chimerax.core.logger.Logger instance
    #   models: chimerax.core.models.Models instance
    session.logger.info("Hello world!")

# CmdDesc contains the command description.  For the
# "hello" command, we expect no arguments.
hello_world_desc = CmdDesc()
