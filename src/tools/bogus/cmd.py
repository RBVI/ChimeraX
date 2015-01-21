# vim: set expandtab ts=4 sw=4:

from chimera.core import cli


def bogus_function(session, args="no arguments"):
    session.logger.info("bogus: %s" % args)
bogus_desc = cli.CmdDesc(optional=[("args", cli.RestOfLine)])
