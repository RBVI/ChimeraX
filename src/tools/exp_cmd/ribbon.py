# vi: set expandtab shiftwidth=4 softtabstop=4:


def initialize(command_name):
    from chimera.core.commands import alias
    alias(None, command_name, "%s $*" % command_name.replace("ribbon", "cartoon"))
