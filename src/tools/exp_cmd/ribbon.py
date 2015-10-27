# vi: set expandtab shiftwidth=4 softtabstop=4:


def initialize(command_name):
    from chimera.core.commands import create_alias
    create_alias(command_name, "%s $*" % command_name.replace("ribbon", "cartoon"))
