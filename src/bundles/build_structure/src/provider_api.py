# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import abc

class StartStructureProvider(metaclass=abc.ABCMeta):
    def __init__(self, session):
        self.session = session

    @abc.abstractmethod
    def command_string(self, widget):
        # Return the command (sub)string corresponding to the settings in the widget.
        # Can return None if the provider doesn't directly add atoms [e.g. links to another tool].
        # If 'new_model_only' was "true" in the Provider tag, then the returned string should be
        # the _entire_ command for opening the model.
        # Otherwise it should be the argument substring of your 'build start' provider's subcommand
        pass

    def execute_command(self, structure, args):
        # Execute this providers subcommand into the given structure and with the given args.
        # This method will not be called if 'new_model_only' was "true" in the Provider tag
        # (i.e. command_string() returns a full command) or if 'indirect' was "true" in the
        # Provider tag (i.e. provider doesn't directly add atoms)
        raise NotImplementedError("Start-structure provider failed to implement 'execute_command'")

    @abc.abstractmethod
    def fill_parameters_widget(self, widget):
        # populate the given widget with controls for specifying the arg values for this providers
        # command
        pass

