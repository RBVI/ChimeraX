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

class NoSaverError(ValueError):
    pass

from chimerax.core.toolshed import ProviderManager
class SaveManager(ProviderManager):
    """Manager for save command"""

    def __init__(self, session):
        self.session = session
        self._savers = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("save command changed")

    def add_provider(self, bundle_info, format_name, compression_okay=True, **kw):
        logger = self.session.logger

        bundle_name = _readable_bundle_name(bundle_info)
        if kw:
            logger.warning("Save-command provider '%s' supplied unknown keywords in"
                " provider description: %s" % (name, repr(kw)))
        try:
            data_format = self.session.data_formats[format_name]
        except KeyError:
            logger.warning("Save-command provider in bundle %s specified unknown data"
                " format '%s'; skipping" % (bundle_name, format_name))
            return
        if data_format in self._savers:
            logger.warning("Replacing file-saver for '%s' from %s bundle with that from"
                " %s bundle" % (data_format.name,
                _readable_bundle_name(self._savers[data_format][0]), bundle_name))
        self._savers[data_format] = (bundle_info, format_name,
            bool_cvt(compression_okay, format_name, bundle_name, "compression_okay"))

    def end_providers(self):
        self.triggers.activate_trigger("save command changed", self)

    def save_args(self, data_format):
        try:
            bundle_info, format_name, *args = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        return bundle_info.run_provider(self.session, format_name, self).save_args

    def hidden_args(self, data_format):
        try:
            bundle_info, format_name, *args = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        return bundle_info.run_provider(self.session, format_name, self).hidden_args

    def save_args_widget(self, data_format):
        try:
            bundle_info, format_name, *args = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        return bundle_info.run_provider(self.session, format_name, self
            ).save_args_widget(self.session)

    def save_args_string_from_widget(self, data_format, widget):
        try:
            bundle_info, format_name, *args = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        return bundle_info.run_provider(self.session, format_name, self
            ).save_args_string_from_widget(widget)

    @property
    def save_data_formats(self):
        return list(self._savers.keys())

    def save_info(self, data_format):
        try:
            return self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)

def _readable_bundle_name(bundle_info):
    name = bundle_info.name
    if name.lower().startswith("chimerax"):
        return name[9:]
    return name

def bool_cvt(val, name, bundle_name, var_name):
    if not isinstance(val, bool):
        try:
            val = eval(val.capitalize())
        except (ValueError, NameError):
            logger.warning("Save provider '%s' in bundle %s specified '%s'"
                " value (%s) that was neither 'true' nor 'false'"
                % (name, bundle_name, var_name, val))
    return val

