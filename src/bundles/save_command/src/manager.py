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

class NoSaverError(ValueError):
    pass

class SaverNotInstalledError(NoSaverError):
    pass

class ProviderInfo:
    def __init__(self, bundle_info, format_name, compression_okay, is_default):
        self.bundle_info = bundle_info
        self.format_name = format_name
        self.compression_okay = compression_okay
        self.is_default = is_default

from chimerax.core.toolshed import ProviderManager
class SaveManager(ProviderManager):
    """Manager for save command"""

    def __init__(self, session, name):
        self.session = session
        self._savers = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("save command changed")
        super().__init__(name)

    def add_provider(self, bundle_info, format_name, compression_okay=True, is_default=True, **kw):
        logger = self.session.logger

        bundle_name = _readable_bundle_name(bundle_info)
        if kw:
            logger.warning("Save-command provider '%s' supplied unknown keywords in"
                " provider description: %s" % (bundle_name, repr(kw)))
        try:
            data_format = self.session.data_formats[format_name]
        except KeyError:
            logger.warning("Save-command provider in bundle %s specified unknown data"
                " format '%s'; skipping" % (bundle_name, format_name))
            return
        if data_format in self._savers:
            if not bundle_info.installed:
                return
            prev_bundle = self._savers[data_format].bundle_info
            if prev_bundle.installed:
                logger.warning("Replacing file-saver for '%s' from %s bundle with that from %s bundle"
                    % (data_format.name, _readable_bundle_name(prev_bundle), bundle_name))
        self._savers[data_format] = ProviderInfo(bundle_info, format_name,
            bool_cvt(compression_okay, format_name, bundle_name, "compression_okay"),
            bool_cvt(is_default, format_name, bundle_name, "is_default"))

    def end_providers(self):
        self.triggers.activate_trigger("save command changed", self)

    def provider_info(self, data_format):
        try:
            return self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)

    def save_args(self, data_format):
        try:
            provider_info = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        if not provider_info.bundle_info.installed:
            raise SaverNotInstalledError("File-saver for format '%s' not installed" % data_format.name)
        return provider_info.bundle_info.run_provider(self.session,
            provider_info.format_name, self).save_args

    def hidden_args(self, data_format):
        try:
            provider_info = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        return provider_info.bundle_info.run_provider(self.session,
            provider_info.format_name, self).hidden_args

    def save_args_widget(self, data_format):
        try:
            provider_info = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        bi = provider_info.bundle_info
        if not bi.installed:
            raise SaverNotInstalledError("File-saver for format '%s' not installed" % data_format.name)
        return bi.run_provider(self.session, provider_info.format_name, self).save_args_widget(self.session)

    def save_args_string_from_widget(self, data_format, widget):
        try:
            provider_info = self._savers[data_format]
        except KeyError:
            raise NoSaverError("No file-saver registered for format '%s'"
                % data_format.name)
        return provider_info.bundle_info.run_provider(self.session,
            provider_info.format_name, self).save_args_string_from_widget(widget)

    def save_data(self, path, **kw):
        """
        Given a file path and possibly format-specific keywords, save a data file based on the
        current session.

        The format name can be provided with the 'format' keyword if the filename suffix of the path
        does not correspond to those for the desired format.
        """
        from .cmd import provider_save
        provider_save(self.session, path, **kw)

    @property
    def save_data_formats(self):
        """
        The names of data formats for which an saver function has been registered.
        """
        return list(self._savers.keys())

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

