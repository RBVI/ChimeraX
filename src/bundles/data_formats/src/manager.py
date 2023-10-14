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

class NoFormatError(ValueError):
    pass

from chimerax.core.toolshed import ProviderManager
class FormatsManager(ProviderManager):
    """
    Manager for data formats.
        Manager can also be used as if it were a { format-name -> data format } dictionary.
    """

    CAT_SCRIPT = "Command script"
    CAT_SESSION = "Session"
    CAT_GENERAL = "General"

    def __init__(self, session, name):
        self.session = session
        self.settings = FormatsManagerSettings(session, "data formats manager")
        self._formats = {}
        self._suffix_to_formats = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("data formats changed")
        self._user_response_cache = {}
        super().__init__(name)

    def add_format(self, bundle_info, name, category, *, suffixes=None, nicknames=None,
            mime_types=None, reference_url=None, insecure=None, encoding=None,
            synopsis=None, allow_directory=False, raise_trigger=True, default_for=None):

        def convert_arg(arg, default=None):
            if arg and isinstance(arg, str):
                if arg == 'None':
                    return None
                if arg == 'true':
                    return True
                if arg == 'false':
                    return False
                return arg.split(',')
            return [] if default is None else default
        suffixes = convert_arg(suffixes)
        nicknames = convert_arg(nicknames, [name.lower()])
        mime_types = convert_arg(mime_types)
        allow_directory = convert_arg(allow_directory, default=False)
        insecure = convert_arg(insecure, default=False)
        default_for = convert_arg(default_for)


        insecure = category == self.CAT_SCRIPT if (insecure is None or insecure is False) else insecure

        logger = self.session.logger
        extra = set(default_for) - set(suffixes)
        if extra:
            from chimerax.core.commands import plural_form
            logger.warning("Data format '%s' (defined by %s) declares default %s (%s) that are not in the"
                " format's list of suffixes (%s)" % (name, bundle_info.name, plural_form(extra, "suffix"),
                ", ".join(list(extra)), ", ".join(suffixes)))
        update_bundle_only = False
        if name in self._formats:
            if not bundle_info.installed:
                return
            prev_bundle = self._formats[name][0]
            if prev_bundle.installed:
                logger.info("Replacing data format '%s' as defined by %s with definition"
                    " from %s" % (name, prev_bundle.name, bundle_info.name))
                del self._formats[name]
            else:
                # usually previously uninstalled bundle getting installed
                update_bundle_only = prev_bundle.name == bundle_info.name
        if update_bundle_only:
            data_format = self._formats[name][1]
        else:
            from .format import DataFormat
            data_format = DataFormat(name, category, suffixes, nicknames, mime_types,
                reference_url, insecure, encoding, synopsis, allow_directory, default_for)
            for suffix in suffixes:
                self._suffix_to_formats.setdefault(suffix.lower(), []).append(data_format)
        self._formats[name] = (bundle_info, data_format)
        if raise_trigger:
            self.triggers.activate_trigger("data formats changed", self)

    def add_provider(self, bundle_info, name, *, category=None, suffixes=None, nicknames=None,
            mime_types=None, reference_url=None, insecure=None, encoding=None, synopsis=None,
            allow_directory=False, default_for=None, **kw):
        logger = self.session.logger
        if kw:
            logger.warning("Data format provider '%s' supplied unknown keywords with format"
                " description: %s" % (name, repr(kw)))
        if category is None:
            logger.warning("Data format provider '%s' didn't specify a category."
                "  Using catch-all category '%s'" % (name, self.CAT_GENERAL))
            category = self.CAT_GENERAL
        self.add_format(bundle_info, name, category, suffixes=suffixes, nicknames=nicknames,
            mime_types=mime_types, reference_url=reference_url,
            insecure=insecure, encoding=encoding, synopsis=synopsis,
            allow_directory=allow_directory, default_for=default_for, raise_trigger=False)

    def open_format_from_suffix(self, suffix, *, clear_cache_before=True, cache_user_responses=True,
            clear_cache_after=False):
        """
        Given a file suffix (starting with a '.'), return the corresponding openable data format.
            Returns None if there is no such format.

        The 'cache_...' keywords are for controlling whether user choices between multiple formats
        are remembered between calls, and except in rare circumstances need not be specified.
        """
        from chimerax.open_command import NoOpenerError
        return self._format_from_suffix(self.session.open_command.provider_info, NoOpenerError, suffix,
            clear_cache_before, cache_user_responses, clear_cache_after)

    def open_format_from_file_name(self, file_name, *, clear_cache_before=True, cache_user_responses=True,
            clear_cache_after=False):
        """
        Given a file name, return the corresponding openable data format.
            Raises NoFormatError if there is no such format.

        The 'cache_...' keywords are for controlling whether user choices between multiple formats
        are remembered between calls, and except in rare circumstances need not be specified.
        """
        "Return data format based on file_name's suffix, ignoring compression suffixes"
        func = lambda suffix, *, ccb=clear_cache_before, cur=cache_user_responses, cca=clear_cache_after, \
            f=self.open_format_from_suffix: f(suffix, clear_cache_before=ccb, cache_user_responses=cur, \
            clear_cache_after=cca)
        return self._format_from_filename(func, file_name)

    def qt_file_filter(self, fmt):
        """
        Given a data format 'fmt', return a string usable as a member of the list argument
        used with the setNameFilters() method of a Qt file dialog.
        """
        return "%s (%s)" % (fmt.synopsis, "*" + " *".join(fmt.suffixes))

    def save_format_from_suffix(self, suffix, *, clear_cache_before=True, cache_user_responses=True,
            clear_cache_after=False):
        """
        Given a file suffix (starting with a '.'), return the corresponding savable data format.
            Returns None if there is no such format.

        The 'cache_...' keywords are for controlling whether user choices between multiple formats
        are remembered between calls, and except in rare circumstances need not be specified.
        """
        from chimerax.save_command import NoSaverError
        return self._format_from_suffix(self.session.save_command.provider_info, NoSaverError, suffix,
            clear_cache_before, cache_user_responses, clear_cache_after)

    def save_format_from_file_name(self, file_name, *, clear_cache_before=True, cache_user_responses=True,
            clear_cache_after=False):
        """
        Given a file name, return the corresponding saveable data format.
            Raises NoFormatError if there is no such format.

        The 'cache_...' keywords are for controlling whether user choices between multiple formats
        are remembered between calls, and except in rare circumstances need not be specified.
        """
        "Return data format based on file_name's suffix, ignoring compression suffixes"
        func = lambda suffix, *, ccb=clear_cache_before, cur=cache_user_responses, cca=clear_cache_after, \
            f=self.save_format_from_suffix: f(suffix, clear_cache_before=ccb, cache_user_responses=cur, \
            clear_cache_after=cca)
        return self._format_from_filename(func, file_name)

    @property
    def formats(self):
        """ Returns a list of all known data formats """
        return [info[1] for info in self._formats.values()]

    def end_providers(self):
        self.triggers.activate_trigger("data formats changed", self)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Data format key is not a string")
        if key in self._formats:
            return self._formats[key][1]
        fallback = None
        for bi, format_data in self._formats.values():
            if key in format_data.nicknames:
                if bi.installed:
                    return format_data
                fallback = format_data
        if fallback is not None:
            return fallback
        raise KeyError("No known data format '%s'" % key)

    def __len__(self):
        return len(self._formats)

    def __iter__(self):
        '''iterator over models'''
        return iter(self.formats)

    def _format_from_filename(self, suffix_func, file_name):
        if '.' in file_name:
            from chimerax import io
            base_name = io.remove_compression_suffix(file_name)
            from os.path import splitext
            root, ext = splitext(base_name)
            if not ext:
                raise NoFormatError("'%s' has only compression suffix; cannot determine"
                    " format from suffix" % file_name)
            data_format = suffix_func(ext)
            if not data_format:
                raise NoFormatError("No known data format for file suffix '%s'" % ext)
        else:
            raise NoFormatError("Cannot determine format for '%s'" % file_name)
        return data_format

    def _format_from_suffix(self, info_func, error_type, suffix,
            clear_cache_before, cache_user_responses, clear_cache_after):
        if clear_cache_before:
            self._user_response_cache.clear()
        if '#' in suffix:
            suffix = suffix[:suffix.index('#')]
        suffix = suffix.lower()
        try:
            try:
                suffix_formats = self._suffix_to_formats[suffix.lower()]
            except KeyError:
                return None
            relevant_formats = []
            for fmt in suffix_formats:
                try:
                    provider_info = info_func(fmt)
                except error_type:
                    continue
                relevant_formats.append(fmt)

            if len(relevant_formats) == 1:
                return relevant_formats[0]

            defaults = [fmt for fmt in relevant_formats if suffix in fmt.default_for]
            if defaults:
                if len(defaults) == 1:
                    return defaults[0]
                self.session.logger.warning(
                    "Multiple formats (%s) declare themselves as default for suffix %s"
                    % (", ".join([fmt.nicknames[0] for fmt in relevant_formats]), suffix))

            if suffix in self._user_response_cache:
                return self._user_response_cache[suffix]
            try:
                preferred_format_name = self.settings.suffix_to_format_name[suffix]
            except KeyError:
                pass
            else:
                for fmt in relevant_formats:
                    if fmt.name == preferred_format_name:
                        return fmt

            if self.session.ui.is_gui and not self.session.in_script:
                from .gui import ask_for_format
                fmt = ask_for_format(self.session, suffix, relevant_formats)
                if cache_user_responses:
                    self._user_response_cache[suffix] = fmt
                return fmt

            from chimerax.core.errors import UserError
            raise UserError("Multiple formats (%s) support %s suffix and none are declared as default; need"
                " to specify format by using 'format' keyword" %
                (" ,".join([fmt.nicknames[0] for fmt in relevant_formats]), suffix))
        finally:
            if clear_cache_after:
                self._user_response_cache.clear()

from chimerax.core.settings import Settings
class FormatsManagerSettings(Settings):
    AUTO_SAVE = {
        'suffix_to_format_name': {}
    }
