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
        self._formats = {}
        self._suffix_to_formats = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("data formats changed")
        super().__init__(name)

    def add_format(self, bundle_info, name, category, *, suffixes=None, nicknames=None,
            mime_types=None, reference_url=None, insecure=None, encoding=None,
            synopsis=None, allow_directory=False, raise_trigger=True):

        def convert_arg(arg, default=None):
            if arg and isinstance(arg, str):
                return arg.split(',')
            return [] if default is None else default
        suffixes = convert_arg(suffixes)
        nicknames = convert_arg(nicknames, [name.lower()])
        mime_types = convert_arg(mime_types)
        insecure = category == self.CAT_SCRIPT if insecure is None else insecure

        logger = self.session.logger
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
                reference_url, insecure, encoding, synopsis, allow_directory)
            for suffix in suffixes:
                self._suffix_to_formats.setdefault(suffix.lower(), []).append(data_format)
        self._formats[name] = (bundle_info, data_format)
        if raise_trigger:
            self.triggers.activate_trigger("data formats changed", self)

    def add_provider(self, bundle_info, name, *, category=None, suffixes=None, nicknames=None,
            mime_types=None, reference_url=None, insecure=None, encoding=None, synopsis=None,
            allow_directory=False, **kw):
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
            allow_directory=allow_directory, raise_trigger=False)

    def open_format_from_suffix(self, suffix):
        """
        Given a file suffix (starting with a '.'), return the corresponding openable data format.
            Returns None if there is no such format.
        """
        from chimerax.open_command import NoOpenerError
        return self._format_from_suffix(self.session.open_command.provider_info,
            NoOpenerError, suffix)

    def open_format_from_file_name(self, file_name):
        """
        Given a file name, return the corresponding openable data format.
            Raises NoFormatError if there is no such format.
        """
        "Return data format based on file_name's suffix, ignoring compression suffixes"
        return self._format_from_filename(self.open_format_from_suffix, file_name)

    def qt_file_filter(self, fmt):
        """
        Given a data format 'fmt', return a string usable as a member of the list argument
        used with the setNameFilters() method of a Qt file dialog.
        """
        return "%s (%s)" % (fmt.synopsis, "*" + " *".join(fmt.suffixes))

    def save_format_from_suffix(self, suffix):
        """
        Given a file suffix (starting with a '.'), return the corresponding savable data format.
            Returns None if there is no such format.
        """
        from chimerax.save_command import NoSaverError
        return self._format_from_suffix(self.session.save_command.provider_info,
            NoSaverError, suffix)

    def save_format_from_file_name(self, file_name):
        """
        Given a file name, return the corresponding saveable data format.
            Raises NoFormatError if there is no such format.
        """
        "Return data format based on file_name's suffix, ignoring compression suffixes"
        return self._format_from_filename(self.save_format_from_suffix, file_name)

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

    def _format_from_suffix(self, info_func, error_type, suffix):
        if '#' in suffix:
            suffix = suffix[:suffix.index('#')]
        try:
            formats = self._suffix_to_formats[suffix.lower()]
        except KeyError:
            return None

        fallback_fmt = fallback_provider_info = None
        for fmt in formats:
            try:
                provider_info = info_func(fmt)
            except error_type:
                pass
            else:
                if provider_info.is_default:
                    return fmt
                if fallback_fmt is None or (provider_info.bundle_info.installed
                and not fallback_provider_info.bundle_info.installed):
                    fallback_fmt = fmt
                    fallback_provider_info = provider_info
        return fallback_fmt
