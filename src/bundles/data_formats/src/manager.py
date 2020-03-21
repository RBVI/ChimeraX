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

from chimerax.core.toolshed import ProviderManager
class FormatsManager(ProviderManager):
    """Manager for data formats"""

    CAT_SCRIPT = "Command script"
    CAT_GENERAL = "General"

    def __init__(self, session):
        self.session = session
        self._formats = {}
        self._suffix_to_format = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("data formats changed")

    def add_format(self, name, category, *, suffixes=None, nicknames=None,
            bundle_info=None, mime_types=None, reference_url=None, insecure=None,
            encoding=None, synopsis=None, allow_directory=False, raise_trigger=True):

        def convert_arg(arg, default=None):
            if arg and isinstance(arg, str):
                return arg.split(',')
            return [] if default is None else default
        suffixes = convert_arg(suffixes)
        nicknames = convert_arg(nicknames, [name.lower()])
        mime_types = convert_arg(mime_types)
        insecure = category == self.CAT_SCRIPT if insecure is None else insecure

        logger = self.session.logger
        if name in self._formats:
            registrant = lambda bi: "unknown registrant" \
                if bi is None else "%s bundle" % bi.name
            logger.info("Replacing data format '%s' as defined by %s with definition"
                " from %s" % (name, registrant(self._formats[name][0]),
                registrant(bundle_info)))
        from .format import DataFormat
        data_format = DataFormat(name, category, suffixes, nicknames, mime_types,
            reference_url, insecure, encoding, synopsis, allow_directory)
        for suffix in suffixes:
            if suffix in self._suffix_to_format:
                other_name = self._suffix_to_format[suffix].name
                if name != other_name:
                    logger.warning("File suffix '%s' for data format %s being replaced by format"
                        " %s" % (suffix, other_name, name))
            self._suffix_to_format[suffix] = data_format
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
        self.add_format(name, category, suffixes=suffixes, nicknames=nicknames,
            bundle_info=bundle_info, mime_types=mime_types, reference_url=reference_url,
            insecure=insecure, encoding=encoding, synopsis=synopsis,
            allow_directory=allow_directory, raise_trigger=False)

    def format_from_suffix(self, suffix):
        if '#' in suffix:
            suffix = suffix[:suffix.index('#')]
        return self._suffix_to_format.get(suffix, None)

    def file_name_to_format(self, file_name):
        "Return data format based on file_name's suffix, ignoring compression suffixes"
        from chimerax.core.errors import UserError
        if '.' in file_name:
            from chimerax import io
            base_name = io.remove_compression_suffix(file_name)
            try:
                dot_pos = base_name.rindex('.')
            except ValueError:
                raise UserError("'%s' has only compression suffix; cannot determine"
                    " format from suffix" % file_name)
            data_format = self.format_from_suffix(base_name[dot_pos:])
            if not data_format:
                raise UserError("No known data format for file suffix '%s'"
                    % base_name[dot_pos:])
        else:
            raise UserError("Cannot determine format for '%s'" % file_name)
        return data_format

    @property
    def formats(self):
        return [info[1] for info in self._formats.values()]

    def end_providers(self):
        self.triggers.activate_trigger("data formats changed", self)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Data format key is not a string")
        if key in self._formats:
            return self._formats[key][1]
        for bi, format_data in self._formats.values():
            if key in format_data.nicknames:
                return format_data
        raise KeyError("No known data format '%s'" % key)

    def __len__(self):
        return len(self._formats)
