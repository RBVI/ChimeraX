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
class OpenManager(ProviderManager):
    """Manager for open command"""

    def __init__(self, session):
        self.session = session
        self._openers = {}
        self._compression_info = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("open command changed")

    def add_provider(self, bundle_info, name, *, type="open", want_path=False, check_path=True,
            compression_suffixes=None, **kw):
        logger = self.session.logger

        type_description = "Open-command" if type == "open" else type.capitalize()
        bundle_name = _readable_bundle_name(bundle_info)
        if kw:
            logger.warning("%s provider '%s' supplied unknown keywords in provider description: %s"
                % (type_description, name, repr(kw)))
        if type == "open":
            try:
                data_format = self.session.data_formats[name]
            except KeyError:
                raise ValueError("Open-command provider in bundle %s specified unknown data format '%s'"
                    % (bundle_name, name))
            if data_format in self._openers:
                logger.warning("Replacing opener for '%s' from %s bundle with that from %s bundle"
                    % (data_format.name, _readable_bundle_name(self._openers[data_format][0]), bundle_name))
            self._openers[data_format] = (bundle_info, want_path, check_path)
        elif type == "fetch":
            pass #TODO
        elif type == "compression":
            suffixes = process_suffixes(compression_suffixes, "compression", name, logger, bundle_name)
            for suffix in suffixes:
                if suffix in self._compression_info:
                    prev_bi, prev_name = self._compression_info[suffix]
                    logger.warning("Duplicate decompression provider registered for file suffix '%s'."
                        " ('%s' from bundle %s and '%s' from bundle %s)" % (prev_name,
                        _readable_bundle_name(prev_bi), name, bundle_name))
                self._compression_info[suffix] = (bundle_info, name)
        else:
            logger.warning("Unknown provider type '%s' with name '%s' from bundle %s"
                % (type, name, bundle_name))

    def end_providers(self):
        self.triggers.activate_trigger("open command changed", self)

    def remove_compression_suffix(self, file_name):
        for suffix in self._compression_info.keys():
            if file_name.endswith(suffix):
                file_name = file_name[:-len(suffix)]
                break
        return file_name

def _readable_bundle_name(bundle_info):
    name = bundle_info.name
    if name.lower().startswith("chimerax"):
        return name[9:]
    return name

def process_suffixes(suffix_string, suffix_type, name, logger, bundle_name, none_okay=False):
    if not suffix_string:
        if not none_okay:
            logger.warning("%s provider for '%s' from bundle %s specified no %s suffixes"
                % (suffix_type.capitalize(), name, bundle_name, suffix_type))
        return []
    suffixes = []
    for suffix in suffix_string.split(','):
        if not suffix:
            logger.warning("Empty %s suffix found in suffix list of provider for '%s'"
                " from bundle %s" % (suffix_type, name, bundle_name))
            continue
        if suffix[0] != '.':
            logger.warning("%s suffixes must start with '.'.  '%s' found in suffix list of provider for"
                " '%s' in bundle %s does not." % (suffix_type, suffix, name, bundle_name))
        suffixes.append(suffix)
    return suffixes
