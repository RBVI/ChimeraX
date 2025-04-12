# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import re
import unicodedata


class MissingInfoError(Exception):
    pass


class ChimeraXClassifier:
    classifier_separator = " :: "

    def __init__(self, name, attrs):
        self.name = name
        self.attrs = attrs

    @property
    def categories(self):
        if "category" in self.attrs:
            return self.attrs["category"]
        if "categories" in self.attrs:
            return self.attrs["categories"]
        raise MissingInfoError(f"No synopsis found for {self.name}")

    @property
    def description(self):
        if "synopsis" in self.attrs:
            return self.attrs["synopsis"]
        if "description" in self.attrs:
            return self.attrs["description"]
        raise MissingInfoError(f"No synopsis found for {self.name}")

    def misc_attrs_to_list(self):
        attrs = []
        for k, v in self.attrs.items():
            formatted_field = k.replace("-", "_")
            if isinstance(v, list):
                if not v:
                    continue
                formatted_val = ",".join([quote_if_necessary(str(val)) for val in v])
            elif isinstance(v, bool):
                formatted_val = quote_if_necessary(str(v).lower())
            else:
                formatted_val = quote_if_necessary(str(v))
            attrs.append("%s:%s" % (formatted_field, formatted_val))
        return attrs


class Tool(ChimeraXClassifier):
    def __init__(self, tool_name: str, attrs: dict[str:str]):
        super().__init__(tool_name, attrs)

    def __str__(self):
        if isinstance(self.categories, str):
            return f"ChimeraX :: Tool :: {self.name} :: {self.categories} :: {self.description}"
        else:
            return f'ChimeraX :: Tool :: {self.name} :: {", ".join(self.categories)} :: {self.description}'


class Command(ChimeraXClassifier):
    def __init__(self, command_name: str, attrs: dict[str:str]):
        super().__init__(command_name, attrs)

    def __str__(self):
        if isinstance(self.categories, str):
            return f"ChimeraX :: Command :: {self.name} :: {self.categories} :: {self.description}"
        else:
            return f'ChimeraX :: Command :: {self.name} :: {", ".join(self.categories)} :: {self.description}'


class Selector(ChimeraXClassifier):
    def __init__(self, selector_name: str, attrs: dict[str:str]):
        super().__init__(selector_name, attrs)

    def __str__(self):
        disp = self.attrs.get('display', '')
        disp = str(disp).lower()
        if disp:
            return f"ChimeraX :: Selector :: {self.name} :: {self.description} :: {disp}"
        return f"ChimeraX :: Selector :: {self.name} :: {self.description}"


class Manager(ChimeraXClassifier):
    default_attrs = {"guiOnly": False, "autostart": True}

    def __init__(self, name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
            attrs['guiOnly'] = attrs.pop('gui-only', False)
        super().__init__(name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        return f"ChimeraX :: Manager :: {self.name} :: {self.classifier_separator.join(attrs)}"


class Provider(ChimeraXClassifier):
    def __init__(self, manager, name, attrs: dict[str:str]):
        self.manager = manager
        super().__init__(name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        if attrs:
            return f"ChimeraX :: Provider :: {self.name} :: {self.manager} :: {self.classifier_separator.join(attrs)}"
        return f"ChimeraX :: Provider :: {self.name} :: {self.manager}"


class DataFormat(Provider):
    default_attrs = {
        "category": "General",
        "encoding": "utf-8",
        "nicknames": None,
        "reference-url": None,
        "suffixes": None,
        "synopsis": None,
        "allow-directory": False,
        "insecure": False,
        "mime-types": [],
    }

    def __init__(self, name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__("data formats", name, attrs)


class FormatReader(Provider):
    default_attrs = {
        "batch": False,
        "check-path": True,
        "is-default": True,
        "pregrouped-structures": False,
        "type": "open",
        "want-path": False,
    }

    def __init__(self, reader_name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__("open command", reader_name, attrs)


class FormatSaver(Provider):
    default_attrs = {"compression-okay": True, "is-default": True}

    def __init__(self, saver_name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__("save command", saver_name, attrs)


class FormatFetcher(Provider):
    default_attrs = {
        "batch": False,
        "check-path": False,
        "is-default": True,
        "pregrouped-structures": False,
        "type": "fetch",
        "want-path": False,
    }

    def __init__(self, name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            name, attrs["format_name"] = attrs.pop("name"), name
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__("open command", name, attrs)


class Preset(Provider):
    default_attrs = {"category": "General"}

    def __init__(self, name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__("presets", name, attrs)


class ToolbarTab(Provider):

    def __init__(self, tab_name, attrs):
        attrs["tab"] = tab_name
        name = "tab-" + tab_name.lower().replace(" ", "-")
        super().__init__("toolbar", name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        return f"ChimeraX :: Provider :: {self.name} :: {self.manager} :: {self.classifier_separator.join(attrs)}"


class ToolbarSection(Provider):
    def __init__(self, tab_name, section_name, attrs):
        self.tab_name = tab_name
        attrs["tab"] = tab_name
        attrs["section"] = section_name
        name = "section-" + section_name.lower().replace(" ", "-")
        super().__init__("toolbar", name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        return f"ChimeraX :: Provider :: {self.name} :: {self.manager} :: {self.classifier_separator.join(attrs)}"


class ToolbarButton(Provider):
    def __init__(self, tab_name, section_name, button_name, attrs):
        attrs["tab"] = tab_name
        attrs["section"] = section_name
        name = "button-" + button_name.lower().replace(" ", "-")
        super().__init__("toolbar", name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        return f"ChimeraX :: Provider :: {self.name} :: {self.manager} :: {self.classifier_separator.join(attrs)}"


class Initialization:
    def __init__(self, type_=None, bundles=None):
        self.type_ = type_
        if not bundles:
            bundles = []
        self.bundles = bundles

    def __str__(self):
        separator = " :: "
        return (
            f"ChimeraX :: InitAfter :: {self.type_} :: {separator.join(self.bundles)}"
        )


def quote_if_necessary(s, additional_special_map={}):
    """quote a string

    So :py:class:`StringArg` treats it like a single value"""
    _internal_single_quote = re.compile(r"'\s")
    _internal_double_quote = re.compile(r'"\s')
    if not s:
        return '""'
    has_single_quote = s[0] == "'" or _internal_single_quote.search(s) is not None
    has_double_quote = s[0] == '"' or _internal_double_quote.search(s) is not None
    has_special = False
    use_single_quote = not has_single_quote and has_double_quote
    special_map = {
        "\a": "\\a",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\v": "\\v",
        "\\": "\\\\",
        ";": ";",
        " ": " ",
    }
    special_map.update(additional_special_map)

    result = []
    for ch in s:
        i = ord(ch)
        if ch == "'":
            result.append(ch)
        elif ch == '"':
            if use_single_quote:
                result.append('"')
            elif has_double_quote:
                result.append('\\"')
            else:
                result.append('"')
        elif ch in special_map:
            has_special = True
            result.append(special_map[ch])
        elif i < 32:
            has_special = True
            result.append("\\x%02x" % i)
        elif ch.strip() == "":
            # non-space and non-newline spaces
            has_special = True
            result.append("\\N{%s}" % unicodedata.name(ch))
        else:
            result.append(ch)
    if has_single_quote or has_double_quote or has_special:
        if use_single_quote:
            return "'%s'" % "".join(result)
        else:
            return '"%s"' % "".join(result)
    return "".join(result)
