# TODO: Modern NLP uses intent recognition to do this, but
# to hack together a limited demo I'm going to do some
# rudimentary keyword recognition and string manipulation
# Here's the strategy:
# - Split the command into words
# - Scan through them to find a word or prhase corresponding to a ChimeraX command
#   Often the first word is going to be the command, but sometimes there are ambiguous
#   prhases like 'make the background color black', but 'background' uniquely identifies
#   the set command.
# - Return an intent that corresponds to a unique hand written parser for that command
from enum import StrEnum


class Intent(StrEnum):
    # one word commands
    SET = "set"
    SELECT = "select"
    VOLUME = "volume"


class CommandParser:
    def parse(self, command: str) -> str:
        words = command.split()
        intent = determine_intent(words)
        if intent == Intent.SET:
            return NaturalLanguageSetParser().parse(command)
        if words[0] == "add" and words[-1] == "selection":
            return NaturalLanguageSelectParser().parse(command)
        elif words[0] == "select":
            return NaturalLanguageSelectParser().parse(command)
        return "parsed: " + command


def determine_intent(words: list[str]) -> Intent:
    try:
        return Intent(words[0])
    except ValueError:
        # Heuristics
        if "background color" in words:
            return Intent.SET
        if words[0] == "add" and words[-1] == "selection":
            return "add_selection"
        elif words[0] == "select":
            return "select"

        return "unknown"


# 'make the background color red'
# 'reset the background color'
# "show the center of rotation"
# "place a marker "

# "change directory"
# Target specifier:
# Hash (#) for Model
# Slash (/) for Chains
# Colon (:) for Residues
# At (@) for Atoms

# number one, chains b through d, and f
# #1/B-D,F


class NaturalLanguageSetParser:
    """Parse natural language into a set command"""

    translations = {
        "reset": "~set",
        "unset": "~set",
        "the": "",
        "to": "",
        "make": "set",
    }
    common_phrases = {
        "background color": "bgColor",
        "background": "bgColor",
        "subdivision value": "subdivision",
    }

    def parse(self, phrase: str) -> str:
        return self.parse_phrase(phrase)

    def parse_phrase(self, phrase: str) -> str:
        words = phrase.replace(",", "")
        for original, new in self.common_phrases.items():
            words = words.replace(original, new)
        individual_words = words.split()
        for word in individual_words:
            if word in self.translations:
                words = words.replace(word, self.translations[word])
            elif word in numbers:
                words = words.replace(word, numbers[word])
        # TODO: If words[0] == "set" and bgColor in words:
        # color = NLColorSpecParser().parse(words[2])
        return " ".join(words.split())


numbers = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

elements = {
    "hydrogen": "h",
    "helium": "he",
    "lithium": "li",
    "beryllium": "be",
    "boron": "b",
    "carbon": "c",
    "nitrogen": "n",
    "oxygen": "o",
    "fluorine": "f",
    "neon": "ne",
    "sodium": "na",
    "magnesium": "mg",
    "aluminum": "al",
    "silicon": "si",
    "phosphorus": "p",
    "sulfur": "s",
    "chlorine": "cl",
    "argon": "ar",
    "potassium": "k",
    "calcium": "ca",
    "scandium": "sc",
    "titanium": "ti",
    "vanadium": "v",
    "chromium": "cr",
    "manganese": "mn",
    "iron": "fe",
    "cobalt": "co",
    "nickel": "ni",
    "copper": "cu",
    "zinc": "zn",
    "gallium": "ga",
    "germanium": "ge",
    "arsenic": "as",
    "selenium": "se",
    "bromine": "br",
    "krypton": "kr",
    "rubidium": "rb",
    "strontium": "sr",
    "yttrium": "y",
    "zirconium": "zr",
    "niobium": "nb",
    "molybdenum": "mo",
    "technetium": "tc",
    "ruthenium": "ru",
    "rhodium": "rh",
    "palladium": "pd",
    "silver": "ag",
    "cadmium": "cd",
    "indium": "in",
    "tin": "sn",
    "antimony": "sb",
    "tellurium": "te",
    "iodine": "i",
    "xenon": "xe",
    "caesium": "cs",
    "cesium": "cs",
    "barium": "ba",
    "lanthanum": "la",
    "cerium": "ce",
    "praseodymium": "pr",
    "neodymium": "nd",
    "prometheum": "pm",
    "samarium": "sm",
    "europium": "eu",
    "gadolinium": "gd",
    "terbium": "tb",
    "dysprosium": "dy",
    "holmium": "ho",
    "erbium": "er",
    "thulium": "tm",
    "ytterbium": "yt",
    "lutetium": "lt",
    "hafnium": "hf",
    "tantalum": "ta",
    "tungsten": "w",
    "rhenium": "re",
    "osmium": "os",
    "iridium": "ir",
    "platinum": "pt",
    "gold": "au",
    "mercury": "hg",
    "thallium": "tl",
    "lead": "pb",
    "bismuth": "bi",
    "polonium": "po",
    "astatine": "at",
    "radon": "rn",
    "francium": "fr",
    "radium": "ra",
    "actinium": "ac",
    "thorium": "th",
    "protactinium": "pa",
    "uranium": "u",
    "neptunium": "np",
    "plutonium": "pu",
    "americium": "am",
    "curium": "cm",
    "berkelium": "bk",
    "californium": "cf",
    "einsteineum": "es",
    "fermium": "fm",
    "mendelevium": "md",
    "nobelium": "no",
    "lawrencium": "lr",
    "rutherfordium": "rf",
    "dubnium": "db",
    "seaborgium": "sg",
    "bohrium": "bh",
    "hassium": "hs",
    "meitnerium": "mt",
    "darmstadtium": "dt",
    "roentgenium": "rg",
    "copernicium": "cn",
    "nihonium": "nh",
    "flerovium": "fl",
    "moscovium": "mc",
    "livermorium": "lv",
    "tennessine": "ts",
    "onganesson": "og",
}


class NaturalLanguageSelectParser:
    common_phrases = {
        "to the selection": "",
        "from the selection": "",
        "select": "",
        "selection": "",
    }

    def parse(self, phrase: str) -> str:
        if phrase == "clear selection":
            return "~select"
        if phrase == "select up":
            return "select up"
        if phrase == "select down":
            return "select down"
        if phrase == "select all":
            return "select all"
        words = phrase.replace(",", "")
        for original, new in self.common_phrases.items():
            words = words.replace(original, new)
        words = words.split()
        if words[0] == "add":
            add = True
        if words[0] == "remove" or words[0] == "unselect":
            subtract = True
        atomspec = NaturalLanguageAtomSpecParser().parse(words[1:])
        if subtract:
            return "select subtract" + atomspec
        if add:
            return "select add" + atomspec
        return "select " + atomspec


class NaturalLanguageAtomSpecParser:
    """Parse natural language into an AtomSpecifier"""

    translations = {
        "number": "#",
        "model": "#",
        "chains": "/",
        "atoms": "@",
        "atoms of": "@",
        "residue": ":",
        "through": "-",
        "submodel": ".",
        "point": ".",
        "to": "-",
    }

    common_phrases = {
        "model number": "model",
        "all the atoms of": "",
        "atoms of": "atoms",
    }

    def parse(self, spec: str) -> str:
        phrases = spec.split("and")
        parsed = []
        for phrase in phrases:
            parsed.append(self.parse_phrase(phrase))
        return " ".join(parsed)

    def parse_phrase(self, spec: str) -> str:
        words = spec.replace(",", "")
        for original, new in self.common_phrases.items():
            words = words.replace(original, new)
        individual_words = words.split()
        for word in individual_words:
            if word in self.translations:
                words = words.replace(word, self.translations[word])
            elif word in numbers:
                words = words.replace(word, numbers[word])
            elif word in elements:
                words = words.replace(word, elements[word])
        return "".join(words).replace(" ", "")


# Kinds of words:
# 1. Words that directly correspond to a simple ChimeraX command that doesn't
# even need an argument
# show
# hide
# exit
# quit
# help
# open
# close
# style
# undo
# redo
#

# Ambiguous phrases:
# show #1 as surface
# display #1 as surface

# 2. Words that have no meaning but may be part of a command
# change the color of the background to red
# color ... background ... red are all that matter from this sentence
# display number one as a surface
# display number one .. surface
# --cmd-- -specifier- -argument-

# Possible flow:
# 1. User speaks command
# 2. Command is recorded
# 3. Command is parsed
# 4. Command is passed to the ChimeraX command parser
# 5. ChimeraX executes command or reports failure back
