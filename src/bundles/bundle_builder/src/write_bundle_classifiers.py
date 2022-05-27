# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os
import warnings
import tomli

class ChimeraXTomlException(Exception):
    pass

from typing import Union, Optional
def write_bundle_classifiers(
        file_: Union[str, bytes, os.PathLike] = "pyproject.toml"
        , dry_run: bool = False
) -> None:
    """Read a pyproject.toml file and convert its ChimeraX metadata to
    Python classifiers."""
    classifiers = [
        "Framework :: ChimeraX"
        , "Intended Audience :: Science/Research"
        , "Programming Language :: Python :: 3"
        , "Topic :: Scientific/Engineering :: Visualization"
        , "Topic :: Scientific/Engineering :: Chemistry"
        , "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]
    try:
        with open(file_, "r") as f:
            pyp_toml = tomli.loads(f.read())
        chimerax_data = pyp_toml['tool']['chimerax']
        classifiers.extend(chimerax_data.get("python-classifiers", []))
        bundle_categories = ",".join(chimerax_data['categories'])
        bundle_name = pyp_toml['project']['name']
        session_tuple = "%s,%s" % (chimerax_data['min-session-version'], chimerax_data['max-session-version'])
        supersedes = ",".join(chimerax_data.get("supersedes", ""))
        custom_init = str(chimerax_data.get("custom-ini", ""))
        bundle_classifier = "ChimeraX :: Bundle :: %s :: %s :: %s :: %s :: %s" % (
            bundle_categories, session_tuple, bundle_name, supersedes, custom_init
        )
        classifiers.append(bundle_classifier)
        # then append managers, then providers, then chimerax-classifiers
        classifiers.extend(chimerax_data.get("chimerax-classifiers", []))
        if dry_run:
            print("\n".join(classifiers))
        else:
            with open("classifiers.txt", "w") as f:
                f.write("\n".join(classifiers))
    except FileNotFoundError as e:
        warnings.warn("Could not read local pyproject.toml, please specify a path in setup.py")
        print(str(e))
        exit(1)
    except tomli.TOMLDecodeError as e:
        print(str(e))
        exit(1)
    except KeyError as e:
        raise ChimeraXTomlException("Required value %s not found in pyproject.toml" % str(e)) from None
