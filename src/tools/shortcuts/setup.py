# vim: set expandtab ts=4 sw=4:

# import distutils.core
# distutils.core.DEBUG = True
from distlib import metadata
print(metadata.METADATA_FILENAME)
metadata.METADATA_FILENAME = "metadata.json"
from setuptools import setup

lib_dir = "/Users/goddard/ucsf/chimera2/build/lib"
pkg_dir = "/Users/goddard/ucsf/chimera2/build/lib/python3.4/site-packages/chimera/shortcuts"

setup(
    name="chimera.shortcuts",
    version="1.0",
    description="Keboard shortcuts",
    author="UCSF RBVI",
    author_email="chimera2@cgl.ucsf.edu",
    url="https://www.rbvi.ucsf.edu/chimera2/",
    package_dir={
        "chimera.shortcuts": pkg_dir,
    },
    packages=[
        "chimera.shortcuts",
        # Add subpackage names here
    ],
    package_data={
        "chimera.shortcuts": ['icons/*.png'],
    },
    data_files=[
        ("", ["timestamp"]),
    ],
    install_requires=[
        "chimera.core",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: MacOS X :: Aqua",
        "Environment :: Win32 (MS Windows)",
        "Environment :: X11 Applications",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows 7",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Vio-Informatics",
        # Each tool should have a single line with up to seven fields:
        #
        # 'Chimera-Tool' :: tool_name :: module_name
        # :: display_name :: commands :: categories
        # :: synopsis
        #
        # 'tool_name', 'module_name' and 'display_name' are strings.
        # 'commands' and 'categories' are comma separated strings.
        # Note that chimera.shortcuts does not need to match the install
        # package name; in particular, it may be a subpackage.
        # Molecule Display is distinct from Keboard shortcuts because
        # a single package may provide multiple tools.
        # 'synopsis' is a short description of the tool.  It is here
        # because it needs to be part of the metadata available for
        # uninstalled tools, so that users can get more than just a
        # name for deciding whether they want the tool or not.
        #
        # Example:
        # "Chimera-Tool :: cmd_line :: chimera.cmd_line "
        # ":: Command Line Interface :: command_line :: General"
        "Chimera-Tool :: molecule_display_shortcuts :: chimera.shortcuts :: Molecule Display :: ks :: General :: Two-letter shortcuts",
        "Chimera-Tool :: graphics_shortcuts :: chimera.shortcuts :: Graphics :: :: General :: Two-letter shortcuts",
    ],
)
