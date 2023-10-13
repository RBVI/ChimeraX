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

"""
Bundle Builder (PEP 517)

This version of bundle builder works a little differently than the previous
generation. Whereas the first bundle builder needed to gather all of the
metadata for a project, newer versions of setuptools will override values
passed to the setup() function with values that are read from pyproject.toml.
This is why we do not need to manually gather authors, URLs, versions anymore
from configuration files, and require that classifiers go under [tool.chimerax]
instead of [project].
"""

# Force import in a particular order since both Cython and
# setuptools patch distutils, and we want Cython to win
import setuptools
import setuptools._distutils as distutils # noqa we know distutils is protected

import collections
import fnmatch
import glob
import importlib
import itertools
import os
import platform
import distutils.ccompiler
import distutils.sysconfig
import distutils.log
import distutils.dir_util
import re
import shutil
import sys
import sysconfig
import tomli
import traceback
import unicodedata
import warnings

from Cython.Build import cythonize

from packaging.version import Version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet, InvalidSpecifier

from setuptools import Extension, find_packages
# From https://stackoverflow.com/questions/35112511/pip-setup-py-bdist-wheel-no-longer-builds-forced-non-pure-wheels
from setuptools.dist import Distribution
from setuptools.extension import Library # noqa

# Even setuptools.build_meta runs setup() behind the scenes...
from setuptools.build_meta import suppress_known_deprecation # noqa import not in __all__

from pkg_resources import get_distribution, DistributionNotFound

# TODO: Verify
# Always import this because it changes the behavior of setuptools
from numpy import get_include as get_numpy_include_dirs

# TODO Fact check
# The compile process is initiated by setuptools and handled
# by numpy.distutils, which eventually calls subprocess.
# On Windows, subprocess invokes CreateProcess.  If a shell
# is used, subprocess sets the STARTF_USESHOWWINDOW flag
# to CreateProcess, assuming that "cmd" is going to create
# a window; otherwise, it does not set the flag and a window
# gets created for each compile and link process.  The code
# below is used to make STARTF_USESHOWWINDOW be set by
# default (written after examining subprocess.py).  The
# default STARTUPINFO class is replaced before calling
# setuptools.setup() and reset after it returns.

import subprocess

try:
    from subprocess import STARTUPINFO
except ImportError:
    MySTARTUPINFO = None
else:
    import _winapi

    class MySTARTUPINFO(STARTUPINFO):
        _original = STARTUPINFO

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.dwFlags |= _winapi.STARTF_USESHOWWINDOW

from .metadata_templates import (
    metadata_preamble, pure_wheel_platforms
)

# Python version was 3.7 in ChimeraX 1.0
CHIMERAX1_0_PYTHON_VERSION = "3.7"

_platforms = {
    "linux": ["linux"]
    , "darwin": ["mac", "macos", "darwin"]
    , "win32": ["win", "windows", "win32"]
}

class MissingInfoError(Exception):
    pass


class _SpecifierWarning(UserWarning):
    pass


def read_toml(file):
    with open(file, 'r') as f:
        return tomli.loads(f.read())


class Bundle:

    def __init__(self, logger, bundle_info):
        self.logger = logger
        self.bundle_info = bundle_info
        project_data = bundle_info['project']
        chimerax_data = bundle_info['chimerax']

        self.pure_python = not (
            bool(chimerax_data.get('extension', {}))
            or bool(chimerax_data.get('library', {}))
            or bool(chimerax_data.get('executable', {}))
        )

        if self.pure_python:
            if not chimerax_data.get('pure', True):
                # The user has no specified extensions, libraries, or executables
                # but they have still specified that their bundle is impure
                self.pure_python = False

        self.limited_api = chimerax_data.get('limited-api', False)

        if 'requires-python' in project_data['dynamic']:
            if self.pure_python:
                # Python-only bundles default to the ChimeraX 1.0
                # version of Python.
                self.python_requirement = SpecifierSet(f'>={CHIMERAX1_0_PYTHON_VERSION}')
            else:
                # Binary files are tied to the current version of Python
                self.python_requirement = SpecifierSet(
                    f'=={".".join(str(num) for num in sys.version_info[:2])}.*'
                )
        else:
            self.python_requirement = project_data.get('requires-python', None)
            if self.python_requirement:
                try:
                    self.python_requirement = SpecifierSet(self.python_requirement)
                    warnings.warn(
                        "If user-set requires-python does not include at least 3.7, "
                        "there may be issues with this bundle in ChimeraX", _SpecifierWarning
                    )
                except InvalidSpecifier as err:
                    raise ValueError("Invalid requires-python specifier: %s" % err)
            else:
                # If they didn't specify it in dynamic and forgot to set it in the bundle
                # metadata, set it to the minimum version ourselves and warn them.
                warnings.warn(
                    "Unset requires-python set to ChimeraX minimum version ('>=3.7'); "
                    "this may become an error in later versions of setuptools", _SpecifierWarning
                )
                self.python_requirement = SpecifierSet(f'>={CHIMERAX1_0_PYTHON_VERSION}')

        self.name = project_data['name']
        if '_' in self.name:
            self.name = self.name.replace('_', '-')
            self.logger.warning(
                "Bundle renamed to %r after replacing "
                "underscores with hyphens." % self.name
            )

        self.bundle_base_name = self.name.replace("ChimeraX-", "")
        if 'module-name-override' in chimerax_data:
            self.module_name = f'chimerax.{chimerax_data.get("module-name-override")}'
        else:
            self.module_name = self.name.replace('-', '.').lower()
        self.dist_info_name = self.name.replace('-', '_')

        # If version is dynamic then we'll attempt to build the wheel and use the version number
        # that setuptools found to check the built wheel
        self.version = None
        if 'version' not in project_data['dynamic']:
            self.version = project_data['version']
            # Check that the version is valid and let the error propagate up if one is thrown
            self.version = str(Version(project_data['version']))

        self.path = os.getcwd()
        build_dir = os.path.join(self.path, 'build')
        # Ensure a clean environment between builds, even when not using build isolation
        shutil.rmtree(build_dir, ignore_errors = True)

        self.egg_info = os.path.join(self.path, self.dist_info_name + ".egg-info")

        dependencies = project_data.get('dependencies', [])

        self.dependencies = []
        for req in dependencies:
            try:
                self.dependencies.append(str(Requirement(req)))
            except ValueError:
                raise ValueError("Bad version specifier (see PEP 440): %r" % req)

        self.requires_python = project_data.get('requires-python', ">=3.7")

        self.min_sess_ver = chimerax_data['min-session-version']
        self.max_sess_ver = chimerax_data['max-session-version']
        # "supercedes" is deprecated in ChimeraX 1.2
        if 'supersedes' in chimerax_data:
            self.supersedes = chimerax_data.get("supersedes")
        elif 'supercedes' in chimerax_data:
            self.supersedes = chimerax_data.get("supercedes")
        else:
            self.supersedes = []
        self.custom_init = str(chimerax_data.get("custom-init", ""))
        self.categories = chimerax_data.get("categories", "")
        self.classifiers = chimerax_data.get("classifiers", [])

        self.tools = []
        self.commands = []
        self.selectors = []
        self.providers = []
        self.data_formats = []
        self.format_readers = []
        self.format_savers = []
        self.file_fetchers = []
        self.presets = []
        self.managers = []
        self.initializations = []
        self.c_modules = []
        self.c_libraries = []
        self.c_executables = []

        if 'tool' in chimerax_data:
            for tool_name, attrs in chimerax_data['tool'].items():
                self.tools.append(Tool(tool_name, attrs))
        if 'command' in chimerax_data:
            for command_name, attrs in chimerax_data['command'].items():
                self.commands.append(Command(command_name, attrs))
        if 'selector' in chimerax_data:
            for selector_name, attrs in chimerax_data['selector'].items():
                self.selectors.append(Selector(selector_name, attrs))
        if 'manager' in chimerax_data:
            for manager_name, attrs in chimerax_data['manager'].items():
                self.managers.append(Manager(manager_name, attrs))
        if 'provider' in chimerax_data:
            for provider_name, attrs in chimerax_data['provider'].items():
                try:
                    manager = attrs.pop('manager')
                except KeyError:
                    raise ValueError("No manager specified for provider %s" % provider_name)
                self.providers.append(Provider(manager, provider_name, attrs))
        if 'data-format' in chimerax_data:
            for format_name, attrs in chimerax_data['data-format'].items():
                if 'open' in attrs:
                    opener = attrs.pop('open')
                    if type(opener) is list:
                        for item in opener:
                            if 'type' in item and item['type'] == 'fetch':
                                self.format_readers.append(FormatFetcher(format_name, item))
                            else:
                                self.format_readers.append(FormatReader(format_name, item))
                    elif type(opener) is bool and opener:
                        self.format_readers.append(FormatReader(format_name, None))
                    else:
                        if 'type' in attrs and attrs['type'] == 'fetch':
                            self.format_readers.append(FormatFetcher(format_name, opener))
                        else:
                            self.format_readers.append(FormatReader(format_name, opener))
                if 'save' in attrs:
                    saver = attrs.pop('save')
                    if type(saver) is list:
                        for item in saver:
                            self.format_savers.append(FormatSaver(format_name, item))
                    elif type(saver) is bool and saver:
                        self.format_savers.append(FormatSaver(format_name, None))
                    else:
                        self.format_savers.append(FormatSaver(format_name, saver))
                self.data_formats.append(DataFormat(format_name, attrs))
        if 'preset' in chimerax_data:
            for preset_name, attrs in chimerax_data['preset'].items():
                self.presets.append(Preset(preset_name, attrs))
        if 'initialization' in chimerax_data:
            init = chimerax_data['initialization']
            if type(init) is list:
                for entry in chimerax_data['initialization']:
                    self.initializations.append(Initialization(entry['type'], entry['bundles']))
            else:
                self.initializations.append(Initialization(init['type'], init['bundles']))
        if 'extension' in chimerax_data:
            for name, attrs in chimerax_data['extension'].items():
                self.c_modules.append(_CModule(name, attrs))
        if 'library' in chimerax_data:
            for name, attrs in chimerax_data['library'].items():
                self.c_libraries.append(_CLibrary(name, attrs))
        if 'executable' in chimerax_data:
            for name, attrs in chimerax_data['executable'].items():
                self.c_executables.append(_CExecutable(name, attrs))

        # TODO: Finalize
        # if 'documentation' in chimerax_data:
        #    for paths in chimerax_data['documentation'].values():
        #        if type(paths) is list:
        #            for path in paths:
        #                self.doc_dirs.append(DocDir(path))
        #        else:
        #            self.doc_dirs.append(DocDir(paths))

        self.datafiles = collections.defaultdict(set)
        self.extra_files = collections.defaultdict(set)
        self.packages = {(self.module_name, "src")}

        raw_package_data = chimerax_data.get('package-data', {})
        platform_package_data = {}
        if 'platform' in raw_package_data:
            for platform in _platforms[sys.platform]:
                try:
                    platform_package_data = raw_package_data["platform"].pop(platform)
                except KeyError:
                    pass

        raw_package_data.pop('platform', None)
        filtered_package_data = raw_package_data
        for key in platform_package_data:
            if key in filtered_package_data:
                filtered_package_data[key].append(platform_package_data[key])
            else:
                filtered_package_data[key] = platform_package_data[key]

        for folder, files in filtered_package_data.items():
            pkg_name = ".".join([self.module_name, folder.replace('src/', '').replace('/', '.')]).rstrip('.')
            if sys.platform == "win32":
                folder = folder.rstrip('/')
            self.packages.add((pkg_name, folder))
            if pkg_name not in self.datafiles:
                self.datafiles[pkg_name] = set(files)
            else:
                curr_files = self.datafiles[pkg_name]
                self.datafiles[pkg_name] = curr_files | set(files)

        raw_extra_files = chimerax_data.get('extra-files', {})
        platform_extra_files = {}
        if 'platform' in raw_extra_files:
            for platform in _platforms[sys.platform]:
                try:
                    platform_extra_files = raw_extra_files["platform"].pop(platform)
                except KeyError:
                    pass

        raw_extra_files.pop('platform', None)
        filtered_extra_files = raw_extra_files
        for key in platform_extra_files:
            if key in filtered_extra_files:
                filtered_extra_files[key].append(platform_extra_files[key])
            else:
                filtered_extra_files[key] = platform_extra_files[key]

        for folder, files in filtered_extra_files.items():
            pkg_name = ".".join([self.module_name, folder.replace('src/', '').replace('/', '.')]).rstrip('.')
            if sys.platform == "win32":
                folder = folder.rstrip('/')
            self.packages.add((pkg_name, folder))
            for file in files:
                # Unlike data files, which takes filenames and wildcards with extensions,
                # extra files needs to take directories or filenames or wildcard filenames
                # or directory/*-type wildcards.
                # If we have a basename, take the basename. If not, take the folder name.
                # core_cpp/logger/*.h --> *.h
                # core_cpp/logger/ --> "" <-- requires os.path.dirname
                # core_cpp/logger --> logger
                maybe_file = os.path.basename(file)
                if not maybe_file:
                    maybe_file = os.path.dirname(file)
                self.datafiles[pkg_name].add(maybe_file)
            # But we need to leave it alone in extra files so we can copy it over
            # into the source tree!
            if pkg_name not in self.extra_files:
                self.extra_files[pkg_name] = set(files)
            else:
                curr_files = self.extra_files[pkg_name]
                self.extra_files[pkg_name] = curr_files | set(files)

        if self.c_libraries:
            pkg_name = ".".join([self.module_name, "lib"])
            self.packages.add((pkg_name, "src/lib"))
            self.datafiles[pkg_name] = {"*"}
        if self.c_executables:
            pkg_name = ".".join([self.module_name, "bin"])
            self.packages.add((pkg_name, "src/bin"))
            self.datafiles[pkg_name] = {"*"}

        self._make_setup_arguments()
        dist = Distribution(attrs=self.setup_arguments)
        bdist_wheel_cmd = dist.get_command_obj('bdist_wheel')
        bdist_wheel_cmd.ensure_finalized()

        distname = bdist_wheel_cmd.wheel_dist_name
        tag = '-'.join(bdist_wheel_cmd.get_tag())
        self._expected_wheel_name = f'{distname}-{tag}.whl'

    @classmethod
    def from_toml_file(cls, logger, toml_file):
        return cls(logger, read_toml(toml_file))

    @classmethod
    def from_path(cls, logger, bundle_path):
        toml_file = os.path.join(os.path.abspath(bundle_path), "pyproject.toml")
        return cls(logger, read_toml(toml_file))

    def make_wheel(self, debug=False):
        self.build_wheel()

    def make_install(self, session, debug=False, user=None, no_deps=None, editable=False):
        if editable:
            wheel = self.build_editable()
        else:
            wheel = self.build_wheel()
        from chimerax.core.commands import run, FileNameArg
        cmd = "toolshed install %s" % FileNameArg.unparse(os.path.join(self.path, 'dist', wheel))
        if user is not None:
            if user:
                cmd += " user true"
            else:
                cmd += " user false"
        if no_deps is not None:
            if no_deps:
                cmd += " noDeps true"
            else:
                cmd += " noDeps false"
        from chimerax.core import toolshed
        ts = toolshed.get_toolshed()
        bundle = ts.find_bundle(self.name, session.logger)
        # version is set in make_wheel()
        if bundle is not None and bundle.version == self.version:
            cmd += " reinstall true"
        run(session, cmd)

    def make_clean(self):
        shutil.rmtree(os.path.join(self.path, "build"), ignore_errors=True)
        shutil.rmtree(os.path.join(self.path, "dist"), ignore_errors=True)
        shutil.rmtree(os.path.join(self.path, "src", "__pycache__"), ignore_errors=True)
        shutil.rmtree(self.egg_info, ignore_errors=True)
        self._clean_extrafiles()
        self._remove_libraries()
        for root, dirnames, filenames in os.walk("src"):
            # Linux, Mac
            for filename in fnmatch.filter(filenames, "*.o"):
                os.remove(os.path.join(root, filename))
            # Windows
            for filename in fnmatch.filter(filenames, "*.obj"):
                os.remove(os.path.join(root, filename))

    def _make_setup_arguments(self):
        def add_argument(name, value):
            if value:
                self.setup_arguments[name] = value

        # Make sure C/C++ libraries (DLLs, shared objects or dynamic
        # libraries) and executables are on the install list
        for lib in self.c_libraries:
            for lib_path in lib.paths():
                self.datafiles[self.module_name + ".lib"].add(lib_path)
        for executable in self.c_executables:
            self.datafiles[self.module_name + ".bin"].add(executable.path())
        self.setup_arguments = {
            "name": self.name,
            "python_requires": str(self.python_requirement)
        }
        if self.version:
            add_argument("version", self.version)
        # Convert our data files sets back to lists, so setuptools will
        # accept them.
        old_datafiles = self.datafiles
        for key, val in old_datafiles.items():
            self.datafiles[key] = list(val)
        add_argument("package_data", self.datafiles)
        # We cannot call find_packages unless we are already
        # in the right directory, and that will not happen
        # until run_setup.  So we do the package stuff there.
        ext_mods = [
            em for em in [
                cm.ext_mod(self.logger, self.module_name, self.dependencies) for cm in self.c_modules
            ] if em is not None
        ]
        if not self.pure_python:
            if sys.platform == "darwin":
                env = "Environment :: MacOS X :: Aqua"
                op_sys = "Operating System :: MacOS :: MacOS X"
            elif sys.platform == "win32":
                env = "Environment :: Win32 (MS Windows)"
                op_sys = "Operating System :: Microsoft :: Windows :: Windows 10"
            else:
                env = "Environment :: X11 Applications"
                op_sys = "Operating System :: POSIX :: Linux"
            platform_classifiers = [env, op_sys]
            if not ext_mods:

                class BinaryDistribution(Distribution):
                    def has_ext_modules(foo): # noqa we don't care that this is static
                        return True

                self.setup_arguments["distclass"] = BinaryDistribution
        else:
            # pure Python
            platform_classifiers = pure_wheel_platforms.split('\n')
        self.setup_arguments["ext_modules"] = cythonize(ext_mods)
        self.classifiers.extend(metadata_preamble.split('\n'))
        self.classifiers.extend(platform_classifiers)
        all_metadata = itertools.chain.from_iterable([
            [self]
            , self.tools
            , self.commands
            , self.selectors
            , self.providers
            , self.data_formats
            , self.format_readers
            , self.format_savers
            , self.file_fetchers
            , self.managers
            , self.presets
            , self.initializations
        ])
        for entry in all_metadata:
            self.classifiers.extend([str(entry)])
        self.setup_arguments["classifiers"] = self.classifiers
        self.setup_arguments["package_dir"], self.setup_arguments["packages"] = self._make_package_arguments()

    def _copy_extrafiles(self):
        for pkg_name, items in self.extra_files.items():
            path = pkg_name.replace(self.module_name, "src").replace('.', '/')
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            for item in items:
                globbed_items = glob.glob(item)
                for entry in globbed_items:
                    destination = os.path.join(path, os.path.basename(entry))
                    if os.path.isdir(entry):
                        shutil.copytree(entry, destination)
                    else:
                        shutil.copyfile(entry, destination)

    # Since we aren't trusting setuptools to compile libraries properly we have
    # to remove them ourselves. Work around the prepare_metadata_for_build_editable
    # bug.
    def _remove_libraries(self):
        for lib in self.c_libraries:
            if lib.static:
                if sys.platform == "win32":
                    os.remove(os.path.join("src/lib/", "".join([lib.name, ".lib"])))
                else:
                    os.remove(os.path.join("src/lib/", "".join(["lib", lib.name, ".a"])))
            else:
                if sys.platform == 'darwin':
                    os.remove(os.path.join("src/lib/", "".join([lib.name, ".dylib"])))
                elif sys.platform == "linux":
                    os.remove(os.path.join("src/lib/", "".join([lib.name, ".so"])))

    def _clean_extrafiles(self):
        for pkg_name, items in self.extra_files.items():
            path = pkg_name.replace(self.module_name, "src").replace('.', '/')
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            for item in items:
                globbed_items = glob.glob(item)
                for entry in globbed_items:
                    destination = os.path.join(path, os.path.basename(entry))
                    if os.path.isdir(entry):
                        shutil.rmtree(destination, ignore_errors=True)
                    else:
                        os.remove(destination)

    def _make_package_arguments(self):
        def add_package(base_package, folder):
            if sys.platform == "win32":
                folder.rstrip('/')
            package_dir[base_package] = folder
            packages.append(base_package)
            # I have no idea why find_packages complains about trailing
            # slashes on Win32 and not Unix
            for sub_pkg in find_packages(folder):
                packages.append(base_package + '.' + sub_pkg)

        package_dir = {}
        packages = []
        for name, folder in self.packages:
            add_package(name, folder)
        return package_dir, packages

    def _run_setup(self, cmd):
        cwd = os.getcwd()
        save = sys.argv
        try:
            if MySTARTUPINFO:
                subprocess.STARTUPINFO = MySTARTUPINFO
            os.chdir(self.path)
            kw = self.setup_arguments.copy()
            kw["package_dir"], kw["packages"] = self._make_package_arguments()
            sys.argv = ["setup.py"] + cmd
            with suppress_known_deprecation():
                dist = setuptools.setup(**kw)
            return dist, True
        except Exception:
            traceback.print_exc()
            return None, False
        finally:
            sys.argv = save
            os.chdir(cwd)
            if MySTARTUPINFO:
                subprocess.STARTUPINFO = MySTARTUPINFO._original # noqa we don't care this is protected

    def _clear_distutils_dir_and_prep_srcdir(self, build_exts = False):
        # HACK: distutils uses a cache to track created directories
        # for a single setup() run.  We want to run setup() multiple
        # times which can remove/create the same directories.
        # So we need to flush the cache before each run.
        try:
            distutils.dir_util._path_created.clear()
        except AttributeError:
            pass
        # Copy additional files into package source tree
        self._copy_extrafiles()
        if build_exts:
            # Build C libraries and executables
            for lib in self.c_libraries:
                lib.compile(self.logger, self.dependencies)
            for executable in self.c_executables:
                executable.compile(self.logger, self.dependencies)

    def _check_output(self, type_="wheel"):
        if type_ == "wheel":
            output = glob.glob(os.path.join(self.path, 'dist', '*.whl'))
        elif type == "sdist":
            if sys.platform == "win32":
                output = glob.glob(os.path.join(self.path, 'dist', '*.zip'))
            else:
                output = glob.glob(os.path.join(self.path, 'dist', '*.tar.gz'))
        else:
            raise ValueError("Unknown output type requested: %s" % type)
        if not output:
            # TODO: Report the sdist name on failure, too
            raise RuntimeError(f"Building wheel failed: {self._expected_wheel_name}")
        else:
            name = output[0]
            if type_ == "wheel":
                print("Distribution is in %s" % os.path.join('./dist', name))
        return name

    def build_wheel(self):
        self._clear_distutils_dir_and_prep_srcdir(build_exts=True)
        setup_args = ["--no-user-cfg", "build"]
        setup_args.extend(["bdist_wheel"])
        dist, built = self._run_setup(setup_args)
        if not self.version:
            self.version = dist.get_version()
        wheel = self._check_output(type_ = "wheel")
        return wheel

    def build_sdist(self):
        self._clear_distutils_dir_and_prep_srcdir()
        setup_args = ["sdist"]
        dist, built = self._run_setup(setup_args)
        if not self.version:
            self.version = dist.get_version()
        sdist = self._check_output(type_ = "sdist")
        return sdist

    # TODO: Remove when pip can glean metadata from build_editable alone
    def build_wheel_for_build_editable(self):
        wheel_name = self.build_wheel()
        wheel_location = os.path.join(self.path, 'dist', wheel_name)
        # Clean everything that was placed in the source directory as a side effect
        self._clean_extrafiles()
        return wheel_location

    def build_editable(self, config_settings = None):
        self._remove_libraries()
        self._clear_distutils_dir_and_prep_srcdir(build_exts=True)
        setup_args = [
            "build_ext"
            , "--inplace"
            , "editable_wheel"
        ]
        if config_settings:
            if 'editable_mode' in config_settings:
                setup_args.extend(["--mode", config_settings["editable_mode"]])
        dist, built = self._run_setup(setup_args)
        if not self.version:
            self.version = dist.get_version()
        wheel = self._check_output(type_ = "wheel")
        return wheel

    def __str__(self):
        name = self.module_name
        sess_tuple = f"{self.min_sess_ver},{self.max_sess_ver}"
        categories = ", ".join(self.categories)
        supersedes = ", ".join(self.supersedes)
        custom_init = "true" if self.custom_init else ""
        return " :: ".join(["ChimeraX", "Bundle", categories, sess_tuple, name, supersedes, custom_init])


class ChimeraXClassifier:
    classifier_separator = " :: "

    def __init__(self, name, attrs):
        self.name = name
        self.attrs = attrs

    @property
    def categories(self):
        if 'category' in self.attrs:
            return self.attrs['category']
        if 'categories' in self.attrs:
            return self.attrs['categories']
        raise MissingInfoError(f"No synopsis found for {self.name}")

    @property
    def description(self):
        if 'synopsis' in self.attrs:
            return self.attrs['synopsis']
        if 'description' in self.attrs:
            return self.attrs['description']
        raise MissingInfoError(f"No synopsis found for {self.name}")

    def misc_attrs_to_list(self):
        attrs = []
        for k, v in self.attrs.items():
            formatted_field = k.replace('-', '_')
            if type(v) is list:
                if not v:
                    continue
                formatted_val = ','.join([quote_if_necessary(str(val)) for val in v])
            elif type(v) is bool:
                formatted_val = quote_if_necessary(str(v).lower())
            else:
                formatted_val = quote_if_necessary(str(v))
            attrs.append("%s:%s" % (formatted_field, formatted_val))
        return attrs


class Tool(ChimeraXClassifier):
    def __init__(self, tool_name: str, attrs: dict[str:str]):
        super().__init__(tool_name, attrs)

    def __str__(self):
        if type(self.categories) == str:
            return f'ChimeraX :: Tool :: {self.name} :: {self.categories} :: {self.description}'
        else:
            return f'ChimeraX :: Tool :: {self.name} :: {", ".join(self.categories)} :: {self.description}'


class Command(ChimeraXClassifier):
    def __init__(self, command_name: str, attrs: dict[str:str]):
        super().__init__(command_name, attrs)

    def __str__(self):
        if type(self.categories) == str:
            return f'ChimeraX :: Command :: {self.name} :: {self.categories} :: {self.description}'
        else:
            return f'ChimeraX :: Command :: {self.name} :: {", ".join(self.categories)} :: {self.description}'


class Selector(ChimeraXClassifier):
    def __init__(self, selector_name: str, attrs: dict[str:str]):
        super().__init__(selector_name, attrs)

    def __str__(self):
        return f'ChimeraX :: Selector :: {self.name} :: {self.description}'


class Manager(ChimeraXClassifier):
    default_attrs = {
        "gui-only": False
        , "autostart": False
    }

    def __init__(self, name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__(name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        return f'ChimeraX :: Manager :: {self.name} :: {self.classifier_separator.join(attrs)}'


class Provider(ChimeraXClassifier):
    def __init__(self, manager, name, attrs: dict[str:str]):
        self.manager = manager
        super().__init__(name, attrs)

    def __str__(self):
        attrs = self.misc_attrs_to_list()
        return f'ChimeraX :: Provider :: {self.name} :: {self.manager} :: {self.classifier_separator.join(attrs)}'


class DataFormat(Provider):
    default_attrs = {
        "category": "General"
        , "encoding": "utf-8"
        , "nicknames": None
        , "reference-url": None
        , "suffixes": None
        , "synopsis": None
        , "allow-directory": False
        , "insecure": False
        , "mime-types": []
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
        "batch": False
        , "check-path": True
        , "is-default": True
        , "pregrouped-structures": False
        , "type": "open"
        , "want-path": False
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
    default_attrs = {
        "compression-okay": True
        , "is-default": True
    }

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
        "batch": False
        , "check-path": False
        , "is-default": True
        , "pregrouped-structures": False
        , "type": "fetch"
        , "want-path": False
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
    default_attrs = {
        "category": "General"
    }

    def __init__(self, name, attrs):
        if not attrs:
            attrs = self.default_attrs
        else:
            for key, val in self.default_attrs.items():
                if key not in attrs:
                    attrs[key] = val
        super().__init__("presets", name, attrs)


class Initialization:
    def __init__(self, type_, bundles):
        self.type_ = type_
        self.bundles = bundles

    def __str__(self):
        separator = " :: "
        return f'ChimeraX :: InitAfter :: {self.type_} :: {separator.join(self.bundles)}'

# TODO: Standardize
# class DocDir:
#    def __init__(self, path):
#        self.path = path
#
#    def __str__(self):
#        return f'ChimeraX :: DocDir :: {self.path}'


class _CompiledCode:
    install_dir = "src"
    output_dir = "src"

    def __init__(self, name, attrs):
        self.name = name
        platform_specific_args = self.get_platform_specific_args(attrs)
        for key, val in platform_specific_args.items():
            if key == "limited-api":
                attrs["limited-api"] = val
            elif key == "target-lang":
                attrs["target-lang"] = val
            else:
                if key in attrs:
                    attrs[key].extend(val)
                else:
                    attrs[key] = val
        self.requires = attrs.get("requires", [])
        source_files = attrs.get("sources", [])
        self.frameworks = attrs.get("frameworks", [])
        self.libraries = attrs.get("libraries", [])
        self.compile_arguments = attrs.get("extra-compile-args", [])
        self.link_arguments = attrs.get("extra-link-args", [])
        self.include_dirs = attrs.get("include-dirs", [])
        self.include_modules = attrs.get("include-modules", [])
        self.include_libraries = attrs.get("library-modules", [])
        self.library_dirs = attrs.get("library-dirs", [])
        self.framework_dirs = attrs.get("framework-dirs", [])
        self.macros = []
        self.target_lang = attrs.get("target-lang", None)
        self.limited_api = attrs.get("limited-api", None)
        defines = attrs.get("define-macros", [])
        self.source_files = []
        for entry in source_files:
            self.source_files.extend(glob.glob(entry))
        for def_ in defines:
            edef = def_.split('=')
            if len(edef) > 2:
                raise TypeError(
                    "Too many arguments for macro "
                    "definition: %s" % edef
                )
            elif len(edef) == 1:
                edef.append(None)
            self.add_macro_define(*edef)
        for undef_ in attrs.get("undef-macros", []):
            self.add_macro_undef(undef_)
        if self.limited_api:
            v = self.limited_api
            if v < CHIMERAX1_0_PYTHON_VERSION:
                v = CHIMERAX1_0_PYTHON_VERSION
            hex_version = (v.major << 24) | (v.minor << 16) | (v.micro << 8)
            self.add_macro_define("Py_LIMITED_API", hex(hex_version))
            self.add_macro_define("CYTHON_LIMITED_API", hex(hex_version))

    def get_platform_specific_args(self, attrs):
        for platform in _platforms[sys.platform]:
            try:
                platform_args = attrs["platform"].pop(platform)
                return platform_args
            except KeyError:
                pass
        else:
            return {}

    def add_include_dir(self, d):
        self.include_dirs.append(d)

    def add_library_dir(self, d):
        self.library_dirs.append(d)

    def add_macro_define(self, m, val):
        # 2-tuple defines (set val to None to define without a value)
        self.macros.append((m, val))

    def add_macro_undef(self, m):
        # 1-tuple of macro name undefines
        self.macros.append((m,))

    def _compile_options(self, logger, dependencies):
        for req in self.requires:
            if not os.path.exists(req):
                raise ValueError("unused on this platform")
        # Add the internal Python include and lib directories
        root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
        inc_dirs = [os.path.join(root, "include")]
        lib_dirs = [os.path.join(root, "lib")]
        # Automatically add this bundle's library directory to lib dirs
        for module in self.include_modules:
            mod = importlib.import_module(module)
            inc_dirs.extend([mod.get_include()])
        for module in self.include_libraries:
            mod = importlib.import_module(module)
            lib_dirs.extend([mod.get_lib()])
        if sys.platform == "darwin":
            libraries = self.libraries
            # Unfortunately, clang on macOS (for now) exits
            # when receiving a -std=c++11 option when compiling
            # a C (not C++) source file, which is why this value
            # is named "cpp_flags" not "compile_flags"
            cpp_flags = ["-std=c++11", "-stdlib=libc++"]
            extra_link_args = ["-F" + d for d in self.framework_dirs]
            for fw in self.frameworks:
                extra_link_args.extend(["-framework", fw])
        elif sys.platform == "win32":
            libraries = []
            for lib in self.libraries:
                if lib.lower().endswith(".lib"):
                    # Strip the .lib since suffixes are handled automatically
                    libraries.append(lib[:-4])
                else:
                    libraries.append("lib" + lib)
            cpp_flags = []
            extra_link_args = []
        else:
            libraries = self.libraries
            cpp_flags = ["-std=c++11"]
            extra_link_args = []
        for req in self.requires:
            if not os.path.exists(req):
                return None
        inc_dirs.extend(self.include_dirs)
        lib_dirs.extend(self.library_dirs)
        extra_link_args.extend(self.link_arguments)
        return (inc_dirs, lib_dirs, self.macros,
                extra_link_args, libraries, cpp_flags)

    def compile_objects(self, logger, dependencies, static, debug=False):
        distutils.log.set_verbosity(1)
        try:
            (inc_dirs, lib_dirs, macros, extra_link_args,
             libraries, cpp_flags) = self._compile_options(logger, dependencies)
        except ValueError:
            print("Error when compiling %s" % self.name)
            return None
        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler)
        if inc_dirs:
            compiler.set_include_dirs(inc_dirs)
        if lib_dirs:
            compiler.set_library_dirs(lib_dirs)
        if libraries:
            compiler.set_libraries(libraries)
        compiler.add_include_dir(distutils.sysconfig.get_python_inc())
        if sys.platform == "win32":
            # Link library directory for Python on Windows
            compiler.add_library_dir(os.path.join(sys.exec_prefix, 'libs'))
        if not static:
            macros.append(("DYNAMIC_LIBRARY", 1))
        # We need to manually separate out C from C++ code here, since clang
        # crashes if -std=c++11 is given as a switch while compiling C code
        c_files = []
        cpp_files = []
        for f in self.source_files:
            lang = compiler.detect_language(f)
            if lang == 'c':
                c_files.append(f)
            elif lang == 'c++':
                cpp_files.append(f)
            else:
                raise RuntimeError("Unsupported language for %s" % f)
        if cpp_files:
            compiler.compile(
                cpp_files,
                extra_preargs=cpp_flags + self.compile_arguments,
                macros=macros, debug=debug
            )
            self.target_lang = "c++"
        if c_files:
            compiler.compile(
                c_files, extra_preargs=self.compile_arguments,
                macros=macros, debug=debug
            )
        objs = compiler.object_filenames(self.source_files)
        return compiler, objs, extra_link_args


class _CModule(_CompiledCode):

    def __init__(self, name, attrs):
        self.name = name
        super().__init__(name, attrs)
        self.major = attrs.get('major-ver', 1)
        self.minor = attrs.get('minor-ver', 0)
        # TODO: Better interface for shared object versioning
        # TODO: self.patch = attrs.get('patch-ver', 0)

    def ext_mod(self, logger, package, dependencies):
        try:
            (inc_dirs, lib_dirs, macros, extra_link_args,
             libraries, cpp_flags) = self._compile_options(logger, dependencies)
            macros.extend(
                [("MAJOR_VERSION", self.major),
                 ("MINOR_VERSION", self.minor)]
            )
        except ValueError:
            return None
        if sys.platform == "linux":
            extra_link_args.append("-Wl,-rpath,$ORIGIN/lib")
        elif sys.platform == "darwin":
            extra_link_args.append("-Wl,-rpath,@loader_path/lib")
        if self.source_files:
            return Extension(
                package + '.' + self.name,
                define_macros=macros,
                extra_compile_args=cpp_flags + self.compile_arguments,
                include_dirs=inc_dirs,
                library_dirs=lib_dirs,
                libraries=libraries,
                extra_link_args=extra_link_args,
                sources=self.source_files,
                py_limited_api=self.limited_api
            )
        else:
            return None


class _CLibrary(_CompiledCode):
    install_dir = "src/lib"
    output_dir = "src/lib"

    def __init__(self, name, attrs):
        if sys.platform == "win32":
            name = 'lib' + name
        else:
            name = name
        super().__init__(name, attrs)
        self.static = attrs.get("static", False)

    def compile(self, logger, dependencies, debug=False):
        compiler, objs, extra_link_args = self.compile_objects(
            logger,
            dependencies,
            self.static,
            debug
        )
        compiler.mkpath(self.output_dir)
        if self.static:
            if sys.platform == 'darwin':
                output_file = os.path.join("src/lib/", "".join(["lib", self.name, ".a"]))
                if os.path.exists(output_file):
                    os.remove(output_file)
            lib = compiler.library_filename(self.name, lib_type="static")
            compiler.create_static_lib(
                objs, self.name, output_dir=self.output_dir,
                target_lang=self.target_lang,
                debug=debug
            )
        else:
            if sys.platform == "darwin":
                # On Mac, we only need the .dylib and it MUST be compiled
                # with "-dynamiclib", not "-bundle".  Hence the giant hack:
                try:
                    n = compiler.linker_so.index("-bundle")
                except ValueError:
                    pass
                else:
                    compiler.linker_so[n] = "-dynamiclib"
                lib = compiler.library_filename(self.name, lib_type="dylib")
                extra_link_args.extend(
                    ["-Wl,-rpath,@loader_path",
                     "-Wl,-install_name,@rpath/%s" % lib]
                )
                compiler.link_shared_object(
                    objs, lib, output_dir=self.output_dir,
                    extra_preargs=extra_link_args,
                    target_lang=self.target_lang,
                    debug=debug
                )
            elif sys.platform == "win32":
                # On Windows, we need both .dll and .lib
                link_lib = compiler.library_filename(self.name, lib_type="static")
                extra_link_args.append("/LIBPATH:%s" % link_lib)
                lib = compiler.shared_object_filename(self.name)
                compiler.link_shared_object(
                    objs, lib, output_dir=self.output_dir,
                    extra_preargs=extra_link_args,
                    target_lang=self.target_lang,
                    debug=debug
                )
            else:
                # On Linux, we only need the .so
                lib = compiler.library_filename(self.name, lib_type="shared")
                extra_link_args.append("-Wl,-rpath,$ORIGIN")
                compiler.link_shared_object(
                    objs, lib, output_dir=self.output_dir,
                    extra_preargs=extra_link_args,
                    target_lang=self.target_lang,
                    debug=debug
                )
        return lib

    def paths(self):
        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler)
        # TODO: Always use the lib prefix
        if sys.platform == "win32":
            lib_name = "lib" + self.name
        else:
            lib_name = self.name
        if self.install_dir:
            lib_name = os.path.join(self.install_dir, lib_name)
        paths = []
        if self.static:
            paths.append(compiler.library_filename(lib_name, lib_type="static"))
        else:
            if sys.platform == "darwin":
                paths.append(
                    compiler.library_filename(
                        lib_name,
                        lib_type="dylib"
                    )
                )
            elif sys.platform == "win32":
                # On Windows we want both .lib and .dll
                paths.append(compiler.shared_object_filename(lib_name))
                paths.append(
                    compiler.library_filename(
                        lib_name,
                        lib_type="static"
                    )
                )
            else:
                paths.append(
                    compiler.library_filename(
                        lib_name,
                        lib_type="shared"
                    )
                )
        return paths


# TODO: Doesn't produce arm64/x86_64 executables on macOS
class _CExecutable(_CompiledCode):
    install_dir = "src/bin"
    output_dir = "src/bin"

    def __init__(self, name, attrs):
        if sys.platform == "win32":
            # Remove .exe suffix because it will be added
            if name.endswith(".exe"):
                name = name[:-4]
        super().__init__(name, attrs)

    def compile(self, logger, dependencies, debug=False):
        compiler, objs, extra_link_args = self.compile_objects(
            logger,
            dependencies,
            False,
            debug
        )
        compiler.mkpath(self.output_dir)
        if sys.platform == "darwin":
            extra_link_args.extend(["-Wl,-rpath,@loader_path"])
            if 'universal2' in sysconfig.get_platform():
                # Don't try to compile ARM binaries on versions of macOS that aren't
                # compatible with Xcode >= 12, the first version that had universal2
                # support, even though universal2 Python can run on macOS as old as
                # 10.9
                mac_ver = platform.mac_ver()[0].split('.')
                mac_ver_major = int(mac_ver[0])
                mac_ver_minor = int(mac_ver[1])
                if (mac_ver_major == 10 and mac_ver_minor > 14) or mac_ver_major > 10:
                    extra_link_args.extend(["-arch", "arm64", "-arch", "x86_64"])
        elif sys.platform == "win32":
            # Remove .exe suffix because it will be added
            if self.name.endswith(".exe"):
                self.name = self.name[:-4]
        else:
            extra_link_args.append("-Wl,-rpath,$ORIGIN")
        compiler.link_executable(
            objs, self.name, output_dir=self.output_dir,
            extra_preargs=extra_link_args,
            target_lang=self.target_lang,
            debug=debug
        )
        return compiler.executable_filename(self.name)

    def path(self):
        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler)
        exec_name = self.name
        if self.install_dir:
            exec_name = os.path.join(self.install_dir, exec_name)
        return compiler.executable_filename(exec_name)


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
        '\a': '\\a',
        '\b': '\\b',
        '\f': '\\f',
        '\n': '\\n',
        '\r': '\\r',
        '\t': '\\t',
        '\v': '\\v',
        '\\': '\\\\',
        ';': ';',
        ' ': ' ',
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
            result.append('\\x%02x' % i)
        elif ch.strip() == '':
            # non-space and non-newline spaces
            has_special = True
            result.append('\\N{%s}' % unicodedata.name(ch))
        else:
            result.append(ch)
    if has_single_quote or has_double_quote or has_special:
        if use_single_quote:
            return "'%s'" % ''.join(result)
        else:
            return '"%s"' % ''.join(result)
    return ''.join(result)
