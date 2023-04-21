# vim: set expandtab ts=4 sw=4:

# Force import in a particular order since both Cython and
# setuptools patch distutils, and we want Cython to win
import setuptools
import setuptools._distutils as distutils
from Cython.Build import cythonize
from packaging.version import Version
from setuptools.build_meta import suppress_known_deprecation # noqa import not in __all__

# Always import this because it changes the behavior of setuptools
from numpy import get_include as get_numpy_include_dirs

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
#
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


# Python version was 3.7 in ChimeraX 1.0
CHIMERAX1_0_PYTHON_VERSION = Version("3.7")


class BundleBuilder:

    def __init__(self, logger, bundle_path=None):
        import os
        self.logger = logger
        if bundle_path is None:
            bundle_path = os.getcwd()
        self.path = bundle_path
        info_file = os.path.join(bundle_path, "bundle_info.xml")
        if not os.path.exists(info_file):
            raise IOError("Bundle info file %s is missing" % repr(info_file))
        try:
            self._read_bundle_info(info_file)
        except ValueError as err:
            raise ValueError("%s: %s" % (info_file, err))
        self._make_paths()

    @classmethod
    def from_path(cls, logger, bundle_path):
        return cls(logger, bundle_path)

    def make_wheel(self, debug=False):
        # HACK: distutils uses a cache to track created directories
        # for a single setup() run.  We want to run setup() multiple
        # times which can remove/create the same directories.
        # So we need to flush the cache before each run.
        self._make_setup_arguments()
        import distutils.dir_util
        try:
            distutils.dir_util._path_created.clear()
        except AttributeError:
            pass
        # Copy additional files into package source tree
        self._copy_extrafiles(self.extrafiles)
        # Build C libraries and executables
        import os
        for lib in self.c_libraries:
            lib.compile(self.logger, self.dependencies, debug=debug)
        for executable in self.c_executables:
            executable.compile(self.logger, self.dependencies, debug=debug)
        setup_args = ["--no-user-cfg", "build"]
        if debug:
            setup_args.append("--debug")
        setup_args.extend(["bdist_wheel"])
        if self._is_pure_python():
            setup_args.extend(["--python-tag", self.tag.interpreter])
        else:
            setup_args.extend(["--plat-name", self.tag.platform])
            if self.limited_api:
                setup_args.extend(["--py-limited-api", self.tag.interpreter])
        dist, built = self._run_setup(setup_args)
        if not built or not os.path.exists(self.wheel_path):
            wheel = os.path.basename(self.wheel_path)
            raise RuntimeError(f"Building wheel failed: {wheel}")
        else:
            print("Distribution is in %s" % self.wheel_path)
        return dist

    def make_editable_wheel(self, debug=False):
        # HACK: distutils uses a cache to track created directories
        # for a single setup() run.  We want to run setup() multiple
        # times which can remove/create the same directories.
        # So we need to flush the cache before each run.
        self._make_setup_arguments()
        import distutils.dir_util
        try:
            distutils.dir_util._path_created.clear()
        except AttributeError:
            pass
        # Copy additional files into package source tree
        self._copy_extrafiles(self.extrafiles)
        # Build C libraries and executables
        import os
        for lib in self.c_libraries:
            lib.compile(self.logger, self.dependencies, debug=debug)
        for executable in self.c_executables:
            executable.compile(self.logger, self.dependencies, debug=debug)
        setup_args = ["build_ext", "--inplace", "editable_wheel"]
        dist, built = self._run_setup(setup_args)
        import glob
        whl_path = glob.glob(os.path.join(self.path, 'dist', '*editable*.whl'))
        if not built or not whl_path:
            raise RuntimeError(f"Building editable wheel failed")
        return whl_path[0]

    def make_install(self, session, debug=False, user=None, no_deps=None, editable=False):
        if editable:
            whl_path = self.make_editable_wheel(debug=debug)
        else:
            _ = self.make_wheel(debug=debug)
            whl_path = self.wheel_path
        from chimerax.core.commands import run, FileNameArg
        cmd = "toolshed install %s" % FileNameArg.unparse(whl_path)
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
        if bundle is not None and bundle.version == self.version:
            cmd += " reinstall true"
        run(session, cmd)

    def make_clean(self):
        import os
        import fnmatch
        self._rmtree(os.path.join(self.path, "build"))
        self._rmtree(os.path.join(self.path, "dist"))
        self._rmtree(os.path.join(self.path, "src", "__pycache__"))
        self._rmtree(self.egg_info)
        for root, dirnames, filenames in os.walk("src"):
            # Static libraries
            for filename in fnmatch.filter(filenames, "*.a"):
                os.remove(os.path.join(root, filename))
            # Linux, Mac
            for filename in fnmatch.filter(filenames, "*.o"):
                os.remove(os.path.join(root, filename))
            # Windows
            for filename in fnmatch.filter(filenames, "*.obj"):
                os.remove(os.path.join(root, filename))

    def dump(self):
        for a in dir(self):
            if a.startswith('_'):
                continue
            v = getattr(self, a)
            if not callable(v):
                print("%s: %s" % (a, repr(v)))

    def _rmtree(self, path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    _mac_platforms = ["mac", "macos", "darwin"]
    _windows_platforms = ["windows", "win32"]
    _linux_platforms = ["linux"]

    def _read_bundle_info(self, bundle_info):
        # Setup platform variable so we can skip non-matching elements
        import sys
        if sys.platform == "darwin":
            # Tested with macOS 10.12
            self._platform_names = self._mac_platforms
        elif sys.platform == "win32":
            # Tested with Cygwin
            self._platform_names = self._windows_platforms
        else:
            # Presumably Linux
            # Tested with Ubuntu 16.04 LTS running in
            #   a singularity container on CentOS 7.3
            self._platform_names = self._linux_platforms
        # Read data from XML file
        self._used_elements = set()
        from lxml.etree import parse
        doc = parse(bundle_info)
        bi = doc.getroot()
        self._get_identifiers(bi)
        self._get_categories(bi)
        self._get_descriptions(bi)
        self._get_datafiles(bi)
        self._get_extrafiles(bi)
        self._get_managers(bi)
        self._get_providers(bi)
        self._get_dependencies(bi)
        self._get_initializations(bi)
        self._get_c_modules(bi)
        self._get_c_libraries(bi)
        self._get_c_executables(bi)
        self._get_packages(bi)
        self._get_classifiers(bi)
        self._check_unused_elements(bi)

    def _get_identifiers(self, bi):
        # TODO: more syntax checking
        from packaging.version import InvalidVersion
        self.name = bi.get("name", '')
        if '_' in self.name:
            self.name = self.name.replace('_', '-')
            self.logger.warning("Bundle renamed to %r after replacing "
                                "underscores with hyphens." % self.name)
        version = bi.get("version", '')
        try:
            # canoncialize version
            version = str(Version(version))
        except InvalidVersion as err:
            raise ValueError("%s line %d" % (err, bi.sourceline))
        self.version = version
        self.package = bi.get("package", '')
        self.min_session = bi.get("minSessionVersion", '')
        self.max_session = bi.get("maxSessionVersion", '')
        # "supercedes" is deprecated in ChimeraX 1.2
        self.supersedes = bi.get("supersedes", '') or bi.get("supercedes", '')
        self.custom_init = bi.get("customInit", '')
        self.pure_python = bi.get("purePython", '')
        self.limited_api = bi.get("limitedAPI", '')
        if self.limited_api:
            try:
                self.limited_api = Version(self.limited_api)
            except InvalidVersion as err:
                raise ValueError("Invalid limitedAPI: %s line %d" % (err, bi.sourceline))
        self.installed_data_dir = bi.get("installedDataDir", '')
        self.installed_include_dir = bi.get("installedIncludeDir", '')
        self.installed_library_dir = bi.get("installedLibraryDir", '')
        self.installed_executable_dir = bi.get("installedExecutableDir", '')

    def _get_categories(self, bi):
        self.categories = []
        deps = self._get_singleton(bi, "Categories")
        for e in self._get_elements(deps, "Category"):
            name = e.get("name")
            if name is None:
                raise ValueError("Missing Category's name at line %d" % e.sourceline)
            self.categories.append(name)

    def _get_descriptions(self, bi):
        self.author = self._get_singleton_text(bi, "Author")
        self.email = self._get_singleton_text(bi, "Email")
        self.url = self._get_singleton_text(bi, "URL")
        self.synopsis = self._get_singleton_text(bi, "Synopsis")
        self.description = self._get_singleton_text(bi, "Description")
        try:
            self.license = self._get_singleton_text(bi, "License")
        except ValueError:
            self.license = None

    def _get_datafiles(self, bi):
        self.datafiles = {}
        for dfs in self._get_elements(bi, "DataFiles"):
            pkg_name = dfs.get("package")
            files = []
            for e in self._get_elements(dfs, "DataFile"):
                filename = self._get_element_text(e)
                files.append(("file", filename))
            for e in self._get_elements(dfs, "DataDir"):
                dirname = self._get_element_text(e)
                files.append(("dir", dirname))
            if files:
                if not pkg_name:
                    pkg_name = self.package
                self.datafiles[pkg_name] = files

    def _get_extrafiles(self, bi):
        self.extrafiles = {}
        for dfs in self._get_elements(bi, "ExtraFiles"):
            pkg_name = dfs.get("package")
            files = []
            for e in self._get_elements(dfs, "ExtraFile"):
                source = e.get("source")
                if source is None:
                    raise ValueError("Missing ExtraFiles's source at line %d" % e.sourceline)
                filename = self._get_element_text(e)
                files.append(("file", source, filename))
            for e in self._get_elements(dfs, "ExtraFileGroup"):
                import os
                import glob
                source = e.get("source")
                if source is None:
                    raise ValueError("Missing ExtraFileGroup's source at line %d" % e.sourceline)
                source_base_dir = os.path.dirname(source)
                while '*' in source_base_dir or '?' in source_base_dir:
                    source_base_dir = os.path.split(source_base_dir)[0]
                dirname = self._get_element_text(e)
                sourcefiles = glob.glob(source, recursive=True)
                if not len(sourcefiles):
                    raise RuntimeError('ExtraFileGroup pattern {} does not match any files!'.format(source))
                for sf in sourcefiles:
                    files.append(("file", sf, os.path.join(dirname, os.path.relpath(sf, source_base_dir))))
            for e in self._get_elements(dfs, "ExtraDir"):
                source = e.get("source")
                if source is None:
                    raise ValueError("Missing ExtraDir's source at line %d" % e.sourceline)
                dirname = self._get_element_text(e)
                files.append(("dir", source, dirname))
            if files:
                if not pkg_name:
                    pkg_name = self.package
                self.extrafiles[pkg_name] = files
                datafiles = [(t[0], t[2]) for t in files]
                try:
                    self.datafiles[pkg_name].extend(datafiles)
                except KeyError:
                    self.datafiles[pkg_name] = datafiles

    def _get_managers(self, bi):
        self.managers = {}
        for mgrs in self._get_elements(bi, "Managers"):
            for e in self._get_elements(mgrs, "Manager"):
                keywords = {}
                if e.attrib:
                    keywords.update(e.attrib.items())
                name = keywords.pop("name", None)
                if name is None:
                    raise ValueError("Missing Manager's name at line %d" % e.sourceline)
                self.managers[name] = keywords

    def _get_providers(self, bi):
        self.providers = {}
        for prvs in self._get_elements(bi, "Providers"):
            default_manager = prvs.get('manager', '')
            for e in self._get_elements(prvs, "Provider"):
                keywords = {}
                if e.attrib:
                    keywords.update(e.attrib.items())
                manager = keywords.pop("manager", '')
                if len(manager) == 0:
                    manager = default_manager
                if len(manager) == 0:
                    raise ValueError("Missing manager from Provider at line %d" % e.sourceline)
                name = keywords.pop("name", None)
                if name is None:
                    raise ValueError("Missing Provider's name at line %d" % e.sourceline)
                self.providers[(manager, name)] = keywords

    def _get_dependencies(self, bi):
        self.dependencies = []
        try:
            deps = self._get_singleton(bi, "Dependencies")
        except ValueError:
            # Dependencies is optional, although
            # ChimeraXCore *should* always be present
            return
        from packaging.requirements import Requirement
        for e in self._get_elements(deps, "Dependency"):
            pkg = e.get("name", '')
            ver = e.get("version", '')
            req = "%s %s" % (pkg, ver)
            try:
                Requirement(req)
            except ValueError:
                raise ValueError("Bad version specifier (see PEP 440): %r" % req)
            self.dependencies.append(req)

    def _get_initializations(self, bi):
        self.initializations = {}
        for inits in self._get_elements(bi, "Initializations"):
            for e in self._get_elements(inits, "InitAfter"):
                i_type = e.get("type")
                if i_type is None:
                    raise ValueError("Missing InitAfter's type at line %d" % e.sourceline)
                bundle = e.get("bundle")
                if bundle is None:
                    raise ValueError("Missing InitAfter's bundle at line %d" % e.sourceline)
                try:
                    self.initializations[i_type].append(bundle)
                except KeyError:
                    self.initializations[i_type] = [bundle]

    def _get_c_modules(self, bi):
        self.c_modules = []
        for cm in self._get_elements(bi, "CModule"):
            mod_name = cm.get("name")
            if mod_name is None:
                raise ValueError("Missing CModule's name at line %d" % cm.sourceline)
            try:
                major = int(cm.get("major_version", ''))
            except ValueError:
                major = 0
            try:
                minor = int(cm.get("minor_version", ''))
            except ValueError:
                minor = 1
            uses_numpy = cm.get("usesNumpy") == "true"
            c = _CModule(mod_name, uses_numpy, major, minor,
                         self.installed_library_dir,
                         self.limited_api)
            self._add_c_options(c, cm)
            self.c_modules.append(c)

    def _get_c_libraries(self, bi):
        self.c_libraries = []
        for lib in self._get_elements(bi, "CLibrary"):
            c = _CLibrary(lib.get("name", ''),
                          lib.get("usesNumpy") == "true",
                          lib.get("static") == "true",
                          self.installed_library_dir,
                          self.limited_api)
            self._add_c_options(c, lib)
            self.c_libraries.append(c)

    def _get_c_executables(self, bi):
        self.c_executables = []
        for lib in self._get_elements(bi, "CExecutable"):
            c = _CExecutable(lib.get("name", ''),
                             self.installed_executable_dir,
                             self.limited_api)
            self._add_c_options(c, lib)
            self.c_executables.append(c)

    def _add_c_options(self, c, ce):
        for e in self._get_elements(ce, "Requires"):
            c.add_require(self._get_element_text(e))
        for e in self._get_elements(ce, "SourceFile"):
            c.add_source_file(self._get_element_text(e))
        for e in self._get_elements(ce, "IncludeDir"):
            c.add_include_dir(self._get_element_text(e))
        for e in self._get_elements(ce, "Library"):
            c.add_library(self._get_element_text(e))
        for e in self._get_elements(ce, "LibraryDir"):
            c.add_library_dir(self._get_element_text(e))
        for e in self._get_elements(ce, "CompileArgument"):
            c.add_compile_argument(self._get_element_text(e))
        for e in self._get_elements(ce, "LinkArgument"):
            c.add_link_argument(self._get_element_text(e))
        for e in self._get_elements(ce, "Framework"):
            c.add_framework(self._get_element_text(e))
        for e in self._get_elements(ce, "FrameworkDir"):
            c.add_framework_dir(self._get_element_text(e))
        for e in self._get_elements(ce, "Define"):
            edef = self._get_element_text(e).split('=')
            if len(edef) > 2:
                raise TypeError("Too many arguments for macro "
                                "definition: %s" % edef)
            elif len(edef) == 1:
                edef.append(None)
            c.add_macro_define(*edef)
        for e in self._get_elements(ce, "Undefine"):
            c.add_macro_undef(self._get_element_text(e))
        if self.limited_api:
            v = self.limited_api
            if v < CHIMERAX1_0_PYTHON_VERSION:
                v = CHIMERAX1_0_PYTHON_VERSION
            hex_version = (v.major << 24) | (v.minor << 16) | (v.micro << 8)
            c.add_macro_define("Py_LIMITED_API", hex(hex_version))
            c.add_macro_define("CYTHON_LIMITED_API", hex(hex_version))

    def _get_packages(self, bi):
        self.packages = []
        try:
            pkgs = self._get_singleton(bi, "AdditionalPackages")
        except ValueError:
            # AdditionalPackages is optional
            return
        for pkg in self._get_elements(pkgs, "Package"):
            pkg_name = pkg.get("name")
            if pkg_name is None:
                raise ValueError("Missing Package's name at line %d" % pkg.sourceline)
            pkg_folder = pkg.get("folder")
            if pkg_folder is None:
                raise ValueError("Missing Package's folder at line %d" % pkg.sourceline)
            self.packages.append((pkg_name, pkg_folder))

    def _get_classifiers(self, bi):
        from chimerax.core.commands import quote_if_necessary
        self.python_classifiers = [
            "Framework :: ChimeraX",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ]
        cls = self._get_singleton(bi, "Classifiers")
        for e in self._get_elements(cls, "PythonClassifier"):
            self.python_classifiers.append(self._get_element_text(e))
        self.chimerax_classifiers = [
            ("ChimeraX :: Bundle :: " + ','.join(self.categories) +
             " :: " + self.min_session + "," + self.max_session +
             " :: " + self.package + " :: " + self.supersedes +
             " :: " + self.custom_init)
        ]
        if self.installed_data_dir:
            self.chimerax_classifiers.append(
                "ChimeraX :: DataDir :: " + self.installed_data_dir)
        if self.installed_include_dir:
            self.chimerax_classifiers.append(
                "ChimeraX :: IncludeDir :: " + self.installed_include_dir)
        if self.installed_library_dir:
            self.chimerax_classifiers.append(
                "ChimeraX :: LibraryDir :: " + self.installed_library_dir)
        if self.installed_executable_dir:
            self.chimerax_classifiers.append(
                "ChimeraX :: ExecutableDir :: " + self.installed_executable_dir)
        for m, kw in self.managers.items():
            args = [m] + ["%s:%s" % (k, quote_if_necessary(v)) for k, v in kw.items()]
            self.chimerax_classifiers.append(
                "ChimeraX :: Manager :: " + " :: ".join(args))
        for (mgr, p), kw in self.providers.items():
            args = [p, mgr] + ["%s:%s" % (k, quote_if_necessary(v)) for k, v in kw.items()]
            self.chimerax_classifiers.append(
                "ChimeraX :: Provider :: " + " :: ".join(args))
        for t, bundles in self.initializations.items():
            args = [t] + bundles
            self.chimerax_classifiers.append(
                "ChimeraX :: InitAfter :: " + " :: ".join(args))
        for e in self._get_elements(cls, "ChimeraXClassifier"):
            classifier = self._get_element_text(e)
            if not classifier.startswith("ChimeraX"):
                classifier = "ChimeraX :: " + classifier
            self.chimerax_classifiers.append(classifier)

    def _is_pure_python(self):
        return (not self.c_modules
                and not self.c_libraries
                and not self.c_executables
                and self.pure_python != "false")

    def _copy_extrafiles(self, files):
        import shutil
        import os
        for pkg_name, entries in files.items():
            for kind, src, dst in entries:
                if kind == "file":
                    filepath = os.path.join("src", dst)
                    dirpath = os.path.dirname(filepath)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    shutil.copyfile(src, filepath)
                elif kind == "dir":
                    dstdir = os.path.join("src", dst.replace('/', os.sep))
                    if os.path.exists(dstdir):
                        shutil.rmtree(dstdir)
                    shutil.copytree(src, dstdir)

    def _expand_datafiles(self, files):
        import os
        datafiles = {}
        for pkg_name, entries in files.items():
            pkg_files = []
            for kind, name in entries:
                if kind == "file":
                    pkg_files.append(name)
                elif kind == "dir":
                    for pknm, folder in self.packages:
                        if pknm == pkg_name:
                            prefix = folder
                            break
                    else:
                        prefix = os.path.join(self.path, "src")
                    prefix_len = len(prefix) + 1
                    root = os.path.join(prefix, name)
                    for dirp, dns, fns in os.walk(root):
                        # Strip leading root component, including separator
                        dp = dirp[prefix_len:]
                        pkg_files.extend([os.path.join(dp, fn) for fn in fns])
            datafiles[pkg_name] = pkg_files
        return datafiles

    def _make_setup_arguments(self):
        def add_argument(name, value):
            if value:
                self.setup_arguments[name] = value
        # Make sure C/C++ libraries (DLLs, shared objects or dynamic
        # libraries) and executables are on the install list
        binary_files = []
        for lib in self.c_libraries:
            for lib_path in lib.paths():
                binary_files.append(("file", lib_path))
        for executable in self.c_executables:
            binary_files.append(("file", executable.path()))
        if binary_files:
            try:
                data_files = self.datafiles[self.package]
            except KeyError:
                self.datafiles[self.package] = binary_files
            else:
                data_files.extend(binary_files)
        if self.limited_api:
            # Limited API was first in Python 3.2
            rel = self.limited_api.release
            if rel < CHIMERAX1_0_PYTHON_VERSION.release:
                rel = CHIMERAX1_0_PYTHON_VERSION.release
        elif binary_files:
            # Binary files are tied to the current version of Python
            import sys
            rel = sys.version_info[:2]
        else:
            # Python-only bundles default to the ChimeraX 1.0
            # version of Python.
            rel = CHIMERAX1_0_PYTHON_VERSION.release
        self.setup_arguments = {"name": self.name,
                                "python_requires": f">={rel[0]}.{rel[1]}"}
        add_argument("version", self.version)
        add_argument("description", self.synopsis)
        add_argument("long_description", self.description)
        add_argument("author", self.author)
        add_argument("author_email", self.email)
        add_argument("url", self.url)
        add_argument("install_requires", self.dependencies)
        add_argument("license", self.license)
        add_argument("package_data", self._expand_datafiles(self.datafiles))
        # We cannot call find_packages unless we are already
        # in the right directory, and that will not happen
        # until run_setup.  So we do the package stuff there.
        ext_mods = [em for em in [cm.ext_mod(self.logger, self.package,
                                             self.dependencies)
                                  for cm in self.c_modules]
                    if em is not None]
        if not self._is_pure_python():
            import sys
            if sys.platform == "darwin":
                env = "Environment :: MacOS X :: Aqua",
                op_sys = "Operating System :: MacOS :: MacOS X"
            elif sys.platform == "win32":
                env = "Environment :: Win32 (MS Windows)"
                op_sys = "Operating System :: Microsoft :: Windows :: Windows 10"
            else:
                env = "Environment :: X11 Applications"
                op_sys = "Operating System :: POSIX :: Linux"
            platform_classifiers = [env, op_sys]
            if not ext_mods:
                # From https://stackoverflow.com/questions/35112511/pip-setup-py-bdist-wheel-no-longer-builds-forced-non-pure-wheels
                from setuptools.dist import Distribution

                class BinaryDistribution(Distribution):
                    def has_ext_modules(foo):
                        return True
                self.setup_arguments["distclass"] = BinaryDistribution
        else:
            # pure Python
            platform_classifiers = [
                "Environment :: MacOS X :: Aqua",
                "Environment :: Win32 (MS Windows)",
                "Environment :: X11 Applications",
                "Operating System :: MacOS :: MacOS X",
                "Operating System :: Microsoft :: Windows :: Windows 10",
                "Operating System :: POSIX :: Linux",
            ]
        self.python_classifiers.extend(platform_classifiers)
        self.setup_arguments["ext_modules"] = cythonize(ext_mods)
        self.setup_arguments["classifiers"] = (self.python_classifiers +
                                               self.chimerax_classifiers)

    def _make_package_arguments(self):
        from setuptools import find_packages

        def add_package(base_package, folder):
            package_dir[base_package] = folder
            packages.append(base_package)
            packages.extend([base_package + "." + sub_pkg
                             for sub_pkg in find_packages(folder)])
        package_dir = {}
        packages = []
        add_package(self.package, "src")
        for name, folder in self.packages:
            add_package(name, folder)
        return package_dir, packages

    def _make_paths(self):
        import os
        from .wheel_tag import tag
        self.tag = tag(self._is_pure_python(), limited=self.limited_api)
        self.bundle_base_name = self.name.replace("ChimeraX-", "")
        bundle_wheel_name = self.name.replace('-', '_')
        wheel = f"{bundle_wheel_name}-{self.version}-{self.tag}.whl"
        self.wheel_path = os.path.join(self.path, "dist", wheel)
        self.egg_info = os.path.join(self.path, bundle_wheel_name + ".egg-info")

    def _run_setup(self, cmd):
        import os
        import sys
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
            import traceback
            traceback.print_exc()
            return None, False
        finally:
            sys.argv = save
            os.chdir(cwd)
            if MySTARTUPINFO:
                subprocess.STARTUPINFO = MySTARTUPINFO._original

    #
    # Utility functions dealing with XML tree
    #
    def _get_elements(self, e, tag):
        tagged_elements = list(e.iter(tag))
        # Mark element as used even for non-applicable platform
        self._used_elements.update(tagged_elements)
        elements = []
        for se in tagged_elements:
            platforms = se.get("platform")
            if not platforms:
                elements.append(se)
            else:
                for platform in platforms.split(','):
                    if platform in self._platform_names:
                        elements.append(se)
                        break
        return elements

    def _get_element_text(self, e):
        return ''.join(e.itertext()).strip()

    def _get_singleton(self, bi, tag):
        elements = list(bi.iter(tag))
        self._used_elements.update(elements)
        if len(elements) > 1:
            raise ValueError("too many %s elements" % repr(tag))
        elif len(elements) == 0:
            raise ValueError("%s element is missing" % repr(tag))
        return elements[0]

    def _get_singleton_text(self, bi, tag):
        return self._get_element_text(self._get_singleton(bi, tag))

    def _check_unused_elements(self, bi):
        for node in bi:
            if node not in self._used_elements:
                if not isinstance(node.tag, str):
                    # skip comments
                    continue
                print("WARNING: unsupported element:", node.tag)


class _CompiledCode:

    def __init__(self, name, uses_numpy, install_dir, limited_api):
        self.name = name
        self.uses_numpy = uses_numpy
        self.requires = []
        self.source_files = []
        self.frameworks = []
        self.libraries = []
        self.compile_arguments = []
        self.link_arguments = []
        self.include_dirs = []
        self.library_dirs = []
        self.framework_dirs = []
        self.macros = []
        self.install_dir = install_dir
        self.target_lang = None
        self.limited_api = limited_api

    def add_require(self, req):
        self.requires.append(req)

    def add_source_file(self, f):
        self.source_files.append(f)

    def add_include_dir(self, d):
        self.include_dirs.append(d)

    def add_library(self, lib):
        self.libraries.append(lib)

    def add_library_dir(self, d):
        self.library_dirs.append(d)

    def add_compile_argument(self, a):
        self.compile_arguments.append(a)

    def add_link_argument(self, a):
        self.link_arguments.append(a)

    def add_framework(self, f):
        self.frameworks.append(f)

    def add_framework_dir(self, d):
        self.framework_dirs.append(d)

    def add_macro_define(self, m, val):
        # 2-tuple defines (set val to None to define without a value)
        self.macros.append((m, val))

    def add_macro_undef(self, m):
        # 1-tuple of macro name undefines
        self.macros.append((m,))

    def _compile_options(self, logger, dependencies):
        import sys
        import os
        for req in self.requires:
            if not os.path.exists(req):
                raise ValueError("unused on this platform")
        # platform-specific
        # Assume Python executable is in ROOT/bin/python
        # and make include directory be ROOT/include
        root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
        inc_dirs = [os.path.join(root, "include")]
        lib_dirs = [os.path.join(root, "lib")]
        if self.uses_numpy:
            inc_dirs.extend([get_numpy_include_dirs()])
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
        from pkg_resources import DistributionNotFound
        for dep in dependencies:
            try:
                d_inc, d_lib = self._get_bundle_dirs(logger, dep)
            except (RuntimeError, DistributionNotFound):
                pass
            else:
                if d_inc:
                    inc_dirs.append(d_inc)
                if d_lib:
                    lib_dirs.append(d_lib)
        extra_link_args.extend(self.link_arguments)
        return (inc_dirs, lib_dirs, self.macros,
                extra_link_args, libraries, cpp_flags)

    def _get_bundle_dirs(self, logger, dep):
        from chimerax.core import toolshed
        from pkg_resources import Requirement, get_distribution
        req = Requirement.parse(dep)
        if not get_distribution(req):
            raise RuntimeError("unsatisfied dependency: %s" % dep)
        ts = toolshed.get_toolshed()
        bundle = ts.find_bundle(req.project_name, logger)
        if not bundle:
            # The requirement is satisfied but is not recognized
            # as a bundle.  Probably just a regular Python package.
            return None, None
        inc = bundle.include_dir()
        lib = bundle.library_dir()
        if not inc and not lib:
            try:
                import importlib
                mod = importlib.import_module(bundle.package_name)
                inc = mod.get_include()
                lib = mod.get_lib()
            # This code does not distinguish between build dependencies and
            # regular dependencies, so must gracefully fail either way
            except (AttributeError, ModuleNotFoundError):
                return None, None
        return inc, lib

    def compile_objects(self, logger, dependencies, static, debug):
        import sys
        import os
        import distutils.ccompiler
        import distutils.sysconfig
        import distutils.log
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
            compiler.compile(cpp_files,
                             extra_preargs=cpp_flags + self.compile_arguments,
                             macros=macros, debug=debug)
            self.target_lang = "c++"
        if c_files:
            compiler.compile(c_files, extra_preargs=self.compile_arguments,
                             macros=macros, debug=debug)
        objs = compiler.object_filenames(self.source_files)
        return compiler, objs, extra_link_args

    def install_locations(self):
        if self.install_dir:
            import os
            output_dir = os.path.join("src", self.install_dir)
            install_dir = '/' + self.install_dir
        else:
            output_dir = "src"
            install_dir = ''
        return output_dir, install_dir


class _CModule(_CompiledCode):

    def __init__(self, name, uses_numpy, major, minor, libdir, limited_api):
        super().__init__(name, uses_numpy, libdir, limited_api)
        self.major = major
        self.minor = minor

    def ext_mod(self, logger, package, dependencies):
        from setuptools import Extension
        try:
            (inc_dirs, lib_dirs, macros, extra_link_args,
             libraries, cpp_flags) = self._compile_options(logger, dependencies)
            macros.extend([("MAJOR_VERSION", self.major),
                           ("MINOR_VERSION", self.minor)])
        except ValueError:
            return None
        import sys
        if self.install_dir:
            install_dir = '/' + self.install_dir
        else:
            install_dir = ''
        if sys.platform == "linux":
            extra_link_args.append("-Wl,-rpath,$ORIGIN%s" % install_dir)
        elif sys.platform == "darwin":
            extra_link_args.append("-Wl,-rpath,@loader_path%s" % install_dir)
        return Extension(package + '.' + self.name,
                         define_macros=macros,
                         extra_compile_args=cpp_flags + self.compile_arguments,
                         include_dirs=inc_dirs,
                         library_dirs=lib_dirs,
                         libraries=libraries,
                         extra_link_args=extra_link_args,
                         sources=self.source_files,
                         py_limited_api=not not self.limited_api)


class _CLibrary(_CompiledCode):

    def __init__(self, name, uses_numpy, static, libdir, limited_api):
        super().__init__(name, uses_numpy, libdir, limited_api)
        self.static = static

    def compile(self, logger, dependencies, debug=False):
        import sys
        compiler, objs, extra_link_args = self.compile_objects(logger,
                                                               dependencies,
                                                               self.static,
                                                               debug)
        output_dir, install_dir = self.install_locations()
        compiler.mkpath(output_dir)
        if sys.platform == "win32":
            # # Link library directory for Python on Windows
            # compiler.add_library_dir(os.path.join(sys.exec_prefix, 'libs'))
            lib_name = "lib" + self.name
        else:
            lib_name = self.name
        if self.static:
            lib = compiler.library_filename(lib_name, lib_type="static")
            compiler.create_static_lib(objs, lib_name, output_dir=output_dir,
                                       target_lang=self.target_lang,
                                       debug=debug)
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
                lib = compiler.library_filename(lib_name, lib_type="dylib")
                extra_link_args.extend(["-Wl,-rpath,@loader_path",
                                        "-Wl,-install_name,@rpath/%s" % lib])
                compiler.link_shared_object(objs, lib, output_dir=output_dir,
                                            extra_preargs=extra_link_args,
                                            target_lang=self.target_lang,
                                            debug=debug)
            elif sys.platform == "win32":
                # On Windows, we need both .dll and .lib
                link_lib = compiler.library_filename(lib_name, lib_type="static")
                extra_link_args.append("/LIBPATH:%s" % link_lib)
                lib = compiler.shared_object_filename(lib_name)
                compiler.link_shared_object(objs, lib, output_dir=output_dir,
                                            extra_preargs=extra_link_args,
                                            target_lang=self.target_lang,
                                            debug=debug)
            else:
                # On Linux, we only need the .so
                lib = compiler.library_filename(lib_name, lib_type="shared")
                extra_link_args.append("-Wl,-rpath,$ORIGIN%s" % install_dir)
                compiler.link_shared_object(objs, lib, output_dir=output_dir,
                                            extra_preargs=extra_link_args,
                                            target_lang=self.target_lang,
                                            debug=debug)
        return lib

    def paths(self):
        import sys
        import os
        import distutils.ccompiler
        import distutils.sysconfig
        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler)
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
                paths.append(compiler.library_filename(lib_name,
                                                       lib_type="dylib"))
            elif sys.platform == "win32":
                # On Windows we want both .lib and .dll
                paths.append(compiler.shared_object_filename(lib_name))
                paths.append(compiler.library_filename(lib_name,
                                                       lib_type="static"))
            else:
                paths.append(compiler.library_filename(lib_name,
                                                       lib_type="shared"))
        return paths


class _CExecutable(_CompiledCode):

    def __init__(self, name, execdir, limited_api):
        import sys
        if sys.platform == "win32":
            # Remove .exe suffix because it will be added
            if name.endswith(".exe"):
                name = name[:-4]
        super().__init__(name, False, execdir, limited_api)

    def compile(self, logger, dependencies, debug=False):
        import sys
        compiler, objs, extra_link_args = self.compile_objects(logger,
                                                               dependencies,
                                                               False,
                                                               debug)
        output_dir, install_dir = self.install_locations()
        compiler.mkpath(output_dir)
        if sys.platform == "darwin":
            extra_link_args.append("-Wl,-rpath,@loader_path")
        elif sys.platform == "win32":
            # Remove .exe suffix because it will be added
            if self.name.endswith(".exe"):
                self.name = self.name[:-4]
        else:
            extra_link_args.append("-Wl,-rpath,$ORIGIN%s" % install_dir)
        compiler.link_executable(objs, self.name, output_dir=output_dir,
                                 extra_preargs=extra_link_args,
                                 target_lang=self.target_lang,
                                 debug=debug)
        return compiler.executable_filename(self.name)

    def path(self):
        import os
        import distutils.ccompiler
        import distutils.sysconfig
        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler)
        exec_name = self.name
        if self.install_dir:
            exec_name = os.path.join(self.install_dir, exec_name)
        return compiler.executable_filename(exec_name)


if __name__ == "__main__" or __name__.startswith("ChimeraX_sandbox"):
    import sys
    bb = BundleBuilder()
    for cmd in sys.argv[1:]:
        if cmd == "wheel":
            bb.make_wheel()
        elif cmd == "install":
            try:
                bb.make_install(session)
            except NameError:
                print("%s only works from ChimeraX, not Python" % repr(cmd))
        elif cmd == "clean":
            bb.make_clean()
        elif cmd == "dump":
            bb.dump()
        else:
            print("unknown command: %s" % repr(cmd))
    raise SystemExit(0)
