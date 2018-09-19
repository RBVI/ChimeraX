# vim: set expandtab ts=4 sw=4:

# Force import in a particular order since the latter two mess
# with the contents of distutils, and we want Cython to win
import distutils
import setuptools
from Cython.Build import cythonize

# Always import this because it changes the behavior of setuptools
from numpy.distutils.misc_util import get_numpy_include_dirs


#
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
try:
    from subprocess import STARTUPINFO
except ImportError:
    MySTARTUPINFO = None
else:
    import subprocess, _winapi
    class MySTARTUPINFO(STARTUPINFO):
        _original = STARTUPINFO
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.dwFlags |= _winapi.STARTF_USESHOWWINDOW


class BundleBuilder:

    def __init__(self, logger, bundle_path=None):
        import os, os.path
        self.logger = logger
        if bundle_path is None:
            bundle_path = os.getcwd()
        self.path = bundle_path
        info_file = os.path.join(bundle_path, "bundle_info.xml")
        if not os.path.exists(info_file):
            raise IOError("Bundle info file %s is missing" % repr(info_file))
        self._read_bundle_info(info_file)
        self._make_paths()
        self._make_setup_arguments()

    def make_wheel(self, test=True, debug=False):
        # HACK: distutils uses a cache to track created directories
        # for a single setup() run.  We want to run setup() multiple
        # times which can remove/create the same directories.
        # So we need to flush the cache before each run.
        import distutils.dir_util
        try:
            distutils.dir_util._path_created.clear()
        except AttributeError:
            pass
        # Copy additional files into package source tree
        self._copy_extrafiles(self.extrafiles)
        # Build C libraries
        import os.path
        for lib in self.c_libraries:
            lib.compile(self.logger, self.dependencies, debug=debug)
        setup_args = ["--no-user-cfg", "build"]
        if debug:
            setup_args.append("--debug")
        if test:
            setup_args.append("test")
        setup_args.extend(["bdist_wheel"])
        built = self._run_setup(setup_args)
        if not built or not os.path.exists(self.wheel_path):
            raise RuntimeError("Building wheel failed")
        else:
            print("Distribution is in %s" % self.wheel_path)

    def make_install(self, session, test=True, debug=False, user=None):
        self.make_wheel(test=test, debug=debug)
        from chimerax.core.commands import run
        cmd = "toolshed install %r" % self.wheel_path
        if user is not None:
            if user:
                cmd += " user true"
            else:
                cmd += " user false"
        run(session, cmd)

    def make_clean(self):
        import os.path, fnmatch
        self._rmtree(os.path.join(self.path, "build"))
        self._rmtree(os.path.join(self.path, "dist"))
        self._rmtree(os.path.join(self.path, "src", "__pycache__"))
        self._rmtree(self.egg_info)
        for root, dirnames, filenames in os.walk("src"):
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
        from xml.dom.minidom import parse
        doc = parse(bundle_info)
        bi = doc.documentElement
        self._get_identifiers(bi)
        self._get_categories(bi)
        self._get_descriptions(bi)
        self._get_datafiles(bi)
        self._get_extrafiles(bi)
        self._get_dependencies(bi)
        self._get_c_modules(bi)
        self._get_c_libraries(bi)
        self._get_packages(bi)
        self._get_classifiers(bi)
        self._check_unused_elements(bi)

    def _get_identifiers(self, bi):
        self.name = bi.getAttribute("name")
        self.version = bi.getAttribute("version")
        self.package = bi.getAttribute("package")
        self.min_session = bi.getAttribute("minSessionVersion")
        self.max_session = bi.getAttribute("maxSessionVersion")
        self.custom_init = bi.getAttribute("customInit")
        self.pure_python = bi.getAttribute("purePython")
        self.installed_data_dir = bi.getAttribute("installedDataDir")
        self.installed_include_dir = bi.getAttribute("installedIncludeDir")
        self.installed_library_dir = bi.getAttribute("installedLibraryDir")

    def _get_categories(self, bi):
        self.categories = []
        deps = self._get_singleton(bi, "Categories")
        for e in self._get_elements(deps, "Category"):
            self.categories.append(e.getAttribute("name"))

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
            pkg_name = dfs.getAttribute("package")
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
            pkg_name = dfs.getAttribute("package")
            files = []
            for e in self._get_elements(dfs, "ExtraFile"):
                source = e.getAttribute("source")
                filename = self._get_element_text(e)
                files.append(("file", source, filename))
            for e in self._get_elements(dfs, "ExtraDir"):
                source = e.getAttribute("source")
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

    def _get_dependencies(self, bi):
        self.dependencies = []
        try:
            deps = self._get_singleton(bi, "Dependencies")
        except ValueError:
            # Dependencies is optional, although
            # ChimeraXCore *should* always be present
            return
        for e in self._get_elements(deps, "Dependency"):
            pkg = e.getAttribute("name")
            ver = e.getAttribute("version")
            self.dependencies.append("%s %s" % (pkg, ver))

    def _get_c_modules(self, bi):
        self.c_modules = []
        for cm in self._get_elements(bi, "CModule"):
            mod_name = cm.getAttribute("name")
            try:
                major = int(cm.getAttribute("major_version"))
            except ValueError:
                major = 0
            try:
                minor = int(cm.getAttribute("minor_version"))
            except ValueError:
                minor = 1
            uses_numpy = cm.getAttribute("usesNumpy") == "true"
            c = _CModule(mod_name, uses_numpy, major, minor,
                         self.installed_library_dir)
            self._add_c_options(c, cm)
            self.c_modules.append(c)

    def _get_c_libraries(self, bi):
        self.c_libraries = []
        for lib in self._get_elements(bi, "CLibrary"):
            c = _CLibrary(lib.getAttribute("name"),
                          lib.getAttribute("usesNumpy") == "true",
                          lib.getAttribute("static") == "true",
                          self.installed_library_dir)
            self._add_c_options(c, lib)
            self.c_libraries.append(c)

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

    def _get_packages(self, bi):
        self.packages = []
        try:
            pkgs = self._get_singleton(bi, "AdditionalPackages")
        except ValueError:
            # AdditionalPackages is optional
            return
        for pkg in self._get_elements(pkgs, "Package"):
            pkg_name = pkg.getAttribute("name")
            pkg_folder = pkg.getAttribute("folder")
            self.packages.append((pkg_name, pkg_folder))

    def _get_classifiers(self, bi):
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
             " :: " + self.package + " :: :: " + self.custom_init)
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
        for e in self._get_elements(cls, "ChimeraXClassifier"):
            self.chimerax_classifiers.append(self._get_element_text(e))

    def _is_pure_python(self):
        return (not self.c_modules and not self.c_libraries
                and self.pure_python != "false")

    def _copy_extrafiles(self, files):
        import shutil, os, os.path
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
        import os, os.path
        datafiles = {}
        for pkg_name, entries in files.items():
            pkg_files = []
            for kind, name in entries:
                if kind == "file":
                    pkg_files.append(name)
                elif kind == "dir":
                    prefix = "src"
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
        # libraries) are on the install list
        for lib in self.c_libraries:
            for lib_path in lib.paths():
                self.datafiles[self.package].append(("file", lib_path))
        self.setup_arguments = {"name": self.name,
                                "python_requires": ">= 3.6"}
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
        import os.path
        from .wheel_tag import tag
        self.tag = tag(self._is_pure_python())
        self.bundle_base_name = self.name.replace("ChimeraX-", "")
        bundle_wheel_name = self.name.replace("-", "_")
        wheel = "%s-%s-%s.whl" % (bundle_wheel_name, self.version, self.tag)
        self.wheel_path = os.path.join(self.path, "dist", wheel)
        self.egg_info = os.path.join(self.path, bundle_wheel_name + ".egg-info")

    def _run_setup(self, cmd):
        import os, sys, setuptools
        cwd = os.getcwd()
        save = sys.argv
        try:
            if MySTARTUPINFO:
                import subprocess
                subprocess.STARTUPINFO = MySTARTUPINFO
            os.chdir(self.path)
            kw = self.setup_arguments.copy()
            kw["package_dir"], kw["packages"] = self._make_package_arguments()
            sys.argv = ["setup.py"] + cmd
            setuptools.setup(**kw)
            return True
        except:
            import traceback
            traceback.print_exc()
            return False
        finally:
            sys.argv = save
            os.chdir(cwd)
            if MySTARTUPINFO:
                import subprocess
                subprocess.STARTUPINFO = MySTARTUPINFO._original

    #
    # Utility functions dealing with XML tree
    #
    def _get_elements(self, e, tag):
        tagged_elements = e.getElementsByTagName(tag)
        # Mark element as used even for non-applicable platform
        self._used_elements.update(tagged_elements)
        elements = []
        for se in tagged_elements:
            platform = se.getAttribute("platform")
            if not platform or platform in self._platform_names:
                elements.append(se)
        return elements

    def _get_element_text(self, e):
        text = ""
        for node in e.childNodes:
            if node.nodeType == node.TEXT_NODE:
                text += node.data
        return text.strip()

    def _get_singleton(self, bi, tag):
        elements = bi.getElementsByTagName(tag)
        self._used_elements.update(elements)
        if len(elements) > 1:
            raise ValueError("too many %s elements" % repr(tag))
        elif len(elements) == 0:
            raise ValueError("%s element is missing" % repr(tag))
        return elements[0]

    def _get_singleton_text(self, bi, tag):
        return self._get_element_text(self._get_singleton(bi, tag))

    def _check_unused_elements(self, bi):
        for node in bi.childNodes:
            if node.nodeType != node.ELEMENT_NODE:
                continue
            if node not in self._used_elements:
                print("WARNING: unsupported element:", node.nodeName)


class _CompiledCode:

    def __init__(self, name, uses_numpy):
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

    def add_require(self, req):
        self.requires.append(req)

    def add_source_file(self, f):
        self.source_files.append(f)

    def add_include_dir(self, d):
        self.include_dirs.append(d)

    def add_library(self, l):
        self.libraries.append(l)

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
        import sys, os.path
        for req in self.requires:
            if not os.path.exists(req):
                raise ValueError("unused on this platform")
        # platform-specific
        # Assume Python executable is in ROOT/bin/python
        # and make include directory be ROOT/include
        root = os.path.dirname(os.path.dirname(sys.executable))
        inc_dirs = [os.path.join(root, "include")]
        lib_dirs = [os.path.join(root, "lib")]
        if self.uses_numpy:
            inc_dirs.extend(get_numpy_include_dirs())
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
        for dep in dependencies:
            d_inc, d_lib = self._get_bundle_dirs(logger, dep)
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
            raise RuntimeError("bundle not found: %s" % req)
            # return None, None
        inc = bundle.include_dir()
        lib = bundle.library_dir()
        return inc, lib


class _CModule(_CompiledCode):

    def __init__(self, name, uses_numpy, major, minor, libdir):
        super().__init__(name, uses_numpy)
        self.major = major
        self.minor = minor
        self.installed_library_dir = libdir

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
        if self.installed_library_dir:
            install_dir = '/' + self.installed_library_dir
        else:
            install_dir = ''
        if sys.platform == "linux":
            extra_link_args.append("-Wl,-rpath,\$ORIGIN%s" % install_dir)
        elif sys.platform == "darwin":
            extra_link_args.append("-Wl,-rpath,@loader_path%s" % install_dir)
        return Extension(package + '.' + self.name,
                         define_macros=macros,
                         extra_compile_args=cpp_flags+self.compile_arguments,
                         include_dirs=inc_dirs,
                         library_dirs=lib_dirs,
                         libraries=libraries,
                         extra_link_args=extra_link_args,
                         sources=self.source_files)


class _CLibrary(_CompiledCode):

    def __init__(self, name, uses_numpy, static, libdir):
        super().__init__(name, uses_numpy)
        self.static = static
        self.installed_library_dir = libdir

    def compile(self, logger, dependencies, debug=False):
        import sys, os, os.path, distutils.ccompiler, distutils.sysconfig
        import distutils.log
        distutils.log.set_verbosity(1)
        try:
            (inc_dirs, lib_dirs, macros, extra_link_args,
             libraries, cpp_flags) = self._compile_options(logger, dependencies)
        except ValueError:
            print("Error when compiling %s" % self.name)
            return None
        if self.installed_library_dir:
            output_dir = os.path.join("src", self.installed_library_dir)
            install_dir = '/' + self.installed_library_dir
        else:
            output_dir = "src"
            install_dir = ''
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
            lib_name = "lib" + self.name
        else:
            lib_name = self.name
        if not self.static:
            macros.append(("DYNAMIC_LIBRARY", 1))
        if sys.platform == "darwin":
            # We need to manually separate out C from C++ code here, since clang
            # crashes if -std=c++11 is given as a switch while compiling C code
            c_files = []
            cpp_files = []
            for f in self.source_files:
                l = compiler.detect_language(f)
                if l == 'c':
                    c_files.append(f)
                elif l == 'c++':
                    cpp_files.append(f)
                else:
                    raise RuntimeError("Unsupported language for %s" % f)
            compiler.compile(cpp_files, extra_preargs=cpp_flags,
                             macros=macros, debug=debug)
            compiler.compile(c_files, extra_preargs=[],
                             macros=macros, debug=debug)
        else:
            compiler.compile(self.source_files, extra_preargs=cpp_flags,
                             macros=macros, debug=debug)
        objs = compiler.object_filenames(self.source_files)
        compiler.mkpath(output_dir)
        if self.static:
            lib = compiler.library_filename(lib_name, lib_type="static")
            compiler.create_static_lib(objs, lib_name, output_dir=output_dir,
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
                                            debug=debug)
            elif sys.platform == "win32":
                # On Windows, we need both .dll and .lib
                link_lib = compiler.library_filename(lib_name, lib_type="static")
                extra_link_args.append("/LIBPATH:%s" % link_lib)
                lib = compiler.shared_object_filename(lib_name)
                compiler.link_shared_object(objs, lib, output_dir=output_dir,
                                            extra_preargs=extra_link_args,
                                            debug=debug)
            else:
                # On Linux, we only need the .so
                lib = compiler.library_filename(lib_name, lib_type="shared")
                extra_link_args.append("-Wl,-rpath,\$ORIGIN%s" % install_dir)
                compiler.link_shared_object(objs, lib, output_dir=output_dir,
                                            extra_preargs=extra_link_args,
                                            debug=debug)
        return lib

    def paths(self):
        import sys, os, os.path, distutils.ccompiler, distutils.sysconfig
        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler)
        if sys.platform == "win32":
            lib_name = "lib" + self.name
        else:
            lib_name = self.name
        if self.installed_library_dir:
            lib_name = os.path.join(self.installed_library_dir, lib_name)
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
