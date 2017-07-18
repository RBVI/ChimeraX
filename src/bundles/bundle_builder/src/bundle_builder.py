# vim: set expandtab ts=4 sw=4:

def distlib_hack(_func):
    # This hack is needed because distlib and wheel do not yet
    # agree on the metadata file name.
    def hack(*args, **kw):
        # import distutils.core
        # distutils.core.DEBUG = True
        # TODO: remove distlib monkey patch when the wheel package
        # implements PEP 426's pydist.json
        from distlib import metadata
        save = metadata.METADATA_FILENAME
        metadata.METADATA_FILENAME = "metadata.json"
        try:
            return _func(*args, **kw)
        finally:
            metadata.METADATA_FILENAME = save
    return hack


class BundleBuilder:

    def __init__(self, bundle_path=None):
        import os, os.path
        if bundle_path is None:
            bundle_path = os.getcwd()
        self.path = bundle_path
        info_file = os.path.join(bundle_path, "bundle_info.xml")
        if not os.path.exists(info_file):
            raise IOError("Bundle info file %s is missing" % repr(info_file))
        self._read_bundle_info(info_file)
        self._make_paths()
        self._make_setup_arguments()

    @distlib_hack
    def make_wheel(self, test=True):
        # HACK: distutils uses a cache to track created directories
        # for a single setup() run.  We want to run setup() multiple
        # times which can remove/create the same directories.
        # So we need to flush the cache before each run.
        import distutils.dir_util
        try:
            distutils.dir_util._path_created.clear()
        except AttributeError:
            pass
        import os.path, shutil
        if test:
            built = self._run_setup(["--no-user-cfg", "test", "bdist_wheel"])
        else:
            built = self._run_setup(["--no-user-cfg", "build", "bdist_wheel"])
        if not built or not os.path.exists(self.wheel_path):
            raise RuntimeError("Building wheel failed")
        else:
            print("Distribution is in %s" % self.wheel_path)

    @distlib_hack
    def make_install(self, session, test=True, user=None):
        self.make_wheel(test=test)
        from chimerax.core.commands import run
        cmd = "toolshed install %s reinstall true" % self.wheel_path
        if user is not None:
            if user:
                cmd += " user true"
            else:
                cmd += " user false"
        run(session, cmd)

    @distlib_hack
    def make_clean(self):
        import os.path, shutil
        self._rmtree(os.path.join(self.path, "build"))
        self._rmtree(os.path.join(self.path, "dist"))
        self._rmtree(os.path.join(self.path, "src/__pycache__"))
        self._rmtree(self.egg_info)

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
        from xml.dom.minidom import parse
        doc = parse(bundle_info)
        bi = doc.documentElement
        self._get_identifiers(bi)
        self._get_categories(bi)
        self._get_descriptions(bi)
        self._get_datafiles(bi)
        self._get_dependencies(bi)
        self._get_c_modules(bi)
        self._get_packages(bi)
        self._get_classifiers(bi)

    def _get_identifiers(self, bi):
        self.name = bi.getAttribute("name")
        self.version = bi.getAttribute("version")
        self.package = bi.getAttribute("package")
        self.min_session = bi.getAttribute("minSessionVersion")
        self.max_session = bi.getAttribute("maxSessionVersion")
        self.custom_init = bi.getAttribute("customInit")
        self.pure_python = bi.getAttribute("purePython")

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

    def _get_datafiles(self, bi):
        import pathlib, os.path
        self.datafiles = {}
        for dfs in self._get_elements(bi, "DataFiles"):
            pkg_name = dfs.getAttribute("package")
            files = []
            for e in self._get_elements(dfs, "DataFile"):
                filename = self._get_element_text(e)
                files.append(filename)
            if files:
                if not pkg_name:
                    pkg_name = self.package
                self.datafiles[pkg_name] = files

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
            c = _CModule(mod_name, major, minor, uses_numpy)
            for e in self._get_elements(cm, "Requires"):
                c.add_require(self._get_element_text(e))
            for e in self._get_elements(cm, "SourceFile"):
                c.add_source_file(self._get_element_text(e))
            for e in self._get_elements(cm, "IncludeDir"):
                c.add_include_dir(self._get_element_text(e))
            for e in self._get_elements(cm, "Library"):
                c.add_library(self._get_element_text(e))
            for e in self._get_elements(cm, "LibraryDir"):
                c.add_library_dir(self._get_element_text(e))
            for e in self._get_elements(cm, "LinkArgument"):
                c.add_link_argument(self._get_element_text(e))
            for e in self._get_elements(cm, "Framework"):
                c.add_framework(self._get_element_text(e))
            for e in self._get_elements(cm, "FrameworkDir"):
                c.add_framework_dir(self._get_element_text(e))
            self.c_modules.append(c)

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
        for e in self._get_elements(cls, "ChimeraXClassifier"):
            self.chimerax_classifiers.append(self._get_element_text(e))

    def _make_setup_arguments(self):
        self.setup_arguments = {
            "name": self.name,
            "version": self.version,
            "description": self.synopsis,
            "long_description": self.description,
            "author": self.author,
            "author_email": self.email,
            "url": self.url,
            "python_requires": ">= 3.6",
            "install_requires": self.dependencies,
        }
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
        self.setup_arguments["package_dir"] = package_dir
        self.setup_arguments["packages"] = packages
        if self.datafiles:
            self.setup_arguments["package_data"] = self.datafiles
        ext_mods = [em for em in [cm.ext_mod(self.package)
				  for cm in self.c_modules]
                    if em is not None]
        if ext_mods or self.pure_python == "false":
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
        self.setup_arguments["ext_modules"] = ext_mods
        self.setup_arguments["classifiers"] = (self.python_classifiers +
                                               self.chimerax_classifiers)

    def _make_paths(self):
        import os.path
        from .wheel_tag import tag
        self.tag = tag(not self.c_modules and self.pure_python != "false")
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
            os.chdir(self.path)
            sys.argv = ["setup.py"] + cmd
            setuptools.setup(**self.setup_arguments)
            return True
        except:
            import traceback
            traceback.print_exc()
            return False
        finally:
            sys.argv = save
            os.chdir(cwd)

    #
    # Utility functions dealing with XML tree
    #
    def _get_elements(self, e, tag):
        elements = []
        for se in e.getElementsByTagName(tag):
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
        if len(elements) > 1:
            raise ValueError("too many %s elements" % repr(tag))
        elif len(elements) == 0:
            raise ValueError("%s element is missing" % repr(tag))
        return elements[0]

    def _get_singleton_text(self, bi, tag):
        return self._get_element_text(self._get_singleton(bi, tag))


class _CModule:

    def __init__(self, name, major, minor, uses_numpy):
        self.name = name
        self.major = major
        self.minor = minor
        self.uses_numpy = uses_numpy
        self.requires = []
        self.source_files = []
        self.frameworks = []
        self.libraries = []
        self.link_arguments = []
        self.include_dirs = []
        self.library_dirs = []
        self.framework_dirs = []

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

    def add_link_argument(self, a):
        self.link_arguments.append(a)

    def add_framework(self, f):
        self.frameworks.append(f)

    def add_framework_dir(self, d):
        self.framework_dirs.append(d)

    def ext_mod(self, package):
        import sys, os.path
        from setuptools import Extension
        # platform-specific
        # Assume Python executable is in ROOT/bin/python
        # and make include directory be ROOT/include
        root = os.path.dirname(os.path.dirname(sys.executable))
        inc_dirs = [os.path.join(root, "include")]
        lib_dirs = [os.path.join(root, "lib")]
        if self.uses_numpy:
            from numpy.distutils.misc_util import get_numpy_include_dirs
            inc_dirs.extend(get_numpy_include_dirs())
        if sys.platform == "darwin":
            libraries = ["-l" + lib for lib in self.libraries]
            compiler_flags = ["-std=c++11", "-stdlib=libc++"]
            extra_link_args = ["-F" + d for d in self.framework_dirs]
            for fw in self.frameworks:
                extra_link_args.extend(["-framework", fw])
        elif sys.platform == "win32":
            libraries = ["lib" + lib for lib in self.libraries]
            compiler_flags = []
            extra_link_args = []
        else:
            libraries = ["-l" + lib for lib in self.libraries]
            compiler_flags = ["-std=c++11"]
            extra_link_args = []
        for req in self.requires:
            if not os.path.exists(req):
                return None
        inc_dirs.extend(self.include_dirs)
        lib_dirs.extend(self.library_dirs)
        extra_link_args.extend(self.link_arguments)
        return Extension(package + '.' + self.name,
                         define_macros=[("MAJOR_VERSION", self.major),
                                        ("MINOR_VERSION", self.minor)],
                         extra_compile_args=compiler_flags,
                         include_dirs=inc_dirs,
                         library_dirs=lib_dirs,
                         libraries=libraries,
                         extra_link_args=extra_link_args,
                         sources=self.source_files)

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
