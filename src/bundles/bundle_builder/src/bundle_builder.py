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
        license_file = os.path.join(bundle_path, "license.txt")
        if not os.path.exists(info_file):
            raise IOError("Bundle license file %s is missing" %
                          repr(license_file))
        self._read_bundle_info(info_file)
        self._make_paths()
        self._make_setup_arguments()

    @distlib_hack
    def make_wheel(self):
        import os.path, shutil
        self._run_setup(["--no-user-cfg", "test", "bdist_wheel"])
        if not os.path.exists(self.wheel_path):
            raise RuntimeError("Building wheel failed")
        else:
            print("Distribution is in %s" % self.wheel_path)

    @distlib_hack
    def make_install(self, session):
        try:
            self.make_wheel()
        except RuntimeError:
            pass
        else:
            from chimerax.core.commands import run
            run(session, "toolshed install %s reinstall true" % self.wheel_path)

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

    def _read_bundle_info(self, bundle_info):
        from xml.dom.minidom import parse
        doc = parse(bundle_info)
        bi = doc.documentElement
        self._get_identifiers(bi)
        self._get_categories(bi)
        self._get_descriptions(bi)
        self._get_dependencies(bi)
        self._get_c_modules(bi)
        self._get_classifiers(bi)

    def _get_identifiers(self, bi):
        self.name = bi.getAttribute("name")
        self.version = bi.getAttribute("version")
        self.package = bi.getAttribute("package")
        self.min_session = bi.getAttribute("minSessionVersion")
        self.max_session = bi.getAttribute("maxSessionVersion")

    def _get_categories(self, bi):
        self.categories = []
        deps = self._get_singleton(bi, "Categories")
        for e in deps.getElementsByTagName("Category"):
            self.categories.append(e.getAttribute("name"))

    def _get_descriptions(self, bi):
        self.author = self._get_singleton_text(bi, "Author")
        self.email = self._get_singleton_text(bi, "Email")
        self.url = self._get_singleton_text(bi, "URL")
        self.synopsis = self._get_singleton_text(bi, "Synopsis")
        self.description = self._get_singleton_text(bi, "Description")

    def _get_singleton_text(self, bi, tag):
        return self._get_element_text(self._get_singleton(bi, tag))

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

    def _get_dependencies(self, bi):
        self.dependencies = []
        deps = self._get_singleton(bi, "Dependencies")
        for e in deps.getElementsByTagName("Dependency"):
            pkg = e.getAttribute("name")
            ver = e.getAttribute("version")
            self.dependencies.append("%s %s" % (pkg, ver))

    def _get_c_modules(self, bi):
        self.c_modules = []
        for cm in bi.getElementsByTagName("CModule"):
            mod_name = cm.getAttribute("name")
            source_files = []
            for e in cm.getElementsByTagName("SourceFile"):
                source_files.append(self._get_element_text(e))
            self.c_modules.append((mod_name, source_files))

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
        for e in cls.getElementsByTagName("PythonClassifier"):
            self.python_classifiers.append(self._get_element_text(e))
        self.chimerax_classifiers = [
            ("ChimeraX :: Bundle :: " + ','.join(self.categories) +
             " :: " + self.min_session + "," + self.max_session +
             " :: " + self.package + " :: :: ")
        ]
        for e in cls.getElementsByTagName("ChimeraXClassifier"):
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
            "package_dir": {self.package: "src"},
            "packages": [self.package],
            "install_requires": self.dependencies,
        }
        if self.c_modules:
            import sys, os.path
            from setuptools import Extension
            # platform-specific
            # Assume Python executable is in ROOT/bin/python
            # and make include directory be ROOT/include
            root = os.path.dirname(os.path.dirname(sys.executable))
            inc_dir = os.path.join(root, "include")
            lib_dir = os.path.join(root, "lib")
            if sys.platform == "darwin":
                # Tested with macOS 10.12
                libraries = []
                compiler_flags = ["-std=c++11", "-stdlib=libc++"]
                env = "Environment :: MacOS X :: Aqua",
                op_sys = "Operating System :: MacOS :: MacOS X"
            elif sys.platform == "win32":
                # Tested with Cygwin
                libraries = ["libatomstruct"]
                compiler_flags = []
                env = "Environment :: Win32 (MS Windows)"
                op_sys = "Operating System :: Microsoft :: Windows :: Windows 10"
            else:
                # Presumably Linux
                # Tested with Ubuntu 16.04 LTS running in
                #   a singularity container on CentOS 7.3
                libraries = []
                compiler_flags = ["-std=c++11"]
                env = "Environment :: X11 Applications"
                op_sys = "Operating System :: POSIX :: Linux"
            ext_mods = [Extension(self.package + '.' + ext_name,
                                  define_macros=[("MAJOR_VERSION", 0),
                                                 ("MINOR_VERSION", 1)],
                                  extra_compile_args=compiler_flags,
                                  include_dirs=[inc_dir],
                                  library_dirs=[lib_dir],
                                  libraries=libraries,
                                  sources=ext_sources)
                        for ext_name, ext_sources in self.c_modules]
            platform_classifiers = [env, op_sys]
        else:
            # pure Python
            ext_mods = []
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
        self.tag = tag(not self.c_modules)
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
        except:
            import traceback
            traceback.print_exc()
        finally:
            sys.argv = save
            os.chdir(cwd)

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
