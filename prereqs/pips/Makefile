PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

include Makefile.pip

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

# qtconsole.patch: Fix crash in completion widget, see Trac#6363
PATCHES = pyopengl_macos11.patch webcolors.patch qtconsole.patch
NETCDF_DYLIBS = libnetcdf.19.dylib libdf.0.dylib libhdf5_hl.200.dylib \
				libhdf5_hl.dylib libjpeg.9.dylib libmfhdf.0.dylib \
				libzip.5.dylib
PYTABLES_DYLIBS = libblosc.1.dylib libhdf5.200.dylib libsnappy.1.dylib \
				  libc++.1.dylib libcrypto.3.dylib libssl.3.dylib

# Notarized macOS application will not launch first time if it contains
# libraries or executables with references to non-system libraries outside the application.
# Remove these inessential libraries from PySide2 that link to external libraries.
# See ChimeraX ticket #4013 for details.
PYSIDE2_MACOS_REMOVE = PySide2/pyside2-lupdate \
			PySide2/Qt/plugins/sqldrivers/libqsqlodbc.dylib \
			PySide2/Qt/plugins/sqldrivers/libqsqlpsql.dylib

all:

install:
	LANG=$(LANG) $(PIP_INSTALL) -r build_requirements.txt
ifeq ($(OS),Darwin)
	# Link cython in Python framework to build/bin, used compiling core/serialize.pyx.
	ln -s ../Library/Frameworks/Python.Framework/Versions/$(PYTHON_VERSION)/bin/cython $(bindir)
endif

list:
	$(PIP) list

app-list:
	$(APP_PIP) list

upgrade:
	$(PIP_INSTALL) -U -r build_requirements.txt

app-install:
	$(APP_PIP_INSTALL) -r app_requirements.txt
ifeq ($(OSARCH),DarwinArm64)
	# Install the Conda versions of Python wheels for arm64 Mac
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/imagecodecs/imagecodecs-2022.2.22-cp39-cp39-macosx_12_0_arm64.whl
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/netcdf4/netCDF4-1.5.8-cp39-cp39-macosx_12_0_arm64.whl
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/pytables/tables-3.7.0-cp39-cp39-macosx_12_0_arm64.whl
	for l in $(NETCDF_DYLIBS); do \
		$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/netcdf4/$$l.arm64 ; \
	done
	for l in $(PYTABLES_DYLIBS); do \
		$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/pytables/$$l.arm64 ; \
	done
	for l in $(NETCDF_DYLIBS) $(PYTABLES_DYLIBS); do \
		mv $$l.arm64 $$l ; \
		chmod +x $$l ; \
	done
	$(APP_PIP_INSTALL) tables-3.7.0-cp39-cp39-macosx_12_0_arm64.whl \
		netCDF4-1.5.8-cp39-cp39-macosx_12_0_arm64.whl \
		imagecodecs-2022.2.22-cp39-cp39-macosx_12_0_arm64.whl
ifdef USE_MAC_FRAMEWORKS
	$(RSYNC) *.dylib $(APP_PYTHON_FRAMEWORK_LIBDIR)
else
	$(RSYNC) *.dylib $(app_shlibdir)
endif
endif
	$(APP_PIP_INSTALL) -r app_pyopengl_requirements.txt
	-for p in $(PATCHES); do \
		patch -d $(APP_PYSITEDIR) -N -p0 < $$p ; \
	done
	-cd $(APP_PYSITEDIR) ; rm -f $(PYSIDE2_MACOS_REMOVE)

app-upgrade:
	$(APP_PIP_INSTALL) -U -r app_requirements.txt
	$(APP_PIP_INSTALL) -U -r app_pyopengl_requirements.txt

# Example targets for patching a pip package. These targets would
# be dependencies of the install and app-install targets.
#
# MPL_VERSION	= 1.5.1
#
# $(PYSITEDIR)/matplotlib:
#	LANG=$(LANG) $(PIP) install matplotlib==$(MPL_VERSION)
#	-for p in $(MPL_PATCHES); do \
#		patch -d $(PYSITEDIR) -N -p0 < $$p ; \
#	done
#
# $(APP_PYSITEDIR)/matplotlib:
#	LANG=$(LANG) $(APP_PIP) install matplotlib==$(MPL_VERSION)
#	-for p in $(MPL_PATCHES); do \
#		patch -d $(APP_PYSITEDIR) -N -p0 < $$p ; \
#	done

clean:
	rm -f pip.log
	make -C package_checker clean

outdated:
	$(PIP) list --outdated --format=columns
	@echo ""
	@echo "Known issues:"
	@echo "  distlib 0.2.7 last version that works with 'toolshed uninstall'"
	@echo "  grako (3.16.5 is the last version that works [see #3460])"
	@echo "  lineprofiler 2.1.2, newer versions 3.0 and above require cmake"
	@echo "  pip, wheel, setuptools versions are set in ../python/Makefile"
	@echo "  jedi (pulled in by line-profiler)"
	@echo "  MarkupSafe (pulled in by Jinja2)"
	@echo "  parso (pulled in by line-profiler via jedi)"
	@echo "  pyzmq (pulled in by ipykernel via jupyter-client)"
	@echo "  snowballstemmer (pulled in by Sphinx)"
	@echo "  tornado (pulled in by ipykernel)"

app-outdated:
	$(APP_PIP) list --outdated --format=columns

src-outdated:
	$(PYTHON_EXE) -m package_checker ../../src
