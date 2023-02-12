PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
-include $(TOP)/.makerc
include Makefile.pip

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

# tcia_utils.patch: Fix needless import of x86_64 zmq on macOS, see Trac#8638
# Remove if tcia_utils removes those imports
PATCHES = tcia_utils.patch tcia_utils_metadata.patch
PYTABLES_DYLIBS = libblosc.1.dylib libhdf5.200.dylib libsnappy.1.dylib \
				  libc++.1.dylib libcrypto.3.dylib libssl.3.dylib

# Notarized macOS application will not launch first time if it contains
# libraries or executables with references to non-system libraries outside the application.
# Remove these inessential libraries from PySide2 that link to external libraries.
# See ChimeraX ticket #4013 for details.
PYSIDE2_MACOS_REMOVE = PySide2/pyside2-lupdate \
			PySide2/Qt/plugins/sqldrivers/libqsqlodbc.dylib \
			PySide2/Qt/plugins/sqldrivers/libqsqlpsql.dylib


GRAKO_WHEEL = grako-3.16.5-py2.py3-none-any.whl
GRAKO_PREREQ = $(PREREQS_ARCHIVE)/grako/$(GRAKO_WHEEL)
ifeq ($(OS),Windows)
TINYARRAY_WHEEL = tinyarray-1.2.4-cp311-cp311-win_amd64.whl
PYOPENGL_ACCELERATE_WHEEL = PyOpenGL_accelerate-3.1.6-cp311-cp311-win_amd64.whl
else
ifeq ($(OS),Linux)
TINYARRAY_WHEEL = tinyarray-1.2.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
PYOPENGL_ACCELERATE_WHEEL =
# PYOPENGL_ACCELERATE_WHEEL = PyOpenGL_accelerate-3.1.6-cp311-cp311-win_amd64.whl??
else
ifeq ($(OS),Darwin)
TINYARRAY_WHEEL = tinyarray-1.2.4-cp311-cp311-macosx_10_9_universal2.whl
PYOPENGL_ACCELERATE_WHEEL = PyOpenGL_accelerate-3.1.6-cp311-cp311-macosx_10_9_universal2.whl
ifeq ($(OSARCH),DarwinArm64)
IMAGECODECS_WHEEL = $(PREREQS_ARCHIVE)/imagecodecs/imagecodecs-2023.1.23-cp311-cp311-macosx_12_0_arm64.whl
PYTABLES_WHEEL = $(PREREQS_ARCHIVE)/pytables/tables-3.8.0-cp311-cp311-macosx_11_0_arm64.whl
else
ifeq ($(OSARCH),DarwinIntel64)
endif
endif
endif
endif
endif

TINYARRAY_PREREQ = $(PREREQS_ARCHIVE)/tinyarray/$(TINYARRAY_WHEEL)
PYOPENGL_ACCELERATE_PREREQ = $(PREREQS_ARCHIVE)/pyopengl/$(PYOPENGL_ACCELERATE_WHEEL)

PLATO_PREREQS = $(TINYARRAY_PREREQ) $(PYOPENGL_ACCELERATE_PREREQ)
PLATO_WHEELS = $(TINYARRAY_WHEEL) $(PYOPENGL_ACCELERATE_WHEEL)

ifeq ($(OS),Windows)
else
ifeq ($(OS),Linux)
else
ifeq ($(OS),Darwin)
ifeq ($(OSARCH),DarwinArm64)
PLATO_PREREQS += $(IMAGECODECS_PREREQ) $(PYTABLES_PREREQ)
PLATO_WHEELS += $(IMAGECODECS_WHEEL) $(PYTABLES_WHEEL)
else
ifeq ($(OSARCH),DarwinIntel64)
endif
endif
endif
endif
endif

all:

install:
	LANG=$(LANG) $(PIP_INSTALL) --no-build-isolation -r build_requirements.txt
	$(FETCH_PREREQ) $(GRAKO_PREREQ)
	$(PIP_INSTALL) $(GRAKO_WHEEL)
ifeq ($(OS),Darwin)
	# Link cython in Python framework to build/bin, used compiling core/serialize.pyx.
	ln -s ../Library/Frameworks/Python.Framework/Versions/$(PYTHON_VERSION)/bin/cython $(bindir)
endif

list:
	$(PIP) list

app-list:
	$(APP_PIP) list

upgrade:
	$(PIP_INSTALL) -U --no-build-isolation -r build_requirements.txt

app-install:
	$(APP_PIP_INSTALL) --no-build-isolation -r app_requirements.txt
	# Will be downloaded from install step
	$(APP_PIP_INSTALL) $(GRAKO_WHEEL)
ifdef DEV_DEPENDENCIES
	$(APP_PIP_INSTALL) --no-build-isolation -r dev_requirements.txt
endif
	for prereq in $(PLATO_PREREQS); do \
		$(FETCH_PREREQ) $$prereq; \
	done
	$(APP_PIP_INSTALL) -r app_pyopengl_requirements.txt
	$(APP_PIP_INSTALL) $(PLATO_WHEELS)
	-for p in $(PATCHES); do \
		patch --ignore-whitespace -d $(APP_PYSITEDIR) -N -p0 < $$p ; \
	done
	-cd $(APP_PYSITEDIR) ; rm -f $(PYSIDE2_MACOS_REMOVE)
	# tcia_utils (a DICOM dependency) pulls in Pandas, but it is patched
	# so that pandas is not needed. When those patches are upstreamed 
	# and pandas is optional this line can be removed
	$(APP_PIP) uninstall -y pandas

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
	-rm -f pip.log $(PYTABLES_DYLIBS) \
		imagecodecs-2022.2.22-cp39-cp39-macosx_12_0_arm64.whl \
		tables-3.7.0-cp39-cp39-macosx_12_0_arm64.whl
	make -C package_checker clean	

outdated:
	$(PIP) list --outdated --format=columns
	@echo ""
	@echo "Known issues:"
	@echo "  distlib 0.2.7 last version that works with 'toolshed uninstall'"
	@echo "  grako (3.16.5 is the last version that works [see #3460])"
	@echo "  lineprofiler 2.1.2, newer versions 3.0 and above require cmake"
	@echo "  jedi (pulled in by line-profiler)"
	@echo "  MarkupSafe (pulled in by Jinja2)"
	@echo "  parso (pulled in by line-profiler via jedi)"
	@echo "  pip, wheel, setuptools versions are set in ../python/Makefile"
	@echo "  PyOpenGL 3.1.6, no source distribution for accelerate"
	@echo "  pyzmq (pulled in by ipykernel via jupyter-client)"
	@echo "  snowballstemmer (pulled in by Sphinx)"
	@echo "  tornado (pulled in by ipykernel)"

app-outdated:
	$(APP_PIP) list --outdated --format=columns

src-outdated:
	$(PYTHON_EXE) -m package_checker ../../src
