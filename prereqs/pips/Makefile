PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
-include $(TOP)/.makerc
include Makefile.pip

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

COMMA = ,
EMPTY =
SPACE = $(EMPTY) $(EMPTY)
SOURCE_WHEELS =
SOURCE_VERSIONS =
WHEEL_CFLAGS ?=

# On macOS, only the x86_64 version of Cython is available from pypi.
# So we install the same version on arm64, but from source.
CXSERVICES_VER = 1.2.3
CXSERVICES_DEPLOYMENT_VER ?= production

# tcia_utils.patch: Fix needless import of x86_64 zmq on macOS, see Trac#8638
# Remove if tcia_utils removes those imports
PATCHES = tcia_utils.patch tcia_utils_metadata.patch
QTSHIM_WHEEL = qtshim-1.1-py3-none-any.whl
CXSERVICES_WHEEL = cxservices-$(CXSERVICES_VER)-py3-none-any.whl
GRAKO_WHEEL = grako-3.16.5-py2.py3-none-any.whl

ifeq ($(OSARCH),DarwinArm64)
CYTHON_BUILD_ARGS= --no-binary :all:
else
CYTHON_BUILD_ARGS=
endif

ifeq ($(OS),Windows) 
ACCELERATE_WHEEL = PyOpenGL_accelerate-3.1.7-cp313-cp313-win_amd64.whl
TABLES_WHEEL = tables-3.10.2.dev0-cp313-cp313-win_amd64.whl
TINYARRAY_WHEEL = tinyarray-1.2.4-cp313-cp313-win_amd64.whl
else
ifeq ($(OS),Linux)
ACCELERATE_WHEEL = PyOpenGL_accelerate-3.1.7-cp313-cp313-manylinux_2_5_x86_64.manylinux1_x86_64.whl
TABLES_WHEEL=
TINYARRAY_WHEEL=
ifeq ($(OSARCH),LinuxArm64)
ACCELERATE_WHEEL = 
endif
else
ifeq ($(OS),Darwin)
WHEEL_CFLAGS="-Wno-error=int-conversion"
ACCELERATE_WHEEL = PyOpenGL_accelerate-3.1.7-cp313-cp313-macosx_10_13_universal2.whl
TABLES_WHEEL=
TINYARRAY_WHEEL=
endif
endif
endif

PLATO_WHEELS = $(GRAKO_WHEEL) $(CXSERVICES_WHEEL) $(QTSHIM_WHEEL) $(ACCELERATE_WHEEL) $(TABLES_WHEEL) $(TINYARRAY_WHEEL)

all:

install:

list:
	$(PIP) list

app-list:
	$(APP_PIP) list

upgrade:

app-install: $(PLATO_WHEELS)
	# Install Cython first because source packages will complain otherwise
	$(APP_PIP_INSTALL) $(CYTHON_BUILD_ARGS) $(shell grep Cython app_requirements.txt)
	# Install numpy second so that PyOpenGL_Accelerate will compile numpy_formathandler
	$(APP_PIP_INSTALL) $(shell grep numpy app_requirements.txt)
	$(APP_PIP_INSTALL) $(shell grep lz4 app_requirements.txt)
	CFLAGS+=$(WHEEL_CFLAGS) $(APP_PIP_INSTALL) --no-build-isolation -r app_requirements.txt
ifneq (,$(SOURCE_WHEELS))
	LANG=$(LANG) $(PIP_INSTALL) --no-build-isolation --no-binary $(subst $(SPACE),$(COMMA),$(SOURCE_WHEELS)) $(SOURCE_VERSIONS)
endif
ifdef DEV_DEPENDENCIES
	$(APP_PIP_INSTALL) --no-build-isolation -r dev_requirements.txt
endif
	$(APP_PIP_INSTALL) $(PLATO_WHEELS)
	-for p in $(PATCHES); do \
		patch --ignore-whitespace -d $(APP_PYSITEDIR) -N -p0 < $$p ; \
	done
	# tcia_utils (a DICOM dependency) pulls in Pandas, but it is patched
	# so that pandas is not needed. When those patches are upstreamed 
	# and pandas is optional this line can be removed
	$(APP_PIP) uninstall -y pandas

app-upgrade:
	$(APP_PIP_INSTALL) -U -r app_requirements.txt

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
	-rm -f pip.log 
	make -C package_checker clean	

outdated:
	$(PIP) list --outdated --format=columns
	@echo ""
	@echo "Known issues:"
	@echo "  tcia_utils versions >1.5.1 make no API changes and add more dependencies to patch out"
	@echo "  distlib 0.2.7 last version that works with 'toolshed uninstall'"
	@echo "  grako (3.16.5 is the last version that works [see #3460])"
	@echo "  lineprofiler 2.1.2, newer versions 3.0 and above require cmake"
	@echo "  jedi (pulled in by line-profiler)"
	@echo "  MarkupSafe (pulled in by Jinja2)"
	@echo "  parso (pulled in by line-profiler via jedi)"
	@echo "  pip, wheel, setuptools versions are set in ../python/Makefile"
	@echo "  pyopenxr use 1.0.3401, newer 1.1.3601 is not supported by SteamVR or Meta App, Aug 2024"
	@echo "  pyzmq (pulled in by ipykernel via jupyter-client)"
	@echo "  snowballstemmer (pulled in by Sphinx)"
	@echo "  tornado (pulled in by ipykernel)"

app-outdated:
	$(APP_PIP) list --outdated --format=columns

src-outdated:
	$(PYTHON_EXE) -m package_checker ../../src

$(TABLES_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/tables/$(TABLES_WHEEL)

$(TINYARRAY_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/tables/$(TINYARRAY_WHEEL)

$(ACCELERATE_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/pyopengl/$(ACCELERATE_WHEEL)

$(GRAKO_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/grako/$(GRAKO_WHEEL)

$(CXSERVICES_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/cxservices-$(CXSERVICES_VER)-$(CXSERVICES_DEPLOYMENT_VER)-py3-none-any.whl
	mv cxservices-$(CXSERVICES_VER)-$(CXSERVICES_DEPLOYMENT_VER)-py3-none-any.whl $(CXSERVICES_WHEEL)

$(QTSHIM_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/qt/$(QTSHIM_WHEEL)
