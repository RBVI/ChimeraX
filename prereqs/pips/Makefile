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
CYTHON_VERSION = $(shell grep Cython $(TOP)/pyproject.toml  | sed -e 's/ ;.*//' -e 's/.*=//' -e 's/.*=//' -e 's/",//')
CXSERVICES_VER = 1.2.3
CXSERVICES_DEPLOYMENT_VER ?= production

# tcia_utils.patch: Fix needless import of x86_64 zmq on macOS, see Trac#8638
# Remove if tcia_utils removes those imports
PATCHES = tcia_utils.patch tcia_utils_metadata.patch grako.patch
ifeq ($(OS),Windows)
ifneq ($(filter $(MSYSTEM),MSYS MINGW64),)
PATCHES += openmm_metadata.patch
else
PATCHES += openmm_metadata_cygwin.patch
endif
endif
ifeq ($(OS),Linux)
ifneq ($(OSARCH),LinuxArm64)
PATCHES += openmm_metadata.patch
endif
endif

CXSERVICES_WHEEL = cxservices-$(CXSERVICES_VER)-py3-none-any.whl

ifeq ($(OSARCH),DarwinArm64)
CYTHON_BUILD_ARGS= --no-binary :all:
else
CYTHON_BUILD_ARGS=
endif

PYOPENGL_SRC = pyopengl_accelerate-3.1.9.tar.gz

ifeq ($(OS),Windows)
else
ifeq ($(OS),Linux)
ifeq ($(OSARCH),LinuxArm64)
PYOPENGL_WHEEL = pyopengl_accelerate-3.1.9-cp311-cp311-linux_aarch64.whl
else
endif
else
ifeq ($(OS),Darwin)
WHEEL_CFLAGS="-Wno-error=int-conversion"
ifeq ($(OSARCH),DarwinArm64)
SCIPY_WHEEL = $(subst ==,-,$(shell grep scipy $(TOP)/pyproject.toml | sed -e 's/[\ ",]//g'))-cp311-cp311-macosx_12_0_arm64.whl
else
ifeq ($(OSARCH),DarwinIntel64)
PYOPENGL_WHEEL = pyopengl_accelerate-3.1.9-cp311-cp311-macosx_12_0_universal2.whl
endif
endif
endif
endif
endif

PLATO_WHEELS = $(CXSERVICES_WHEEL)

all:

install:

list:
	$(PIP) list

app-list:
	$(APP_PIP) list

upgrade:

ifneq ($(filter $(OSARCH),DarwinIntel64 LinuxArm64),)
app-install: app_requirements.txt $(PLATO_WHEELS) patched-pyopengl
else
ifeq ($(OSARCH),DarwinArm64)
app-install: app_requirements.txt $(PLATO_WHEELS) $(SCIPY_WHEEL)
else
app-install: app_requirements.txt $(PLATO_WHEELS)
endif
endif
	# Install Cython first so PyOpenGL_accelerate builds
	$(APP_PIP_INSTALL) $(CYTHON_BUILD_ARGS) $(shell grep Cython $(TOP)/pyproject.toml | sed -e 's/[\ #",]//g')
	$(APP_PIP_INSTALL) $(shell grep numpy app_requirements.txt | sed -e 's/[\ #",]//g')
ifeq ($(OSARCH),DarwinArm64)
	$(APP_PIP_INSTALL) ./$(SCIPY_WHEEL)
endif
	CFLAGS+=$(WHEEL_CFLAGS) $(APP_PIP_INSTALL) --no-build-isolation -r app_requirements.txt
ifneq ($(filter $(OSARCH),DarwinIntel64 LinuxArm64),)
	$(APP_PIP_INSTALL) $(PYOPENGL_WHEEL)
else
	$(APP_PIP_INSTALL) $(shell grep PyOpenGL-accelerate $(TOP)/pyproject.toml | sed -e 's/[\ #",]//g')
endif
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
ifneq (,$(filter $(OS),Windows Linux))
ifneq ($(OSARCH),LinuxArm64)
	# openmm-cuda-12 brings in ~600MB of CUDA dependencies it does not use
	$(APP_PIP) uninstall -y nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cufft-cu12
endif
ifeq ($(OSARCH),Linux)
	$(APP_PIP) uninstall -y nvidia-nvjitlink-cu12
endif
endif

app_requirements.txt:
	$(APP_PYTHON_BIN) make_requirements.py

uv-install: $(PLATO_WHEELS)
	uv pip install -r $(TOP)/pyproject.toml --all-extras
	uvx pip download --platform=macosx_12_0_arm64 --no-deps $(shell grep scipy $(TOP)/pyproject.toml | sed 's/[\ #",]//g')
ifeq ($(OSARCH),DarwinArm64)
	uv pip install ./$(SCIPY_WHEEL)
endif
	-for p in $(PATCHES); do \
		patch --ignore-whitespace -d $(TOP)/.venv/lib/python3.11/site-packages/ -N -p0 < $$p ; \
	done

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
	-rm -f app_requirements.txt
	-rm -f pyopengl_accelerate-3.1.9.tar.gz
	-rm -rf pyopengl_accelerate-3.1.9
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
	@echo "  pyzmq (pulled in by ipykernel via jupyter-client)"
	@echo "  snowballstemmer (pulled in by Sphinx)"
	@echo "  tornado (pulled in by ipykernel)"

app-outdated:
	$(APP_PIP) list --outdated --format=columns

src-outdated:
	$(PYTHON_EXE) -m package_checker ../../src

ifneq ($(filter $(OSARCH),DarwinIntel64 LinuxArm64),)
$(PYOPENGL_SRC):
	$(APP_PIP) download --no-build-isolation --no-binary=:all: $(shell grep PyOpenGL $(TOP)/pyproject.toml | sed 's/[\ ",#]//g')

patched-pyopengl: $(PYOPENGL_SRC)
	tar --no-same-owner -xf $(PYOPENGL_SRC)
	patch -N -p0 < pyopengl_accelerate.patch
	$(APP_PYTHON_BIN) -m build --wheel pyopengl_accelerate-3.1.9
	mv pyopengl_accelerate-3.1.9/dist/$(PYOPENGL_WHEEL) .
endif

$(SCIPY_WHEEL):
	$(APP_PIP) download --platform=macosx_12_0_arm64 --no-deps $(shell grep scipy $(TOP)/pyproject.toml | sed 's/[\ ",]//g')

$(CYTHON_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cython/$(CYTHON_WHEEL)

$(CXSERVICES_WHEEL):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/cxservices-$(CXSERVICES_VER)-$(CXSERVICES_DEPLOYMENT_VER)-py3-none-any.whl
	mv cxservices-$(CXSERVICES_VER)-$(CXSERVICES_DEPLOYMENT_VER)-py3-none-any.whl $(CXSERVICES_WHEEL)
