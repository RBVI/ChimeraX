# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

TOP = .
TOP := $(shell (cd "$(TOP)"; pwd))
NO_SUBDIR_INSTALL=1
NO_SUBDIR_TEST=1
SUBDIRS = prereqs src
-include .makerc
include $(TOP)/mk/config.make
include $(TOP)/mk/subdir.make
include $(TOP)/mk/detectOS.make

all:
	@echo "'make install' to build everything" && exit 1

ifdef WIN32
install:	vsdefined
else
install:
endif
	@echo 'Started install at' `date` on `hostname`
	$(MAKE) build-dirs
ifdef NO_PREBUILT
	$(MAKE) -C prereqs install
else
	$(MAKE) -C prereqs install-prebuilt
endif
	$(MAKE) -C prereqs app-install
	$(MAKE) build-app-dirs
	$(MAKE) -C src install
	$(MAKE) -C docs install
ifndef WIN32
	# Admin privileges are needed on Windows 10
	# To enable, follow the instructions at
	# https://stackoverflow.com/a/65504258/12208118
	# then you can make -C vdocs by hand if you like
	$(MAKE) -C vdocs install
endif
	$(APP_PYTHON_EXE) utils/clean_app.py
	$(APP_PYTHON_EXE) -m pip check
ifeq ($(OS),Darwin)
	# update Info.plist with data formats provided by bundles
	$(MAKE) -C src/apps/ChimeraX install-plist
endif
ifdef FLATPAK_DIST
	-${APP_PYTHON_EXE} -m compileall -d CHIMERAX ${app_libdir}
	${APP_EXE} --exit --nogui --silent --cmd 'linux flatpak-files edu.ucsf.rbvi.ChimeraX'
endif
	@echo 'Finished install at' `date`

uv-build: build-dirs
	uv pip install -r pyproject.toml --all-extras
	UV_BUILD=1 $(MAKE) -C prereqs/pips uv-install
	uv pip install PySide6 shiboken6
	PYTHON=python $(MAKE) -C src/bundles uv-install

uv-install: build-dirs uv-build
	UV_BUILD=1 $(MAKE) -e USE_MAC_FRAMEWORKS='' -C src/apps/ChimeraX
	UV_BUILD=1 $(MAKE) -e USE_MAC_FRAMEWORKS='' -C src/apps/ChimeraX uv-install

install-rbvi:
	$(MAKE) PYQT_LICENSE=commercial install

ifdef WIN32
build-minimal:	vsdefined
else
build-minimal:
endif
	$(MAKE) build-dirs
	$(MAKE) build-app-dirs
ifdef WIN32
	$(MAKE) -C prereqs/win32 app-install
endif
	$(MAKE) -C prereqs/Python install
	$(MAKE) -C prereqs/pips install
	$(MAKE) -C prereqs/Python app-install
	$(MAKE) -C prereqs/pips app-install
	$(MAKE) -C prereqs/ambertools app-install
	$(MAKE) -C src/bundles install
	$(MAKE) -C src/apps/ChimeraX install

test src.test: testimports
	$(MAKE) -C src test

testimports:
	$(APP_EXE) --exit --nogui --silent utils/cxtestimports.py

ifdef FLATPAK_DIST
SCRIPT_COVERAGE_ARGS := $(if $(USE_COVERAGE),-c -s -f,)
else
SCRIPT_COVERAGE_ARGS := $(if $(USE_COVERAGE),-c -s,)
endif
COVERAGE_ARGS := $(if $(USE_COVERAGE),--cov=chimerax --cov-append,)
SILENT_COVERAGE_ARGS := $(if $(USE_COVERAGE),$(COVERAGE_ARGS) --cov-report=,)

pytest-install:
	$(APP_PYTHON_EXE) -I -m pip install pytest $(if $(USE_COVERAGE),pytest-cov,)

clean-coverage:
	-rm .coverage
	-rm -rf chimerax

prepare-coverage:
	cp -r $(APP_PYSITEDIR)/chimerax .

report-coverage:
ifdef UV_BUILD
	python -m coverage report -i
else
	$(APP_PYTHON_EXE) -m coverage report -i
endif

pytest-both-exes:
ifdef UV_BUILD
	./tests/env.sh $(SCRIPT_COVERAGE_ARGS) -u
else
	./tests/env.sh $(SCRIPT_COVERAGE_ARGS)
endif

pytest-wheel:
ifdef UV_BUILD
	python -m pytest -m "wheel" $(SILENT_COVERAGE_ARGS) tests/test_imports_wheel.py
	python -m pytest -m "wheel" $(SILENT_COVERAGE_ARGS)
else
	$(APP_PYTHON_EXE) -m pytest -m "wheel" $(SILENT_COVERAGE_ARGS) tests/test_imports_wheel.py
	$(APP_PYTHON_EXE) -m pytest -m "wheel" $(SILENT_COVERAGE_ARGS)
endif

pytest-app:
ifdef UV_BUILD
	python -m pytest -m "wheel" $(SILENT_COVERAGE_ARGS) tests/test_imports_wheel.py
	python -m pytest -m "wheel" $(SILENT_COVERAGE_ARGS)
else
	$(APP_PYTHON_EXE) -m pytest -m "not wheel" $(SILENT_COVERAGE_ARGS) tests/test_imports_app.py
	$(APP_PYTHON_EXE) -m pytest -m "not wheel" $(SILENT_COVERAGE_ARGS)
endif

ifdef USE_COVERAGE
pytest: clean-coverage prepare-coverage pytest-both-exes pytest-wheel pytest-app report-coverage
else
pytest: pytest-both-exes pytest-wheel pytest-app
endif

install-common-wheels:
	$(APP_PYTHON_EXE) -m pip install wheels/*.whl

sync:
	mkdir -p $(build_prefix)/sync/
	$(MAKE) -C src/bundles sync

.PHONY: venv
venv:
	if [ -x $(APP_PYTHON_BIN) ] && [ ! -x .venv ]; then \
		$(APP_PYTHON_BIN) -m venv .venv --system-site-packages ; \
		echo 'Virtual environment created in .venv' ; \
		echo 'source .venv/bin/activate to activate it on Linux or macOS' ; \
		echo 'or source .venv/Scripts/activate to activate it on Windows' ; \
		exit ; \
	else \
		if [ -x .venv ]; then \
			echo '.venv already exists' ; \
			exit ; \
		fi ; \
		echo 'Build ChimeraX before creating your virtual environment'; \
		exit ; \
	fi

ifdef WIN32
vsdefined:
	@if [ -z $${VSINSTALLDIR+x} ]; then \
		echo 'Visual Studio not found.  Run ". vsvars.sh"' ; \
		false; \
	fi
endif

docs.install:
	$(MAKE) -C docs install

vdocs.install:
	$(MAKE) -C vdocs install


build-dirs:
	-mkdir -p $(build_prefix) $(bindir) $(libdir) $(includedir) $(datadir) \
		$(build_prefix)/sync/ $(tmpdir)
ifndef WIN32
	-cd $(build_prefix) && ln -nfs lib lib64
endif
ifneq ($(libdir), $(shlibdir))
	-mkdir $(shlibdir)
endif
ifeq ($(OS),Darwin)
	-mkdir -p $(frameworkdir) $(build_prefix)/Library
	#-cd $(build_prefix)/Library && ln -nfs ../Frameworks .
endif

build-app-dirs:
	-mkdir -p $(app_prefix) $(app_bindir) $(app_datadir)
ifneq ($(OS),Windows)
	-mkdir -p $(app_libdir) $(app_includedir)
endif
ifeq ($(OS),Darwin)
	-mkdir -p $(app_prefix)/MacOS $(app_prefix)/Resources \
		$(app_frameworkdir)
else
	-mkdir -p $(APP_PYSITEDIR)
endif

distclean: clean
	-rm -rf build
	-$(MAKE) -C src clean
	-$(MAKE) -C docs clean
	-$(MAKE) -C vdocs clean
	-rm -rf prereqs/prebuilt-*.tar.bz2

reallyclean:
	rm -rf $$(git status --short --ignored --porcelain=v1 | sed -e '/^!!/!d' -e 's/^!! //')
	# for linux:
	rm -rf .cache .config

clean:
	-rm -rf $(APP_FILENAME)
	-rm -rf $(build_prefix)

build-from-scratch: distclean
ifdef INSTALL_RBVI
	$(MAKE) install-rbvi
else
	$(MAKE) install
endif

# Linux debugging:

gltrace:
	rm -f $(APP_NAME).trace
	apitrace trace $(app_bindir)/$(APP_NAME) $(DATA_FILE)

dumptrace:
	@apitrace dump $(APP_NAME).trace

SNAPSHOT_TAG = develop

# create a source snapshot
snapshot:
ifeq (,$(SNAPSHOT_DIR))
	$(error set SNAPSHOT_DIR on command line)
endif
	mkdir $(SNAPSHOT_DIR)
	echo "branch: $(SNAPSHOT_TAG)" > $(SNAPSHOT_DIR)/last-commit
	git show --summary --date=iso --pretty=fuller $(SNAPSHOT_TAG) >> $(SNAPSHOT_DIR)/last-commit
	git archive $(SNAPSHOT_TAG) | tar -C $(SNAPSHOT_DIR) -xf -
