TOP = ..
TOP := $(shell (cd "$(TOP)"; pwd))
include $(TOP)/mk/config.make
PYTHON = ../ChimeraX.app/Contents/bin/python3.9

BUNDLE ?=

wheel: clean chimerax-copy pyproject.toml
	touch chimerax/__init__.py
	rm -rf chimerax/ui
	python -m build

pyproject.toml: ensure-chimerax pyproject.toml.in
	$(APP_PYTHON_BIN) -I filter_deps.py

chimerax-link: ensure-chimerax
	ln -s $(APP_PYSITEDIR)/chimerax .

chimerax-copy: ensure-chimerax
	cp -r $(APP_PYSITEDIR)/chimerax .

copy-bundle: ensure-chimerax
	cp -r $(APP_PYSITEDIR)/chimerax/${BUNDLE} chimerax/${BUNDLE}

bundle-diff:
	diff -r chimerax/${BUNDLE} ../src/bundles/${BUNDLE}/src

ensure-chimerax:
	@[ -e $(TOP)/$(APP_FILENAME) ] || (echo \
		"ChimeraX not found; run 'make install' first before building the wheel." && exit 1)

chimerax-diff:
	diff chimerax $(APP_PYSITEDIR)/chimerax 

clean:
	\rm -rf dist build chimerax.egg-info htmlcov chimerax

test:
	python -m coverage run -m pytest --rootdir=. tests/core
	python -m coverage run -m pytest --rootdir=. --ignore=tests/core

.PHONY: venv
venv:
	if [ -x $(APP_PYTHON_BIN) ] && [ ! -x .venv ]; then \
		$(APP_PYTHON_BIN) -m venv .venv; \
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
