TOP = ..
TOP := $(shell (cd "$(TOP)"; pwd))
include $(TOP)/mk/config.make
PYTHON = ../ChimeraX.app/Contents/bin/python3.9

BUNDLE ?=

wheel: chimerax
	touch $(APP_PYSITEDIR)/chimerax/__init__.py
	mv $(APP_PYSITEDIR)/chimerax/ui $(APP_PYSITEDIR)/chimerax_ui
	python -m build
	mv $(APP_PYSITEDIR)/chimerax_ui $(APP_PYSITEDIR)/chimerax/ui
	rm $(APP_PYSITEDIR)/chimerax/__init__.py

chimerax:
	ln -s $(APP_PYSITEDIR)/chimerax .

chimerax-copy:
	cp -r $(APP_PYSITEDIR)/chimerax .

copy-bundle:
	cp -r $(APP_PYSITEDIR)/chimerax/${BUNDLE} chimerax/${BUNDLE}

bundle-diff:
	diff -r chimerax/${BUNDLE} ../src/bundles/${BUNDLE}/src

clean:
	\rm -rf dist build chimerax.egg-info
	\rm chimerax

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
