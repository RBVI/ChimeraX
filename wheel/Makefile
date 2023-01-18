TOP = ..
TOP := $(shell (cd "$(TOP)"; pwd))
include $(TOP)/mk/config.make
PYTHON = ../ChimeraX.app/Contents/bin/python3.9

BUNDLE ?=

wheel: chimerax
	$(PYTHON) setup.py bdist_wheel

chimerax:
	ln -s ../ChimeraX.app/Contents/lib/python3.9/site-packages/chimerax .

chimerax-copy:
	cp -r ../ChimeraX.app/Contents/lib/python3.9/site-packages/chimerax .

copy-bundle:
	cp -r ../ChimeraX.app/Contents/lib/python3.9/site-packages/chimerax/${BUNDLE} chimerax/${BUNDLE}

bundle-diff:
	diff -r chimerax/${BUNDLE} ../src/bundles/${BUNDLE}/src

clean:
	\rm -rf dist build chimerax.egg-info
	\rm chimerax

test:
	python -m pytest --rootdir=.

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
