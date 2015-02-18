FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

PIP	= $(PYTHON_EXE) -m pip --log pip.log
APP_PIP	= $(APP_PYTHON_EXE) -m pip --log pip.log

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

all:

install: $(SOURCE)
	LANG=$(LANG) $(PIP) install -r build_requirements.txt

list:
	$(PIP) list

upgrade:
	$(PIP) install -U -r build_requirements.txt

app-install:
	LANG=$(LANG) $(APP_PIP) install -r app_requirements.txt

clean:
	rm -f pip.log

outdated:
	$(PIP) list --allow-external wxpython-phoenix --outdated

