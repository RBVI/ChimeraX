FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

PIP	= $(PYTHON_EXE) -m pip --log pip.log
APP_PIP	= $(APP_PYTHON_EXE) -m pip --log pip.log

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

PATCHES	= matplotlib-1.4.3-wxphoenix.patch

all:

install: $(SOURCE)
	LANG=$(LANG) $(PIP) install -r build_requirements.txt
	for p in $(PATCHES); do \
		(cd $(PYSITEDIR) && patch -N -p0) < $$p ; \
	done

list:
	$(PIP) list

upgrade:
	$(PIP) install -U -r build_requirements.txt

app-install:
	$(APP_PIP) install -r app_requirements.txt
	for p in $(PATCHES); do \
		(cd $(APP_PYSITEDIR) && patch -N -p0) < $$p ; \
	done

clean:
	rm -f pip.log

outdated:
	$(PIP) list --allow-external wxpython-phoenix --outdated --pre

