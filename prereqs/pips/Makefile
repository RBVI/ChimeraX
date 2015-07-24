FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

PIP	= $(PYTHON_EXE) -m pip --log pip.log
APP_PIP	= $(APP_PYTHON_EXE) -m pip --log pip.log

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

MPL_VERSION	= 1.4.3
MPL_PATCHES	= matplotlib-$(MPL_VERSION)-wxphoenix.patch

all:

install:	$(PYSITEDIR)/matplotlib
	LANG=$(LANG) $(PIP) install -r build_requirements.txt

$(PYSITEDIR)/matplotlib:
	LANG=$(LANG) $(PIP) install matplotlib==$(MPL_VERSION)
	-for p in $(MPL_PATCHES); do \
		patch -d $(PYSITEDIR) -N -p0 < $$p ; \
	done

list:
	$(PIP) list

upgrade:
	$(PIP) install -U -r build_requirements.txt

app-install:	$(APP_PYSITEDIR)/matplotlib
	$(APP_PIP) install -r app_requirements.txt

$(APP_PYSITEDIR)/matplotlib:
	LANG=$(LANG) $(APP_PIP) install matplotlib==$(MPL_VERSION)
	-for p in $(MPL_PATCHES); do \
		patch -d $(APP_PYSITEDIR) -N -p0 < $$p ; \
	done

clean:
	rm -f pip.log

outdated:
	$(PIP) list --allow-external wxpython-phoenix --outdated --pre

