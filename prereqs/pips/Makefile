PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

include Makefile.pip

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

all:

install:
	LANG=$(LANG) $(PIP) install -r build_requirements.txt

list:
	$(PIP) list

upgrade:
	$(PIP) install -U -r build_requirements.txt

app-install:
	$(APP_PIP) install -r app_requirements.txt

app-upgrade:
	$(APP_PIP) install -U -r app_requirements.txt

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
	rm -f pip.log

outdated:
	$(PIP) list --outdated --format=columns

app-outdated:
	$(APP_PIP) list --outdated --format=columns

src-outdated:
	$(PYTHON_EXE) -m package_checker ../../src
