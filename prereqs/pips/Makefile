FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Force LANG to UTF-8 so that grako will install successfully
LANG	= en_US.UTF-8

all:

install: $(SOURCE)
	LANG=$(LANG) $(PYTHON_EXE) -m pip install -r build_requirements.txt

app-install:
	LANG=$(LANG) $(APP_PYTHON_EXE) -m pip install -r app_requirements.txt

clean:
