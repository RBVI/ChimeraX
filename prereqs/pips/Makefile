FOREIGN_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

all:

install: $(SOURCE)
	$(PYTHON_EXE) -m pip install -r build_requirements.txt

app-install:
	$(APP_PYTHON_EXE) -m pip install -r app_requirements.txt

clean:
