PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

NAME = cxservices
VERSION = 1.0
SOURCE = client
PIPINSTALL_ARGS = --upgrade-strategy only-if-needed

ifdef WIN32
WHEEL = `cygpath -m $(SOURCE)/dist/*.whl`
else
WHEEL = $(SOURCE)/dist/*.whl
endif

all:

install:

app-install: install
	cd $(SOURCE) && $(APP_PYTHON_EXE) setup.py $(SETUP_ARGS) bdist_wheel
	$(APP_PYTHON_EXE) -m pip install $(PIPINSTALL_ARGS) $(WHEEL)

clean:
	rm -rf $(SOURCE)/build
	
distclean:	clean
	rm -rf $(SOURCE)/dist $(SOURCE)/$(NAME).egg-info

# Must have Java installed and on PATH

$(SOURCE):	cxservices.yml cxservices.config cxservices.patch
	rm -rf $(SOURCE)
	java -jar swagger-codegen-cli.jar generate \
		-DpackageName=$(NAME) \
		--input-spec cxservices.yml \
		--config cxservices.config \
		--lang python \
		--output $(SOURCE)
	cd $(SOURCE) && patch -p1 < ../cxservices.patch

cxservices.config:	cxservices.config.in
	sed -e 's/VERSION/$(VERSION)/' \
	    -e 's/PACKAGE/$(NAME)/' cxservices.config.in > cxservices.config

# "-DpackageName=cxservices_client" is needed to set the package name
# because "packageName" in cxservices.config does nothing

#
# Useful commands I can't remember
#

help:
	java -jar swagger-codegen-cli.jar -h

meta:
	java -jar swagger-codegen-cli.jar meta

langs:
	java -jar swagger-codegen-cli.jar langs

config:
	java -jar swagger-codegen-cli.jar config-help -l python
