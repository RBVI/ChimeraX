PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

NAME = cxservices
VERSION = 1.0
SOURCE = client
PIPINSTALL_ARGS = --upgrade-strategy only-if-needed

SWAGGER_CODEGEN = swagger-codegen-cli.jar

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

$(SOURCE): $(SWAGGER_CODEGEN) cxservices.yml cxservices.config cxservices.patch
	rm -rf $(SOURCE)
	java -jar $(SWAGGER_CODEGEN) generate \
		-DpackageName=$(NAME) \
		--input-spec cxservices.yml \
		--config cxservices.config \
		--lang python \
		--output $(SOURCE)
	cd $(SOURCE) && patch -p1 < ../cxservices.patch

cxservices.config:	cxservices.config.in
	sed -e 's/VERSION/$(VERSION)/' \
	    -e 's/PACKAGE/$(NAME)/' cxservices.config.in > cxservices.config

# "-DpackageName=cxservices" is needed to set the package name
# because "packageName" in cxservices.config does nothing

#
# Useful commands I can't remember
#

help: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) -h

meta: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) meta

langs: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) langs

config: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) config-help -l python

$(SWAGGER_CODEGEN):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/$(SWAGGER_CODEGEN)
