PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make
include ../pips/Makefile.pip

NAME = cxservices
VERSION = 1.1
SOURCE = client
PIPINSTALL_ARGS = --upgrade-strategy only-if-needed

# From https://mvnrepository.com/artifact/io.swagger.codegen.v3/swagger-codegen-cli
SWAGGER_VERSION=3.0.27
SWAGGER_CODEGEN = swagger-codegen-cli-$(SWAGGER_VERSION).jar

RBVI_SERVER := http:\/\/webservices.rbvi.ucsf.edu\/cxservices\/api\/v1\/
RBVI_DESC := Production server

LOCAL_SERVER := http:\/\/localhost:8000\/
LOCAL_DESC := Local test server

ifdef WIN32
WHEEL = `cygpath -m $(SOURCE)/dist/*.whl`
else
WHEEL = $(SOURCE)/dist/*.whl
endif

.PHONY: app-install app-reinstall cxservices.config cxservices.yml production local-test

app-install: $(SOURCE)
	cd $(SOURCE) && $(APP_PYTHON_EXE) setup.py $(SETUP_ARGS) bdist_wheel
	$(APP_PYTHON_EXE) -m pip install $(PIPINSTALL_ARGS) $(WHEEL)

app-reinstall: $(SOURCE)
	cd $(SOURCE) && $(APP_PYTHON_EXE) setup.py $(SETUP_ARGS) bdist_wheel
	$(APP_PYTHON_EXE) -m pip install $(PIPINSTALL_ARGS) $(WHEEL) --force-reinstall

clean:
	rm -rf $(SOURCE)/build

confclean:
	-rm -f cxservices.config
	-rm -f cxservices.yml

distclean:	clean
	rm -rf $(SOURCE)/dist $(SOURCE)/$(NAME).egg-info

local-test: distclean confclean
	LOCAL_ENV=1 $(MAKE) $(SOURCE)
	$(MAKE) app-reinstall

production: distclean confclean
	$(MAKE) $(SOURCE)
	$(MAKE) app-reinstall

# Must have Java installed and on PATH
$(SOURCE): $(SWAGGER_CODEGEN) cxservices.yml cxservices.config cxservices.patch
	-rm -rf $(SOURCE)
	java -jar $(SWAGGER_CODEGEN) generate \
		-DpackageName=$(NAME) \
		--input-spec cxservices.yml \
		--config cxservices.config \
		--lang python \
		--output $(SOURCE)
	cd $(SOURCE) && patch -p1 < ../cxservices.patch

cxservices.config: cxservices.config.in
	sed -e 's/VERSION/$(VERSION)/' \
	    -e 's/PACKAGE/$(NAME)/' cxservices.config.in > cxservices.config

cxservices.yml: cxservices.yml.in
ifdef LOCAL_ENV
	sed -e 's/SERVER_URL/$(LOCAL_SERVER)/' \
	    -e 's/SERVER_DESC/$(LOCAL_DESC)/' $< > $@
else
	sed -e 's/SERVER_URL/$(RBVI_SERVER)/' \
	    -e 's/SERVER_DESC/$(RBVI_DESC)/' $< > $@
endif

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
