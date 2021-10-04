TOP	= ../../..
include $(TOP)/mk/config.make

.PHONY: cxservices.yml cxservices.config app-install app-reinstall

# To test changes on a local server:
# LOCAL_ENV=1 make app-reinstall
# to switch back to the remote server:
# make app-reinstall

# Information and targets necessary to build and upload cxservices_client
NAME = cxservices
SOURCE = client
API_VERSION = 1
PIPINSTALL_ARGS = --upgrade-strategy only-if-needed
# From https://mvnrepository.com/artifact/io.swagger.codegen.v3/swagger-codegen-cli
SWAGGER_VERSION=3.0.27
SWAGGER_CODEGEN = swagger-codegen-cli-$(SWAGGER_VERSION).jar

REMOTE_URL=http:\/\/webservices.rbvi.ucsf.edu\/cxservices\/api\/v$(API_VERSION)/
REMOTE_DESC=Production server

LOCAL_URL=http:\/\/localhost:8000\/
LOCAL_DESC=Local test server

# Information and targets necessary to install cxservices_client
BUNDLE_VERSION = 1.1

ifdef WIN32
WHEEL = `cygpath -m $(SOURCE)/dist/*.whl`
else
WHEEL = $(SOURCE)/dist/*.whl
endif

app-install: 
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/
	cd $(SOURCE) && $(APP_PYTHON_EXE) setup.py $(SETUP_ARGS) bdist_wheel
	$(APP_PYTHON_EXE) -m pip install $(PIPINSTALL_ARGS) $(WHEEL)

app-reinstall: $(SOURCE)
	cd $(SOURCE) && $(APP_PYTHON_EXE) setup.py $(SETUP_ARGS) bdist_wheel
	$(APP_PYTHON_EXE) -m pip install $(PIPINSTALL_ARGS) $(WHEEL) --force-reinstall

clean:
	rm -rf $(SOURCE)/build

distclean:	clean
	rm -rf $(SOURCE)/dist $(SOURCE)/$(NAME).egg-info

# Must have Java installed and on PATH
# "-DpackageName=cxservices" is needed to set the package name
# because "packageName" in cxservices.config does nothing
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
	sed -e 's/VERSION/$(BUNDLE_VERSION)/' \
	    -e 's/PACKAGE/$(NAME)/' cxservices.config.in > cxservices.config

cxservices.yml: cxservices.yml.in
ifdef LOCAL_ENV
	sed -e 's/SERVER_URL/$(REMOTE_URL)/' \
	    -e 's/SERVER_DESC/$(REMOTE_DESC)/' cxservices.yml.in > cxservices.yml
else
	sed -e 's/SERVER_URL/$(LOCAL_URL)/' \
	    -e 's/SERVER_DESC/$(LOCAL_DESC)/' cxservices.yml.in > cxservices.yml
endif

$(SWAGGER_CODEGEN):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/cxservices/swagger/$(SWAGGER_CODEGEN)

upload_new_version:
	$(RSYNC) $(SOURCE)/dist/.whl $(PREREQS_UPLOAD)/cxservices/

#
# Other Useful commands
#
help: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) -h

meta: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) meta

langs: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) langs

config: $(SWAGGER_CODEGEN)
	java -jar $(SWAGGER_CODEGEN) config-help -l python
