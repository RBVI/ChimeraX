include ../Makefile.bundle

DATAFILES = src/server.pem

build wheel install app-install:	$(DATAFILES)

src/server.pem:	ssl.conf
	# This creates a self-signed cert
	# https://stackoverflow.com/questions/10175812/how-to-create-a-self-signed-certificate-with-openssl
	openssl req -new -x509 -keyout src/server.pem -out src/server.pem \
		-days 365 -nodes -config ssl.conf
