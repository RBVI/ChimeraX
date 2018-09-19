include ../Makefile.bundle

DATAFILES = src/static src/server.pem

wheel install app-install:	$(DATAFILES)

src/server.pem:	ssl.conf
	# This creates a self-signed cert
	# https://stackoverflow.com/questions/10175812/how-to-create-a-self-signed-certificate-with-openssl
	openssl req -new -x509 -keyout server.key -out src/server.pem \
		-days 365 -nodes -config ssl.conf
