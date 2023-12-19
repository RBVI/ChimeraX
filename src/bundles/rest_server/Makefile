include ../Makefile.bundle

DATAFILES = src/static src/server.pem

build wheel install app-install:	$(DATAFILES)

src/server.pem:	ssl.conf
	# This creates a self-signed cert
	# https://stackoverflow.com/questions/10175812/how-to-create-a-self-signed-certificate-with-openssl
	# with both key and cert in the same file
	openssl req -new -x509 -keyout src/server.pem -out src/server.pem \
		-days 365 -nodes -config ssl.conf

clean:
	-rm -rf src/server.pem
