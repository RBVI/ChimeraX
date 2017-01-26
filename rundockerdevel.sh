#!/bin/bash

# Will need current user's name so ssh can work
user=$(id -n -u)

# Docker image and container names
IMAGE_NAME="${user}-chimera-image"
CONTAINER_NAME="${user}-chimera"
BRANCH=develop

usage() {
	echo "usage: $0 [-b branch] [-i] [-s]"
	echo "  -b BRANCH  use given BRANCH (default 'develop')"
	echo "  -i         just initialize docker image only, '$IMAGE_NAME', for later use"
	echo "  -s         skip cloning chimerax repository"
	exit 2
}

while getopts b:is opt
do
	case $opt in
	b)
		BRANCH=$OPTARG
		;;
	i)
		INIT_ONLY=true
		;;
	s)
		SKIP_CLONE=true
		;;
	*)
		usage
		;;
	esac
done

if docker info > /dev/null 2>&1
then
	if [ -z "$INIT_ONLY" ]
	then
		echo "Building interactive docker setup for one-time use"
		echo ""
	else
		echo "Building docker image for later use"
		echo ""
	fi
else
	echo "Docker daemon needs to be running first"
	exit 1
fi

cleanup_on_exit() {
	if [ -n "$SETUP_DIR" ]
	then
		cd /
		echo ""
		echo "Removing temporary directory"
		rm -rf $SETUP_DIR
	fi
	if [ -n "$INIT_ONLY" ]
	then
		echo "Run 'docker run -it --name $CONTAINER_NAME $IMAGE_NAME bash' to build/debug chimera"
		echo "Remember to remove the container and image when done:"
		echo "  docker rm $CONTAINER_NAME"
		echo "  docker rmi $IMAGE_NAME"
		exit
	fi
	echo "Removing docker container:"
	docker rm $CONTAINER_NAME
	echo "Removing docker image:"
	docker rmi $IMAGE_NAME
}
trap cleanup_on_exit EXIT

SETUP_DIR=$(mktemp -d)
cd $SETUP_DIR

# need user's ssh credentials
echo "Copying ~/.ssh"
cp -rp ~/.ssh .

if [ -z "$SKIP_CLONE" ]
then
	# git clone in advance in case ssh credentials need a password
	echo "Checking out ChimeraX"
	git clone --depth 1 --single-branch --branch $BRANCH plato.cgl.ucsf.edu:/usr/local/projects/chimerax/git/chimerax.git
fi

cat > Dockerfile << EOF
# Use ChimeraX Linux development environment
FROM rbvi/chimerax-devel:1.0
# create root password so su will work, and create non-root user
RUN echo root:chimerax | chpasswd && useradd -m ${user}
WORKDIR /home/${user}
# copy build context into user's home directory
COPY . ./
# Everything that is copied is owned by root
RUN chown -R ${user} .
USER ${user}
CMD make install > build.out 2>&1
EOF

echo "Building docker image"
docker build -q --force-rm -f Dockerfile -t $IMAGE_NAME .

if [ -z "$INIT_ONLY" ]
then
	echo "Docker container and image will be removed when done.  So"
	echo "remember to commit results, and/or scp them back before exiting."
	echo ""
	docker run -it --name $CONTAINER_NAME $IMAGE_NAME bash
fi
