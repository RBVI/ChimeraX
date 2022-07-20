#/usr/bin/ev bash
ROOT=$(dirname $(dirname -- $0))

UBUNTU_VER=$(cat /etc/os-release | grep VERSION_ID | cut -d= -f2 | sed -e s/\"//g)

case $UBUNTU_VER in
	'18.04')
		PREREQ_FILE="${ROOT}/utils/ubuntu/1804.txt"
		;;
	'20.04')
		PREREQ_FILE="${ROOT}/utils/ubuntu/2004.txt"
		;;
	'22.04')
		PREREQ_FILE="${ROOT}/utils/ubuntu/2204.txt"
		;;
esac

mapfile -t packages < "$PREREQ_FILE" ; sudo apt-get install -y "${packages[@]}"
