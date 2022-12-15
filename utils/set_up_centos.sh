#/usr/bin/ev bash
ROOT=$(dirname $(dirname -- $0))

if [ -f /etc/centos-release ]; then
	CENTOS_VER=$(cat /etc/centos-release | cut -d ' ' -f4 | cut -d. -f1)
elif [ -f /etc/rocky-release ]; then
	CENTOS_VER=$(cat /etc/rocky-release | cut -d ' ' -f4 | cut -d. -f1)
else
	echo "Not on CentOS 8, CentOS 9, Rocky 8, or Rocky 9; try another script."
	exit 1
fi

dnf install -y git-all
dnf install -y epel-release

case $CENTOS_VER in
	'8')
		dnf config-manager --set-enabled powertools
		dnf update -y
		PREREQ_FILE="${ROOT}/utils/centos/8.txt"
		;;
	'9')
		PREREQ_FILE="${ROOT}/utils/centos/9.txt"
		;;
esac

mapfile -t packages < "$PREREQ_FILE" ; dnf install -y "${packages[@]}"
