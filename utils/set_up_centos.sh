#/usr/bin/ev bash
ROOT=$(dirname $(dirname -- $0))

CENTOS_VER=$(cat /etc/centos-release | cut -d ' ' -f4 | cut -d. -f1)

case $CENTOS_VER in
	'8')
		PREREQ_FILE="${ROOT}/utils/centos/8.txt"
		;;
	'9')
		PREREQ_FILE="${ROOT}/utils/centos/9.txt"
		;;
esac

mapfile -t packages < "$PREREQ_FILE" ; dnf install -y "${packages[@]}"
