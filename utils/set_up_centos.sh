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

dnf install -y epel-release

# TODO: drop --exclude='mesa-lib*' once Rocky publishes a mesa-libOSMesa in CRB
# matching the mesa version in AppStream. As of Nov 2025, AppStream shipped
# mesa-libGL-25.2.7-4.el9 while CRB's mesa-libOSMesa was still at 25.0.7-3.el9_7,
# so updating mesa makes the libOSMesa install in the prereq list unresolvable.
case $CENTOS_VER in
	'8')
		dnf config-manager --set-enabled powertools
		dnf update -y --exclude='mesa-lib*'
		dnf -y --setopt=exclude='*.i?86' group install "Development Tools"
		PREREQ_FILE="${ROOT}/utils/centos/8.txt"
		;;
	'9')
		dnf install 'dnf-command(config-manager)'
		/usr/bin/crb enable
		dnf update -y --exclude='mesa-lib*'
		dnf -y --setopt=exclude='*.i?86' group install "Development Tools"
		PREREQ_FILE="${ROOT}/utils/centos/9.txt"
		;;
	'10')
		dnf install 'dnf-command(config-manager)'
		/usr/bin/crb enable
		dnf update -y --exclude='mesa-lib*'
		dnf -y --setopt=exclude='*.i?86' group install "Development Tools"
		PREREQ_FILE="${ROOT}/utils/centos/10.txt"
		;;
esac

mapfile -t packages < "$PREREQ_FILE" ; dnf --setopt=exclude='*.i?86' install -y "${packages[@]}"
