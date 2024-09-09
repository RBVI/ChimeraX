#!/usr/bin/env bash

while getopts sb: flag; do
	case "${flag}" in
	b) BUNDLE=${OPTARG} ;;
	*)
		echo "Usage: $0 [-b <bundle>]"
		exit 1
		;;
	esac
done

if [ -z "${BUNDLE}" ] && [ -z "${SESSION_DATA}" ]; then
	echo "Usage: $0 [-b <bundle>]"
	exit 1
fi

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

if [ -n "${BUNDLE}" ]; then

	BUNDLEDIR="${ROOT}/src/bundles/${BUNDLE}"

	if [ ! -e "${BUNDLEDIR}" ]; then
		echo "Directory ${BUNDLEDIR} doesn't exist! Did you misspell the bundle name?"
		exit 1
	fi

	DATADIR="${ROOT}/src/bundles/${BUNDLE}/tests/data"

	mkdir -p "${DATADIR}"
	wget -np -nH -r --cut-dirs=4 --reject="index.html*" --reject="robots.txt" https://rbvi.ucsf.edu/chimerax/data/test-data/"${BUNDLE}"/ -P "${DATADIR}"

	exit 0
fi
