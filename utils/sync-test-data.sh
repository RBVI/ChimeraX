#!/usr/bin/env bash

while getopts ub: flag; do
	case "${flag}" in
	b) BUNDLE=${OPTARG} ;;
  u) UPDATE=true;;
	*)
		echo "Usage: $0 [-b <bundle>] [-u]"
		exit 1
		;;
	esac
done

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

if [ -n "${BUNDLE}" ]; then

	BUNDLEDIR="${ROOT}/src/bundles/${BUNDLE}"

	if [ ! -e "${BUNDLEDIR}" ]; then
		echo "Directory ${BUNDLEDIR} doesn't exist! Did you misspell the bundle name?"
		exit 1
	fi

	DATADIR="${ROOT}/src/bundles/${BUNDLE}/tests/data"

  WGET_ARGS=" -np -nH -r --cut-dirs=4 --reject="index.html*" --reject="robots.txt" "

  if [ "${UPDATE}" = true ]; then
    # Add the 'no clobber' flag to skip previously downloaded files
    WGET_ARGS="-nc "${WGET_ARGS}""
  fi

	mkdir -p "${DATADIR}"
	wget ${WGET_ARGS} https://rbvi.ucsf.edu/chimerax/data/test-data/"${BUNDLE}"/ -P "${DATADIR}"

	exit 0
fi
