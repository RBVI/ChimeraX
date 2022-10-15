#!/usr/bin/env bash
ROOT=$(dirname $(dirname -- $0))

cd ${ROOT}

mapfile -t BUNDLES < <(find src/bundles -maxdepth 1 -type d | sed -n '1d;p' | sort)

mkdir ${ROOT}/docs/devel/chimerax || true

cd ${ROOT}/docs/devel/chimerax 

for bundle in "${BUNDLES[@]}"; do 
	readarray -d/ -t bundle_fields <<< "${bundle}"
	bundle_name=${bundle_fields[2]%$'\n'}
	ln -s ../../../${bundle%$'\n'}/src ${bundle_name};
done
