#!/usr/bin/env bash

# Try not to run by hand.
ROOT=$(readlink -f $(dirname $(dirname $(dirname -- $0))))

cd ${ROOT}

# maxdepth 3 catches rotamer_libs/*/src but not md_crds/gromacs/xdrfile/src
while IFS=$'\n' read -r line; do
	BUNDLE_SRC_FOLDERS+=("$line")
done < <(find src/bundles -maxdepth 3 -type d -name "src" | sort)
while IFS=$'\n' read -r line; do
	BUNDLE_DEV_DOC_FOLDERS+=("$line")
done < <(find src/bundles/**/docs -maxdepth 3 -type d -name "devel" | sort)
while IFS=$'\n' read -r line; do
	BUNDLE_USR_DOC_FOLDERS+=("$line")
done < <(find src/bundles/**/docs -maxdepth 3 -type d -name "user" | sort)

PREFIX="devel"
if [[ $DEV_DOCS_IN_ROOT ]]; then
	# For readthedocs
	PREFIX=""
fi

if [[ ! $INTERNAL_CHIMERAX ]]; then
	mkdir ${ROOT}/docs/${PREFIX}/chimerax
	cd ${ROOT}/docs/${PREFIX}/chimerax
	for bundle in "${BUNDLE_SRC_FOLDERS[@]}"; do
		IFS=/ read -a bundle_fields <<<${bundle}
		bundle_name=${bundle_fields[2]%$'\n'}
		if [ $bundle_name == "rotamer_libs" ]; then
			bundle_name="$(tr [A-Z] [a-z] <<<${bundle_fields[3],,%'\n'}_rotamer_lib)"
		fi
		ln -s ${ROOT}/${bundle%$'\n'} ${bundle_name}
	done
fi

cd ${ROOT}
ln -s ${ROOT}/src/apps docs/${PREFIX}/

mkdir ${ROOT}/docs/${PREFIX}/modules
cd ${ROOT}/docs/${PREFIX}/modules
for bundle in "${BUNDLE_DEV_DOC_FOLDERS[@]}"; do
	IFS=/ read -a bundle_fields <<<"${bundle}"
	bundle_name=${bundle_fields[2]%$'\n'}
	ln -s ${ROOT}/${bundle%$'\n'} ${bundle_name}
done

# TODO: Figure out what to do about the user doc folders
# if [[ $DEV_DOCS_IN_ROOT ]]; then
#     cd ${ROOT}/docs
#     for bundle in "${BUNDLE_USR_DOC_FOLDERS[@]}"; do
#	      readarray -d/ -t bundle_fields <<< "${bundle}"
#	      bundle_name=${bundle_fields[2]%$'\n'}
#         ln -s ${ROOT}${bundle%$'\n'} ${bundle_name};
#     done
# fi
