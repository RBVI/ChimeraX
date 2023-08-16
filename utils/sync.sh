#!/usr/bin/env bash
STASHED=0
read -r -p "Stash changes before proceeding? [Y/n]" res
case $res in
  [Yy]*)
    STASH_OUTPUT="$(git stash)"
    if [ "${STASH_OUTPUT}" = "No local changes to save" ]
    then
      STASHED=0
    else
      STASHED=1
    fi
    ;;
  *);;
esac

PULL_OUTPUT="$(git pull)"
if [ "${PULL_OUTPUT}" = "Already up to date." ]
then
  echo "${PULL_OUTPUT}"
  exit 0;
fi
BUNDLES=$(echo "${PULL_OUTPUT}" | grep "src/bundles" | trim | sed -e )
for bundle in $BUNDLES
do
  make -C src/bundles/"${bundle}" install
done
case $STASHED in
  1)
    git stash pop;;
esac
