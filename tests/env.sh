#!/usr/bin/env bash

# Run Pytest twice on this ONE file to test whether chimerax.core.__main__.init()
# runs the same whether it's called by running the ChimeraX binary or by
# running ChimeraX.app/bin/python -I -m chimerax.core

while getopts csf flag; do
  case "${flag}" in
  c) COVERAGE=true ;;
  s) COV_SILENT=true ;;
  f) FLATPAK=true ;;
  esac
done

if [ "$COVERAGE" = true ]; then
  COVERAGE_ARGS="--cov=chimerax"
  if [ "$COV_SILENT" = true ]; then
    COVERAGE_ARGS="${COVERAGE_ARGS} --cov-report="
  fi
else
  COVERAGE_ARGS=""
fi

CHIMERAX_PYTHON_BIN=
CHIMERAX_BIN=

case $OSTYPE in
linux-gnu)
  if [ "$FLATPAK" = true ]; then
    CHIMERAX_PYTHON_BIN=$(ls /app/bin/python3.1*)
    CHIMERAX_BIN=$(ls /app/bin/ChimeraX)
  else
    CHIMERAX_PYTHON_BIN=$(ls ./ChimeraX.app/bin/python3.1*)
    CHIMERAX_BIN=./ChimeraX.app/bin/ChimeraX
  fi
  ;;
msys | cygwin)
  CHIMERAX_PYTHON_BIN=./ChimeraX.app/bin/python.exe
  CHIMERAX_BIN=./ChimeraX.app/bin/ChimeraX-console.exe
  ;;
darwin*)
  CHIMERAX_PYTHON_BIN=$(ls ./ChimeraX.app/Contents/bin/python3.1*)
  CHIMERAX_BIN=./ChimeraX.app/Contents/bin/ChimeraX
  ;;
esac

echo "OSTYPE: ${OSTYPE}"
echo "CHIMERAX_PYTHON_BIN: ${CHIMERAX_PYTHON_BIN}"

if [ ! -e "${CHIMERAX_PYTHON_BIN}" ]; then
  echo "Looked for ChimeraX Python at ${CHIMERAX_PYTHON_BIN}"
  echo "No ChimeraX Python binary found" && exit 1
fi
if [ ! -e "${CHIMERAX_BIN}" ]; then
  echo "Looked for ChimeraX binary at ${CHIMERAX_BIN}"
  echo "No ChimeraX binary found" && exit 1
fi

echo "Running Pytest on tests/test_env.py (ChimeraX)"
${CHIMERAX_BIN} -I -m pytest tests/test_env.py ${COVERAGE_ARGS}

echo "Running Pytest on tests/test_env.py (Python)"
${CHIMERAX_PYTHON_BIN} -I -m pytest tests/test_env.py ${COVERAGE_ARGS}
