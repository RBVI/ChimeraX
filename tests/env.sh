#!/usr/bin/env bash

# Run Pytest twice on this ONE file to test whether chimerax.core.__main__.init()
# runs the same whether it's called by running the ChimeraX binary or by
# running ChimeraX.app/bin/python -I -m chimerax.core

echo "Running Pytest on tests/test_env.py (ChimeraX)"
./ChimeraX.app/Contents/bin/ChimeraX -I -m pytest tests/test_env.py

echo "Running Pytest on tests/test_env.py (Python)"
./ChimeraX.app/Contents/bin/python3.11 -I -m pytest tests/test_env.py
