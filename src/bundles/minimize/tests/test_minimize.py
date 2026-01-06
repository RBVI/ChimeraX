# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.minimize.cmd import _minimize
import pytest


def test_minimize_without_gaff_type(test_production_session):
    """Test that minimize gives a clear error (not AttributeError) when gaff_type is missing.
    
    This tests the fix for AttributeError: 'Atom' object has no attribute 'gaff_type'
    when minimize is run without first running addcharge. The fix ensures we get a
    UserError with a helpful message instead of an AttributeError.
    """
    
    session = test_production_session
    # Open a small structure (alanine dipeptide or similar)
    # Using a PDB ID that should be available
    run(session, "open 2gbp")
    structure = session.models[0]
    
    # Verify structure is loaded
    assert structure is not None, "Failed to open structure"
    assert structure.num_atoms > 0, "Structure has no atoms"
    
    # Verify atoms don't have gaff_type assigned (they shouldn't unless addcharge was run)
    # Note: gaff_type attribute may be registered but set to None
    atoms_with_gaff = [a for a in structure.atoms 
                       if hasattr(a, 'gaff_type') and getattr(a, 'gaff_type', None) is not None]
    # Most atoms should not have gaff_type assigned initially
    # Allow for some edge cases, but most should not have it
    assert len(atoms_with_gaff) < len(structure.atoms) * 0.1, \
        "Too many atoms have gaff_type before addcharge - test may be invalid"
    
    # Add hydrogens first (minimize may need them)
    run(session, "addh #1")
    
    # Run minimize - should raise UserError (not AttributeError) when gaff_type is missing
    try:
        run(session, "minimize #1 dockPrep false maxSteps 10 liveUpdates false")
        assert False, "Expected UserError when gaff_type is missing"
    except UserError as e:
        # Verify it's a helpful error message, not an AttributeError
        assert "gaff_type" in str(e) or "addcharge" in str(e).lower(), \
            f"Error message should mention gaff_type or addcharge, got: {e}"
    except AttributeError as e:
        if "'Atom' object has no attribute 'gaff_type'" in str(e):
            raise AssertionError("AttributeError still occurs - fix may not be working") from e
        raise


def test_minimize_with_gaff_type(test_production_session):
    """Test that minimize works when atoms have gaff_type assigned (normal case)."""
    
    session = test_production_session
    # Open a structure
    run(session, "open 2gbp")
    structure = session.models[0]
    
    # Add hydrogens first (required for addcharge)
    run(session, "addh #1")
    
    # Add charges (which also assigns gaff_type)
    # Note: addcharge may prompt for adding hydrogens, but we already added them
    # The command should work without interactive prompts since hydrogens are present
    try:
        run(session, "addcharge #1")
    except Exception:
        # If addcharge fails (e.g., due to interactive prompt), skip this test
        # The important test is that minimize works when gaff_type IS present
        pytest.skip("addcharge requires interactive input or failed")
    
    # Verify atoms now have gaff_type
    atoms_with_gaff = [a for a in structure.atoms if hasattr(a, 'gaff_type') and getattr(a, 'gaff_type', None) is not None]
    assert len(atoms_with_gaff) > 0, "Expected atoms with gaff_type after addcharge"
    
    # Run minimize with dockPrep false and a small number of steps
    run(session, "minimize #1 dockPrep false maxSteps 10 liveUpdates false")
    
    # Verify the structure still exists
    assert structure.num_atoms > 0, "Structure lost atoms after minimization"


def test_minimize_gaff_type_error_message(test_production_session):
    """Test that _minimize raises UserError (not AttributeError) when gaff_type is missing.
    
    This directly tests the code path where gaff_type is missing to ensure
    we get a helpful error message instead of an AttributeError.
    """
    
    session = test_production_session
    # Open a small structure
    run(session, "open 2gbp")
    structure = session.models[0]
    
    # Verify no atoms have gaff_type assigned (attribute may exist but be None)
    for atom in structure.atoms[:10]:  # Check first 10 atoms
        gaff_type = getattr(atom, 'gaff_type', None)
        assert gaff_type is None, \
            f"Atom {atom} should not have gaff_type before addcharge, but has {gaff_type}"
    
    # Add hydrogens first (minimize may need them)
    run(session, "addh #1")
    
    # Run minimize directly (bypassing dockPrep) with minimal steps
    # This should raise UserError (not AttributeError) when gaff_type is missing
    try:
        _minimize(session, structure, live_updates=False, log_energy=False, max_steps=5)
        assert False, "Expected UserError when gaff_type is missing"
    except UserError as e:
        # Verify it's a helpful error message
        assert "gaff_type" in str(e) or "addcharge" in str(e).lower(), \
            f"Error message should mention gaff_type or addcharge, got: {e}"
    except AttributeError as e:
        if "'Atom' object has no attribute 'gaff_type'" in str(e) or \
           "'Atom' object has no attribute 'charge'" in str(e):
            raise AssertionError("AttributeError still occurs - fix may not be working") from e
        raise

