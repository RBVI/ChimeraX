# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Tests for the display_state command in the list_info bundle."""

import pytest
import json


# Commands to set up test structures
setup_structure = [
    "open 2tpk autostyle false",  # Small structure, ~400 residues
]


def test_display_state_basic(test_production_session):
    """Test that display_state returns valid JSON with expected structure."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Run display_state and get JSON result
    result = run(test_production_session, "info display_state")
    
    # Verify we got a JSONResult
    assert result is not None
    
    # Parse the JSON
    data = json.loads(result.json_result)
    
    # Should be a list of models
    assert isinstance(data, list)
    assert len(data) >= 1
    
    # Check first model has expected keys
    model = data[0]
    assert 'id' in model
    assert 'name' in model
    assert 'type' in model
    
    # Close all models
    run(test_production_session, "close")


def test_display_state_hidden_model(test_production_session):
    """Test that hidden models have minimal output with 'hidden' key."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide the model
    run(test_production_session, "hide #1 target m")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    
    model = data[0]
    assert model.get('hidden') == True, "Hidden model should have hidden=True"
    
    # Hidden model should NOT have detailed keys
    assert 'chains' not in model, "Hidden model should not have chains key"
    assert 'ligands' not in model, "Hidden model should not have ligands key"
    assert 'ions' not in model, "Hidden model should not have ions key"
    
    run(test_production_session, "close")


def test_display_state_shown_model_structure(test_production_session):
    """Test display_state output structure for a visible AtomicStructure."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Make sure it's visible
    run(test_production_session, "show #1 target m")
    
    # Show ribbons for all chains
    run(test_production_session, "show #1 target c")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    
    # Find the AtomicStructure model
    model = data[0]
    assert model.get('type') in ('AtomicStructure', 'Structure')
    assert 'hidden' not in model, "Visible model should not have hidden key"
    
    # Should have chains with ribbons displayed
    assert 'chains' in model, "Visible model with ribbons should have chains"
    
    for chain in model['chains']:
        assert 'id' in chain
        assert 'polymer_type' in chain
        # Should have ribbons since we showed them
        assert 'ribbons' in chain, f"Chain {chain['id']} should have ribbons"
        assert 'spec' in chain['ribbons'], "Ribbons should have spec"
    
    run(test_production_session, "close")


def test_display_state_atoms_only(test_production_session):
    """Test that chains only appear when something is displayed."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide everything
    run(test_production_session, "hide #1 target abcs")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # Should not have chains since nothing is displayed
    assert 'chains' not in model or len(model.get('chains', [])) == 0, \
        "No chains should appear when nothing is displayed"
    
    # Now show atoms for chain A only
    run(test_production_session, "show #1/A target a")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # Should now have chain A with atoms
    assert 'chains' in model
    chain_ids = [c['id'] for c in model['chains']]
    assert 'A' in chain_ids, "Chain A should be in output"
    
    chain_a = [c for c in model['chains'] if c['id'] == 'A'][0]
    assert 'atoms' in chain_a, "Chain A should have atoms displayed"
    assert 'spec' in chain_a['atoms'], "Atoms should have spec"
    
    run(test_production_session, "close")


def test_display_state_partial_display(test_production_session):
    """Test that partial displays generate correct specs."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms first
    run(test_production_session, "hide #1 target a")
    
    # Show atoms only for residues 1-10 in chain A
    run(test_production_session, "show #1/A:1-10 target a")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    if chain_a and 'atoms' in chain_a:
        spec = chain_a['atoms']['spec']
        assert spec is not None, "Spec should not be None for partial display"
        # Spec should reference model 1 and not be the full chain
        assert '#1' in spec, f"Spec '{spec}' should contain model reference"
    
    run(test_production_session, "close")


def test_display_state_no_models(test_production_session):
    """Test display_state when no models are loaded."""
    from chimerax.core.commands import run
    
    # Ensure no models are loaded
    run(test_production_session, "close")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    
    # Should return empty list
    assert isinstance(data, list)
    assert len(data) == 0


def test_display_state_only_displayed_ligands(test_production_session):
    """Test that only displayed ligands appear in output."""
    from chimerax.core.commands import run
    
    # Open a structure with ligands (2tpk has ligands)
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # First hide all atoms (including ligands)
    run(test_production_session, "hide #1 target a")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # Ligands key should not be present when ligands are hidden
    assert 'ligands' not in model or len(model.get('ligands', [])) == 0, \
        "Ligands should not appear when hidden"
    
    # Show ligand atoms
    run(test_production_session, "show ligand target a")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # Now ligands should appear if they exist
    if model.get('ligands'):
        for lig in model['ligands']:
            assert 'name' in lig
            assert 'spec' in lig
            # Should NOT have a 'displayed' boolean key
            assert 'displayed' not in lig, "Presence in output implies displayed"
    
    run(test_production_session, "close")


def test_display_state_only_displayed_ions(test_production_session):
    """Test that only displayed ions appear in output."""
    from chimerax.core.commands import run
    
    # Open structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all including ions
    run(test_production_session, "hide #1 target a")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # Ions should not appear when hidden
    assert 'ions' not in model or len(model.get('ions', [])) == 0, \
        "Ions should not appear when hidden"
    
    # Show ions
    run(test_production_session, "show ions target a")
    
    result = run(test_production_session, "info display_state")
    data = json.loads(result.json_result)
    model = data[0]
    
    # If ions exist and are shown, they should appear without 'displayed' key
    if model.get('ions'):
        for ion in model['ions']:
            assert 'name' in ion
            assert 'spec' in ion
            # Should NOT have a 'displayed' boolean key
            assert 'displayed' not in ion, "Presence in output implies displayed"
    
    run(test_production_session, "close")


# Test commands list for parametrized testing
display_state_commands = [
    ["open 2tpk autostyle false", "info display_state"],
    ["open 2tpk autostyle false", "hide #1 target m", "info display_state"],
    ["open 2tpk autostyle false", "show #1 target c", "info display_state"],
    ["open 2tpk autostyle false", "hide #1 target abc", "show #1/A:1-20 target a", "info display_state"],
]


@pytest.mark.parametrize("commands", display_state_commands)
def test_display_state_command_sequence(test_production_session, commands):
    """Test that display_state command works correctly in various scenarios."""
    from chimerax.core.commands import run
    
    for command in commands:
        result = run(test_production_session, command)
        
        # If this was the display_state command, verify it returns valid JSON
        if "display_state" in command:
            assert result is not None, "display_state should return a result"
            data = json.loads(result.json_result)
            assert isinstance(data, list), "Result should be a list"
    
    # Clean up
    run(test_production_session, "close")
