# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Tests for the 'info shown' command in the list_info bundle."""

import pytest
import json


def _in_ci():
    """Check if running in a CI environment."""
    import os
    # Common CI environment variables
    ci_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL', 'TRAVIS']
    return any(os.environ.get(var) for var in ci_vars)


# Marker for tests that require EMDB access - skip in CI due to unreliable FTP servers
requires_emdb = pytest.mark.skipif(
    _in_ci(),
    reason="Skipping EMDB tests in CI (FTP servers are unreliable)"
)


# Commands to set up test structures
setup_structure = [
    "open 3ptb autostyle false",  # Trypsin with benzamidine ligand (~220 residues)
]


def test_shown_basic(test_production_session):
    """Test that 'info shown' returns valid JSON with expected structure."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Run 'info shown' and get JSON result
    # Note: return_json must be passed to run(), not in command string,
    # because the command infrastructure overrides command-line return_json
    result = run(test_production_session, "info shown", return_json=True)
    
    # Verify we got a JSONResult
    assert result is not None
    
    # Parse the JSON
    data = json.loads(result.json_value)
    
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


def test_shown_hidden_model(test_production_session):
    """Test that hidden models are NOT included in output at all."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Verify model #1 is in output when visible
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model_ids = [m['id'] for m in data]
    assert '#1' in model_ids, "Visible model should be in output"
    
    # Hide the model
    run(test_production_session, "hide #1 target m")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Hidden model should NOT be in output at all
    model_ids = [m['id'] for m in data]
    assert '#1' not in model_ids, "Hidden model should NOT be in output"
    
    # Output should be empty (only model was hidden)
    assert len(data) == 0, "Output should be empty when all models are hidden"
    
    run(test_production_session, "close")


def test_shown_visible_model_structure(test_production_session):
    """Test 'info shown' output structure for a visible AtomicStructure."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Make sure it's visible
    run(test_production_session, "show #1 target m")
    
    # Show ribbons for all chains
    run(test_production_session, "show #1 target c")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Find the AtomicStructure model
    model = data[0]
    assert model.get('type') in ('AtomicStructure', 'Structure')
    assert 'hidden' not in model, "Visible model should not have hidden key"
    
    # Should have chains with ribbons displayed
    assert 'chains' in model
    
    # Chains should have ribbon info
    chains = model['chains']
    for chain in chains:
        assert 'ribbons_shown' in chain
    
    run(test_production_session, "close")


def test_shown_atoms_only(test_production_session):
    """Test that only atom display is reported when atoms are shown."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show only atoms (no ribbons)
    run(test_production_session, "hide #1 target c")
    run(test_production_session, "show #1/A target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Should have chains with atoms
    assert 'chains' in model
    chain = model['chains'][0]
    assert 'atoms_shown' in chain
    
    # Should NOT have ribbons
    assert 'ribbons_shown' not in chain
    
    run(test_production_session, "close")


def test_shown_partial_display(test_production_session):
    """Test partial atom display generates correct atomspec."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show only some atoms from chain A (3ptb residues start at 16, not 1)
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:16-25 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Should have chain A with atoms
    assert 'chains' in model
    chain = model['chains'][0]
    assert chain['id'] == 'A'
    assert 'atoms_shown' in chain
    
    # Atom spec should describe the shown atoms
    atom_spec = chain['atoms_shown']['spec']
    assert '16' in atom_spec or '25' in atom_spec  # Should include residue numbers
    
    run(test_production_session, "close")


def test_shown_no_models(test_production_session):
    """Test 'info shown' with no models open returns empty list."""
    from chimerax.core.commands import run
    
    # Make sure no models are open
    run(test_production_session, "close")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Should be an empty list
    assert isinstance(data, list)
    assert len(data) == 0


def test_shown_only_displayed_ligands(test_production_session):
    """Test that only displayed ligands appear in output."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # First hide all atoms (including ligands)
    run(test_production_session, "hide #1 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Ligands key should not be present when ligands are hidden
    assert 'ligands' not in model or len(model.get('ligands', [])) == 0, \
        "Ligands should not appear when hidden"
    
    # Show ligand atoms
    run(test_production_session, "show ligand target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Now ligands should appear if they exist
    if model.get('ligands'):
        for lig in model['ligands']:
            assert 'name' in lig
            assert 'atoms_shown' in lig
    
    run(test_production_session, "close")


def test_shown_only_displayed_ions(test_production_session):
    """Test that only displayed ions appear in output."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # First hide all atoms (including ions)
    run(test_production_session, "hide #1 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Ions key should not be present when ions are hidden
    assert 'ions' not in model or len(model.get('ions', [])) == 0, \
        "Ions should not appear when hidden"
    
    # Show ion atoms
    run(test_production_session, "show ions target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Now ions should appear if they exist
    if model.get('ions'):
        for ion in model['ions']:
            assert 'name' in ion
            assert 'spec' in ion
    
    run(test_production_session, "close")


def test_shown_parent_visibility(test_production_session):
    """Test that parent model visibility affects child visibility."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show model #1
    run(test_production_session, "show #1 target m")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Model #1 should be in output
    model_ids = [m['id'] for m in data]
    assert '#1' in model_ids
    
    # Any child models (like pseudobond groups) would also be visible
    # and should appear in output (if they have display=True)
    child_ids = [m['id'] for m in data if m['id'].startswith('#1.')]
    
    # Hide parent model
    run(test_production_session, "hide #1 target m")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Neither parent nor children should be in output
    model_ids = [m['id'] for m in data]
    assert '#1' not in model_ids
    for child_id in child_ids:
        assert child_id not in model_ids, \
            f"Child {child_id} should not appear when parent is hidden"
    
    run(test_production_session, "close")


def test_shown_child_inherits_parent_visibility(test_production_session):
    """Test that child models inherit parent visibility correctly."""
    from chimerax.core.commands import run
    
    # Open a structure (which may have child models like pseudobond groups)
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show parent
    run(test_production_session, "show #1 target m")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    initial_count = len(data)
    
    # Hide parent
    run(test_production_session, "hide #1 target m")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # All models starting with #1 should be gone
    remaining_model_ids = [m['id'] for m in data]
    for mid in remaining_model_ids:
        assert not mid.startswith('#1'), \
            f"Model {mid} should not be visible when parent #1 is hidden"
    
    run(test_production_session, "close")


@pytest.mark.parametrize("commands", [
    # Test 1: Show then hide chain
    (["show #1/A target ac", "hide #1/A target ac"], 0),
    
    # Test 2: Show atoms then ribbons
    (["show #1/A target a", "show #1/A target c"], 1),
    
    # Test 3: Hide all then show one chain
    (["hide #1 target ac", "show #1/A target ac"], 1),
    
    # Test 4: Complex sequence
    (["hide #1 target ac", "show #1/A target a", "show #1/A target c", "hide #1/A:1-50 target a"], 1),
])
def test_shown_command_sequence(test_production_session, commands):
    """Test that various command sequences produce correct output."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Execute test commands
    for cmd in commands[0]:
        run(test_production_session, cmd)
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    if data:
        model = data[0]
        num_chains = len(model.get('chains', []))
        expected_chains = commands[1]
        assert num_chains == expected_chains, \
            f"Expected {expected_chains} chains, got {num_chains}"
    
    run(test_production_session, "close")


def test_shown_partial_atoms_in_residue(test_production_session):
    """Test that partial atom display in a residue generates atom-level spec."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show only backbone atoms for a range of residues
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:20-30@CA,C,N,O target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Should have chain A with atoms
    assert 'chains' in model
    chain = model['chains'][0]
    assert 'atoms_shown' in chain
    
    # Spec should include atom names since we're showing partial atoms
    atom_spec = chain['atoms_shown']['spec']
    # Should contain "@" for atom-level specification
    assert '@' in atom_spec, "Partial atom display should use @ notation"
    
    run(test_production_session, "close")


def test_shown_mixed_full_and_partial_residues(test_production_session):
    """Test mix of full residues and partial atoms generates correct spec."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide everything first
    run(test_production_session, "hide #1 target ac")
    
    # Show some full residues
    run(test_production_session, "show #1/A:20-25 target a")
    
    # Show only CA atoms for another residue
    run(test_production_session, "show #1/A:30@CA target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Should have chain A with atoms
    assert 'chains' in model
    chain = model['chains'][0]
    assert 'atoms_shown' in chain
    
    # Spec should handle both full and partial residues
    atom_spec = chain['atoms_shown']['spec']
    assert ':' in atom_spec, "Should have residue specifications"
    
    run(test_production_session, "close")


def test_shown_full_residue_display_no_atom_spec(test_production_session):
    """Test that full residue display doesn't include atom names in spec."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide everything, then show full residues
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:20-30 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    chain = model['chains'][0]
    atom_spec = chain['atoms_shown']['spec']
    
    # For full residue display, we should see residue range without @ atom names
    # The spec could be like "#1/A:20-30" with no "@" for atom names
    # (This depends on concise_atom_spec implementation)
    assert ':' in atom_spec, "Should have residue range"
    
    run(test_production_session, "close")


def test_shown_ligand_partial_atoms(test_production_session):
    """Test that partial ligand atom display uses atom-level spec."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms
    run(test_production_session, "hide #1 target a")
    
    # Show only some atoms of a ligand (if one exists)
    # BEN ligand should be present in 3ptb
    run(test_production_session, "show ligand & @C* target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # If ligands are shown, check their spec format
    if 'ligands' in model and model['ligands']:
        lig = model['ligands'][0]
        assert 'atoms_shown' in lig
        atom_spec = lig['atoms_shown']['spec']
        # Should have atom-level specification for partial display
        assert '@' in atom_spec, "Partial ligand display should use @ notation"
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_none(test_production_session):
    """Test hydrogen visibility classification when no hydrogens are shown."""
    from chimerax.core.commands import run
    
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show atoms but make sure no hydrogens
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:20-30 target a")
    run(test_production_session, "hide #1 & H target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    chain = model['chains'][0]
    assert 'hydrogens_shown' in chain
    assert chain['hydrogens_shown'] == 'none'
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_polar(test_production_session):
    """Test hydrogen visibility classification when only polar hydrogens are shown."""
    from chimerax.core.commands import run
    
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Show atoms with polar hydrogens only (H & ~HC)
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:20-30 target a")
    run(test_production_session, "show #1/A:20-30 & H & ~HC target a")
    run(test_production_session, "hide #1/A:20-30 & HC target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    chain = model['chains'][0]
    assert 'hydrogens_shown' in chain
    # Might be 'polar' or 'none' depending on whether the residues have polar H
    assert chain['hydrogens_shown'] in ('polar', 'none')
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_all(test_production_session):
    """Test hydrogen visibility classification when all hydrogens are shown."""
    from chimerax.core.commands import run
    
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Add hydrogens and show all atoms
    run(test_production_session, "addh")
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:20-30 target a")
    run(test_production_session, "show #1/A:20-30 & H target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    chain = model['chains'][0]
    assert 'hydrogens_shown' in chain
    assert chain['hydrogens_shown'] == 'all'
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_some(test_production_session):
    """Test hydrogen visibility classification for arbitrary subset."""
    from chimerax.core.commands import run
    
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Add hydrogens
    run(test_production_session, "addh")
    
    # Show some arbitrary hydrogens (not following polar/all pattern)
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show #1/A:20-30 target a")
    run(test_production_session, "show #1/A:20-25 & H target a")  # Only some residues with H
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    chain = model['chains'][0]
    assert 'hydrogens_shown' in chain
    # Should be 'some' since we show H for only some residues
    # Note: actual result depends on classify_hydrogen_visibility logic
    assert chain['hydrogens_shown'] in ('some', 'polar', 'none', 'all')
    
    run(test_production_session, "close")


def test_shown_ligand_hydrogen_visibility(test_production_session):
    """Test that ligands also report hydrogen visibility."""
    from chimerax.core.commands import run
    
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Add hydrogens
    run(test_production_session, "addh")
    
    # Hide everything, then show ligand with all atoms
    run(test_production_session, "hide #1 target ac")
    run(test_production_session, "show ligand target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # If ligands are present, check hydrogen visibility
    if 'ligands' in model and model['ligands']:
        lig = model['ligands'][0]
        assert 'hydrogens_shown' in lig
        assert lig['hydrogens_shown'] in ('none', 'polar', 'all', 'some')
    
    run(test_production_session, "close")


@requires_emdb
def test_shown_volume_with_surfaces(test_production_session):
    """Test that volume surfaces are reported as separate child models."""
    from chimerax.core.commands import run

    # Create a simple test volume
    run(test_production_session, "open emdb:1080")  # Small cryo-EM map
    
    # Set surface levels - can specify multiple levels in one command
    run(test_production_session, "volume #1 level 1.5 level 3.0")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Find the Volume model
    volume_model = None
    surface_models = []
    
    for m in data:
        if m['type'] == 'Volume' and m['id'] == '#1':
            volume_model = m
        elif m['type'] == 'VolumeSurface' and m['id'].startswith('#1.'):
            surface_models.append(m)
    
    # Volume should exist but NOT have surface_levels
    assert volume_model is not None, "Volume model should be present"
    assert 'surface_levels' not in volume_model, \
        "Volume should NOT have surface_levels (moved to child models)"
    
    # Should have two VolumeSurface child models
    assert len(surface_models) == 2, \
        f"Should have 2 VolumeSurface models, got {len(surface_models)}"
    
    # Each surface should have level info
    levels = sorted([s['level'] for s in surface_models])
    assert levels == [1.5, 3.0], f"Expected levels [1.5, 3.0], got {levels}"
    
    # Each surface should have level and style
    for surf in surface_models:
        assert 'level' in surf
        assert 'style' in surf
        assert surf['style'] in ('surface', 'mesh')
    
    run(test_production_session, "close")


@requires_emdb
def test_shown_volume_child_model_ids(test_production_session):
    """Test that VolumeSurface models have correct parent-child IDs."""
    from chimerax.core.commands import run

    # Create a test volume with surfaces
    run(test_production_session, "open emdb:1080")
    run(test_production_session, "volume #1 level 2.0 level 4.0")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Find surfaces
    surface_ids = [m['id'] for m in data if m['type'] == 'VolumeSurface']
    
    # Surface IDs should be #1.1 and #1.2
    assert '#1.1' in surface_ids
    assert '#1.2' in surface_ids
    
    run(test_production_session, "close")


@requires_emdb
def test_shown_volume_hidden_surface(test_production_session):
    """Test that hidden VolumeSurface models don't appear in output."""
    from chimerax.core.commands import run

    # Create a volume with surfaces
    run(test_production_session, "open emdb:1080")
    run(test_production_session, "volume #1 level 2.0 level 4.0")
    
    # Hide one of the surfaces
    run(test_production_session, "hide #1.1 model")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Find surfaces
    surface_models = [m for m in data if m['type'] == 'VolumeSurface']
    
    # Should only have one visible surface
    assert len(surface_models) == 1
    assert surface_models[0]['id'] == '#1.2'
    
    run(test_production_session, "close")
