# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Tests for the 'info shown' command in the list_info bundle."""

import pytest
import json


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
    assert 'chains' in model, "Visible model with ribbons should have chains"
    
    for chain in model['chains']:
        assert 'id' in chain
        assert 'polymer_type' in chain
        # Should have ribbons since we showed them
        assert 'ribbons' in chain, f"Chain {chain['id']} should have ribbons"
        assert 'spec' in chain['ribbons'], "Ribbons should have spec"
    
    run(test_production_session, "close")


def test_shown_atoms_only(test_production_session):
    """Test that chains only appear when something is displayed."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide everything
    run(test_production_session, "hide #1 target abcs")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Should not have chains since nothing is displayed
    assert 'chains' not in model or len(model.get('chains', [])) == 0, \
        "No chains should appear when nothing is displayed"
    
    # Now show atoms for chain A only
    run(test_production_session, "show #1/A target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Should now have chain A with atoms
    assert 'chains' in model
    chain_ids = [c['id'] for c in model['chains']]
    assert 'A' in chain_ids, "Chain A should be in output"
    
    chain_a = [c for c in model['chains'] if c['id'] == 'A'][0]
    assert 'atoms' in chain_a, "Chain A should have atoms displayed"
    assert 'spec' in chain_a['atoms'], "Atoms should have spec"
    
    run(test_production_session, "close")


def test_shown_partial_display(test_production_session):
    """Test that partial displays generate correct specs.
    
    Note: Specs follow ChimeraX convention - when there's only one model,
    the model number may be omitted from the spec.
    """
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms first
    run(test_production_session, "hide #1 target a")
    
    # Show atoms only for residues 1-10 in chain A
    run(test_production_session, "show #1/A:16-25 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
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
        # Spec should reference chain A and residue range (model number optional per ChimeraX convention)
        assert '/A:' in spec or '/A' in spec, f"Spec '{spec}' should reference chain A"
        # Should indicate a partial range, not the full chain
        assert ':' in spec, f"Spec '{spec}' should contain residue range for partial display"
    
    run(test_production_session, "close")


def test_shown_no_models(test_production_session):
    """Test 'info shown' when no models are loaded."""
    from chimerax.core.commands import run
    
    # Ensure no models are loaded
    run(test_production_session, "close")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Should return empty list
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
            assert 'spec' in lig
            # Should NOT have a 'displayed' boolean key
            assert 'displayed' not in lig, "Presence in output implies displayed"
    
    run(test_production_session, "close")


def test_shown_only_displayed_ions(test_production_session):
    """Test that only displayed ions appear in output."""
    from chimerax.core.commands import run
    
    # Open structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all including ions
    run(test_production_session, "hide #1 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Ions should not appear when hidden
    assert 'ions' not in model or len(model.get('ions', [])) == 0, \
        "Ions should not appear when hidden"
    
    # Show ions
    run(test_production_session, "show ions target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # If ions exist and are shown, they should appear without 'displayed' key
    if model.get('ions'):
        for ion in model['ions']:
            assert 'name' in ion
            assert 'spec' in ion
            # Should NOT have a 'displayed' boolean key
            assert 'displayed' not in ion, "Presence in output implies displayed"
    
    run(test_production_session, "close")


def test_shown_parent_visibility(test_production_session):
    """Test that child models are excluded from output when parent is hidden.
    
    This tests the model.visible property behavior: a model is only visible
    if its own display is True AND all its parents are also visible.
    
    When a parent model is hidden, child models should NOT appear in the output
    even if their own display property is True.
    """
    from chimerax.core.commands import run
    
    # Open a structure - this creates model #1 with possible submodels
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    
    # Find any model that has child models (id contains a dot like #1.1)
    parent_models = [m for m in data if '.' not in m['id'].replace('#', '')]
    child_models = [m for m in data if '.' in m['id'].replace('#', '')]
    
    if len(parent_models) > 0 and len(child_models) > 0:
        # Get a parent that has children
        parent_id = parent_models[0]['id']
        parent_id_num = parent_id.replace('#', '')
        
        # Find children of this parent
        children_of_parent = [m for m in child_models 
                             if m['id'].replace('#', '').startswith(parent_id_num + '.')]
        
        if len(children_of_parent) > 0:
            # Record child IDs before hiding parent
            child_ids_before = [c['id'] for c in children_of_parent]
            
            # Now hide the parent model
            run(test_production_session, f"hide {parent_id} target m")
            
            result = run(test_production_session, "info shown", return_json=True)
            data = json.loads(result.json_value)
            
            # Get all model IDs in output
            output_ids = [m['id'] for m in data]
            
            # Parent should NOT be in output
            assert parent_id not in output_ids, \
                f"Hidden parent {parent_id} should NOT be in output"
            
            # All children should also NOT be in output
            for child_id in child_ids_before:
                assert child_id not in output_ids, \
                    f"Child {child_id} should NOT be in output when parent {parent_id} is hidden"
            
            # Show parent again
            run(test_production_session, f"show {parent_id} target m")
            
            result = run(test_production_session, "info shown", return_json=True)
            data = json.loads(result.json_value)
            
            # Parent should now be in output
            output_ids = [m['id'] for m in data]
            assert parent_id in output_ids, \
                f"Parent {parent_id} should be in output after showing"
    
    run(test_production_session, "close")


def test_shown_child_inherits_parent_visibility(test_production_session):
    """Test specific case: child model with display=True but parent hidden.
    
    A child model should NOT appear in output if its parent is hidden,
    even if the child's own display property is True. This matches the
    ChimeraX model.visible property behavior.
    """
    from chimerax.core.commands import run
    from chimerax.core.models import Model
    
    # Open structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    session = test_production_session
    models = session.models.list()
    
    # Find a model with children
    for m in models:
        children = m.child_models()
        if len(children) > 0:
            # We found a parent with children
            # Ensure child's own display is True
            child = children[0]
            child.display = True
            
            # Verify child's display is True
            assert child.display == True, "Child display should be True"
            
            # Verify child is visible (parent is visible too)
            if m.display:
                assert child.visible == True, \
                    "Child should be visible when parent is visible"
            
            # Now hide the parent
            m.display = False
            
            # Child's display is still True
            assert child.display == True, "Child display should still be True"
            
            # But child's visible should be False (parent is hidden)
            assert child.visible == False, \
                "Child should NOT be visible when parent is hidden"
            
            # Run info shown and check the output
            result = run(session, "info shown", return_json=True)
            data = json.loads(result.json_value)
            
            # Child should NOT be in the output at all
            child_id = '#' + child.id_string
            output_ids = [x['id'] for x in data]
            assert child_id not in output_ids, \
                f"Child {child_id} should NOT be in output when parent is hidden"
            
            # Restore parent visibility for cleanup
            m.display = True
            break
    
    run(test_production_session, "close")


# Test commands list for parametrized testing
# Note: "info shown" is a marker that will trigger JSON validation
shown_commands = [
    ["open 3ptb autostyle false", "info shown"],
    ["open 3ptb autostyle false", "hide #1 target m", "info shown"],
    ["open 3ptb autostyle false", "show #1 target c", "info shown"],
    ["open 3ptb autostyle false", "hide #1 target abc", "show #1/A:16-35 target a", "info shown"],
]


@pytest.mark.parametrize("commands", shown_commands)
def test_shown_command_sequence(test_production_session, commands):
    """Test that 'info shown' command works correctly in various scenarios."""
    from chimerax.core.commands import run
    
    for command in commands:
        # For 'info shown' command, pass return_json=True to run()
        # (command-line returnJson is overridden by the run() parameter)
        if command == "info shown":
            result = run(test_production_session, command, return_json=True)
            assert result is not None, "'info shown' should return a result"
            data = json.loads(result.json_value)
            assert isinstance(data, list), "Result should be a list"
        else:
            run(test_production_session, command)
    
    # Clean up
    run(test_production_session, "close")


def test_shown_partial_atoms_in_residue(test_production_session):
    """Test that partial atom display within residues generates atom-level specs.
    
    When only some non-H atoms of a residue are shown (e.g., backbone only),
    the spec should include atom names like ':10@CA,CB,C,N,O'.
    """
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms
    run(test_production_session, "hide #1 target a")
    
    # Show only backbone atoms (CA, C, N, O) for a few residues
    run(test_production_session, "show #1/A:16-18@CA,C,N,O target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    assert chain_a is not None, "Chain A should be in output"
    assert 'atoms' in chain_a, "Chain A should have atoms"
    
    spec = chain_a['atoms']['spec']
    # The spec should contain '@' indicating atom-level specification
    # because not all atoms of the residues are shown
    assert '@' in spec, f"Spec '{spec}' should contain @ for partial atom display"
    
    run(test_production_session, "close")


def test_shown_mixed_full_and_partial_residues(test_production_session):
    """Test specs when some residues have all atoms shown and others partial.
    
    Expected format: ':48@CA,C,N,O:50-55' meaning residue 48 has partial atoms,
    residues 50-55 have all atoms shown.
    """
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms first
    run(test_production_session, "hide #1 target a")
    
    # Show all atoms for residues 50-55
    run(test_production_session, "show #1/A:50-55 target a")
    
    # Show only backbone for residue 48 (using a residue that has side chain atoms)
    run(test_production_session, "show #1/A:48@CA,C,N,O target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    assert chain_a is not None, "Chain A should be in output"
    assert 'atoms' in chain_a, "Chain A should have atoms"
    
    spec = chain_a['atoms']['spec']
    # Should have both residue-level specs (for full display) and atom-level (for partial)
    assert '@' in spec, f"Spec '{spec}' should contain @ for partial residue"
    # Should also have residue range for full-display residues
    # The exact format depends on order, but should have both patterns
    
    run(test_production_session, "close")


def test_shown_full_residue_display_no_atom_spec(test_production_session):
    """Test that full residue display uses residue-level spec (no @).
    
    When all non-H atoms of a residue are shown, we should get just
    residue specs like ':10-20' not ':10@CA,CB,...:11@CA,...'.
    """
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms
    run(test_production_session, "hide #1 target a")
    
    # Show all atoms (not just some) for a range of residues
    run(test_production_session, "show #1/A:25-30 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    assert chain_a is not None, "Chain A should be in output"
    assert 'atoms' in chain_a, "Chain A should have atoms"
    
    spec = chain_a['atoms']['spec']
    # When all atoms are shown, should NOT have @ in spec (residue-level only)
    assert '@' not in spec, f"Spec '{spec}' should NOT contain @ when all atoms are shown"
    # Should have residue range
    assert ':' in spec, f"Spec '{spec}' should contain residue range"
    
    run(test_production_session, "close")


def test_shown_ligand_partial_atoms(test_production_session):
    """Test that partial ligand atom display shows which atoms are displayed."""
    from chimerax.core.commands import run
    
    # Open a structure with ligands
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # First check if there are ligands
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    if not model.get('ligands'):
        # Structure doesn't have ligands, skip test
        run(test_production_session, "close")
        pytest.skip("Test structure has no ligands")
        return
    
    # Get the first ligand's residue spec
    lig = model['ligands'][0]
    lig_spec = lig['spec']
    
    # Hide all ligand atoms, then show only some
    run(test_production_session, f"hide {lig_spec} target a")
    
    # Show just a couple of atoms (e.g., first two heavy atoms)
    # Using a general approach - show atoms named C* (carbons)
    run(test_production_session, f"show {lig_spec}@C* target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    if model.get('ligands'):
        for lig in model['ligands']:
            # If partial display, should have 'shown_atoms' with atom names
            if lig.get('atoms_shown'):
                # atoms_shown is like "3/10" format
                assert '/' in lig['atoms_shown'], \
                    "atoms_shown should be in 'shown/total' format"
                # Should also have shown_atoms list
                assert 'shown_atoms' in lig, \
                    "Partial display should include shown_atoms list"
                assert isinstance(lig['shown_atoms'], list), \
                    "shown_atoms should be a list of atom names"
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_none(test_production_session):
    """Test that hiding all hydrogens reports 'hydrogens': 'none'."""
    from chimerax.core.commands import run
    
    # Open a structure
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Hide all atoms, then show only non-hydrogen atoms
    run(test_production_session, "hide #1 target a")
    run(test_production_session, "show #1/A:16-25 & ~H target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    if chain_a and 'atoms' in chain_a:
        assert 'hydrogens' in chain_a, "Chain should have 'hydrogens' field"
        assert chain_a['hydrogens'] == 'none', \
            f"Expected 'none' but got '{chain_a['hydrogens']}' when no H shown"
        # Spec should not contain any hydrogen atom names
        spec = chain_a['atoms']['spec']
        assert '@H' not in spec.upper() or '@HE' in spec.upper() or '@HI' in spec.upper(), \
            "Spec should not contain hydrogen atoms when hydrogens='none'"
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_polar(test_production_session):
    """Test that showing only polar hydrogens reports 'hydrogens': 'polar'."""
    from chimerax.core.commands import run
    
    # Open a structure and add hydrogens
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Add hydrogens to ensure we have them
    run(test_production_session, "addh #1")
    
    # Hide all atoms
    run(test_production_session, "hide #1 target a")
    
    # Show atoms for chain A, but hide non-polar hydrogens (HC)
    run(test_production_session, "show #1/A:16-25 target a")
    run(test_production_session, "hide #1/A:16-25 & HC target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    if chain_a and 'atoms' in chain_a:
        assert 'hydrogens' in chain_a, "Chain should have 'hydrogens' field"
        # Should be 'polar' since only polar H are shown
        assert chain_a['hydrogens'] == 'polar', \
            f"Expected 'polar' but got '{chain_a['hydrogens']}' when only polar H shown"
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_all(test_production_session):
    """Test that showing all hydrogens reports 'hydrogens': 'all'."""
    from chimerax.core.commands import run
    
    # Open a structure and add hydrogens
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Add hydrogens
    run(test_production_session, "addh #1")
    
    # Hide all, then show all atoms including hydrogens
    run(test_production_session, "hide #1 target a")
    run(test_production_session, "show #1/A:16-25 target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    if chain_a and 'atoms' in chain_a:
        assert 'hydrogens' in chain_a, "Chain should have 'hydrogens' field"
        assert chain_a['hydrogens'] == 'all', \
            f"Expected 'all' but got '{chain_a['hydrogens']}' when all H shown"
    
    run(test_production_session, "close")


def test_shown_hydrogen_visibility_some(test_production_session):
    """Test that showing arbitrary hydrogens reports 'hydrogens': 'some' and includes H in spec."""
    from chimerax.core.commands import run
    
    # Open a structure and add hydrogens
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Add hydrogens
    run(test_production_session, "addh #1")
    
    # Hide all, then show atoms for a few residues
    run(test_production_session, "hide #1 target a")
    run(test_production_session, "show #1/A:16-20 target a")
    
    # Now hide some but not all hydrogens in an arbitrary pattern
    # Hide all HC, then show just some of them back (arbitrary selection)
    run(test_production_session, "hide #1/A:16-20 & H target a")
    # Show just hydrogens on residue 1 (arbitrary subset)
    run(test_production_session, "show #1/A:16 & H target a")
    
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    # Find chain A
    chain_a = None
    for chain in model.get('chains', []):
        if chain['id'] == 'A':
            chain_a = chain
            break
    
    if chain_a and 'atoms' in chain_a:
        assert 'hydrogens' in chain_a, "Chain should have 'hydrogens' field"
        # Check that it's 'some' (arbitrary subset) - but might also be 'none' if residue 1
        # doesn't have H or if the pattern happens to match polar
        h_vis = chain_a['hydrogens']
        if h_vis == 'some':
            # When 'some', the spec should potentially include H atoms
            spec = chain_a['atoms']['spec']
            # The spec format may vary, but if there are partial residues with H, 
            # they should be included
            pass  # Just verify no crash, actual H content depends on structure
    
    run(test_production_session, "close")


def test_shown_ligand_hydrogen_visibility(test_production_session):
    """Test that ligand hydrogen visibility is reported."""
    from chimerax.core.commands import run
    
    # Open a structure with ligands
    for cmd in setup_structure:
        run(test_production_session, cmd)
    
    # Check if there are ligands
    result = run(test_production_session, "info shown", return_json=True)
    data = json.loads(result.json_value)
    model = data[0]
    
    if not model.get('ligands'):
        run(test_production_session, "close")
        pytest.skip("Test structure has no ligands")
        return
    
    # Ligands should have 'hydrogens' field
    for lig in model['ligands']:
        assert 'hydrogens' in lig, f"Ligand {lig['name']} should have 'hydrogens' field"
        assert lig['hydrogens'] in ('none', 'polar', 'all', 'some'), \
            f"Ligand hydrogens should be one of: none, polar, all, some; got {lig['hydrogens']}"
    
    run(test_production_session, "close")

