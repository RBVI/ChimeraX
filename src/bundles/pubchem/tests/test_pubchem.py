import os

import pytest


def _in_ci():
    """Check if running in a CI environment."""
    ci_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL', 'TRAVIS']
    return any(os.environ.get(var) for var in ci_vars)


@pytest.mark.skipif(
    _in_ci(),
    reason="Skipping PubChem test in CI (external server is unreliable)"
)
def test_pubchem(test_production_session):
    from chimerax.core.commands import run
    run(test_production_session, "open pubchem:2519")
