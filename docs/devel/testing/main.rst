.. vim: set expandtab shiftwidth=4 softtabstop=4 syntax=rst:

..
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

Testing
=======
ChimeraX currently has both a hand-rolled test suite and a Pytest suite; 
it is preferred that additional tests use Pytest.


``pytest``
----------

The ``pytest`` test suite can be invoked from the command line, but it is easier to invoke
it using ChimeraX's top level ``Makefile``. To run the tests, simply run: ::

    make pytest

in the root directory. This runs three rounds of tests:

1. ``tests/test_env.py`` with both Python and the ChimeraX binary, to ensure there's
   no difference between running ChimeraX and calling ``python -m chimerax.core``.

2. Wheel tests; i.e. all tests that are marked with ``@pytest.mark.wheel``

3. All other tests, using a production session produced in ``conftest.py``

In the latter two cases, the appropriate ``tests/test_imports_{app,wheel}.py`` is run before 
the rest of the suite, since if a module is unimportable then that is a critical error that 
must be addressed before other failures.

Test Order
^^^^^^^^^^
In general it is good practice if tests do not depend on each other, but in our case different runs of the
test suite do have an order. Wheel tests must be run first, for several reasons:

1. Creating the session that is used for testing the ChimeraX app changes values found in the top-level ``chimerax`` 
   namespace

2. During session creation, a global ``chimerax.core.toolshed.Toolshed`` object is produced, which has its
   own problems detailed below

It would be possible to run these tests at the same time if these issues were addressed.

Configuration
^^^^^^^^^^^^^
The configuration for ``pytest`` is found in the top level ``pyproject.toml``. 

Pytest is configured to look for tests in both the top-level ``tests`` folder and any
bundle's ``tests`` folder.

Custom markers include: 

- ``wheel``: used to mark tests that should be run in the same environment that would be 
  present if a user had downloaded the ChimeraX PyPi package. 

Please send email to ``chimerax-programmers@cgl.ucsf.edu`` before changing the configuration.

Uploading Test Data
---------------------
ChimeraX deals with a huge variety of data, much of which is held in large files. To avoid making the
ChimeraX repo too large, test data is not stored in the repo. It is instead stored on Plato, the RBVI's
server cluster. To add test data to Plato, place it in the ``test-data`` folder, which is next to the
``prereqs`` folder, in a subdirectory that has the same name as the bundle directory in the ChimeraX 
repository.

Downloading Test Data
---------------------
There is a script at ``utils/sync-test-data.sh`` that will download test data from Plato given the
name of a bundle's folder, e.g. ::

    ./utils/sync-test-data.sh -b dicom

will download the test data for the ``dicom`` bundle to ``src/bundles/dicom/tests/data``.

Adding Tests
------------
First, add a folder called ``tests`` in your bundle's directory. Any Python file in this directory
named ``test_*.py`` or ``*_test.py`` will be picked up and run by Pytest. Then, in those files, 
any function that starts with ``test_`` will be run. ::

    def test_something():
        # test code here

If your test requires a session, the recommended way to do this is to use the ``test_production_session`` fixture.
There is no need to import this fixture, as it is automatically available in all test files. ::
    
    def test_something(test_production_session):
        # test code here

If you need to parametrize the test to run over several different cases, or need to mark the test as for the wheel,
you'll need to ``import pytest`` ::

    import pytest

    @pytest.mark.parametrize('input,expected', [('a', 'b'), ('c', 'd')])
    def test_something(test_production_session, input, expected):
        # test code here

If your test uses the data directory, you can enumerate test cases using this pattern so that when the
test case is printed in the Pytest log it shows just the name of the file or folder and no the whole 
absolute path to the data: ::

    import os
    import pytest

    test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    test_cases = os.listdir(test_data_dir)

    @pytest.mark.parametrize('data', test_cases)
    def test_something(test_production_session, data):
        test_file = os.path.join(test_data_dir, data)
        # test code here


Finally, if you have a set of tests that do require data, it's best to mark those tests to be skipped if the
data is not present. ::

    import os
    import pytest

    test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(test_data_dir):
        pytest.skip("Skipping <BUNDLE> tests because test data is unavailable.",
        allow_module_level=True,
    )

    test_cases = os.listdir(test_data_dir)

    @pytest.mark.parametrize('data', test_cases)
    def test_something(test_production_session, data):
        test_file = os.path.join(test_data_dir, data)
        # test code here

Wheel Import Tests
------------------
The wheel import tests attempt to import every module in ChimeraX that is destined for inclusion in the
PyPi package. The list of modules that are *not* included can be found in ``utils/wheel/filter_modules.py``.

The wheel is created by copying the ``chimerax`` directory out of a built ChimeraX and then running that
script to delete excluded modules. To keep data centralized, add exclusions to that file instead of 
``tests/test_imports_wheel.py``.

