# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def run_user_startup_scripts(session, directory = None):

    if directory is None:
        from os import environ
        directory = environ.get('CHIMERAX_START', '~/chimerax_start')
        
    from os import path, listdir
    dir = path.expanduser(directory)
    if not path.isdir(dir):
        return
    
    # Add startup directory to end of module search path.
    import sys
    sys.path.append(dir)

    # Exec Python files in startup directory.
    dlist = listdir(dir)
    pyfiles = [f for f in dlist if f.endswith('.py')]
    for filename in pyfiles:
        p = path.join(dir, filename)
        try:
            execfile(p, session)
        except Exception as e:
            session.logger.warning('Error opening %s:\n%s' % (p, str(e)))

    # Try importing modules in startup directory.
    import importlib
    for f in dlist:
        p = path.join(dir, f)
        if path.isdir(p) and path.join(p, '__init__.py'):
            try:
                m = importlib.import_module(f)
            except Exception as e:
                session.logger.warning('Error opening module %s:\n%s' % (p, str(e)))
                m = None
            if hasattr(m, 'start'):
                try:
                    m.start(session)
                except Exception as e:
                    session.logger.warning('Error calling start() method for module %s:\n%s' % (p, str(e)))

def execfile(path, session):
    with open(path, 'rb') as f:
        code = compile(f.read(), path, 'exec')
        exec(code, {'session':session})
