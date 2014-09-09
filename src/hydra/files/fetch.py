# -----------------------------------------------------------------------------
#
def fetch_from_database(id, dbname, session):
    '''
    Fetch a database entry with a specified id code and return a list of models.
    The models are not added to the scene.  The allowed databases are those
    registered with the register_fetch_database() routine.
    '''
    dbn = dbname.lower()
    databases = session.databases
    if dbn in databases:
        mlist = databases[dbn].fetch_id(id, session)
    else:
        session.show_status('Unknown database %s' % dbname)
        mlist = []
        # TODO warn.
    return mlist

# -----------------------------------------------------------------------------
#
def register_fetch_database(dbname, fetch_func, example_id, home_page, info_url, session):
    '''
    Register a database so that files can be fetched using the fetch_from_database() routine.
    The fetching function takes a single argument, the entry identifier (a string) and returns
    a list of models.
    '''
    db = Database(dbname, fetch_func, example_id, home_page, info_url)
    sdb = session.databases
    sdb[dbname] = sdb[dbname.lower()] = db

# -----------------------------------------------------------------------------
#
class Database:
    def __init__(self, name, fetch_func, example_id, home_page, info_url):
        self.name = name
        self.fetch_func = fetch_func
        self.example_id = example_id
        self.home_page = home_page
        self.info_url = info_url
    def fetch_id(self, id, session):
        return self.fetch_func(id, session)

# -----------------------------------------------------------------------------
# Download file from web.
#
def fetch_file(url, name, session, minimum_file_size = None, save_dir = '',
               save_name = '', uncompress = False, ignore_cache = False):
        """
        This is a helper routine for fetching files using an http url.
        It caches fetched files, uses cached files when available, and
        supports decompressing files.

        A fetched non-local file that doesn't get cached will be
           removed when Chimera exits

        If 'ignore_cache' is True, then cached files will not be used,
           but the fetched file will still be cached.
        """
        
        if save_name and not ignore_cache:
                path = fetch_local_file(save_dir, save_name)
                if path:
                        return path, {}

        session.show_status('Fetching %s' % (name,))

#       from chimera import tasks
#       task = tasks.Task("Fetch %s" % name, modal=True)
        def report_cb(barrived, bsize, fsize):
                if fsize > 0:
                        percent = min(100.0,(100.0*barrived*bsize)/fsize)
                        prog = '%s %.0f%% of %s' % (name, percent, byte_text(fsize))
                else:
                        prog = '%s %s received' % (name, byte_text(barrived*bsize))
                session.show_status(prog)
#                task.updateStatus(prog)

        from urllib import request
        try:
                path, headers = request.urlretrieve(url, reporthook = report_cb)
        except IOError as v:
            raise IOError('Error fetching %s: %s' % (name, str(v)))
#       finally:
#               task.finished()         # Remove from tasks panel

        # Check if page is too small, indicating error return.
        if minimum_file_size != None:
                import os
                if os.stat(path).st_size < minimum_file_size:
#                       from chimera import NonChimeraError
#                       raise NonChimeraError('%s not available.' % name)
                    raise IOError('%s not available.' % name)

        if uncompress:
                if path.endswith('.gz'):
                        upath = path[:-3]
                elif uncompress == 'always':
                        upath = path + '.gunzip'
                else:
                        upath = None
                if upath:
                        session.show_status('Uncompressing %s' % name)
                        gunzip(path, upath)
                        session.show_status('')
                        path = upath

        if save_name:
                spath = save_fetched_file(path, save_dir, save_name, session)
                if spath:
                        path = spath

        # if not (url.startswith("file:") or (save_name and spath)):
        #       from OpenSave import osTemporaryFile
        #       import os
        #       tmpPath = osTemporaryFile(suffix=os.path.splitext(path)[1])
        #       # Windows doesn't like rename to an existing file, so...
        #       if os.path.exists(tmpPath):
        #               os.unlink(tmpPath)
        #       import shutil
        #       shutil.move(path, tmpPath)
        #       path = tmpPath

        return path, headers

# -----------------------------------------------------------------------------
#
def byte_text(b):

        if b >= 1024*1024:
                return '%.1f Mbytes' % (float(b)/(1024*1024))
        elif b >= 1024:
                return '%.1f Kbytes' % (float(b)/1024)
        return '%d bytes' % int(b)

# -----------------------------------------------------------------------------
#
def fetch_directory(create = False):
    from os.path import expanduser
    dir = expanduser('~/Downloads/Hydra')
    from os.path import isdir
    if create and not isdir(dir):
        import os
        try:
            os.mkdir(dir)
        except (OSError, IOError):
            return None
    return dir

# -----------------------------------------------------------------------------
#
def fetch_local_file(save_dir, save_name):

        dir = fetch_directory()
        if not dir:
                return None
        from os.path import join, isfile
        path = join(dir, save_dir, save_name)
        if not isfile(path):
                return None
        return path

# -----------------------------------------------------------------------------
#
def save_fetched_file(path, save_dir, save_name, session):

        spath = save_location(save_dir, save_name)
        if spath is None:
                return None
        session.show_status('Copying %s to download directory' % save_name)
        import shutil
        try:
                shutil.copyfile(path, spath)
        except IOError:
                return None
        session.show_status('')
        return spath

# -----------------------------------------------------------------------------
#
def save_location(save_dir, save_name):

        dir = fetch_directory(create = True)
        if not dir:
                return None
        from os.path import join, dirname, isdir
        spath = join(dir, save_dir, save_name)
        sdir = dirname(spath)
        if not isdir(sdir):
                import os
                try:
                        os.mkdir(sdir)
                except (OSError, IOError):
                        return None
        return spath

# -----------------------------------------------------------------------------
#
def gunzip(gzpath, path):

        import gzip
        gzf = gzip.open(gzpath)
        f = open(path, 'wb')
        f.write(gzf.read())
        f.close()
        gzf.close()
