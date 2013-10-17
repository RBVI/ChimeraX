# -----------------------------------------------------------------------------
#
def fetch_from_database(id, dbname):
    dbn = dbname.lower()
    global databases
    if dbn in databases:
        mlist = databases[dbn].fetch_id(id)
    else:
        from ..ui.gui import show_status
        show_status('Unknown database %s' % dbname)
        mlist = []
        # TODO warn.
    return mlist

# -----------------------------------------------------------------------------
#
databases = {}
def register_fetch_database(dbname, fetch_func, example_id, home_page, info_url):
    db = Database(dbname, fetch_func, example_id, home_page, info_url)
    global databases
    databases[dbname] = databases[dbname.lower()] = db

# -----------------------------------------------------------------------------
#
class Database:
    def __init__(self, name, fetch_func, example_id, home_page, info_url):
        self.name = name
        self.fetch_func = fetch_func
        self.example_id = example_id
        self.home_page = home_page
        self.info_url = info_url
    def fetch_id(self, id):
        return self.fetch_func(id)

# -----------------------------------------------------------------------------
# Download file from web.
#
def fetch_file(url, name, minimum_file_size = None, save_dir = '',
                        save_name = '', uncompress = False, ignore_cache = False):
        """a fetched non-local file that doesn't get cached will be
           removed when Chimera exits

           if 'ignore_cache' is True, then cached values will be ignored,
           though the retrieved values will still be cached if appropriate
        """
        
        if save_name and not ignore_cache:
                path = fetch_local_file(save_dir, save_name)
                if path:
                        return path, {}

        from ..ui.gui import show_status
        show_status('Fetching %s' % (name,))

#       from chimera import tasks
#       task = tasks.Task("Fetch %s" % name, modal=True)
        def report_cb(barrived, bsize, fsize):
                if fsize > 0:
                        percent = min(100.0,(100.0*barrived*bsize)/fsize)
                        prog = '%s %.0f%% of %s' % (name, percent, byte_text(fsize))
                else:
                        prog = '%s %s received' % (name, byte_text(barrived*bsize))
                show_status(prog)
#                task.updateStatus(prog)

        from urllib import request
        try:
                path, headers = request.urlretrieve(url, reporthook = report_cb)
        except IOError as v:
                from chimera import NonChimeraError
                raise NonChimeraError('Error fetching %s: %s' % (name, str(v)))
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
                        show_status('Uncompressing %s' % name)
                        gunzip(path, upath)
                        show_status('')
                        path = upath

        if save_name:
                spath = save_fetched_file(path, save_dir, save_name)
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
def save_fetched_file(path, save_dir, save_name):

        spath = save_location(save_dir, save_name)
        if spath is None:
                return None
        from ..ui.gui import show_status
        show_status('Copying %s to download directory' % save_name)
        import shutil
        try:
                shutil.copyfile(path, spath)
        except IOError:
                return None
        show_status('')
        return spath

# -----------------------------------------------------------------------------
#
def save_fetched_data(data, save_dir, save_name):

        spath = save_location(save_dir, save_name)
        if spath is None:
                return None
        from ..ui.gui import show_status
        show_status('Saving %s to download directory' % save_name)
        try:
                f = open(spath, 'wb')
                f.write(data)
                f.close()
        except IOError:
                return None
        show_status('')
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
