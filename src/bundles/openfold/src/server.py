def start_server(runs_directory, openfold_exe, host = None, port = 30173,
                 device = 'gpu', gpus = None, extra_options = []):
    # Get the hostname
    if not host:
        host = _default_host()
    
    import socket
    server_socket = socket.socket()
    server_socket.bind((host, port))

    server_socket.listen()
    log(f'OpenFold server listening at {host}, port {port}')

    prediction_queue = create_prediction_queue(gpus)
    
    while True:
        client_socket, address = server_socket.accept()  # Accept new connection
        returned_result = False
        
        try:
            zip_data = read_socket_data(client_socket)
            if is_zip_data(zip_data):
                log(f'Read openfold input zip data, {len(zip_data)} bytes')
                zip_path, job_id = write_zip_file(runs_directory, zip_data)
                log(f'Wrote zip file for server job {zip_path}, job id {job_id}')
                p = OpenFoldPrediction(job_id, zip_path, openfold_exe, client_socket, address,
                                       device=device, extra_options=extra_options)
                prediction_queue.put(p)
                msg = f'Job id: {job_id}'.encode('utf-8')
            elif zip_data.startswith(b'status: '):
                job_id = zip_data[8:].decode('utf-8')
                log(f'Status check for job {job_id}')
                returned_result, msg = check_status(client_socket, runs_directory, job_id)
            else:
                log(f'Invalid server connection, {len(zip_data)} bytes')
                msg = b'Invalid server request'
        except Exception as e:
            import traceback
            msg = f'Error: {str(e)}\n\n{traceback.format_exc()}'.encode('utf-8')

        client_socket.sendall(msg)
        try:
            client_socket.shutdown(socket.SHUT_RDWR)
        except Exception as e:
            pass  # The client already closed the socket.
        client_socket.close()

        if returned_result:
            remove_job_files(runs_directory, job_id)

    server_socket.close()

def create_prediction_queue(gpus = None):
    import queue
    prediction_queue = queue.Queue()

    gpu_ids = gpus if gpus else [None]
    for gpu_id in gpu_ids:
        from threading import Thread
        t = Thread(target=run_queued_predictions, args=(prediction_queue,), kwargs = {'gpu_id': gpu_id})
        t.daemon = True # Daemon threads don't block the main program from exiting
        t.start()

    return prediction_queue
    
def run_queued_predictions(prediction_queue, gpu_id = None):
    while True:
        try:
            p = prediction_queue.get()
            p.run(gpu_id = gpu_id)
        except Exception as e:
            log(f'error running prediction queue: {str(e)}')

def read_socket_data(connection, buffer_size = 1024*1024):
    blocks = []
    while True:
        data = connection.recv(buffer_size)
        if data == b'':
            break
        blocks.append(data)
    data = b''.join(blocks)
    return data

def is_zip_data(zip_data):
    return zip_data[:4] == bytearray([0x50, 0x4B, 0x03, 0x04])

def write_zip_file(runs_directory, zip_data):
    from tempfile import NamedTemporaryFile
    tf = NamedTemporaryFile(dir = runs_directory, prefix = 'openfold_job_', suffix = '.zip', delete = False)
    zip_path = tf.name
    tf.write(zip_data)
    tf.close()
    from os.path import basename
    job_id = basename(zip_path).removeprefix('openfold_job_').removesuffix('.zip')
    return zip_path, job_id

def make_zip_file_from_directory(folder_path, zip_path):
    # Get the root directory length for relative paths
    root_len = len(folder_path) + 1
    from os.path import join
    import zipfile, os
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = join(root, file)
                # Write file with its path relative to the original folder
                zipf.write(file_path, file_path[root_len:])

class OpenFoldPrediction:
    def __init__(self, job_id, zip_path, openfold_exe, socket, address, device = 'gpu', extra_options = []):
        self._job_id = job_id
        self._zip_path = zip_path
        self._openfold_exe = openfold_exe
        self._socket = socket
        self._address = address
        self._device = device		# 'gpu' or 'cpu'
        self._extra_options = extra_options  # e.g. for low mem ['--use_cpu_memory', '--inplace_operations']
    def run(self, gpu_id = None):
        try:
            run_openfold_prediction(self._zip_path, self._openfold_exe, device = self._device, gpu_id = gpu_id,
                                 extra_options = self._extra_options)
            log(f'prediction {self._job_id} finished')
        except Exception as e:
            self._report_error(e)
    def _report_error(self, e):
        import traceback
        msg = f'Error running job {self._job_id}: {str(e)}\n\n{traceback.format_exc()}'
        log(msg)
        run_dir = self._zip_path.removesuffix('.zip')
        from os.path import join
        with open(join(run_dir, 'error'), 'w') as f:
            f.write(msg)

def run_openfold_prediction(zip_path, openfold_exe, device = 'gpu', gpu_id = None, extra_options = []):
    from zipfile import ZipFile
    zf = ZipFile(zip_path)

    run_dir = zip_path.removesuffix('.zip')
    from os import mkdir
    mkdir(run_dir)

    log(f'extracting prediction input to {run_dir}')
    zf.extractall(run_dir)

    log('running openfold')
    run_openfold(run_dir, openfold_exe, device=device, gpu_id=gpu_id, extra_options=extra_options)

    from os.path import join, basename, dirname
    job_id = basename(zip_path).removesuffix('.zip').removeprefix('openfold_job_')
    results_zip = join(dirname(run_dir), f'openfold_results_{job_id}.zip')
    log(f'creating openfold results zip file {results_zip}')
    results_zip_tmp = results_zip + '.tmp'
    make_zip_file_from_directory(run_dir, results_zip_tmp)
    from os import rename
    rename(results_zip_tmp, results_zip)

def run_openfold(directory, openfold_exe, device = 'gpu', gpu_id = None, extra_options = []):
    env = {}
    from sys import platform
    if platform == 'darwin':
        # On Mac PyTorch uses MPS (metal performance shaders) but not all functions are implemented
        # on the GPU (Feb 10, 2025) so PYTORCH_ENABLE_MPS_FALLBACK=1 allows these to run on the CPU.
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        # On Mac the huggingface.co URLs get SSL certificate errors unless we setup
        # certifi root certificates.
        import certifi
        env["SSL_CERT_FILE"] = certifi.where()
    elif platform == 'linux':
        # Torch needs ninja to be in the path to compile CUDA code.
        # ninja is in the openfold virtual environment bin directory.
        from os.path import dirname
        bin_dir = dirname(openfold_exe)
        from os import environ
        exe_path = environ.get('PATH', '')
        env['PATH'] = f'{bin_dir}:{exe_path}'

    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu_id

    if len(env) == 0:
        env = None
        
    from os.path import join
    command_file = join(directory, 'command')
    with open(command_file, 'r') as f:
        cmd_string = f.read()
    command_args = cmd_string.split()
    command_args[0] = openfold_exe  # Use server openfold executable location

    device_args = [arg for arg in command_args if arg.startswith('--device=')]
    for arg in device_args:
        command_args.remove(arg)
    command_args.append(f'--device={device}')

    from sys import platform
    command_args.extend(extra_options)
    log(f'Command: {" ".join(command_args)}')

    from time import time
    t_start = time()
    from subprocess import Popen, PIPE
    p = Popen(command_args, cwd = directory,
              stdout = PIPE, stderr = PIPE, env=env,
              creationflags = _no_subprocess_window())
    stdout, stderr = p.communicate()
    t_end = time()
    
    with open(join(directory, 'stdout'), 'wb') as f:
        f.write(stdout)
    with open(join(directory, 'stderr'), 'wb') as f:
        f.write(stderr)
    with open(join(directory, 'time'), 'w') as f:
        f.write('%.2f' % (t_end - t_start))

def _no_subprocess_window():
    '''The Python subprocess module only has the CREATE_NO_WINDOW flag on Windows.'''
    from sys import platform
    if platform == 'win32':
        from subprocess import CREATE_NO_WINDOW
        flags = CREATE_NO_WINDOW
    else:
        flags = 0
    return flags

def check_status(client_socket, runs_directory, job_id):
    returned_result = False
    from os.path import join, exists
    job_dir = join(runs_directory, f'openfold_job_{job_id}')
    if exists(job_dir):
        results_zip = join(runs_directory, f'openfold_results_{job_id}.zip')
        if exists(results_zip):
            with open(results_zip, 'rb') as f:
                msg = f.read()
                returned_result = True
        else:
            from os.path import join, exists
            error_file = join(job_dir, 'error')
            if exists(error_file):
                with open(error_file, 'rb') as f:
                    msg = f.read()
                returned_result = True
            else:
                msg = b'Running'
    else:
        if exists(job_dir + '.zip'):
            msg = b'Not yet started'
        else:
            msg = b'No such job'
    return returned_result, msg

def remove_job_files(runs_directory, job_id):
    from os.path import join, exists
    job_zip = join(runs_directory, f'openfold_job_{job_id}.zip')
    job_dir = job_zip.removesuffix('.zip')
    results_zip = join(runs_directory, f'openfold_results_{job_id}.zip')
    from os import remove
    if exists(job_zip):
        remove(job_zip)
    if exists(results_zip):
        remove(results_zip)
    if exists(job_dir):
        from shutil import rmtree
        rmtree(job_dir)
    log(f'removed job files {job_id}')
    
def predict_on_server(run_dir, host = None, port = 30173):
    zip_path = run_dir + '.zip'
    make_zip_file_from_directory(run_dir, zip_path)
    with open(zip_path, 'rb') as f:
        zip_data = f.read()

    if not host:
        host = _default_host()
    print(f'Sending job request to server ({len(zip_data)} bytes)')

    try:
        msg = send_to_server(zip_data, host, port)
    except ConnectionRefusedError:
        from chimerax.core.errors import UserError
        raise UserError(f'Could not connect to {host}:{port}, connection refused')
    if msg.startswith(b'Job id: '):
        job_id = msg[8:].decode('utf-8')
        print(f'Server {host}:{port} queued job {job_id}')
    elif len(msg) == 0:
        raise RuntimeError(f'Did not receive job id from server {host}:{port}')
    else:
        raise RuntimeError(f'Server {host}:{port} did not return job id: {msg.decode("utf-8")}')

    from os.path import join
    with open(join(run_dir, 'server'), 'w') as f:
        f.write(f'{host} {port} {job_id}')

    return job_id

def get_results(job_id, run_dir, host = None, port = 30173):
    try:
        msg = send_to_server(f'status: {job_id}'.encode('utf-8'), host, port)
    except Exception as e:
        msg = f'Error: {str(e)}'.encode('utf-8')
    if is_zip_data(msg):
        zip_result_path = f'{run_dir}_output.zip'
        with open(zip_result_path, 'wb') as f:
            f.write(msg)

        from zipfile import ZipFile
        zf = ZipFile(zip_result_path)
        zf.extractall(run_dir)
        return 'Done'

    return msg.decode('utf-8')

def _default_host():
    import socket
    host = socket.gethostname()
    host = socket.gethostbyname(host)  # IP address
    return host

def send_to_server(zip_data, host, port, timeout = 10.0):
    if host is None:
        host = _default_host()

    import socket
    client_socket = socket.socket()  # Instantiate
    client_socket.settimeout(timeout)
    try:
        client_socket.connect((host, port))  # Connect to the server
    except socket.gaierror as e:
        from chimerax.core.errors import UserError
        raise UserError(f'Invalid host "{host}"  {e}') 
    except TimeoutError as e:
        raise RuntimeError(f'Connection to host "{host}" was not made within {timeout} seconds') 

    client_socket.sendall(zip_data)
    client_socket.shutdown(socket.SHUT_WR)

    response = read_socket_data(client_socket)
    client_socket.close()  # Close the connection

    return response

def log(msg):
    from sys import stderr
    from datetime import datetime
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stderr.write(f'{time_stamp}: {msg}\n')

def openfold_server_start(session,
                       host = None, port = 30173,
                       openfold_exe = None,
                       jobs_directory = '~/openfold_server_jobs',
                       device = 'gpu',
                       gpus = None,
                       extra_options = None,
                       server_log = 'openfold_server_log'):
    if host is None:
        host = _default_host()

    from os.path import expanduser, join
    jobs_directory = expanduser(jobs_directory)
    from os.path import exists
    if not exists(jobs_directory):
        from os import mkdir
        mkdir(jobs_directory)

    if openfold_exe is None:
        from .settings import _openfold_settings
        settings = _openfold_settings(session)
        from .install import find_executable
        openfold_exe = find_executable(settings.openfold_install_location, 'run_openfold')
    openfold_exe = expanduser(openfold_exe)

    from chimerax.core.python_utils import chimerax_python_executable
    python_exe = chimerax_python_executable()

    cmd = [python_exe, __file__, '--host', host, '--port', str(port),
           '--openfold_exe', openfold_exe, '--jobs_directory', jobs_directory]
    if device == 'cpu':
        cmd.append('--cpu')
    if gpus:
        cmd.extend(['--gpus', gpus])
    if extra_options:
        cmd.extend(['--extra_options', f'"{extra_options}"'])
        
    f = open(join(jobs_directory, server_log), 'w')

    import subprocess
    # Create a server process that will continue running after ChimeraX exits.
    from sys import platform
    if platform == 'win32':
        p = subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                             stdin=subprocess.DEVNULL, stdout=f, stderr=f)
    else:
        p = subprocess.Popen(cmd, start_new_session = True, stdin=subprocess.DEVNULL, stdout=f, stderr=f)

    msg = f'Started OpenFold server process {p.pid}: {" ".join(cmd)}'
    session.logger.info(msg)

    try:
        stdout, stderr = p.communicate(timeout = 2)
    except subprocess.TimeoutExpired:
        return

    msg = f'Server process exited with code {p.returncode}\n\nstdout: {stdout}\n\nstderr: {stderr}'
    session.logger.error(msg)

def openfold_server_list(session):
    lines = [f'Job {job_id} on server {host} port {port}'
             for run_dir, host, port, job_id in _active_server_jobs(session)]
    if len(lines) == 0:
        lines.append('No active openfold prediction server jobs')
    msg = '\n'.join(lines)
    session.logger.info(msg)

def _active_server_jobs(session):
    from .settings import _openfold_settings
    settings = _openfold_settings(session)
    run_dirs = settings.active_server_jobs
    jobs = _server_jobs_info(run_dirs)
    return jobs

def _server_jobs_info(run_directories):
    jobs = []
    from os.path import join, exists
    for run_dir in run_directories:
        path = join(run_dir, 'server')
        if exists(path):
            with open(path, 'r') as f:
                host, port, job_id = f.read().split()
                jobs.append((run_dir, host, int(port), job_id))
    return jobs

def openfold_server_fetch(session, run_directory = None, open = True):
    lines = []
    done = set()

    if run_directory:
        jobs = _server_jobs_info([run_directory])
    else:
        jobs = _active_server_jobs(session)

    for run_dir, host, port, job_id in jobs:
        msg = get_results(job_id, run_dir, host=host, port=port)
        if msg == 'Done':
            done.add(run_dir)
        lines.append(f'{run_dir} jobs {job_id} host {host} port {port}: {msg}')

    if len(lines) == 0:
        lines.append('No active openfold prediction server jobs')
        
    msg = '\n'.join(lines)
    session.logger.info(msg)

    if done:
        from .settings import _openfold_settings
        settings = _openfold_settings(session)
        active = [run_dir for run_dir in settings.active_server_jobs if run_dir not in done]
        settings.active_server_jobs = active

    if open:
        for run_dir in done:
            from .history import PredictionDirectory
            p = PredictionDirectory(run_dir)
            p.open_structures(session)

    return done

def register_openfold_server_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, IntArg, StringArg, BoolArg, OpenFolderNameArg
    desc = CmdDesc(
        keyword = [('host', StringArg),
                   ('port', IntArg),
                   ('openfold_exe', StringArg),
                   ('jobs_directory', StringArg),
                   ('device', EnumOf(['gpu', 'cpu'])),
                   ('gpus', StringArg),
                   ('extra_options', StringArg),
                   ],
        synopsis = 'Start a OpenFold prediction server',
        url = 'https://www.rbvi.ucsf.edu/chimerax/data/openfold-feb2026/openfold_server.html'
    )
    register('openfold server start', desc, openfold_server_start, logger=logger)

    desc = CmdDesc(
        synopsis = 'List active OpenFold prediction server jobs',
        url = 'https://www.rbvi.ucsf.edu/chimerax/data/openfold-feb2026/openfold_server.html'
    )
    register('openfold server list', desc, openfold_server_list, logger=logger)

    desc = CmdDesc(
        optional = [('run_directory', OpenFolderNameArg)],
        keyword = [('open', BoolArg)],
        synopsis = 'Fetch results for active OpenFold prediction server jobs',
        url = 'https://www.rbvi.ucsf.edu/chimerax/data/openfold-feb2026/openfold_server.html'
    )
    register('openfold server fetch', desc, openfold_server_fetch, logger=logger)

def _start_server():
    description = ('The ChimeraX OpenFold structure prediction can run predictions on another machine'
                   ' for faster predictions of larger structures.  This command starts the server.'
                   ' First use ChimeraX to install OpenFold with the ChimeraX command "openfold install".'
                   ' To run predictions on the server from another machine use the "User server..."'
                   ' option in the ChimeraX OpenFold prediction user interface or the useServer option'
                   ' of the ChimeraX "openfold predict" command.')

    from argparse import ArgumentParser
    p = ArgumentParser(prog = 'ChimeraX OpenFold Server', description = description)
    p.add_argument('--host',
                   help = 'Host name or IP address for listening.  Default uses Python socket module gethostname() and gethostbyname().')
    p.add_argument('--port', type = int, default = 30173,
                   help = 'Port number for connecting to server.  Default 30173.')
    from os.path import expanduser
    p.add_argument('--openfold_exe', default = expanduser('~/openfold3/bin/run_openfold'),
                   help = 'Path to the openfold executable.  Default is ~/openfold3/bin/run_openfold')
    p.add_argument('--jobs_directory', default = expanduser('~/openfold_server_jobs'),
                   help = 'Directory where to place prediction job zip files and directories.  The directory will be created if it does not exist.  A log file openfold_server_log will also be created in this directory.  Default is ~/openfold_server_jobs')
    p.add_argument('--cpu', action = 'store_true', default = False,
                   help = 'Whether to run predictions on CPU instead of GPU.  Default false.')
    p.add_argument('--gpus',
                   help = 'Comma separated list of integer CUDA GPU ids.  Can run a separate prediction on each GPU in parallel.  By default uses just one GPU.')
    p.add_argument('--extra_options',
                   help = 'Extra openfold options to add for each prediction.  For example, "--use_cpu_memory" to handle larger predictions with low memory OpenFold variants.')
    args = p.parse_args()
    device = 'cpu' if args.cpu else 'gpu'
    gpus = args.gpus.split(',') if args.gpus else None
    extra_options = args.extra_options.removeprefix('"').removesuffix('"').split() if args.extra_options else []

    from os.path import exists
    if not exists(args.jobs_directory):
        from os import mkdir
        mkdir(args.jobs_directory)

    start_server(args.jobs_directory, args.openfold_exe, args.host, args.port,
                 device = device, gpus = gpus, extra_options = extra_options)

if __name__ == '__main__':
    _start_server()

