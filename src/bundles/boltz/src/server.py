def start_server(runs_directory, boltz_exe, host = None, port = 30172):
    # Get the hostname
    if not host:
        host = _default_host()
    
    import socket
    server_socket = socket.socket()
    server_socket.bind((host, port))

    server_socket.listen()  # Listen for up to 2 incoming connections

    while True:
        conn, address = server_socket.accept()  # Accept new connection
        print("Connection from: " + str(address))

        zip_data = read_socket_data(conn)
        print(f'read boltz input zip data, {len(zip_data)} bytes')
        if zip_data:
            try:
                zip_result = run_boltz_prediction(runs_directory, boltz_exe, zip_data)
            except Exception as e:
                import traceback
                msg = f'Error: {str(e)}\n\n{traceback.format_exc()}'
                conn.send(msg.encode())
            else:
                print(f'sending boltz results back to client, {len(zip_result)} bytes')
                conn.send(zip_result)

        print(f'finished boltz server job, closing connection {address}')
        conn.close()

    server_socket.close()

def read_socket_data(connection, buffer_size = 1024*1024):
    blocks = []
    while True:
        data = connection.recv(buffer_size)
        if data == b'':
            break
        blocks.append(data)
    data = b''.join(blocks)
    return data

def run_boltz_prediction(runs_directory, boltz_exe, zip_data):
    from tempfile import NamedTemporaryFile
    tf = NamedTemporaryFile(dir = runs_directory, prefix = 'boltz_job_', suffix = '.zip', delete = False)
    path = tf.name
    tf.write(zip_data)
    tf.close()
    print(f'wrote zip file for server job {path}')
    from zipfile import ZipFile
    zf = ZipFile(path)

    run_dir = path.removesuffix('.zip')
    from os import mkdir
    mkdir(run_dir)

    print(f'extracting prediction input to {run_dir}')
    zf.extractall(run_dir)

    print ('running boltz')
    run_boltz(run_dir, boltz_exe)

    from os.path import join, basename
    job_id = basename(path).removesuffix('.zip').removeprefix('boltz_job_')
    results_zip = join(runs_directory, f'boltz_results_{job_id}.zip')
    print (f'creating boltz results zip file {results_zip}')
    make_zip_file_from_directory(run_dir, results_zip)

    with open(results_zip, 'rb') as f:
        zip_results_data = f.read()
    return zip_results_data

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

def run_boltz(directory, boltz_exe):
    from sys import platform
    if platform == 'darwin':
        env = {}
        # On Mac PyTorch uses MPS (metal performance shaders) but not all functions are implemented
        # on the GPU (Feb 10, 2025) so PYTORCH_ENABLE_MPS_FALLBACK=1 allows these to run on the CPU.
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        # On Mac the huggingface.co URLs get SSL certificate errors unless we setup
        # certifi root certificates.
        import certifi
        env["SSL_CERT_FILE"] = certifi.where()
    else:
        env = None

    from os.path import join, basename
    command_file = join(directory, 'command')
    with open(command_file, 'r') as f:
        cmd_string = f.read()
    command_args = cmd_string.split()
    command_args[0] = boltz_exe  # Use server boltz executable location
    command_args[2] = basename(command_args[2])  # Don't use absolute path to .yaml file from client machine.

    from subprocess import Popen, PIPE
    # To continue to run even if ChimeraX exits use start_new_session=True
    p = Popen(command_args, cwd = directory,
              stdout = PIPE, stderr = PIPE, env=env,
              creationflags = _no_subprocess_window())
    stdout, stderr = p.communicate()

    with open(join(directory, 'stdout'), 'wb') as f:
        f.write(stdout)
    with open(join(directory, 'stderr'), 'wb') as f:
        f.write(stderr)

def _no_subprocess_window():
    '''The Python subprocess module only has the CREATE_NO_WINDOW flag on Windows.'''
    from sys import platform
    if platform == 'win32':
        from subprocess import CREATE_NO_WINDOW
        flags = CREATE_NO_WINDOW
    else:
        flags = 0
    return flags

def predict_on_server(run_dir, host = None, port = 30172):
    zip_path = run_dir + '.zip'
    make_zip_file_from_directory(run_dir, zip_path)
    with open(zip_path, 'rb') as f:
        zip_data = f.read()

    if not host:
        host = _default_host()
    zip_result = send_to_server(zip_data, host, port)

    zip_result_path = f'{run_dir}_output.zip'
    with open(zip_result_path, 'wb') as f:
        f.write(zip_result)

    from zipfile import ZipFile
    zf = ZipFile(zip_result_path)
    zf.extractall(run_dir)

def _default_host():
    import socket
    host = socket.gethostname()
    host = socket.gethostbyname(host)  # IP address
    return host

def send_to_server(zip_data, host, port):
    if host is None:
        host = _default_host()

    import socket
    client_socket = socket.socket()  # Instantiate
    client_socket.connect((host, port))  # Connect to the server

    print (f'sending prediction input to server, {len(zip_data)} bytes')
    client_socket.send(zip_data)
    client_socket.shutdown(socket.SHUT_WR)
    print ('finished sending prediction input to server')

    print ('waiting for prediction results from server')
    zip_result = read_socket_data(client_socket)
    if zip_result.startswith(b'Error:'):
        print (f'got error server: {zip_result.decode("utf-8")}')
    else:
        print (f'got prediction results from server, {len(zip_result)} bytes')
    client_socket.close()  # Close the connection

    return zip_result

def boltz_server(session, operation = None,
                 host = None, port = 30172,
                 boltz_exe = None,
                 jobs_directory = '~/boltz_server_jobs',
                 server_log = 'boltz_server_log'):
    if host is None:
        host = _default_host()
    from os.path import expanduser
    jobs_directory = expanduser(jobs_directory)
    if boltz_exe is None:
        from .settings import _boltz_settings
        settings = _boltz_settings(session)
        from .install import find_executable
        boltz_exe = find_executable(settings.boltz22_install_location, 'boltz')
    boltz_exe = expanduser(boltz_exe)
    from chimerax.core.python_utils import chimerax_python_executable
    python_exe = chimerax_python_executable()

    cmd = [python_exe, __file__, 'start', host, str(port),
           boltz_exe, jobs_directory]

    from os.path import expanduser, exists, join
    jobs_directory = expanduser(jobs_directory)
    if not exists(jobs_directory):
        from os import mkdir
        mkdir(jobs_directory)
    f = open(join(jobs_directory, server_log), 'w')
    import subprocess
    # Create a server process that will continue running after ChimeraX exits.
    from sys import platform
    if platform == 'win32':
        p = subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
#                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, stdout=f, stderr=f)

    else:
        p = subprocess.Popen(cmd, start_new_session = True, stdin=subprocess.DEVNULL, stdout=f, stderr=f)
    msg = f'Started Boltz server process {p.pid}: {" ".join(cmd)}'
    session.logger.info(msg)
    try:
        stdout, stderr = p.communicate(timeout = 2)
    except subprocess.TimeoutExpired:
        return

    msg = f'Server process exited with code {p.returncode}\n\nstdout: {stdout}\n\nstderr: {stderr}'
    session.logger.error(msg)
    
def register_boltz_server_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, IntArg, StringArg
    desc = CmdDesc(
        optional = [('operation', EnumOf(['start']))],
        keyword = [('host', StringArg),
                   ('port', IntArg),
                   ('boltz_exe', StringArg),
                   ('jobs_directory', StringArg),
                   ],
        synopsis = 'Start a Boltz prediction server',
        url = 'help:boltz_help.html'
    )
    register('boltz server', desc, boltz_server, logger=logger)

    
if __name__ == '__main__':
    from sys import argv
    print(f'server called with argv: {argv}')
    run_dir = argv[1]
    if run_dir == 'start':
        host = argv[2]
        port = int(argv[3])
        boltz_exe = argv[4]
        jobs_directory = argv[5]
        start_server(jobs_directory, boltz_exe, host, port)
    else:
        host = argv[2]
        port = int(argv[3])
        predict_on_server(run_dir, host, port)
