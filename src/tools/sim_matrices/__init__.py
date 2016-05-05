def read_matrix_file(file_name, style="protein"):
	mfile = open(file_name, "r")

	matrix = {}
	if style == "protein":
		min_chars = 20
	else:
		min_chars = 4
	divisor = 1.0
	while 1: # skip through header
		line = mfile.readline()
		if not line:
			raise ValueError("No matrix found in %s" % file_name)
		chars = line.split()
		if chars[:2] == ["#", "Divisor:"]:
			divisor = float(chars[2])
			if divisor <= 0:
				raise ValueError("Bad divisor (%g) in matrix %s" % (divisor, file_name))
		if len(chars) < min_chars:
			continue
		if max([len(cs) for cs in chars]) > 1:
			continue

		for char in chars:
			fields = mfile.readline().split()
			if len(fields) != len(chars) + 1:
				raise ValueError("Unexpected number of fields on line %d of matrix in %s" % (
					chars.index(char)+1, file_name))
			if fields[0] != char:
				raise ValueError("Line %d of %s is %s, expected %s" % (chars.index(char)+1,
					file_name, fields[0], char)
			for i, c in enumerate(chars):
				matrix[(fields[0], c)] = float(fields[i+1]) / divisor
		# assume '?' is an unknown residue...
		for char in chars:
			matrix[(char, '?')] = matrix[('?', char)] = 0.0
		matrix[('?', '?')] = 0.0
		# treat pyrrolysine ('O') and selenocysteine ('U') as
		# 'X' and 'C' respectively (see Chimera 1 Trac #12828 for reasoning)
		if style == "protein":
			for char in chars + ["?"]:
				matrix[(char, 'O')] = matrix[('O', char)] = matrix[(char, 'X')]
				matrix[(char, 'U')] = matrix[('U', char)] = matrix[(char, 'C')]
			matrix[('O', 'U')] = matrix[('U', 'O')] = matrix[('X', 'C')]
			matrix[('O', 'O')] = matrix[('X', 'X')]
			matrix[('U', 'U')] = matrix[('C', 'C')]
		mfile.close()
		break
	return matrix

_matrices = _matrix_files = None

def matrices(session):
	if _matrices == None:
		_init(session)
	return _matrices

def matrix_files(session):
	if _matrix_files == None:
		_init(session)
	return _matrix_files

def _init(session):
	global _matrices, _matrix_files
	_matrices = {}
	_matrix_files = {}

	import os
	import chimerax
	for parent_dir in [os.path.dirname(__file__), chimerax.app_dirs_unversioned.user_data_dir]:
		mat_dir = os.path.join(parent_dir, "matrices")
		if not os.path.exists(mat_dir):
			continue
		for matrix_file in os.listdir(mat_dir):
			if matrix_file.startswith("."):
				# some editors and file system mounters create files that start
				# with '.' that should be ignored
				continue
			if matrix_file.endswith(".matrix"):
				ftype = "protein"
				name = matrix_file[:-7]
			elif matrix_file.endswith(".dnamatrix"):
				ftype = "dna"
				name = matrix_file[:-10]
			else:
				continue
			path = os.path.join(mat_dir, matrix_file)
			try:
				_matrices[name] = read_matrix_file(path, style=ftype)
			except ValueError:
				chimera.replyobj.reportException()
			_matrix_files[name] = path
	if not _matrices:
		chimerax.logger.warning("No matrices found by sim_matrices module.")
