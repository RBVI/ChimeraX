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

class ModelingError(ValueError):
    pass

def modeller_copy(seq):
	from copy import copy
	mseq = copy(seq)
	mseq.name = mseq.name[:16].replace(' ', '_')
	mseq.characters = "".join([c.upper() if c.isalpha() else '-' for c in mseq.characters])
	return mseq

def opal_safe_file_name(fn):
    return fn.replace(':', '_').replace(' ', '_').replace('|', '_')

def structure_save_name(s):
    return s.name.replace(':', '_').replace(' ', '_') + "_" + s.id_string

def chain_save_name(chain):
    return structure_save_name(chain.structure) + '/' + chain.chain_id.replace(' ', '_')

def regularized_seq(aseq, chain):
    mmap = aseq.match_maps[chain]
    rseq = modeller_copy(aseq)
    rseq.description = "structure:" + chain_save_name(chain)
    seq_chars = list(rseq.characters)
    from chimerax.atomic import Sequence
    from chimerax.pdb import standard_polymeric_res_names as std_res_names
    in_seq_hets = []
    num_res = 0
    for ungapped in range(len(aseq.ungapped())):
        gapped = aseq.ungapped_to_gapped(ungapped)
        if ungapped not in mmap:
            seq_chars[gapped] = '-'
        else:
            r = mmap[ungapped]
            num_res += 1
            if r.name not in std_res_names:
                in_seq_hets.append(r.name)
                seq_chars[gapped] = '.'
            else:
                seq_chars[gapped] = Sequence.rname3to1(mmap[ungapped].name)
    s = chain.structure
    het_set = getattr(s, 'in_seq_hets', set())
    # may want to preserve all-HET chains, so don't auto-exclude them
    if num_res != len(in_seq_hets):
        het_set.update(in_seq_hets)
    s.in_seq_hets = het_set
    rseq.characters = "".join(seq_chars)
    return rseq

def get_license_key(session, license_key):
    from .settings import get_settings
    settings = get_settings(session)
    if license_key is None:
        license_key = settings.license_key
    else:
        settings.license_key = license_key
    if license_key is None:
        from chimerax.core.errors import UserError
        raise UserError("No Modeller license key provided."
            " Get a license key by registering at the Modeller web site.")
    return license_key

def write_modeller_scripts(license_key, num_models, het_preserve, water_preserve, hydrogens, fast,
        loop_info, custom_script, temp_path, thorough_opt, dist_restraints_path, *, version=2):
    """Function to prepare the Modeller scripts.

    Returns (path-to-Modeller-script, path-to-Modeller-XML-config-file,
        temporary dir [tempfile.TemporaryDirectory-like instance])
    """

    if temp_path:
        # not auto-destroyed
        class FakeTmpDir:
            name = temp_path
        temp_dir = FakeTmpDir()
    else:
        import tempfile
        temp_dir = tempfile.TemporaryDirectory(dir=temp_path)

    # Write out a config file
    # Remember to bump the (non-xml) version number when changing contents
    import os
    with open(os.path.join(temp_dir.name, "ModellerScriptConfig.xml"), 'w') as config_file:
        print(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<modeller9v8>\n'
            '\t<key>%s</key>\n'
            '\t<version>%d</version>\n'
            '\t<numModel>%s</numModel>\n'
            '\t<hetAtom>%s</hetAtom>\n'
            '\t<water>%s</water>\n'
            '\t<allHydrogen>%s</allHydrogen>\n'
            '\t<veryFast>%s</veryFast>\n'
            '\t<loopInfo>%s</loopInfo>\n'
            '</modeller9v8>' % (license_key, version, num_models, int(het_preserve),
                int(water_preserve), int(hydrogens), int(fast), repr(loop_info)), file=config_file)

    if custom_script:
        return custom_script, config_file.name, temp_dir

    with open(os.path.join(temp_dir.name, "ModellerModelling.py"), 'w') as script_file:
        # header: read in from script_head.py
        pkg_dir = os.path.dirname(__file__)
        with open(os.path.join(pkg_dir, "script_head.py"), 'r') as f:
            script_file.write(f.read())

        # write body of script based on arg values
        body = '\n'
        if het_preserve:
            body += '# Read in HETATM records from template PDBs\n'
            body += 'env.io.hetatm = True\n\n'
        if water_preserve:
            body += '# Read in water molecules from template PDBs \n'
            body += 'env.io.water = True\n\n'

        body += '# create a subclass of automodel or loopmodel, MyModel.\n'
        body += '# user can further modify the MyModel class, to override certain routines.\n'

        if hydrogens:
            body += 'class MyModel(allhmodel):'
        elif loop_info:
            method_prefix, loop_data = loop_info
            res_range = ",\n".join(["\t\t\tself.residue_range('%s', '%s')" % (start, end)
                                    for start, end in loop_data])
            body += 'class MyModel(%sloopmodel):' % method_prefix
            body += """
    def select_loop_atoms(self):
        from modeller import selection
        return selection(%s)

    def select_atoms(self):
        from modeller import selection
        return selection(%s)

""" % (res_range, res_range)
        else:
            body += 'class MyModel(automodel):'

        if dist_restraints_path:
            body += """
    def customised_function(self): pass

    def special_restraints(self, aln):
        %s

    #code overrides the special_patches method.
    # e.g. to include the addtional disulfides.
    #def special_patches(self, aln):
    """	% _process_dist_restraints(dist_restraints_path)
        else:	# put in a commented-out line for special restraints
            body += """
    def customised_function(self): pass
    #code overrides the special_restraints method

    #def special_restraints(self, aln):

    #code overrides the special_patches method.
    # e.g. to include the addtional disulfides.
    #def special_patches(self, aln):
    """

        if loop_info:
            body += """
a = MyModel(env, sequence = tarSeq,
        # alignment file with template codes and target sequence
        alnfile = 'alignment.ali',
        # name of initial PDB template
        knowns = template[0])

# one fixed model to base loops on
a.starting_model = 1
a.ending_model = 1

# %s loop models
loopRefinement = True
a.loop.starting_model = 1
a.loop.ending_model = %s
a.loop.assess_methods=(assess.DOPE, assess.GA341, assess.normalized_dope)

""" % (num_models, num_models)
        else:
            body += """
a = MyModel(env, sequence = tarSeq,
        # alignment file with template codes and target sequence
        alnfile = 'alignment.ali',
        # PDB codes of the templates
        knowns = template)
# index of the first model
a.starting_model = 1
# index of the last model
a.ending_model = %s
loopRefinement = False

""" % num_models

        if fast:
            body += '# To get an approximate model very quickly\n'
            body += 'a.very_fast()\n\n'

        if thorough_opt:
            body += """
# perform thorough optimization
a.library_schedule = autosched.normal
a.max_var_iterations = 500
a.md_level = refine.slow

"""

        script_file.write(body)

        # Tail part: contains the a.make and data output
        with open(os.path.join(pkg_dir, "script_tail.py"), 'r') as f:
            script_file.write(f.read())

    return script_file.name, config_file.name, temp_dir

#TODO: handle chain IDs?  Does Modeller support that for this purpose?
def _process_dist_restraints(filename):
    """
    Parses the distance restraints file specified by user and returns code that needs
    to be injected into the "special_restraints" method of the Modeller input script.
    The distance restraints file needs four space-seperated numbers on each line.
    Format: res1 res2 dist stdev
    res1 and res2 can be residue ranges (e.g. res1 = 123-789), in which case atoms
    pseudoatom will be created by Modeller.
    """

    from chimerax.core.errors import UserError
    # this function will check whether a residue number is valid
    def verify_residue(value):
        try:
            res = int(value)
        except ValueError:
            raise UserError('The residue nr. %s specified in the additional distance restraints file'
                ' is not an integer.' % value)
        if res <= 0:
            raise UserError('The residue nr. %d specified in the additional distance restraints file'
                ' needs to be greater than 0.' % res)
        return res

    # check whether the specified path is a file:
    from os.path import isfile
    if not isfile(filename):
        raise UserError('The user-specified additional distance restraints file "%s" does not exist'
            " (or isn't a file)." % filename)

    # initialize code that will be returned:
    headcode = """
                rsr = self.restraints
                atm = self.atoms
"""
    maincode = ""

    # parse file:
    with open(filename, 'r') as distrestrfile:
        i = 0 # number of pseudoatoms
        pseudodict = {} # to avoid having duplicate pseudoatoms
        for line in distrestrfile:

            try:
                residues1, residues2, dist, stdev = line.strip().split()
            except TypeError:
                raise UserError('The line "%s" specified in the additional distance restraints file'
                    ' is not exactly four space-seperated values.' % line)

            # check whether dist and stdev are ok:    
            try:
                dist = float(dist)
                stdev = float(stdev)
            except ValueError:
                raise UserError('The distance %s or standard deviation %s specified in the additional'
                    ' distance restraints file is not a real number.' % (dist, stdev))
            if stdev <= 0:
                raise UserError('The standard deviation %f specified in the additional distance restraints'
                    ' file needs to be greater than 0.' % stdev)

            # check whether residue ranges or single residues where specified:
            atoms = []
            for residues in [residues1, residues2]:
                if '-' in residues: # looks like a residue range was specified -> verify
                    try:
                        res1, res2 = residues.split('-')
                    except TypeError:
                        raise UserError('The residue range %s specified in the additional distance'
                            ' restraints file is not valid.' % residues)
                    resA = verify_residue(res1)
                    resB = verify_residue(res2)
                    if (resA, resB) not in pseudodict:
                        i += 1
                        atom = 'pseudo' + str(i)
                        # add to dict:
                        pseudodict[(resA, resB)] = atom
                        # add pseudoatoms to output code:
                        headcode +=    """
                        %s = pseudo_atom.gravity_center(self.residue_range('%d:','%d:'))
                        rsr.pseudo_atoms.append(%s)
        """ % (atom, resA, resB, atom)
                    else:
                        atom = pseudodict[(resA, resB)]
                else: # hopefully, a single residue was specified -> verify    
                    res = verify_residue(residues)
                    atom = "atm['CA:" + str(res) +"']"
                atoms.append(atom)

            # add restraints line to output
            maincode += """
                    rsr.add(forms.gaussian(group=physical.xy_distance, feature=features.distance(%s,%s), mean=%f, stdev=%f))
""" % (atoms[0], atoms[1], dist, stdev)


    # concatenate and return output code:
    return headcode + maincode

from chimerax.core.session import State
class RunModeller(State):

    def __init__(self, session, match_chains, num_models, target_seq_name, targets, *, res_numberings=None):
        self.session = session
        self.match_chains = match_chains
        self.num_models = num_models
        self.target_seq_name = target_seq_name
        self.targets = targets
        self.chain_ids = match_chains.chain_ids
        self.res_numberings = res_numberings

    def process_ok_models(self, ok_models_text, stdout_text, get_pdb_model):
        ok_models_lines = ok_models_text.rstrip().split('\n')
        headers = [h.strip() for h in ok_models_lines[0].split('\t')][1:]
        for i, hdr in enumerate(headers):
            if hdr.endswith(" score"):
                headers[i] = hdr[:-6]
        from chimerax.core.utils import string_to_attr
        attr_names = [string_to_attr(hdr, prefix="modeller_") for hdr in headers]
        from chimerax.atomic import AtomicStructure
        for attr_name in attr_names:
            AtomicStructure.register_attr(self.session, attr_name, "Modeller", attr_type=float)

        from chimerax import match_maker as mm
        models = []
        match_okay = True
        for i, line in enumerate(ok_models_lines[1:]):
            fields = line.strip().split()
            pdb_name, scores = fields[0], [float(f) for f in fields[1:]]
            model = get_pdb_model(pdb_name)
            for attr_name, val in zip(attr_names, scores):
                setattr(model, attr_name, val)
            model.name = self.target_seq_name
            if model.num_chains == len(self.chain_ids):
                model.chains.chain_ids = self.chain_ids
            if self.res_numberings is not None and len(self.res_numberings) == model.num_chains:
                for chain, renumbering in zip(model.chains, self.res_numberings):
                    new_numbers, new_inserts = renumbering
                    existing = chain.existing_residues
                    existing.numbers = new_numbers
                    existing.insertion_codes = new_inserts
            if model.num_chains == len(self.match_chains):
                pairings = list(zip(self.match_chains, model.chains))
                mm.match(self.session, mm.CP_SPECIFIC_SPECIFIC, pairings, mm.defaults['matrix'],
                    mm.defaults['alignment_algorithm'], mm.defaults['gap_open'], mm.defaults['gap_extend'],
                    cutoff_distance=mm.defaults['iter_cutoff'])
            else:
                match_okay = False
            models.append(model)
        if not match_okay:
            self.session.logger.warning("The number of model chains does not match the number used from"
                " the template structure(s) [which can be okay if you closed or modified template"
                " structures while the job was running], so no superposition of the models onto the"
                " templates was performed.")

        reset_alignments = []
        for alignment, target_seq in self.targets:
            alignment.associate(models, seq=target_seq)
            if alignment.auto_associate:
                alignment.auto_associate = False
                reset_alignments.append(alignment)
        self.session.models.add_group(models, name=self.target_seq_name + " models")
        for alignment in reset_alignments:
            alignment.auto_associate = True

        if self.session.ui.is_gui:
            from .tool import ModellerResultsViewer
            ModellerResultsViewer(self.session, "Modeller Results", models, attr_names)

    def take_snapshot(self, session, flags):
        """For session/scene saving"""
        return {
            'chain_ids': self.chain_ids,
            'match_chains': self.match_chains,
            'num_models': self.num_models,
            'res_numberings': self.res_numberings,
            'target_seq_name': self.target_seq_name,
            'targets': self.targets,
        }

    def set_state_from_snapshot(self, data):
        self.chain_ids = data.get('chain_ids', None)
        self.match_chains = data['match_chains']
        self.num_models = data['num_models']
        self.res_numberings = data.get('res_numberings', None)
        self.target_seq_name = data['target_seq_name']
        self.target = data['targets']

class ModellerWebService(RunModeller):

    def __init__(self, session, match_chains, num_models, target_seq_name, input_file_map, config_name,
            targets, **kw):

        super().__init__(session, match_chains, num_models, target_seq_name, targets, **kw)
        self.input_file_map = input_file_map
        self.config_name = config_name

        self.job = None

    def run(self, *, block=False):
        self.job = ModellerJob(self.session, self, self.config_name, self.input_file_map, block)

    def take_snapshot(self, session, flags):
        """For session/scene saving"""
        return {
            'base data': super().take_snapshot(session, flags),
            'input_file_map': self.input_file_map,
            'config_name': self.config_name,
        }

    @staticmethod
    def restore_snapshot(session, data):
        inst = ModellerWebService(session, None, None, None, data['input_file_map'], data['config_name'],
            None, None)
        inst.set_state_from_snapshot(data['base data'])

from chimerax.webservices.opal_job import OpalJob
class ModellerJob(OpalJob):

    OPAL_SERVICE = "Modeller9v8Service"
    SESSION_SAVE = True

    def __init__(self, session, caller, command, input_file_map, block):
        super().__init__(session)
        self.caller = caller
        self.start(self.OPAL_SERVICE, command, input_file_map=input_file_map, blocking=block)

    def monitor(self):
        super().monitor()
        stdout = self.get_file("stdout.txt")
        num_done = stdout.count('# Heavy relative violation of each residue is written to:')
        num_done = max(stdout.count('>> Normalized DOPE z score') - 1, 0)
        status = self.session.logger.status
        tsafe = self.session.ui.thread_safe
        if not num_done:
            tsafe(status, "No models generated yet")
        else:
            tsafe(status, "%d of %d models generated" % (num_done, self.caller.num_models))

    def next_check(self):
        return 15

    def on_finish(self):
        logger = self.session.logger
        logger.info("Modeller job ID %s finished" % self.job_id)
        if not self.exited_normally():
            err = self.get_file("stderr.txt")
            if self.fail_callback:
                self.fail_callback(self, err)
                return
            if err:
                raise RuntimeError("Modeller failure; standard error:\n" + err)
            else:
                raise RuntimeError("Modeller failure with no error output")
        try:
            model_info = self.get_file("ok_models.dat")
        except KeyError:
            try:
                stdout = self.get_file("stdout.txt")
                stderr = self.get_file("stderr.txt")
            except KeyError:
                raise RuntimeError("No output from Modeller")
            logger.info("<br><b>Modeller error output</b>", is_html=True)
            logger.info(stderr)
            logger.info("<br><b>Modeller run output</b>", is_html=True)
            logger.info(stdout)
            from chimerax.core.errors import NonChimeraError
            raise NonChimeraError("No output models from Modeller; see log for Modeller text output.")
        try:
            stdout = self.get_file("stdout.txt")
        except KeyError:
            raise RuntimeError("No standard output from Modeller job")
        def get_pdb_model(fname):
            from io import StringIO
            try:
                pdb_text = self.get_file(fname)
            except KeyError:
                raise RuntimeError("Could not find Modeller out PDB %s on server" % fname)
            from chimerax.pdb import open_pdb
            return open_pdb(self.session, StringIO(pdb_text), fname)[0][0]
        self.caller.process_ok_models(model_info, stdout, get_pdb_model)
        self.caller = None

