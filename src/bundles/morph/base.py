class Morph:

        def __init__(self, mols, log, method = 'corkscrew', rate = 'linear', frames = 20, cartesian = False):
                # TODO: grab preferences before initializing default options
                options = {
                        "method": method,
                        "rate": rate,
                        "frames": frames,
                        "cartesian": cartesian,
                }
                self.actions = [(mol, options) for mol in mols]
                self.motion = None
                self.addhMap = {}
                self.log = log

        def __del__(self):
                self.motion = None
                self.actions = []

        def addAction(self, mol, after=None, **kw):
                if after is None:
                        self.actions.append((mol, kw))
                        where = len(self.actions) - 1
                else:
                        self.actions.insert(after + 1, (mol, kw))
                        where = after + 1
                return where

        def moveActionUp(self, which):
                mol, opts = self.actions[which]
                prevMol, prevOpts = self.actions[which - 1]
                self.actions[which - 1] = (mol, prevOpts)
                self.actions[which] = (prevMol, opts)

        def removeAction(self, which):
                del self.actions[which]

        def setOptions(self, which, **kw):
                mol, opts = self.actions[which]
                opts.update(kw)

        def makeTrajectory(self, minimize=False, steps=60):
                #
                # It's an all-or-nothing deal.  If any step requires
                # minimization, all molecules must have hydrogens
                # because the interpolation routines requires a 1-1
                # atom mapping between the two conformations
                #
                if minimize:
                        try:
                                molMap = self._addHydrogens()
                        except ValueError:
                                from chimera.baseDialog import AskYesNoDialog
                                from chimera.tkgui import app
                                d = AskYesNoDialog(
                                        "Cannot minimize with "
                                                "non-identical models.\n"
                                        "Continue without minimization?",
                                        default="Yes")
                                if d.run(app) != "yes":
                                        return None, None
                                minimize = False
                if not minimize:
                        molMap = {}
                        for mol, options in self.actions:
                                molMap[mol] = mol

                import copy
                mol, options = self.actions[0]
                initOptions = copy.copy(options)
                if self.motion is None:
                        from .motion import MolecularMotion
                        self.motion = MolecularMotion(molMap[mol],
                                                        mol.position,
                                                        minimize=minimize,
                                                        steps=steps,
                                                        **initOptions)
                else:
                        self.motion.reset(minimize=minimize, steps=steps,
                                                        **initOptions)
                prevMol = mol
                prevOptions = copy.copy(options)

                for i in range(1, len(self.actions)):
                        msg = "Computing interpolation %d\n" % i
                        self.log.status(msg)
                        mol, options = self.actions[i]
                        prevOptions.update(options)
                        try:
                                self.motion.interpolate(molMap[mol],
                                                        mol.position,
                                                        **prevOptions)
                        except ValueError as msg:
                                raise
                                self.motion = None
                                from chimerax.core.errors import UserError
                                raise UserError("cannot interpolate models: %s"
                                                % msg)
                        prevMol = mol
                return self.motion.trajectory(), self.motion.xform

        def _addHydrogens(self):
                import AddH
                kw = {
                        "delSolvent": False,
                        "nogui": True,
                        "addHFunc": AddH.simpleAddHydrogens,
                }
                import DockPrep
                from .util import mapAtoms, copyMolecule
                refMol = None
                for mol, options in self.actions:
                        if mol in self.addhMap:
                                refMol = mol
                if refMol is None:
                        refMol = self.actions[0][0]
                        self.log.info("Add hydrogens to %s\n"
                                                % refMol.oslIdent())
                        m, aMap, rMap = copyMolecule(refMol, copyPBG=False)
                        DockPrep.prep([m], **kw)
                        self.addhMap[refMol] = m
                for mol, options in self.actions:
                        if mol in self.addhMap:
                                continue
                        amap = mapAtoms(mol, refMol,
                                        ignoreUnmatched=True)
                        for a, refa in amap.iteritems():
                                a.idatmType = refa.idatmType
                        self.log.info("Add hydrogens to %s\n"
                                                % mol.oslIdent())
                        m, aMap, rMap = copyMolecule(mol, copyPBG=False)
                        DockPrep.prep([m], **kw)
                        self.addhMap[mol] = m
                return self.addhMap
