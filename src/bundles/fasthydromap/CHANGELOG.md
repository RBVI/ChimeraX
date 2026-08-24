# Changelog

## 0.1.1

- Exclude nucleic acids, waters, ions, and ligands from the temporary
  FastHydroMap prediction input without deleting them from the open ChimeraX
  model.
- Report how many non-protein residues were omitted.
- Warn during prediction when DNA, RNA, waters, ions, or ligands are ignored,
  and when histidine tautomer/protonation states cannot be represented.
- Add a ChimeraX tutorial explaining why Fdewet is context-aware, what the
  PC1-PC3 sign conventions represent, and how to control the maps.
- Show clickable quick-start and help commands when the managed installation
  finishes.

## 0.1.0

- Initial ChimeraX Toolshed release.
- Predict and color Fdewet, PC1, PC2, and PC3 residue maps.
- Install FastHydroMap and PyTorch in an isolated, bundle-managed environment.
