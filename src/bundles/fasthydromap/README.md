# FastHydroMap for ChimeraX

![FastHydroMap overview](https://raw.githubusercontent.com/samlobe/FastHydroMap/main/images/FastHydroMap_image.png)

This ChimeraX bundle predicts per-residue FastHydroMap Fdewet and water-structure
PC maps and colors the corresponding atoms, cartoons, and molecular surfaces.
Non-protein components such as nucleic acids, waters, ions, and ligands remain in
the ChimeraX scene but are excluded from prediction.

Install the bundle from ChimeraX's **Tools > More Tools...** page or run:

```text
toolshed install FastHydroMap
```

The first use requires a one-time installation of FastHydroMap and PyTorch into
an isolated environment managed by the bundle:

```text
fasthydromap install
```

Open a protein structure and run:

```text
fasthydromap #1
```

The `quantity` option accepts `fdewet`, `pc1`, `pc2`, or `pc3`. For example:

```text
fasthydromap #1 quantity pc1 target cs
```

See `help fasthydromap` inside ChimeraX for all options and installation notes.
For a complete walkthrough, including why Fdewet is context-aware and how to
interpret the PC1-PC3 sign conventions, see the
[FastHydroMap ChimeraX tutorial](https://github.com/samlobe/FastHydroMap/blob/main/docs/ChimeraX_tutorial.md).

FastHydroMap is developed at <https://github.com/samlobe/FastHydroMap>.
The method is described in Lobo, Najafi, Shea, and Shell,
[*Context-Aware Hydrophobicity Modeling: HydroMap and FastHydroMap*](https://doi.org/10.64898/2026.06.07.730647).
