# FastHydroMap for ChimeraX

This ChimeraX bundle predicts per-residue FastHydroMap Fdewet and water-structure
PC maps and colors the corresponding atoms, cartoons, and molecular surfaces.

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

FastHydroMap is developed at <https://github.com/samlobe/FastHydroMap>.
