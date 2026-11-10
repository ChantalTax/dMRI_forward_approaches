# Simulating the diffusion MRI signal

© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.

Interactive [marimo](https://marimo.io) notebooks with exercises, worked answers and visualisations.

| Ch | Topic | Run in your browser |
|----|-------|---------------------|
| 1 | Foundations: notation, diffusion equation, interface and boundary conditions, eigenmodes, displacement cumulants, Bloch–Torrey, the full multi-compartment problem and its weak form, length scales, Feynman–Kac | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ChantalTax/dMRI_forward_approaches/blob/add-simulation-ch1/simulation/ch1_foundations.py/wasm) |
| 2 | Series expansions: matrix methods (MCF, multiple propagator, number of modes) and the cumulant expansion (Gaussian phase, kurtosis, finite pulses) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ChantalTax/dMRI_forward_approaches/blob/add-simulation-ch1/simulation/ch2_series_expansions.py/wasm) |
| 3 | Analytical formulas: the PGSE parameter space, the narrow-pulse (q,t) map, the long-pulse (g,T) map (free, motional narrowing, localisation), when the Gaussian phase approximation holds, relaxing walls and exchange | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ChantalTax/dMRI_forward_approaches/blob/add-simulation-ch1/simulation/ch3_analytical_formulas.py/wasm) |
| 4 | Grids and meshes: the method of lines, finite differences (time stepping, stability, jump probabilities and the matrix form, periodic cells), finite elements (weak form, the full system with relaxation and membranes, modes on a mesh), curved walls | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ChantalTax/dMRI_forward_approaches/blob/add-simulation-ch1/simulation/ch4_grids_and_meshes.py/wasm) |
| 5 | Monte Carlo | coming |

<!-- The molab links point to the add-simulation-ch1 branch. After merging, replace blob/add-simulation-ch1/ with blob/main/. -->

The "Open in molab" link runs the notebook in your browser (WebAssembly). Nothing needs to be installed. The first load takes a minute while Python, NumPy and Matplotlib download. On a phone or tablet the browser version is heavy; a laptop works best.

Answers are folded under each exercise. The **Teacher view** switch at the top shows all answers at once.

## Run locally

```bash
pip install marimo numpy matplotlib
marimo run simulation/ch1_foundations.py    # lesson view, code hidden
marimo edit simulation/ch1_foundations.py   # editing view
marimo run simulation/ch2_series_expansions.py
marimo run simulation/ch3_analytical_formulas.py
marimo run simulation/ch4_grids_and_meshes.py
```

With [uv](https://docs.astral.sh/uv/) the dependencies listed at the top of the notebook are installed for you:

```bash
uvx marimo run --sandbox simulation/ch1_foundations.py
```
