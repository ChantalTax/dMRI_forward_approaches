# Simulating the diffusion MRI signal

© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.

Interactive [marimo](https://marimo.io) notebooks with exercises, worked answers and visualisations.

| Ch | Topic | Run in your browser |
|----|-------|---------------------|
| 1 | Foundations: notation, diffusion equation, interface and boundary conditions, eigenmodes, Bloch–Torrey, length scales, Feynman–Kac | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ChantalTax/dMRI_forward_approaches/blob/main/simulation/ch1_foundations.py/wasm) |
| 2 | Analytical formulas: the regimes where the signal has a closed form | coming |
| 3 | Matrix methods: eigenmodes coupled by the gradient | coming |
| 4 | Finite differences and finite elements | coming |
| 5 | Monte Carlo | coming |

The "Open in molab" link runs the notebook in your browser (WebAssembly). Nothing needs to be installed. The first load takes a minute while Python, NumPy and Matplotlib download. On a phone or tablet the browser version is heavy; a laptop works best.

Answers are folded under each exercise. The **Teacher view** switch at the top shows all answers at once.

## Run locally

```bash
pip install marimo numpy matplotlib
marimo run simulation/ch1_foundations.py    # lesson view, code hidden
marimo edit simulation/ch1_foundations.py   # editing view
```

With [uv](https://docs.astral.sh/uv/) the dependencies listed at the top of the notebook are installed for you:

```bash
uvx marimo run --sandbox simulation/ch1_foundations.py
```
