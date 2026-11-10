# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib"]
# ///
# © 2026 Chantal Tax. All rights reserved.
# Shared for viewing only; not licensed for reuse or redistribution.

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="dMRI simulation · Ch 3 · Analytical formulas",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def expm(A):
        """Matrix exponential by scaling and squaring with a Taylor series (no SciPy needed)."""
        A = np.asarray(A)
        nrm = np.abs(A).sum(axis=0).max()
        s = max(0, int(np.ceil(np.log2(nrm))) + 1) if nrm > 0 else 0
        X = A / 2.0**s
        E = np.eye(A.shape[0], dtype=X.dtype)
        term = E.copy()
        for k in range(1, 20):
            term = term @ X / k
            E = E + term
        for _ in range(s):
            E = E @ E
        return E

    # Small NumPy versions of the Bessel functions we need (no SciPy, so it runs in the browser).
    _tau_q = np.linspace(0, np.pi, 257)
    _w_q = np.full(_tau_q.size, _tau_q[1])
    _w_q[[0, -1]] /= 2

    def jv(m, x):
        """Bessel function J_m(x), integer m, from its integral representation."""
        x = np.asarray(x, float)
        integrand = np.cos(m * _tau_q - x[..., None] * np.sin(_tau_q))
        return (integrand @ _w_q) / np.pi

    def sph_j1(x):
        """Spherical Bessel function j_1(x)."""
        x = np.asarray(x, float)
        return np.sin(x) / x**2 - np.cos(x) / x

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    C_MAIN = "#2a6fdb"   # exact / main curve
    C_FREE = "#9aa5b1"   # free-diffusion reference
    C_ALT = "#d9622b"    # approximation
    C_GREEN = "#2e9e6a"  # third formula
    C_PURPLE = "#8a5cd1" # fourth formula
    # gamma in units that fit µm, ms and mT/m: phase (rad) = GAMMA * g[mT/m] * x[µm] * t[ms]
    GAMMA = 2.6752e-4
    D0 = 2.0             # µm²/ms, used throughout this chapter
    return (
        C_ALT,
        C_FREE,
        C_GREEN,
        C_MAIN,
        C_PURPLE,
        D0,
        GAMMA,
        ListedColormap,
        expm,
        jv,
        mo,
        np,
        plt,
        sph_j1,
    )


@app.cell(hide_code=True)
def _(D0, GAMMA, expm, np):
    def slab_B(L, N):
        """Eigenvalues λ_n and the matrix B_nm = ∫ u_n (x − L/2) u_m dx of a slab, n, m = 0..N−1."""
        n = np.arange(N)
        m, k = n[:, None], n[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            B = (L / np.pi**2) * ((((-1.0) ** (m - k)) - 1) / (m - k) ** 2 + (((-1.0) ** (m + k)) - 1) / (m + k) ** 2)
            j = np.maximum(m, k)
            B0 = np.sqrt(2) * L * (((-1.0) ** j) - 1) / (j * np.pi) ** 2
        B = np.where((m == 0) | (k == 0), B0, B)
        B = np.where(m == k, 0.0, B)
        return (n * np.pi / L) ** 2, B

    def pgse_mcf(G, L, delta, Delta, N=40, D=D0):
        """Exact PGSE echo in a slab with Grebenkov's MCF (Chapter 2): three matrix exponentials."""
        lam, B = slab_B(L, N)
        Lam = np.diag(lam)
        P1 = expm(-(D * Lam + 1j * GAMMA * G * B) * delta)
        P2 = expm(-D * Lam * (Delta - delta))
        P3 = expm(-(D * Lam - 1j * GAMMA * G * B) * delta)
        return (P3 @ P2 @ P1)[0, 0]

    def pgse_box(Gam, dt, Dt, N=40):
        """Exact PGSE echo in a slab in dimensionless units (L = D = 1).

        Gam = γgL³/D, dt = Dδ/L², Dt = DΔ/L² (dt and Dt may be arrays of the same shape).
        One eigendecomposition of DΛ + iγgB serves every δ and Δ, so a whole column of a map
        costs one call."""
        lam, B = slab_B(1.0, N)
        mu, V = np.linalg.eig(np.diag(lam) + 1j * Gam * B)
        Vi = np.linalg.inv(V)
        dt = np.asarray(dt, float)[..., None]
        Dt = np.asarray(Dt, float)[..., None]
        c = (np.exp(-mu * dt) * Vi[:, 0]) @ V.T                 # first lobe
        c = np.exp(-lam * (Dt - dt)) * c                         # gap: diffusion only
        w = c @ np.conj(Vi).T                                    # second lobe (−g): conjugate modes
        return (np.exp(-np.conj(mu) * dt) * w) @ np.conj(V)[0]

    def gpa_box(Gam, dt, Dt, nmax=2001):
        """ln E of a PGSE in a slab in the Gaussian phase approximation (Chapter 2), dimensionless."""
        n = np.arange(1, nmax, 2)
        w, a = 8 / (n * np.pi) ** 4, (n * np.pi) ** 2
        dt = np.asarray(dt, float)[..., None]
        Dt = np.asarray(Dt, float)[..., None]
        cross = 0.5 * (np.exp(-a * (Dt - dt)) + np.exp(-a * (Dt + dt))) - np.exp(-a * Dt)
        I = (4 * (a * dt - 1 + np.exp(-a * dt)) - 4 * cross) / a**2
        return -0.5 * np.asarray(Gam, float) ** 2 * np.sum(w * I, -1)

    def slab_np(qL, tau, nmax=300):
        """Narrow-pulse signal of a slab (Tanner & Stejskal 1968); qL and tau = Dt/L² broadcast."""
        z = np.asarray(qL, float)[..., None]
        tau = np.asarray(tau, float)[..., None]
        n = np.arange(1, nmax + 1)
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        E0 = 2 * (1 - np.cos(z)) / z**2
        den = z**2 - (n * np.pi) ** 2
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        terms = 4 * z**2 * np.exp(-((n * np.pi) ** 2) * tau) * (1 - (-1.0) ** n * np.cos(z)) / den**2
        return E0[..., 0] + terms.sum(-1)

    def slab_cumulants(tau, N=1500):
        """Displacement D(t)/D and K(t) in a slab (Chapter 1); tau = D t / L²."""
        tau = np.asarray(tau, float)
        n = np.arange(1, N + 1)[:, None]
        k = np.pi * n
        odd = n % 2 == 1
        c1 = np.where(odd, -2 * np.sqrt(2) / k**2, 0.0)
        c3 = np.where(odd, np.sqrt(2) * (-1.5 / k**2 + 12 / k**4), 0.0)
        c2 = np.where(~odd, 2 * np.sqrt(2) / k**2, 0.0)
        e = np.exp(-(k**2) * tau[None, :])
        X2 = 1 / 6 - 2 * (e * c1 * c1).sum(0)
        X4 = 2 / 80 + 6 / 144 - 8 * (e * c3 * c1).sum(0) + 6 * (e * c2 * c2).sum(0)
        return X2 / (2 * tau), X4 / X2**2 - 3

    return gpa_box, pgse_box, pgse_mcf, slab_B, slab_cumulants, slab_np


@app.cell(hide_code=True)
def _(mo):
    show_answers = mo.ui.switch(label="Teacher view: show all answers", value=False)
    mo.vstack(
        [
            mo.md(
                r"""
    # Chapter 3 · Analytical formulas: a map of the PGSE parameter space

    *© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.*

    *Course: simulating the diffusion MRI signal. Chapter 2 computed the signal exactly, for any gradient. This chapter asks where it has a closed form.*

    **What you'll take away**

    1. A PGSE experiment in a pore is fixed by **three dimensionless numbers**, $D\Delta/L^2$, $D\delta/L^2$ and $\gamma gL^3/D$. Every closed formula lives in one corner of this box.
    2. On the **narrow-pulse face** the signal is $\langle e^{iqX}\rangle$. The $(q,t)$ map has four regions: free diffusion, short times (surface-to-volume ratio), long times (the pore form factor and diffraction), and the region where cumulants work.
    3. On the **long-pulse face** the $(g,T)$ map has three regimes, free diffusion, motional narrowing and localisation, which meet at a triple point.
    4. **Inside the box**, for real PGSE sequences: how the edge of the cumulant region moves with $\delta/\Delta$, and a rule for it, tested against the exact signal.
    5. Two **extra axes**: walls that relax magnetization (Brownstein–Tarr) and exchange between compartments (Kärger).
    6. A **one-page crib sheet** with every formula, each one checked against the exact solutions of Chapter 2.

    Every exercise has hints and a worked answer, folded away. Try the exercise first, then open the hint, then the answer.
    """
            ),
            show_answers,
        ]
    )
    return (show_answers,)


@app.cell(hide_code=True)
def _(mo, show_answers):
    def maths(text, title="The maths"):
        """A derivation in the main text, folded away."""
        return mo.accordion({f"▸ {title}": mo.md(text)})

    def hint(text, title="Hint"):
        return mo.accordion({f"▸ {title}": mo.md(text)})

    def answer(text, title="Answer"):
        """Worked answers: folded for students, shown inline in teacher view."""
        if show_answers.value:
            return mo.callout(mo.md(f"**{title}**\n\n" + text), kind="success")
        return mo.accordion({f"▸ {title}": mo.md(text)})

    def exercise(text, number):
        return mo.callout(mo.md(f"### ✎ Exercise {number}\n\n" + text), kind="info")

    return answer, exercise, hint, maths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The course at a glance

    | Ch | Topic | Status |
    | --- | --- | --- |
    | 1 | Foundations: notation, diffusion equation, interface and boundary conditions, eigenmodes, displacement cumulants; Bloch–Torrey; the full multi-compartment problem and its weak form; length scales; Feynman–Kac | done |
    | 2 | Series expansions: matrix methods (MCF, multiple propagator, number of modes) and the cumulant expansion (Gaussian phase, kurtosis, finite pulses) | done |
    | **3** | **Analytical formulas:** the PGSE box, the narrow-pulse $(q,t)$ map, the long-pulse $(g,T)$ map, the inside of the box, relaxing walls and exchange | this notebook |
    | 4 | Grids and meshes: the method of lines, finite differences, time stepping and stability, periodic cells, finite elements, curved walls | done |
    | 5 | Monte Carlo: simulating the random walk | coming |

    **Units:** lengths in µm, times in ms, gradients in mT/m, $\gamma = 2.6752\times10^{-4}$ rad/(ms·mT/m·µm). $D = 2$ µm²/ms throughout. **Running example:** a slab of width $L = 8$ µm with the gradient across it, so $L^2/D = 32$ ms. Where a formula exists for cylinders and spheres too, it is given alongside.

    **How to read this chapter.** Each section gives a formula, the region where it holds, and a figure that **computes** that region by comparing the formula with the exact signal of Chapter 2. The maps are not sketches. Each line on them is a length-scale estimate, and the coloured regions show where the formulas actually work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you need from Chapters 1 and 2

    | Idea | Formula |
    | --- | --- |
    | Length scales | pore size $L$; diffusion length $\ell_D = \sqrt{Dt}$; gradient length $\ell_g = (D/\gamma g)^{1/3}$; phase wavelength $1/q$ |
    | Averages $\langle\cdot\rangle$ | the mean over many spins, each starting at a uniformly random $x_0$ in the pore (volume $V$) and following its own random path: $\langle f\rangle = \lim_{N\to\infty}\frac1N\sum_{j=1}^N f_j$ |
    | … of the displacement | $X = x(t) - x_0$: $\ \langle f(X)\rangle = \int f(X)\,\bar P(X,t)\,dX$, with the average propagator $\bar P(X,t) = \frac1V\int_V P(x_0 + X\mid x_0,t)\,dx_0$ |
    | Displacement cumulants | $\langle X^2\rangle = \int X^2\bar P(X,t)\,dX = 2D(t)\,t$, $\ K(t) = \langle X^4\rangle/\langle X^2\rangle^2 - 3$, with $\langle X^4\rangle = \int X^4\bar P(X,t)\,dX$ |
    | Slab, limits | $D(t)\to D$ and $K\to0$ for $t\to0$; $\ \langle X^2\rangle\to L^2/6$ and $K\to-3/5$ for $t\to\infty$ |
    | Slab modes | $u_0 = 1/\sqrt L$, $\ u_n = \sqrt{2/L}\cos(n\pi x/L)$, $\ \lambda_n = (n\pi/L)^2$, $\ \tau_1 = L^2/(\pi^2D)$ |
    | Path picture | $E = \langle e^{-i\varphi}\rangle = \lim_{N\to\infty}\frac1N\sum_{j=1}^N e^{-i\varphi_j}$, $\ \varphi_j = \gamma\int_0^T g(t)\,x_j(t)\,dt$ for spin $j$ with path $x_j(t)$ |
    | Exact PGSE (MCF) | $E = [e^{-(D\Lambda - i\gamma G\mathbf B)\delta}\,e^{-D\Lambda(\Delta-\delta)}\,e^{-(D\Lambda + i\gamma G\mathbf B)\delta}]_{00}$ |
    | Gaussian phase (GPA) | $\ln E \approx -\frac12\langle\varphi^2\rangle$, $\ \langle\varphi^2\rangle = \gamma^2\iint g(t_1)g(t_2)\,\langle x(t_1)x(t_2)\rangle\,dt_1dt_2$, with $\langle x(t_1)x(t_2)\rangle = \frac1V\iint x_0\,x\,P(x\mid x_0,\lvert t_2-t_1\rvert)\,dx\,dx_0 = \sum_n B_{0n}^2\,e^{-D\lambda_n\lvert t_1-t_2\rvert}$ |
    | Verified in Ch 2 | GPA 5% off at $G \approx 205$ mT/m for $\delta = 10$, $\Delta = 20$ ms; complex zero of $E(G)$ near $463 + 20i$ mT/m |

    **Refresh first**

    - Fourier transform of a box of width $L$: $\int_0^L e^{iqx}dx\,/L = e^{iqL/2}\,\operatorname{sinc}(qL/2)$, with $\operatorname{sinc}z = \sin z/z$. Its zeros are at $qL = 2\pi, 4\pi, \dots$
    - $\ln(1+\epsilon) = \epsilon - \epsilon^2/2 + \dots$, so a 5% error in $E$ is an error of 0.05 in $\ln E$.
    - The Airy function $\mathrm{Ai}(\xi)$ solves $y'' = \xi y$ and decays for $\xi\to+\infty$. The first zero of $\mathrm{Ai}'$ is at $a_1' = -1.0188$.
    - *Your notes:* 03-PDE, the fundamental solution and the method of images (short times); heat conduction through layers in series (tortuosity).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3.0 · The PGSE box

    A PGSE sequence in a pore has five dimensional inputs: $\Delta$, $\delta$, $g$, the pore size $L$ and the diffusivity $D$. Only three dimensionless combinations matter. The natural units are the pore size for length and the pore time $L^2/D$ for time:

    | Number | Definition | As a ratio of lengths | Running example ($\delta = 10$, $\Delta = 20$ ms, $G = 205$ mT/m) |
    | --- | --- | --- | --- |
    | diffusion time | $\tilde\Delta = D\Delta/L^2$ | $(\ell_\Delta/L)^2$, with $\ell_\Delta = \sqrt{D\Delta}$ | 0.63 |
    | pulse duration | $\tilde\delta = D\delta/L^2$ | $(\ell_\delta/L)^2$ | 0.31 |
    | gradient strength | $\Gamma = \gamma gL^3/D$ | $(L/\ell_g)^3$ | 14.0, so $\ell_g = 3.3$ µm |

    The box is the set $0 < \tilde\delta \le \tilde\Delta$, $\Gamma > 0$. It has two faces where the physics simplifies:

    - **The narrow-pulse face**, $\tilde\delta\to0$. With $\Gamma\tilde\delta = \gamma g\delta L = qL$ held fixed, the gradient and the pulse length merge into one number, $q = \gamma g\delta$. The face is the **$(q,t)$ plane** (Section 3a).
    - **The long-pulse face**, $\tilde\delta = \tilde\Delta$. The two lobes touch, and the effective gradient is $+g$ for $T/2$ and then $-g$ for $T/2$, with $T = 2\delta$. This is a spin echo in a constant gradient. The face is the **$(g,T)$ plane** (Section 3b).

    Real sequences sit **inside the box** (Section 3c). The sliders below place a sequence in it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    box_delta = mo.ui.slider(0.5, 40, step=0.5, value=10, label="δ (ms)", show_value=True)
    box_Delta = mo.ui.slider(1, 80, step=1, value=20, label="Δ (ms)", show_value=True)
    box_G = mo.ui.slider(10, 1500, step=5, value=205, label="G (mT/m)", show_value=True)
    box_L = mo.ui.slider(2, 20, step=1, value=8, label="Pore size L (µm)", show_value=True)
    mo.vstack([mo.hstack([box_delta, box_Delta], justify="start", gap=2), mo.hstack([box_G, box_L], justify="start", gap=2)])
    return box_Delta, box_G, box_L, box_delta


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, GAMMA, box_Delta, box_G, box_L, box_delta, mo, np, plt):
    _L, _de, _G = box_L.value, box_delta.value, box_G.value
    _De = max(box_Delta.value, _de)
    _dt, _Dt = D0 * _de / _L**2, D0 * _De / _L**2
    _Gam = GAMMA * _G * _L**3 / D0
    _lg = (D0 / (GAMMA * _G)) ** (1 / 3)

    _fig, _a = plt.subplots(figsize=(6.4, 4.2))
    _x = np.logspace(-3, 1.5, 200)
    _a.fill_between(_x, _x, 40, color=C_FREE, alpha=0.12, lw=0)
    _a.plot(_x, _x, color=C_ALT, lw=2.2)
    _a.axvline(1e-3, color=C_MAIN, lw=4)
    _a.text(0.15, 2.4, "long-pulse face  δ = Δ\n(g,T) map, Section 3b", color=C_ALT, fontsize=8.5, rotation=0)
    _a.text(1.25e-3, 12, "narrow-pulse face\nδ → 0: (q,t) map\nSection 3a", color=C_MAIN, fontsize=8.5)
    _a.text(3e-3, 0.3, "inside the box:\nreal PGSE, Section 3c", color="0.35", fontsize=8.5)
    _a.axhline(1, color="0.5", lw=0.8, ls=":")
    _a.axvline(1, color="0.5", lw=0.8, ls=":")
    _a.text(12, 1.12, "Δ = L²/D", color="0.4", fontsize=8)
    _a.plot(_dt, _Dt, "o", ms=10, color="k", mfc="white", mew=2)
    _a.annotate(f"Γ = γgL³/D = {_Gam:.1f}\nL/ℓ_g = {_L / _lg:.2f}", (_dt, _Dt), xytext=(12, -28), textcoords="offset points", fontsize=8.5)
    _a.set(xscale="log", yscale="log", xlim=(1e-3, 30), ylim=(1e-2, 30), xlabel="δ̃ = Dδ/L²", ylabel="Δ̃ = DΔ/L²", title="Where your sequence sits (δ ≤ Δ: shaded half)")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"δ̃ = **{_dt:.3g}**, Δ̃ = **{_Dt:.3g}**, Γ = **{_Gam:.3g}**. Length scales: L = {_L} µm, "
                f"ℓ_Δ = √(DΔ) = **{np.sqrt(D0 * _De):.2f} µm**, ℓ_δ = √(Dδ) = {np.sqrt(D0 * _de):.2f} µm, ℓ_g = **{_lg:.2f} µm**, "
                f"q = γGδ = {GAMMA * _G * _de:.3f} µm⁻¹ (qL = {GAMMA * _G * _de * _L:.2f})."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Use the sliders to compute $(\tilde\delta, \tilde\Delta, \Gamma)$ for $L = 8$ µm and three sequences: a clinical one ($\delta = 20$, $\Delta = 40$ ms, $G = 80$ mT/m), a Connectom-type one ($\delta = 8$, $\Delta = 20$ ms, $G = 300$ mT/m) and a preclinical one ($\delta = 2$, $\Delta = 10$ ms, $G = 1000$ mT/m). Which face is each sequence closest to?

    **(b)** Show that on the narrow-pulse face only $\Gamma\tilde\delta$ survives, and that it equals $qL$.

    **(c)** Show that on the long-pulse face the three numbers become the three length scales $L$, $\ell_D = \sqrt{DT}$ and $\ell_g$, with $\Gamma = (L/\ell_g)^3$ and $DT/L^2 = (\ell_D/L)^2$.
    """, "1 · Three numbers")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ With $L^2/D = 32$ ms and $\Gamma = \gamma GL^3/D = 0.0685\,G$ (G in mT/m):

    | Sequence | $\tilde\delta$ | $\tilde\Delta$ | $\Gamma$ | $\delta/\Delta$ | Closest face |
    | --- | --- | --- | --- | --- | --- |
    | clinical | 0.63 | 1.25 | 5.5 | 0.5 | long-pulse ($\delta \gtrsim L^2/D$) |
    | Connectom-type | 0.25 | 0.63 | 20.5 | 0.4 | in between |
    | preclinical | 0.063 | 0.31 | 68.5 | 0.2 | narrow-pulse |

    → Only the preclinical sequence comes near the narrow-pulse face. For an 8 µm pore, clinical pulses are long. For a 2 µm axon the clinical $\tilde\delta$ is 16 times larger still, so axon-diameter experiments always live near the long-pulse face.

    **(b)** ⇒ The phase from a narrow pulse is $\gamma g\delta\,x$. In units of $L$ it is $(\gamma g\delta L)(x/L) = (\Gamma\tilde\delta)(x/L)$. Letting $\tilde\delta\to0$ with $\Gamma\tilde\delta$ fixed keeps that phase, and the pulse length itself no longer appears. So only $qL = \Gamma\tilde\delta$ and $\tilde\Delta$ remain.

    **(c)** ⇒ $\Gamma = \gamma gL^3/D = L^3/\ell_g^3$, since $\ell_g^3 = D/(\gamma g)$. With $T = 2\delta$, $DT/L^2 = \ell_D^2/L^2$. Two numbers, and on this face they say which of the three lengths is the smallest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3a · The narrow-pulse face: the $(q,t)$ map

    ## From the path picture to $\langle e^{iqX}\rangle$

    Suppose both gradient pulses are so short that spins do not move while they are on. The first pulse gives each spin the phase $\gamma g\delta\,x(0)$. After the refocusing pulse, the second removes $\gamma g\delta\,x(\Delta)$. So $\varphi = -qX$, with $q = \gamma g\delta$ and the displacement $X = x(\Delta) - x(0)$, and

    $$E(q,\Delta) = \big\langle e^{iqX}\big\rangle = \int \bar P(X,\Delta)\,e^{iqX}\,dX ,$$

    the characteristic function of the displacement: the Fourier transform of the average propagator $\bar P$ of Chapter 1 ([Stejskal & Tanner 1965](https://doi.org/10.1063/1.1695690); [Tanner & Stejskal 1968](https://doi.org/10.1063/1.1670306)). In powers of $q$, the cumulants of $\ln E$ are exactly the displacement cumulants, $\langle X^2\rangle$, $K(t)$, …. With finite pulses this identity breaks, because the phase then depends on the whole path, not only on its two end points (Chapter 2, Exercise 6).

    → **When is a pulse narrow?** A spin must not move far during a pulse compared with the two lengths the signal resolves: $\sqrt{D\delta} \ll L$ and $\sqrt{D\delta} \ll 1/q$. In addition $\delta \ll \Delta$.

    ## The exact narrow-pulse signal of a slab

    Put the eigenmode propagator of Chapter 1 into the Fourier integral. Every mode contributes its own decay $e^{-D\lambda_nt}$, weighted by how strongly it overlaps with the phase pattern $e^{iqx}$:

    $$E(q,t) = \sum_n \frac{\lvert\tilde u_n(q)\rvert^2}{L}\,e^{-D\lambda_nt},\qquad \tilde u_n(q) = \int_0^L u_n(x)\,e^{iqx}\,dx .$$

    For a slab this gives the closed form of [Tanner & Stejskal (1968)](https://doi.org/10.1063/1.1670306):

    $$E(q,t) = \frac{2\,(1-\cos qL)}{(qL)^2} + 4\,(qL)^2\sum_{n\ge1} e^{-n^2\pi^2Dt/L^2}\;\frac{1-(-1)^n\cos qL}{\big[(qL)^2-(n\pi)^2\big]^2}.$$

    It depends on $qL$ and $Dt/L^2$ only: these are the two coordinates of the narrow-pulse face. The first term is the $t\to\infty$ limit. The sum, which dies out on the pore time $\tau_1 = L^2/(\pi^2D)$, carries all the dynamics. Chapter 2's MCF reproduces it as $\delta\to0$; the check below uses $\delta = 0.05$ ms.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ① **Propagator.** $P(x\mid x_0,t) = \sum_n u_n(x)u_n(x_0)e^{-D\lambda_nt}$ (Chapter 1).

    ② **Average over starting points.** Spins start uniformly, with density $1/L$:

    $$E = \frac1L\int_0^L\!\!\int_0^L P(x\mid x_0,t)\,e^{iq(x-x_0)}\,dx\,dx_0 = \frac1L\sum_n e^{-D\lambda_nt}\Big(\int u_n(x)e^{iqx}dx\Big)\Big(\int u_n(x_0)e^{-iqx_0}dx_0\Big) .$$

    ⇒ The modes are real, so the second integral is the complex conjugate of the first: $E = \sum_n e^{-D\lambda_nt}\lvert\tilde u_n(q)\rvert^2/L$.

    ③ **Mode $n = 0$.** $\tilde u_0 = (e^{iqL}-1)/(iq\sqrt L)$, so $\lvert\tilde u_0\rvert^2/L = 2(1-\cos qL)/(qL)^2 = \operatorname{sinc}^2(qL/2)$.

    ④ **Modes $n\ge1$.** Write $\cos k_nx = \frac12(e^{ik_nx}+e^{-ik_nx})$ with $k_n = n\pi/L$, and integrate:

    $$\tilde u_n = \sqrt{\tfrac2L}\;\frac{iq\,\big[1-(-1)^ne^{iqL}\big]}{k_n^2-q^2} ,\qquad \frac{\lvert\tilde u_n\rvert^2}{L} = \frac{2q^2\,\big[2-2(-1)^n\cos qL\big]}{L^2\,(k_n^2-q^2)^2} = \frac{4(qL)^2\,\big[1-(-1)^n\cos qL\big]}{\big[(qL)^2-(n\pi)^2\big]^2}.$$

    → At $q = 0$ only $n = 0$ survives, with weight 1, so $E = 1$. ✓

    → **Every term is non-negative.** The narrow-pulse echo of a closed pore can approach zero but never becomes negative. That is no longer true for finite pulses.

    → Same structure as the MSD weights of Chapter 1: there the weights were $\lvert\int u_n\,x\rvert^2$, here they are $\lvert\int u_n\,e^{iqx}\rvert^2$. Expanding $e^{iqx}$ in powers of $q$ turns one into the other.
    """, title="Derivation (Tanner–Stejskal)")
    return


@app.cell(hide_code=True)
def _(D0, GAMMA, mo, np, pgse_mcf, slab_np):
    _L = 8.0
    _rows = []
    for _qq in (0.25, 0.5, 1.5):
        _q = _qq * 2 * np.pi / _L
        _G = _q / (GAMMA * 0.05)
        _E_mcf = pgse_mcf(_G, _L, 0.05, 20.0, N=60).real
        _E_ts = slab_np(_q * _L, D0 * 20.0 / _L**2)
        _rows.append(f"| {_qq:.2f} | {_G:,.0f} | {_E_mcf:.4f} | {float(_E_ts):.4f} |")
    mo.md(
        "**Check against Chapter 2** (slab L = 8 µm, Δ = 20 ms; MCF with δ = 0.05 ms and 60 modes):\n\n"
        "| qL/2π | G (mT/m) | MCF, δ = 0.05 ms | Tanner–Stejskal, δ → 0 |\n| --- | --- | --- | --- |\n"
        + "\n".join(_rows)
        + "\n\nThe remaining difference shrinks roughly in proportion to δ (it is 2% at δ = 0.2 ms)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    np_check = mo.ui.radio(
        options=[
            "Yes: near qL = 2π the echo becomes slightly negative",
            "No: it can approach zero, but never goes below it",
        ],
        label="**Predict first:** at a finite diffusion time, can the narrow-pulse echo of a slab become negative?",
    )
    np_check
    return (np_check,)


@app.cell(hide_code=True)
def _(mo, np_check):
    if np_check.value is None:
        _out = mo.md("")
    elif np_check.value.startswith("No"):
        _out = mo.callout(mo.md("**Right.** Each term of the mode sum is a squared overlap times a decaying exponential, so it is never negative. A zero is reached only as t → ∞, where only the form factor sinc²(qL/2) is left. At finite t the other modes fill the dips in. In Chapter 2 the echo with finite pulses did cross zero (near 464 mT/m): that cannot happen here."), kind="success")
    else:
        _out = mo.callout(mo.md("**Look at the derivation** (step ④): every term is a squared magnitude times $e^{-D\\lambda_nt}$. What is the smallest value such a sum can take?"), kind="warn")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The map

    The figure compares four closed formulas with the exact Tanner–Stejskal signal at every point $(qL, Dt/L^2)$. Each pixel is coloured by the **simplest formula that is within the tolerance**, tried in this order:

    1. **free** diffusion, $E = e^{-q^2Dt}$;
    2. **short times**, a Gaussian with the wall-corrected $D(t)$ of Mitra et al. (below);
    3. **form factor**, the $t\to\infty$ limit $\operatorname{sinc}^2(qL/2)$;
    4. **Gaussian cumulant**, $e^{-q^2\langle X^2\rangle/2}$, using the exact $\langle X^2\rangle$ of Chapter 1.

    White means that none of them works: there you need the full solution (Chapter 2, 4 or 5). Two grey zones are excluded. **Light grey**: $E > 0.97$, where the signal has hardly decayed and every formula trivially agrees. **Dark grey**: $E < 10^{-3}$, below a typical noise floor.

    The lines are the length-scale estimates: $\ell_D = L$ (horizontal), $q\ell_D = 1$ (diagonal), $qL = 1$ and $qL = 2\pi$ (vertical), and $q X_{\text{rms}} = 1$ (curved, with $X_{\text{rms}} = \sqrt{\langle X^2\rangle}$). Pick a time with the slider to see the signal along a horizontal cut.
    """)
    return


@app.cell(hide_code=True)
def _(np, slab_cumulants, slab_np):
    # The (q,t) map: exact narrow-pulse slab signal and four closed formulas on one grid.
    qt_x = np.logspace(np.log10(0.03), np.log10(4.0), 120)        # qL / 2π
    qt_tau = np.logspace(-3, 1, 110)                  # Dt / L²
    _z = 2 * np.pi * qt_x[None, :]
    _t = qt_tau[:, None]
    qt_E = slab_np(_z, _t, nmax=160)
    _d, _ = slab_cumulants(qt_tau, N=1500)
    qt_X2 = 2 * _d * qt_tau                           # ⟨X²⟩ / L²
    _mitra = np.clip(1 - 8 / (3 * np.sqrt(np.pi)) * np.sqrt(_t), 0, None)
    qt_forms = {
        "free": np.exp(-(_z**2) * _t),
        "short times": np.exp(-(_z**2) * _t * _mitra),
        "form factor": (2 * (1 - np.cos(_z)) / _z**2) * np.ones_like(_t),
        "Gaussian cumulant": np.exp(-(_z**2) * qt_X2[:, None] / 2),
    }
    return qt_E, qt_X2, qt_forms, qt_tau, qt_x


@app.cell(hide_code=True)
def _(mo):
    qt_logt = mo.ui.slider(-3, 1, step=0.05, value=-0.5, label="log₁₀(Dt/L²) for the cut", show_value=True)
    qt_tol = mo.ui.dropdown(options={"2%": 0.02, "5%": 0.05, "10%": 0.10}, value="5%", label="Tolerance")
    qt_show = mo.ui.dropdown(
        options=["all (simplest that works)", "free", "short times", "form factor", "Gaussian cumulant"],
        value="all (simplest that works)",
        label="Colour",
    )
    mo.hstack([qt_logt, qt_tol, qt_show], justify="start", gap=2)
    return qt_logt, qt_show, qt_tol


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_FREE,
    C_GREEN,
    C_MAIN,
    C_PURPLE,
    ListedColormap,
    mo,
    np,
    plt,
    qt_E,
    qt_X2,
    qt_forms,
    qt_logt,
    qt_show,
    qt_tau,
    qt_tol,
    qt_x,
    slab_cumulants,
    slab_np,
):
    _tol = qt_tol.value
    _names = list(qt_forms)
    _cols = [C_MAIN, C_GREEN, C_ALT, C_PURPLE]
    _ok = {k: np.abs(f - qt_E) < _tol * np.abs(qt_E) for k, f in qt_forms.items()}
    _trivial = qt_E > 0.97
    _floor = np.abs(qt_E) < 1e-3
    _lab = np.zeros(qt_E.shape)
    if qt_show.value.startswith("all"):
        for _i in reversed(range(4)):
            _lab = np.where(_ok[_names[_i]], _i + 1, _lab)
    else:
        _i = _names.index(qt_show.value)
        _lab = np.where(_ok[_names[_i]], _i + 1, 0)
    _lab = np.where(_trivial | _floor, np.nan, _lab)
    _cmap = ListedColormap(["#ffffff"] + _cols)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    _a1.pcolormesh(qt_x, qt_tau, _lab, cmap=_cmap, vmin=-0.5, vmax=4.5, alpha=0.55, shading="nearest", rasterized=True)
    _a1.pcolormesh(qt_x, qt_tau, np.where(_trivial, 1.0, np.nan), cmap=ListedColormap(["#e9ebee"]), shading="nearest", rasterized=True)
    _a1.pcolormesh(qt_x, qt_tau, np.where(_floor, 1.0, np.nan), cmap=ListedColormap(["#b9bec5"]), shading="nearest", rasterized=True)
    _zz = 2 * np.pi * qt_x
    _a1.axhline(1, color="k", lw=0.9)
    _a1.plot(qt_x, 1 / _zz**2, color="k", lw=0.9, ls="--")
    _a1.axvline(1 / (2 * np.pi), color="k", lw=0.8, ls=":")
    _a1.axvline(1, color="k", lw=0.8, ls=":")
    _a1.plot(1 / np.sqrt(qt_X2) / (2 * np.pi), qt_tau, color="k", lw=1.6, ls="-.")
    _a1.axhline(10**qt_logt.value, color="k", lw=0.8, alpha=0.6)
    _a1.text(2.1, 1.15, "ℓ_D = L", fontsize=8)
    _a1.text(0.033, 1.4e-3, "light grey: E > 0.97\ndark grey: E < 10⁻³", fontsize=7.5, color="0.3")
    _a1.text(0.6, 1.4e-3, "qℓ_D = 1", fontsize=8, rotation=-36)
    _a1.text(1.05, 3, "qL = 2π", fontsize=8)
    _a1.text(0.165, 4, "qL = 1", fontsize=8)
    _a1.text(0.2, 0.012, "qX_rms = 1", fontsize=8)
    _a1.set(xscale="log", yscale="log", xlim=(qt_x[0], qt_x[-1]), ylim=(qt_tau[0], qt_tau[-1]), xlabel="qL / 2π", ylabel="Dt / L²", title=f"Which formula works (within {int(_tol * 100)}%)")
    _a1.grid(False)
    for _k, _c in zip(_names, _cols):
        _a1.plot([], [], "s", color=_c, alpha=0.55, ms=9, label=_k)
    _a1.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))

    _tau = 10**qt_logt.value
    _z = np.linspace(0.02, 4 * 2 * np.pi, 600)
    _E = slab_np(_z, _tau, nmax=300)
    _d, _K = slab_cumulants(np.array([_tau]))
    _X2 = 2 * _d[0] * _tau
    _k4 = _K[0] * _X2**2
    _x = _z / (2 * np.pi)
    _a2.semilogy(_x, np.clip(_E, 1e-6, None), color="k", lw=2.2, label="exact (Tanner–Stejskal)")
    _a2.semilogy(_x, np.exp(-(_z**2) * _tau), color=C_MAIN, lw=1.4, label="free")
    if _tau < 0.1:
        _a2.semilogy(_x, np.exp(-(_z**2) * _tau * (1 - 8 / (3 * np.sqrt(np.pi)) * np.sqrt(_tau))), color=C_GREEN, lw=1.4, ls="--", label="short times (Mitra)")
    _a2.semilogy(_x, np.clip(2 * (1 - np.cos(_z)) / _z**2, 1e-6, None), color=C_ALT, lw=1.4, label="form factor sinc²(qL/2)")
    _a2.semilogy(_x, np.exp(-(_z**2) * _X2 / 2), color=C_PURPLE, lw=1.4, ls="--", label="Gaussian cumulant")
    _a2.semilogy(_x, np.exp(-(_z**2) * _X2 / 2 + _z**4 * _k4 / 24), color=C_PURPLE, lw=1.0, ls=":", label="cumulants 2 + 4")
    _a2.axvline(1 / np.sqrt(_X2) / (2 * np.pi), color="k", lw=0.8, ls="-.")
    _a2.set(xscale="log", xlim=(0.02, 4), ylim=(1e-4, 1.3), xlabel="qL / 2π", ylabel="E", title=f"Cut at Dt/L² = {_tau:.3g}")
    _a2.legend(frameon=False, fontsize=7.5, loc="lower left")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"At Dt/L² = {_tau:.3g}: D(t)/D = **{_d[0]:.3f}**, K(t) = **{_K[0]:+.3f}**, X_rms = **{np.sqrt(_X2):.3f} L**. "
                f"For the running example (L = 8 µm, D = 2 µm²/ms) this is t = {_tau * 32:.3g} ms."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the map.** The lines and the coloured regions agree on the **shape** of each boundary. The constants differ. The four regions:

    | Region | Where (5%) | Line that estimates its edge | What sets the edge |
    | --- | --- | --- | --- |
    | free | below a line parallel to $q\ell_D = 1$, a factor of 5–10 lower in $t$ | $q\ell_D = 1$ | the fraction of spins that have met a wall, $\sim\ell_D/L$ |
    | short times | up to $Dt \approx 0.1\,L^2$, and up to $q\ell_D \approx 1$ | $\ell_D = L$ and $q\ell_D = 1$ | the two walls start to see each other; non-Gaussian displacement |
    | form factor | $Dt \gtrsim 0.2$–$0.5\,L^2$ at $qL \gtrsim 2$; all $q$ once $Dt \gtrsim L^2$ | $\ell_D = L$ | the slowest mode, $e^{-Dt/\tau_1}$, has died out |
    | Gaussian cumulant | left of $qX_{\text{rms}}\approx 1.1$–$1.8$ | $qX_{\text{rms}} = 1$, or $q\min(\ell_D, L) = 1$ | the fourth cumulant, then the first zero |

    The white band between the cumulant region and the form-factor region, around $qL \approx 2\pi$ at intermediate times, is where no closed formula works. It is exactly where diffraction is still forming.

    The next four subsections derive each formula and its region.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ① Short times: free diffusion, the $b$-matrix, and the surface-to-volume ratio

    **Free diffusion.** Far from any wall $\bar P$ is a Gaussian of variance $2Dt$, so $E = e^{-q^2Dt}$. For narrow pulses $b = q^2\Delta$; for finite pulses $b = q^2(\Delta-\delta/3)$ ([Stejskal & Tanner 1965](https://doi.org/10.1063/1.1695690)). In an anisotropic medium the Gaussian has a covariance $2\mathbf Dt$ and

    $$E = e^{-\mathbf b:\mathbf D} = \exp\Big(-\sum_{ij}b_{ij}D_{ij}\Big),\qquad b_{ij} = q_iq_j\,t ,$$

    the **$b$-matrix** of DTI ([Basser, Mattiello & LeBihan 1994](https://doi.org/10.1006/jmrb.1994.1037)). *Your notes:* the anisotropic membrane diffusion $D_{11}\ll D_{22}$ of the physiology course is this tensor.

    **Walls, to first order.** At short times only spins within about $\ell_D$ of a wall notice it, and they move less. Their fraction is proportional to the surface-to-volume ratio $S/V$, which gives the result of [Mitra, Sen, Schwartz & Le Doussal (1992)](https://doi.org/10.1103/PhysRevLett.68.3555) and [Mitra, Sen & Schwartz (1993)](https://doi.org/10.1103/PhysRevB.47.8565):

    $$\frac{D(t)}{D} = 1 - \frac{4}{3\sqrt\pi\,d}\,\frac SV\,\sqrt{Dt} + O(t),$$

    in $d$ dimensions, for any pore shape. Slab: $S/V = 2/L$, $d = 1$. Cylinder (in the cross-section): $S/V = 2/R$, $d = 2$. Sphere: $S/V = 3/R$, $d = 3$. The first measurements of $S/V$ this way were in porous media ([Latour et al. 1993](https://doi.org/10.1006/jmra.1993.1056)).

    **The kurtosis appears at the same order.** A slab at short times is a mixture: a bulk that diffuses freely, and a thin layer at each wall whose displacements are narrower. A mixture of Gaussian-like distributions of different widths has **positive** excess kurtosis. Exercise 2 derives, for a slab,

    $$K(t) \approx \frac{16}{5\sqrt\pi}\,\frac{\sqrt{Dt}}{L} = 1.805\,\frac{\sqrt{Dt}}{L}.$$

    **Edge enhancement.** The same physics, seen locally: the right-hand panel shows the signal of each starting position $x_0$ across the slab. Spins near a wall are attenuated less, so in a high-resolution image the edges of a pore light up (Hyslop & Lauterbur 1991; Pütz, Barsky & Schulten 1992; [de Swiet 1995](https://doi.org/10.1006/jmrb.1995.1141)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    st_logt = mo.ui.slider(-3, -0.5, step=0.05, value=-2.0, label="log₁₀(Dt/L²) for the profile", show_value=True)
    st_E = mo.ui.slider(0.02, 0.6, step=0.02, value=0.1, label="Bulk signal e^{−q²Dt} for the profile", show_value=True)
    mo.hstack([st_logt, st_E], justify="start", gap=2)
    return st_E, st_logt


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_GREEN, C_MAIN, mo, np, plt, slab_cumulants, st_E, st_logt):
    _tau = np.logspace(-4, 0.5, 300)
    _d, _K = slab_cumulants(_tau, N=4000)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    _a1.semilogx(_tau, _d, color=C_MAIN, lw=2.2, label="D(t)/D, exact")
    _a1.semilogx(_tau, 1 - 8 / (3 * np.sqrt(np.pi)) * np.sqrt(_tau), color=C_MAIN, lw=1.2, ls="--", label="1 − (8/3√π)·√(Dt)/L  (Mitra)")
    _a1.semilogx(_tau, _K, color=C_ALT, lw=2.2, label="K(t), exact")
    _a1.semilogx(_tau, 16 / (5 * np.sqrt(np.pi)) * np.sqrt(_tau), color=C_ALT, lw=1.2, ls="--", label="(16/5√π)·√(Dt)/L")
    _a1.semilogx(_tau, 1 / (12 * _tau), color=C_FREE, lw=1.2, ls=":", label="L²/(12Dt): fully restricted")
    _a1.axhline(-0.6, color=C_ALT, lw=0.8, ls=":")
    _a1.axvline(10**st_logt.value, color="k", lw=0.8, ls=":")
    _a1.set(ylim=(-0.75, 1.1), xlabel="Dt / L²", title="Short-time laws and their range")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower left")

    # local signal E(x0): the echo of the spins that started at x0
    _L, _N = 1.0, 260
    _t = 10**st_logt.value
    _q = np.sqrt(-np.log(st_E.value) / _t)
    _x = np.linspace(0, _L, 801)
    _w = np.full(_x.size, _x[1])
    _w[[0, -1]] /= 2
    _n = np.arange(_N)[:, None]
    _u = np.where(_n == 0, 1 / np.sqrt(_L), np.sqrt(2 / _L) * np.cos(_n * np.pi * _x / _L))
    _lam = (np.arange(_N) * np.pi / _L) ** 2
    _Uq = (_u * (_w * np.exp(1j * _q * _x))).sum(1)
    _Ex = ((_u * (np.exp(-_lam * _t) * _Uq)[:, None]).sum(0)) * np.exp(-1j * _q * _x)
    _a2.plot(_x, _Ex.real, color=C_GREEN, lw=2.2, label="Re E(x₀): what adds up to the echo")
    _a2.plot(_x, np.abs(_Ex), color=C_GREEN, lw=1.2, ls="--", label="|E(x₀)|: what a magnitude image shows")
    _a2.axhline(st_E.value, color=C_FREE, lw=1.2, ls="--", label="bulk: e^{−q²Dt}")
    _a2.axvspan(0, np.sqrt(_t), color=C_FREE, alpha=0.15, lw=0)
    _a2.axvspan(1 - np.sqrt(_t), 1, color=C_FREE, alpha=0.15, lw=0, label="within ℓ_D of a wall")
    _a2.set(xlabel="x₀ / L", ylabel="local signal E(x₀)", ylim=(0, 1.05), title=f"Edge enhancement at Dt/L² = {_t:.3g}, qL = {_q:.1f}")
    _a2.legend(frameon=False, fontsize=7.5, loc="upper center")
    _fig.tight_layout()
    _Emean = float(np.real((_Ex * _w).sum()))
    mo.vstack([_fig, mo.md(f"Mean of the local signal over the slab = the measured echo = **{_Emean:.4f}** (bulk value {st_E.value:.2f}). The walls add a layer of brighter signal about ℓ_D thick.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Take one wall at $x = 0$, a spin starting at height $x_0$, and free Brownian increments $W\sim\mathcal N(0, 2Dt)$. By the method of images (your 03-PDE notes) the reflected walker ends at $\lvert x_0 + W\rvert$, so its displacement is $X = \lvert x_0 + W\rvert - x_0$.

    **(a)** Show that $X^2 - W^2 = 4x_0^2 + 4x_0W$ when $x_0 + W < 0$, and $0$ otherwise. Average over $W$ and over $x_0$ (uniform, per unit wall area) to find the excess $\int_0^\infty\langle X^2 - W^2\rangle dx_0$. Then put two such walls on a slab of width $L$ and recover Mitra's $D(t)/D = 1 - \frac{8}{3\sqrt\pi}\frac{\sqrt{Dt}}{L}$.

    **(b)** Do the same for $X^4 - W^4$ and show that $K(t) \approx \frac{16}{5\sqrt\pi}\frac{\sqrt{Dt}}{L}$.

    **(c)** In the left panel, up to which $Dt/L^2$ is each short-time line within 5% of the exact curve? Why does the kurtosis line fail first?

    **(d)** Move the profile sliders. How thick is the bright layer? Show that for a spin starting exactly at the wall the real part of its signal equals the bulk value $e^{-q^2Dt}$. Where does the extra echo come from, then?
    """, "2 · Short times from the method of images")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - Scale lengths by $\sigma = \sqrt{2Dt}$: $x_0 = \sigma u$, $W = \sigma w$. The excesses become $\sigma^3 I_2$ and $\sigma^5 I_4$ with pure numbers $I_2$, $I_4$.
    - Swap the order of integration: integrate over $u$ from 0 to $-w$ first, for each $w < 0$. You need $\int_{-\infty}^0 \lvert w\rvert^k\,\phi(w)\,dw$ with $\phi$ the standard normal density: $\frac12, \frac1{\sqrt{2\pi}}, \frac12, \frac{2}{\sqrt{2\pi}}, \frac32, \frac{8}{\sqrt{2\pi}}$ for $k = 0..5$.
    - For $K$, expand $\langle X^4\rangle/\langle X^2\rangle^2 - 3$ to first order in the small excesses.
    - (d) At $x_0 = 0$, $X = \lvert W\rvert$. Compare $\langle\cos(q\lvert W\rvert)\rangle$ with $\langle\cos(qW)\rangle$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ If $x_0 + W\ge0$ nothing is reflected and $X = W$. If $x_0 + W < 0$, $X = -2x_0 - W$, so $X^2 - W^2 = 4x_0^2 + 4x_0W$.

    ⇒ Scale with $\sigma$: excess $= \sigma^3 I_2$, $\ I_2 = \int_{-\infty}^0\phi(w)\int_0^{-w}(4u^2 + 4uw)\,du\,dw$. The inner integral is $\frac43\lvert w\rvert^3 - 2\lvert w\rvert^3 = -\frac23\lvert w\rvert^3$, so $I_2 = -\frac23\cdot\frac{2}{\sqrt{2\pi}} = -\frac{4}{3\sqrt{2\pi}}$.

    ⇒ Two walls, spins spread over $L$: $\langle X^2\rangle = \sigma^2 + 2\sigma^3I_2/L$, so

    $$\frac{D(t)}{D} = \frac{\langle X^2\rangle}{\sigma^2} = 1 + \frac{2\sigma I_2}{L} = 1 - \frac{8}{3\sqrt{2\pi}}\frac{\sqrt{2Dt}}{L} = 1 - \frac{8}{3\sqrt\pi}\frac{\sqrt{Dt}}{L}. \checkmark$$

    → In Mitra's form: $\frac{4}{3\sqrt\pi\,d}\frac SV = \frac{4}{3\sqrt\pi}\cdot\frac2L$. ✓

    **(b)** ⇒ For $x_0 + W < 0$: $X^4 - W^4 = (2x_0 + W)^4 - W^4$. The inner integral over $u\in[0,\lvert w\rvert]$ of $(2u + w)^4 - w^4$ is $\frac{(2u+w)^5}{10}\Big|_0^{\lvert w\rvert} - \lvert w\rvert^5 = \frac{\lvert w\rvert^5}{10} + \frac{\lvert w\rvert^5}{10} - \lvert w\rvert^5 = -\frac45\lvert w\rvert^5$, so $I_4 = -\frac45\cdot\frac{8}{\sqrt{2\pi}} = -\frac{32}{5\sqrt{2\pi}}$.

    ⇒ $\langle X^4\rangle = 3\sigma^4 + 2\sigma^5I_4/L$ and $\langle X^2\rangle^2 = \sigma^4(1 + 4\sigma I_2/L + \dots)$, so

    $$K = \frac{3 + 2\sigma I_4/L}{1 + 4\sigma I_2/L} - 3 \approx \frac{2\sigma}{L}\big(I_4 - 6I_2\big) = \frac{2\sigma}{L}\cdot\frac{8/5}{\sqrt{2\pi}} = \frac{16}{5\sqrt\pi}\frac{\sqrt{Dt}}{L} = 1.805\,\frac{\sqrt{Dt}}{L}.$$

    The modes of Chapter 1 give 1.79 at $Dt = 10^{-4}L^2$ and 1.75 at $10^{-3}L^2$, approaching 1.805.

    **(c)** ⇒ $D(t)$: the line is exact (to $10^{-6}$) up to $Dt\approx0.03\,L^2$ and still within 0.2% at $0.08\,L^2$. The two walls are independent until the spins from one wall reach the other. $K(t)$: within 5% only up to $Dt\approx 2\times10^{-3}L^2$; it peaks at $+0.20$ near $0.03\,L^2$ and changes sign near $0.08\,L^2$, on its way to $-3/5$.

    → The kurtosis is a ratio of a small excess to $\langle X^2\rangle^2$. Its leading correction is of relative order $\sqrt{Dt}/L$ already, while for $D(t)$ the $O(t)$ term vanishes for a flat wall (it involves the curvature and the surface relaxivity, [Mitra et al. 1993](https://doi.org/10.1103/PhysRevB.47.8565)).

    **(d)** ⇒ The bright layer is about $\ell_D$ thick (the shaded bands); the real part peaks at roughly $x_0\approx\ell_D/2$ to $\ell_D$.

    ⇒ At $x_0 = 0$ the displacement is $X = \lvert W\rvert$. The cosine is even, so $\langle\cos(q\lvert W\rvert)\rangle = \langle\cos qW\rangle = e^{-q^2Dt}$: the real part is exactly the bulk value. The magnitude is larger only because $\langle\sin(q\lvert W\rvert)\rangle\ne0$. The displacement is one-sided, which gives a phase, and the phases from the two walls cancel in the echo.

    → The extra echo comes from spins a little away from the wall, $0 < x_0\lesssim\ell_D$. The paths that would have carried them far past the wall are folded back, so their displacements are shorter than free ones. A magnitude image (dashed curve) shows the bright edge more strongly than the echo does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ② Long times: closed pores, tortuosity and disorder

    **A closed pore.** Once $\ell_D \gg L$, a spin has forgotten where it started. Its end point is uniform in the pore and independent of its start, so the displacement $X = x - x_0$ is the difference of two independent uniform positions. For a slab, $\bar P(X)$ is a **triangle** of half-width $L$:

    $$\bar P(X) = \frac{L-\lvert X\rvert}{L^2},\qquad \langle X^2\rangle = \frac{L^2}{6},\qquad D(t) = \frac{\langle X^2\rangle}{2t} = \frac{L^2}{12\,t},\qquad K = -\frac35 .$$

    Its Fourier transform is the form factor of ③, $E = \operatorname{sinc}^2(qL/2)$. In a closed pore $D(t)$ falls to zero as $1/t$, and $K$ stays negative: the displacement is bounded, so the distribution has lighter tails than a Gaussian.

    → **Motional narrowing is not on this face.** "Motional narrowing" means the phase is averaged *while the gradient is on*. With $\delta\to0$ nothing is averaged during a pulse. The long-time narrow-pulse limit is the form factor; motional narrowing belongs to the long-pulse face (Section 3b).

    **A connected medium.** If spins can leave the pore, through permeable membranes or through gaps between obstacles, $D(t)$ does not go to zero but to a finite $D_\infty < D$. The ratio $D/D_\infty$ is the **tortuosity**. For a stack of membranes with spacing $L$ and permeability $\kappa$ (the interface condition of Chapter 1), the resistances to flux add in series (*your notes:* heat conduction through layers in series):

    $$\frac1{D_\infty} = \frac1D + \frac1{\kappa L} .$$

    **How $D(t)$ approaches $D_\infty$** carries the structure. [Novikov et al. (2014)](https://doi.org/10.1073/pnas.1316944111) showed that the tail is a power law set by the type of disorder:

    | Structure | Long-time approach | Example |
    | --- | --- | --- |
    | closed pore | $D(t) \simeq L^2/(12t)\to0$ (slab) | cell bodies, impermeable |
    | periodic (ordered) | $D(t) - D_\infty \propto 1/t$ | a regular lattice of obstacles |
    | short-range disorder in $d$ dimensions | $D(t) - D_\infty \propto t^{-d/2}$ | 1D: random beads or membranes along an axon, $t^{-1/2}$ |
    | general | $t^{-\vartheta}$, $\ \vartheta = (p + d)/2$ | $p$ = structural exponent of the disorder |

    The $1/t$ of an ordered medium is the general constant offset in $\langle X^2\rangle = 2D_\infty t + c$. Short-range disorder decays more slowly, so it dominates. The $t^{-1/2}$ law along axons is what [Burcaw et al. (2015)](https://doi.org/10.1016/j.neuroimage.2015.03.061) used to model $D(t)$ in white matter.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive the triangle $\bar P(X)$ for $\ell_D\gg L$. Compute $\langle X^2\rangle$ and $\langle X^4\rangle$, and show that $K = -3/5$. Then show that its Fourier transform is $\operatorname{sinc}^2(qL/2)$, without integrating the triangle.

    **(b)** Derive $1/D_\infty = 1/D + 1/(\kappa L)$ from a steady flux $J$ through many cells: in each cell, the concentration drops across the water layer and jumps across the membrane. Evaluate it for the running example ($L = 8$ µm, $D = 2$ µm²/ms) with $\kappa = 0.1$ µm/ms. When is the membrane the bottleneck?

    **(c)** In the left panel of the short-time figure, from which $Dt/L^2$ on is $L^2/(12Dt)$ within 5% of the exact $D(t)/D$? Compare with $\tau_1 = L^2/(\pi^2D)$.
    """, "3 · Long times")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) The density of a sum (or difference) of independent variables is the convolution of their densities. The Fourier transform of a convolution is the product of the transforms (*your notes:* physiology, convolution theorem).
    - (b) In one cell of width $L$: $\Delta C_{\text{water}} = JL/D$ and $\Delta C_{\text{membrane}} = J/\kappa$. Define $D_\infty$ by $J = D_\infty\,\Delta C_{\text{cell}}/L$.
    - (c) From Chapter 1, $\langle X^2\rangle/L^2 = \frac16 - \frac{16}{\pi^4}e^{-\pi^2Dt/L^2} - \dots$, and $D(t)/D = \langle X^2\rangle/(2Dt)$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $x$ and $x_0$ are independent and uniform on $[0,L]$, so the density of $X = x - x_0$ is the box convolved with its mirror image: a triangle, $\bar P(X) = (L-\lvert X\rvert)/L^2$ on $\lvert X\rvert\le L$.

    ⇒ $\langle X^2\rangle = 2\int_0^L X^2\frac{L-X}{L^2}dX = 2L^2\big(\frac13-\frac14\big) = \frac{L^2}{6}$; $\ \langle X^4\rangle = 2L^4\big(\frac15-\frac16\big) = \frac{L^4}{15}$.

    ⇒ $K = \frac{L^4/15}{L^4/36} - 3 = \frac{36}{15} - 3 = -\frac35$. ✓

    ⇒ Fourier: $\langle e^{iqX}\rangle = \langle e^{iqx}\rangle\langle e^{-iqx_0}\rangle = \lvert\langle e^{iqx}\rangle\rvert^2 = \operatorname{sinc}^2(qL/2)$, using the box transform from the "refresh first" box. The form factor is the squared Fourier transform of the pore shape, because the end points are independent.

    **(b)** ⇒ Per cell, $\Delta C_{\text{cell}} = JL/D + J/\kappa$. With $J = D_\infty\Delta C_{\text{cell}}/L$: $\ L/D_\infty = L/D + 1/\kappa$, so $1/D_\infty = 1/D + 1/(\kappa L)$. ✓

    ⇒ $D/(\kappa L) = 2/(0.1\times8) = 2.5$, so $D_\infty = D/3.5 = 0.57$ µm²/ms, a tortuosity of 3.5.

    → The membrane is the bottleneck when $D/(\kappa L)\gg1$, i.e. $\kappa L/D\ll1$. That is the same condition under which the Kärger model of Section 3d applies: exchange limited by the barrier, with well-mixed compartments.

    **(c)** ⇒ The ratio of $D(t)/D$ to $L^2/(12Dt)$ is $\langle X^2\rangle/(L^2/6)$. With $\langle X^2\rangle/L^2 = \frac16 - \frac{16}{\pi^4}e^{-\pi^2Dt/L^2} - \dots$, the correction is 5% of $\frac16$ when $e^{-\pi^2Dt/L^2} = 0.05\pi^4/96 = 0.051$, at $Dt/L^2 = 0.30$, i.e. $t \approx 3\tau_1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ③ Large $q$: diffraction and the pore form factor

    For $t\gg\tau_1$ the signal is the squared Fourier transform of the pore shape, the **form factor**, as Exercise 3a showed for the slab. It has zeros, the **diffusion–diffraction** minima, and their positions give the pore size without any model of the dynamics ([Callaghan et al. 1991](https://doi.org/10.1038/351467a0); q-space imaging: [Cory & Garroway 1990](https://doi.org/10.1002/mrm.1910140303)):

    | Pore | Form factor $E(q,\infty)$ | First zero | Zero in units of the width $w$ |
    | --- | --- | --- | --- |
    | slab, width $L$ | $\operatorname{sinc}^2(qL/2)$ | $qL = 2\pi$ | $qw/2\pi = 1$ |
    | cylinder, radius $R$, $q\perp$ axis | $\big[2J_1(qR)/(qR)\big]^2$ | $qR = 3.832$ | $qw/2\pi = 1.22$ |
    | sphere, radius $R$ | $\big[3j_1(qR)/(qR)\big]^2$ | $qR = 4.493$ | $qw/2\pi = 1.43$ |

    *Memory aid:* the 1.22 of the cylinder is the **1.22 of the Airy disk** in optics, the diffraction pattern of a circular aperture. It is the same Fourier transform of a disk.

    At finite $t$ the higher modes fill the zeros in: every term of the Tanner–Stejskal sum is non-negative. The left panel shows the slab at several diffusion times. The right panel compares the three shapes at the same width.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dif_logt = mo.ui.slider(-1.5, 1, step=0.05, value=-0.5, label="log₁₀(DΔ/L²) for the highlighted curve", show_value=True)
    dif_logt
    return (dif_logt,)


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_GREEN, C_MAIN, dif_logt, jv, mo, np, plt, slab_np, sph_j1):
    _z = np.linspace(0.01, 3.2 * 2 * np.pi, 800)
    _x = _z / (2 * np.pi)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    for _tau, _al in ((0.05, 0.35), (0.15, 0.5), (0.5, 0.7)):
        _a1.semilogy(_x, np.clip(slab_np(_z, _tau, nmax=300), 1e-8, None), color=C_FREE, lw=1.0, alpha=_al)
        _a1.text(0.93, slab_np(2 * np.pi * 0.995, _tau, nmax=300), f"{_tau:g}", fontsize=7.5, color="0.4", ha="right", va="center")
    _tsel = 10**dif_logt.value
    _Esel = slab_np(_z, _tsel, nmax=300)
    _a1.semilogy(_x, np.clip(_Esel, 1e-8, None), color=C_MAIN, lw=2.2, label=f"DΔ/L² = {_tsel:.3g}")
    _a1.semilogy(_x, np.clip(2 * (1 - np.cos(_z)) / _z**2, 1e-8, None), color="k", lw=1.0, ls="--", label="form factor (Δ → ∞)")
    for _k in (1, 2, 3):
        _a1.axvline(_k, color="k", lw=0.6, ls=":")
    _a1.set(ylim=(1e-5, 1.3), xlabel="qL / 2π", ylabel="E", title="Slab: the dips fill in at short Δ (grey: 0.05, 0.15, 0.5)")
    _a1.legend(frameon=False, fontsize=8, loc="upper right")

    _xx = np.linspace(0.005, 3.2, 700)
    _ff_slab = np.sinc(_xx) ** 2                              # q w / 2π with w = L
    _qR = np.pi * _xx                                          # w = 2R  ->  qR = π · (q w / 2π)
    _ff_cyl = (2 * jv(1, _qR) / _qR) ** 2
    _ff_sph = (3 * sph_j1(_qR) / _qR) ** 2
    _a2.semilogy(_xx, np.clip(_ff_slab, 1e-8, None), color=C_MAIN, lw=1.8, label="slab")
    _a2.semilogy(_xx, np.clip(_ff_cyl, 1e-8, None), color=C_GREEN, lw=1.8, label="cylinder (q ⊥ axis)")
    _a2.semilogy(_xx, np.clip(_ff_sph, 1e-8, None), color=C_ALT, lw=1.8, label="sphere")
    for _z0, _c in ((1.0, C_MAIN), (1.2197, C_GREEN), (1.4303, C_ALT)):
        _a2.axvline(_z0, color=_c, lw=0.8, ls=":")
    _a2.set(ylim=(1e-5, 1.3), xlabel="q w / 2π   (w = L or 2R: the pore width)", ylabel="E(q, ∞)", title="Form factors at the same width")
    _a2.legend(frameon=False, fontsize=8, loc="upper right")
    _fig.tight_layout()
    _E2pi = float(slab_np(2 * np.pi - 1e-4, _tsel, nmax=400))
    mo.vstack([_fig, mo.md(f"Depth of the first slab dip at DΔ/L² = {_tsel:.3g}: E(qL = 2π) = **{_E2pi:.2e}** (running example: Δ = {32 * _tsel:.3g} ms).")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** A narrow-pulse experiment on a sample of identical pores shows its first diffraction minimum at $q = 0.785$ µm⁻¹. What is the pore width if the pores are slabs, cylinders, or spheres?

    **(b)** Show from the Tanner–Stejskal sum that at the first zero, $qL = 2\pi$,

    $$E(2\pi, t) = \frac{32}{9\pi^2}e^{-\pi^2Dt/L^2} + \frac12 e^{-4\pi^2Dt/L^2} + \dots$$

    How long must $\Delta$ be, for the running example, before the dip reaches $10^{-2}$? Check with the slider.

    **(c)** $\ln\operatorname{sinc}^2(qL/2)$ is the long-time limit of the cumulant series in $q$. Where is it singular? What does that say about the largest $q$ at which any truncated cumulant expansion can work at long times? Relate this to the complex zero of $E(G)$ of Chapter 2.
    """, "4 · Diffraction")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (b) At $qL = 2\pi$, $\cos qL = 1$, so the odd-$n$ terms have $1-(-1)^n\cos qL = 2$ and the even ones vanish, except $n = 2$, where the denominator vanishes too. Take the limit $qL\to2\pi$ for that term.
    - (c) A power series converges up to the nearest singularity in the complex plane. $\ln f$ is singular where $f = 0$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Slab: $L = 2\pi/q = 8.0$ µm. Cylinder: $R = 3.832/q = 4.88$ µm, width 9.8 µm. Sphere: $R = 4.493/q = 5.72$ µm, width 11.4 µm. The same dip means different sizes: the **shape** must be known, or found from the ratio of the second to the first zero.

    **(b)** ⇒ Odd $n$: $4(2\pi)^2\cdot2/[(2\pi)^2 - (n\pi)^2]^2 = 32/[\pi^2(4-n^2)^2]$, which is $32/(9\pi^2) = 0.360$ for $n = 1$ and $32/(25\pi^2)$ for $n = 3$. The $n = 3$ term decays as $e^{-9\pi^2Dt/L^2}$, so it is negligible.

    ⇒ $n = 2$: put $qL = 2\pi + \epsilon$. Then $1 - \cos qL\approx\epsilon^2/2$ and $(qL)^2 - 4\pi^2\approx4\pi\epsilon$, so the term tends to $4(2\pi)^2(\epsilon^2/2)/(16\pi^2\epsilon^2) = \frac12$. ✓

    ⇒ The dip reaches $10^{-2}$ when $0.360\,e^{-\pi^2Dt/L^2}\approx0.01$: $Dt/L^2 = \ln(36)/\pi^2 = 0.36$, i.e. $\Delta\approx11.6$ ms for $L = 8$ µm. About $3.6\,\tau_1$. The slider gives $E(2\pi) = 1.00\times10^{-2}$ at $D\Delta/L^2 = 0.363$.

    **(c)** ⇒ $\ln\operatorname{sinc}^2(qL/2)$ is singular where $\operatorname{sinc}(qL/2) = 0$, first at $qL = 2\pi$. So at long times the cumulant series in $q$ converges only for $qL < 2\pi$, i.e. $qX_{\text{rms}} < 2\pi/\sqrt6 = 2.57$. Adding cumulants cannot take you past the first diffraction minimum.

    → Chapter 2 found the same thing for finite pulses: the radius of convergence of the cumulant series in $G$ is set by the nearest zero of $E(G)$. For narrow pulses at long times that zero is on the real axis, at $qL = 2\pi$. For finite pulses or finite $t$ it moves off the real axis ($463 + 20i$ mT/m in the Chapter 2 example), so $E$ only dips without reaching zero, but the series still stops there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ④ The cumulant boundary

    The cumulant series is an expansion in the phase spread, $q$ times the typical displacement. The typical displacement is the root-mean-square $X_{\text{rms}} = \sqrt{\langle X^2\rangle} = \sqrt{2D(t)\,t}$, which runs from $\sqrt{2Dt}$ at short times to $L/\sqrt6$ at long times. So the region where the Gaussian (second cumulant) works is

    $$q\,X_{\text{rms}}\lesssim 1 \qquad\Longleftrightarrow\qquad q\min(\ell_D, L)\lesssim 1\ \text{ (up to constants)} \qquad\Longleftrightarrow\qquad b\,D(t)\lesssim\tfrac12 ,$$

    since $q^2X_{\text{rms}}^2 = 2q^2D(t)\,t = 2bD(t)$ for narrow pulses.

    **How large is "$\lesssim 1$"?** The first correction is the fourth cumulant. With $\kappa_4 = K\,X_{\text{rms}}^4$,

    $$\ln E = -\frac{(qX_{\text{rms}})^2}{2} + \frac{K\,(qX_{\text{rms}})^4}{24} - \dots$$

    The Gaussian is 5% off when the second term reaches 0.05: $(qX_{\text{rms}})^* \approx (1.2/\lvert K\rvert)^{1/4}$. A scan of the exact slab signal gives:

    | $Dt/L^2$ | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | $\ge1$ |
    | --- | --- | --- | --- | --- | --- | --- |
    | $K(t)$ | +0.09 | +0.16 | +0.20 | −0.08 | −0.53 | −0.60 |
    | $(qX_{\text{rms}})^*$, 5% | 1.76 | 1.58 | 1.48 | 1.80 | 1.19 | 1.14 |
    | $(1.2/\lvert K\rvert)^{1/4}$ | 1.89 | 1.67 | 1.57 | 1.97 | 1.23 | 1.19 |

    So **$qX_{\text{rms}}\approx1$ is a safe rule**: the Gaussian fails between 1.1 and 1.8, and the kurtosis predicts where. The curved line on the map is $qX_{\text{rms}} = 1$. This refines the cruder $q\min(\sqrt{Dt}, L)\approx1$, which has the right shape but a constant that drifts from about 1.2 at short times to 2.8 at long times. Near $Dt\approx0.1L^2$, where $K$ changes sign, the kurtosis estimate is only an upper bound: there the sixth cumulant takes over.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Use the map's right-hand panel (set $Dt/L^2 = 3$). Where do the solid black and the dashed purple curves part by 5%? Express it as $qL$ and as $qX_{\text{rms}}$, and compare with $(1.2/\lvert K\rvert)^{1/4}$.

    **(b)** Translate the rule into $b$-values for the running example at long times, and for free water ($D = 2$ µm²/ms) at short times. Why is the long-time limit not a statement about $b$ alone?

    **(c)** Does adding the fourth cumulant (dotted purple) extend the range? By how much, and what limits it?
    """, "5 · Where the cumulants stop")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ At long times they part at $qL\approx2.8$, i.e. $qL/2\pi\approx0.45$. With $X_{\text{rms}} = L/\sqrt6 = 0.408L$: $qX_{\text{rms}}\approx1.14$. The kurtosis estimate with $K = -3/5$ is $(1.2/0.6)^{1/4} = 1.19$. ✓

    **(b)** ⇒ $qX_{\text{rms}}\le1$ means $bD(t)\le\frac12$. At long times $D(t) = L^2/(12t)$, so $bD(t) = q^2L^2/12$, independent of $t$: the condition is on $qL\le\sqrt6 = 2.45$, not on $b$. For $L = 8$ µm that is $q\le0.31$ µm⁻¹.

    ⇒ At short times $D(t)\approx D$, so $b\lesssim1/(2D) = 0.25$ ms/µm² $= 250$ s/mm² for the strict rule, and up to $(1.75)^2/(2D) = 0.77$ ms/µm² $= 770$ s/mm² for 5% at $Dt = 0.003L^2$, where $K$ is small.

    → In a closed pore the long-time signal stops depending on $\Delta$: increasing $\Delta$ at fixed $q$ increases $b$ but not the attenuation. A rule written in $b$ alone hides this.

    **(c)** ⇒ At long times the dotted curve stays within 5% of the exact one up to $qL\approx3.8$ ($qX_{\text{rms}}\approx1.57$) instead of 2.8. It can never pass $qL = 2\pi$ ($qX_{\text{rms}} = 2.57$), where the series diverges (Exercise 4c). More cumulants buy less and less.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3b · The long-pulse face: the $(g,T)$ map

    Put $\delta = \Delta = T/2$: the effective gradient is $+g$ for $T/2$ and $-g$ for $T/2$, a spin echo in a constant gradient. The three lengths are $L$, $\ell_D = \sqrt{DT}$ and $\ell_g = (D/\gamma g)^{1/3}$, and **whichever is the smallest sets the regime** ([de Swiet & Sen 1994](https://doi.org/10.1063/1.467127); [Grebenkov 2007](https://doi.org/10.1103/RevModPhys.79.1077)):

    | Smallest length | Regime | Signal | In length ratios | First reference |
    | --- | --- | --- | --- | --- |
    | $\ell_D$ | **free** | $\ln E = -\gamma^2g^2DT^3/12$ | $-\frac1{12}(\ell_D/\ell_g)^6$ | [Carr & Purcell 1954](https://doi.org/10.1103/PhysRev.94.630) |
    | $L$ | **motional narrowing** | $\ln E = -\gamma^2g^2L^4T/(120D)$ | $-\frac1{120}(L/\ell_g)^4(\ell_D/\ell_g)^2$ | [Robertson 1966](https://doi.org/10.1103/PhysRev.151.273); [Neuman 1974](https://doi.org/10.1063/1.1680931) |
    | $\ell_g$ | **localisation** | $E \approx C\,\frac{\ell_g}{L}\,e^{-\frac{\lvert a_1'\rvert}{2}DT/\ell_g^2}$ | $\ln(C\ell_g/L) - 0.509\,(\ell_D/\ell_g)^2$ | [Stoller, Happer & Dyson 1991](https://doi.org/10.1103/PhysRevA.44.7459); [Hürlimann et al. 1995](https://doi.org/10.1006/jmra.1995.1091) |

    with $\lvert a_1'\rvert = 1.0188$ (the first zero of $\mathrm{Ai}'$) and $C\approx5.89$ for a slab with two walls. That value is computed here from the MCF of Chapter 2 (it converges to 5.888); [Moutal et al. (2019)](https://doi.org/10.1016/j.jmr.2019.06.016) give the theory of the prefactor.

    The boundaries are $\ell_D = L$, $\ell_g = L$ and $\ell_g = \ell_D$. The last one is also $q\ell_D = 1$ if you write $q = \gamma gT$: from $\ell_g = \ell_D$, $\gamma g = D/\ell_D^3$, so $\gamma gT\,\ell_D = DT/\ell_D^2 = 1$. The three lines meet at the **triple point** $\ell_D = \ell_g = L$.

    → **Three different laws.** Free diffusion decays as $g^2T^3$. Motional narrowing decays as $g^2T$: the spins sample the whole pore many times, and the phase does a random walk with steps $\gamma gL\cdot L^2/D$. Localisation decays as $g^{2/3}T$: only the spins in a layer of thickness $\ell_g$ at each wall survive, and they lose phase slowly because the wall stops them from wandering along the gradient. This is the same "edge enhancement" as in ①, now at strong gradients.

    The map below is computed from the exact MCF signal of Chapter 2 on a grid of $(L/\ell_g, \ell_D/L)$. The signal spans many decades, so a formula counts as working when its **exponent** $\ln E$ is within the tolerance. The right panel is a cut at fixed gradient: $-\ln E$ against $T$, on log axes, where the three laws show up as slopes of 3 and 1.
    """)
    return


@app.cell(hide_code=True)
def _(np, pgse_box):
    # (g,T) map on the long-pulse face: exact ln E on a grid of L/ℓ_g (columns) and ℓ_D/L (rows).
    gt_x = np.logspace(np.log10(0.2), np.log10(15.0), 38)    # L / ℓ_g
    gt_y = np.logspace(-1.6, 1.0, 50)                         # ℓ_D / L
    _T = gt_y**2
    gt_E = np.empty((gt_y.size, gt_x.size))
    for _j, _xv in enumerate(gt_x):
        _N = int(min(7 * _xv + 30, 140))
        gt_E[:, _j] = pgse_box(_xv**3, _T / 2, _T / 2, N=_N).real
    _X, _Y = np.meshgrid(gt_x, gt_y)
    _G, _TT = _X**3, _Y**2
    gt_lnF = {
        "free": -(_G**2) * _TT**3 / 12,
        "motional narrowing": -(_G**2) * _TT / 120,
        "localisation": np.log(5.89 / _X) - 0.5094 * _TT * _X**2,
    }
    return gt_E, gt_lnF, gt_x, gt_y


@app.cell(hide_code=True)
def _(mo):
    gt_cut = mo.ui.slider(0.3, 1.15, step=0.05, value=0.75, label="log₁₀(L/ℓ_g) for the cut", show_value=True)
    gt_tol = mo.ui.dropdown(options={"5%": 0.05, "10%": 0.10, "20%": 0.20}, value="10%", label="Tolerance on ln E")
    mo.hstack([gt_cut, gt_tol], justify="start", gap=2)
    return gt_cut, gt_tol


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_GREEN,
    C_MAIN,
    ListedColormap,
    gt_E,
    gt_cut,
    gt_lnF,
    gt_tol,
    gt_x,
    gt_y,
    mo,
    np,
    pgse_box,
    plt,
):
    _tol = gt_tol.value
    _lnE = np.log(np.clip(np.abs(gt_E), 1e-300, None))
    _valid = np.abs(gt_E) > 1e-10
    _names = list(gt_lnF)
    _cols = [C_MAIN, C_GREEN, C_ALT]
    _lab = np.zeros(gt_E.shape)
    for _i in reversed(range(3)):
        _ok = (np.abs(gt_lnF[_names[_i]] - _lnE) < _tol * np.abs(_lnE)) & _valid
        _lab = np.where(_ok, _i + 1, _lab)
    _lab = np.where(_valid, _lab, np.nan)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    _a1.pcolormesh(gt_x, gt_y, _lab, cmap=ListedColormap(["#ffffff"] + _cols), vmin=-0.5, vmax=3.5, alpha=0.5, shading="nearest", rasterized=True)
    _a1.pcolormesh(gt_x, gt_y, np.where(_valid, np.nan, 1.0), cmap=ListedColormap(["#b9bec5"]), shading="nearest", rasterized=True)
    _a1.pcolormesh(gt_x, gt_y, np.where(gt_E > 0.99, 1.0, np.nan), cmap=ListedColormap(["#e9ebee"]), shading="nearest", rasterized=True)
    _cs = _a1.contour(gt_x, gt_y, np.log10(np.clip(np.abs(gt_E), 1e-12, 1)), levels=[-6, -3, -2, -1, np.log10(0.5), np.log10(0.9)], colors="0.35", linewidths=0.7)
    _a1.clabel(_cs, fmt={-6: "10⁻⁶", -3: "10⁻³", -2: "10⁻²", -1: "0.1", np.log10(0.5): "0.5", np.log10(0.9): "0.9"}, fontsize=7)
    _a1.axvline(1, color="k", lw=1.0)
    _a1.axhline(1, color="k", lw=1.0)
    _a1.plot(gt_x, 1 / gt_x, color="k", lw=1.0, ls="--")
    _a1.plot(1, 1, "o", color="k", ms=7, mfc="white", mew=1.6)
    _a1.axvline(10**gt_cut.value, color="k", lw=0.8, alpha=0.5, ls=":")
    _a1.text(8.0, 0.04, "free", color=C_MAIN, fontsize=10, weight="bold")
    _a1.text(0.24, 4.0, "motional\nnarrowing", color=C_GREEN, fontsize=10, weight="bold")
    _a1.text(5.5, 1.2, "localisation", color=C_ALT, fontsize=10, weight="bold")
    _a1.text(1.06, 0.2, "ℓ_g = L", fontsize=8)
    _a1.text(0.21, 1.1, "ℓ_D = L", fontsize=8)
    _a1.text(2.4, 0.33, "ℓ_g = ℓ_D", fontsize=8, rotation=-40)
    _a1.text(1.08, 0.82, "triple point", fontsize=7.5)
    _a1.text(7.5, 5.5, "E < 10⁻¹⁰", fontsize=7.5, color="0.25")
    _a1.text(0.22, 0.4, "light grey:\nE > 0.99", fontsize=7.5, color="0.35")
    _a1.set(xscale="log", yscale="log", xlim=(gt_x[0], gt_x[-1]), ylim=(gt_y[0], gt_y[-1]), xlabel="L / ℓ_g  =  (γgL³/D)^{1/3}", ylabel="ℓ_D / L  =  √(DT)/L", title=f"Long-pulse face: where each law holds (ln E within {int(_tol * 100)}%)")
    _a1.grid(False)

    _xc = 10**gt_cut.value
    _Gam = _xc**3
    _T = np.logspace(-2.6, 2, 160)
    _N = int(min(7 * _xc + 40, 160))
    _E = pgse_box(_Gam, _T / 2, _T / 2, N=_N).real
    _ok = np.abs(_E) > 1e-12
    _a2.loglog(_T[_ok], -np.log(_E[_ok]), color="k", lw=2.2, label="exact (MCF)")
    _a2.loglog(_T, _Gam**2 * _T**3 / 12, color=C_MAIN, lw=1.4, ls="--", label="free: Γ²T³/12")
    _a2.loglog(_T, _Gam**2 * _T / 120, color=C_GREEN, lw=1.4, ls="--", label="motional narrowing: Γ²T/120")
    _loc = 0.5094 * _T * _xc**2 - np.log(5.89 / _xc)
    _a2.loglog(_T[_loc > 0], _loc[_loc > 0], color=C_ALT, lw=1.4, ls="--", label="localisation")
    _a2.axvline(1, color="0.5", lw=0.8, ls=":")
    _a2.axvline(1 / _xc**2, color="0.5", lw=0.8, ls="-.")
    _a2.text(1.08, 2e-3, "ℓ_D = L", fontsize=7.5, color="0.4")
    _a2.text(1.08 / _xc**2, 2e-2, "ℓ_D = ℓ_g", fontsize=7.5, color="0.4")
    _a2.set(ylim=(1e-3, 40), xlim=(_T[0], _T[-1]), xlabel="DT / L²  (T = 2δ)", ylabel="−ln E", title=f"Cut at L/ℓ_g = {_xc:.2f}  (Γ = {_Gam:.1f})")
    _a2.legend(frameon=False, fontsize=7.5, loc="upper left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"For the running example (L = 8 µm, D = 2 µm²/ms) this cut is G = {_Gam * 2 / (2.6752e-4 * 512):.0f} mT/m, ℓ_g = {8 / _xc:.2f} µm; DT/L² = 1 is T = 32 ms. Dark grey: signal below 10⁻¹⁰, where the exact value is not computed reliably. Light grey: E > 0.99, no measurable attenuation.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the map.** Each law holds deep inside its own corner, and the three lines put the corners in the right places. Around the lines there are bands, a factor of 2 to 3 wide, where no law holds: the crossovers. Two of them are worth knowing:

    - **Free → localisation** happens at $\ell_D\approx1.6\,\ell_g$, close to the line $\ell_g = \ell_D$. But the localisation law becomes accurate only for $L/\ell_g\gtrsim3$–$4$, with the signal already below about $10^{-2}$.
    - **Motional narrowing → localisation** happens near $L/\ell_g\approx2$–$3$, not at $L = \ell_g$. Exercise 6 shows where the factor comes from.

    The free region is small. Like the short-time correction in ①, the walls change the exponent by a relative amount proportional to $\ell_D/L$, so the free law is within 10% only for $\ell_D\lesssim0.07\,L$. Where the attenuation is also measurable ($E < 0.99$), that leaves the bottom-right corner, at strong gradients.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Write the three exponents in length ratios (the table above), and find where each pair is equal. Ignore the logarithmic prefactor of the localisation law. How far from the lines $\ell_D = L$, $\ell_g = L$ and $\ell_g = \ell_D$ do the pairs cross?

    **(b)** Derive the motional-narrowing law from the Gaussian phase approximation of Chapter 2. For $T\gg\tau_1$, show that $\ln E = -\gamma^2g^2T\sum_n B_{0n}^2/(D\lambda_n)$, and evaluate the sum for a slab with $\sum_{n\ \text{odd}}n^{-6} = \pi^6/960$. Then keep the next term and show that $\ln E = -\frac{\gamma^2g^2L^4}{120D}\big(T - \frac{17}{56}\frac{L^2}{D}\big)$. Use $\sum_{n\ \text{odd}}n^{-8} = 17\pi^8/161280$.

    **(c)** The same sum with the Bessel weights of Chapter 1 gives $\frac7{96}R^4$ for a cylinder and $\frac8{175}R^4$ for a sphere ([Neuman 1974](https://doi.org/10.1063/1.1680931)). For the same $\gamma g$, $T$ and $D$, which of slab ($L = 2R$), cylinder and sphere attenuates most? Why does the comparison favour the slab?

    **(d)** Why is it called motional narrowing?
    """, "6 · The long-pulse map")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (b) In Chapter 2, $\langle\varphi^2\rangle = \gamma^2\iint g(t_1)g(t_2)\sum_nB_{0n}^2e^{-D\lambda_n\lvert t_1-t_2\rvert}dt_1dt_2$. For a constant $\lvert g\rvert$ over a long time, $\iint e^{-a\lvert t_1-t_2\rvert}\approx 2T/a - 2/a^2$ per lobe pair; the closed form for the PGSE with $\delta = \Delta$ is $I = 4\delta/a - 6/a^2$ for $a\delta\gg1$.
    - (b) Slab: $B_{0n}^2 = 8L^2/(n\pi)^4$ and $\lambda_n = (n\pi/L)^2$, odd $n$.
    - (d) Think of the NMR line width of a spin whose precession frequency fluctuates quickly.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Free = MN: $\frac1{12}(\ell_D/\ell_g)^6 = \frac1{120}(L/\ell_g)^4(\ell_D/\ell_g)^2$ ⇒ $\ell_D^4 = L^4/10$ ⇒ $\ell_D = 0.56\,L$.

    ⇒ Free = localisation: $\frac1{12}(\ell_D/\ell_g)^6 = 0.509(\ell_D/\ell_g)^2$ ⇒ $(\ell_D/\ell_g)^4 = 6.1$ ⇒ $\ell_D = 1.57\,\ell_g$.

    ⇒ MN = localisation: $\frac1{120}(L/\ell_g)^4 = 0.509$ ⇒ $L/\ell_g = 2.8$.

    → The lines are right in shape. The constants $1/12$, $1/120$ and $0.509$ shift the crossings by factors between 0.56 and 2.8. The largest shift, at $\ell_g = L$, comes from the small $1/120$: motional narrowing is very efficient, so localisation has to be strong before it wins.

    **(b)** ⇒ With $\delta = \Delta = T/2$, the Chapter 2 formula gives, per mode, $I_n = 4\delta/a_n - 6/a_n^2 + O(e^{-a_n\delta})$, with $a_n = D\lambda_n$. So

    $$\ln E = -\frac{\gamma^2g^2}2\sum_n B_{0n}^2\Big(\frac{2T}{D\lambda_n} - \frac{6}{(D\lambda_n)^2}\Big) .$$

    ⇒ Slab: $\sum_n\frac{B_{0n}^2}{\lambda_n} = \sum_{\text{odd}}\frac{8L^4}{(n\pi)^6} = \frac{8L^4}{\pi^6}\cdot\frac{\pi^6}{960} = \frac{L^4}{120}$. So $\ln E = -\gamma^2g^2L^4T/(120D)$. ✓

    ⇒ Next term: $3\sum_n\frac{B_{0n}^2}{\lambda_n^2} = 3\cdot\frac{8L^6}{\pi^8}\cdot\frac{17\pi^8}{161280} = \frac{17L^6}{6720} = \frac{L^4}{120}\cdot\frac{17}{56}L^2$. So $T\to T - \frac{17}{56}\frac{L^2}{D}$. The map uses the leading law, which is within 10% only for $DT/L^2\gtrsim3$. With the shift the law is within 1% already at $DT/L^2 = 1$.

    **(c)** ⇒ Coefficients of $\gamma^2g^2T/D$: slab $L^4/120 = 16R^4/120 = 0.133R^4$; cylinder $7R^4/96 = 0.073R^4$; sphere $8R^4/175 = 0.046R^4$.

    → The slab attenuates most. The motional-narrowing coefficient is $\sum_n B_{0n}^2/\lambda_n$, a variance of $x$ multiplied by a correlation time. Both are largest when the pore extends furthest along the gradient over the most of its volume: a slab of full width $2R$ everywhere, against a disk or ball that is that wide only through its centre.

    **(d)** ⇒ A spin's precession frequency is $\gamma gx$. Moving fast across the pore, it sees a frequency that fluctuates by about $\gamma gL$ with a correlation time $\tau_c\sim L^2/D$. Its phase does a random walk with $T/\tau_c$ steps of size $\gamma gL\tau_c$, so $\langle\varphi^2\rangle\sim\gamma^2g^2L^2\tau_cT = \gamma^2g^2L^4T/D$.

    → The decay rate $\sim(\gamma gL)^2\tau_c$ is much smaller than the frequency spread $\gamma gL$ when $\gamma gL\,\tau_c\ll1$, i.e. $L\ll\ell_g$. In spectroscopy the resonance line is then **narrowed** by the motion. Same physics, same name ([Robertson 1966](https://doi.org/10.1103/PhysRev.151.273)).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Near one wall, rescale $x = \ell_g\xi$ and $t = (\ell_g^2/D)s$. Show that the Bloch–Torrey equation $\partial_tm = D\partial_x^2m - i\gamma gxm$ becomes the universal equation $\partial_sm = \partial_\xi^2m - i\xi m$, with no parameters left. Its slowest-decaying solution with $\partial_\xi m = 0$ at the wall is an Airy function, with decay rate $\frac{\lvert a_1'\rvert}2 = 0.509$ in units of $D/\ell_g^2$. Show that this rate scales as $g^{2/3}$.

    **(b)** Why is the prefactor proportional to $\ell_g/L$?

    **(c)** For the running example with $\delta = \Delta = 20$ ms ($T = 40$ ms), from which $G$ on is the localisation law within 20% of the exact signal? How large is $E$ there? Use the cut panel, or the table in the answer.
    """, "7 · Localisation")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\partial_t = (D/\ell_g^2)\partial_s$ and $\partial_x^2 = \ell_g^{-2}\partial_\xi^2$, so $\frac{D}{\ell_g^2}\partial_sm = \frac{D}{\ell_g^2}\partial_\xi^2m - i\gamma g\ell_g\xi m$. Divide by $D/\ell_g^2$: the last coefficient is $\gamma g\ell_g^3/D = 1$ by the definition of $\ell_g$. ✓

    ⇒ The rate is $0.509\,D/\ell_g^2 = 0.509\,D\,(\gamma g/D)^{2/3} = 0.509\,D^{1/3}(\gamma g)^{2/3}$: it scales as $g^{2/3}$, against $g^2$ in every other regime. Doubling the gradient raises the decay rate by only $2^{2/3} = 1.59$.

    → *Your notes* (QM, the triangular well; 03-PDE, similarity variables): the operator $-\partial_\xi^2 + i\xi$ is the Schrödinger operator of a particle in a linear potential, with an imaginary slope. Its eigenvalues are the Airy zeros rotated by $e^{\pm i\pi/3}$ in the complex plane. The real part, $\lvert a_1'\rvert\cos(\pi/3) = \lvert a_1'\rvert/2$, is the decay rate. The operator is not self-adjoint, which is why the modes are complex ([Stoller et al. 1991](https://doi.org/10.1103/PhysRevA.44.7459)).

    **(b)** ⇒ The surviving magnetization lives in a layer of thickness $\sim\ell_g$ at each wall. Its share of the total is $\sim2\ell_g/L$. The bulk, further than a few $\ell_g$ from a wall, has dephased completely. $C = 5.89$ collects the factor 2 and the shape of the Airy layer.

    **(c)** ⇒ Exact MCF against the law for $L = 8$ µm, $T = 40$ ms:

    | $G$ (mT/m) | $\ell_g$ (µm) | $L/\ell_g$ | exact $E$ | law / exact |
    | --- | --- | --- | --- | --- |
    | 380 | 2.70 | 3.0 | $1.2\times10^{-3}$ | 5.9 |
    | 1000 | 1.96 | 4.1 | $2.5\times10^{-5}$ | 1.34 |
    | 1500 | 1.71 | 4.7 | $1.3\times10^{-6}$ | 0.84 |
    | 3000 | 1.36 | 5.9 | $2.2\times10^{-10}$ | 1.06 |

    → Within 20% only from about 1500 mT/m on, where $E\sim10^{-6}$. The exponent $\ln E$ is right to 3% already at 1000 mT/m, but the prefactor converges more slowly. In practice localisation is seen with strong static gradients and short echo times ([Hürlimann et al. 1995](https://doi.org/10.1006/jmra.1995.1091)), or in larger pores (Moutal et al. 2019).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3c · Inside the box: real PGSE sequences

    Between the two faces there are no new closed formulas. What remains is the question of Chapter 2: **up to which gradient can I trust the Gaussian phase (cumulant) approximation?** On the two faces the answer is known:

    - narrow pulses: $q\min(\ell_D, L)\lesssim1$ (Section 3a ④);
    - long pulses: the smallest of $\ell_D$ and $L$ must stay below about $\ell_g$ (Section 3b), i.e. $\min(\ell_D, L)/\ell_g\lesssim1$.

    **A rule of thumb to test.** A natural guess for the interior is

    $$H = \gamma gL\cdot\min\big(\delta,\ L^2/D\big)\ \lesssim\ 1 .$$

    It reads: the phase difference across the pore that builds up during one lobe, where the lobe counts only up to the pore time (after that the spins average over the pore). In dimensionless form $H = \Gamma\min(\tilde\delta, 1)$. For long pulses it gives $\Gamma = (L/\ell_g)^3\lesssim1$, the motional-narrowing edge. For narrow pulses it gives $qL\lesssim1$.

    **A refinement.** At short $\Delta$ the walls are hardly felt, and free diffusion is Gaussian at any gradient. So the length in the rule should be the one the spins actually explore, $\ell = \min(\sqrt{D\Delta}, L)$:

    $$H_2 = \gamma g\,\ell\cdot\min\big(\delta,\ \ell^2/D\big)\ \lesssim\ 1,\qquad \ell = \min(\sqrt{D\Delta}, L).$$

    On the narrow-pulse face $H_2 = q\ell$, the cumulant boundary of 3a. On the long-pulse face $H_2 = (\ell/\ell_g)^3$, which gives both the free–localisation line ($\ell_D = \ell_g$) and the motional narrowing–localisation line ($L = \ell_g$) of 3b.

    The map shows the exact error of the GPA, $\lvert E_{\text{GPA}}/E - 1\rvert$, over the $(L/\ell_g, \ell_\Delta/L)$ plane at a fixed ratio $\delta/\Delta$. The solid black curve is the 5% contour. Dashed: $H = 1$ and $H = 3$. Dash-dotted: $H_2 = 1$ and $H_2 = 3$. At $\delta/\Delta = 0.5$ a ring marks the running example ($\delta = 10$, $\Delta = 20$ ms, $G = 205$ mT/m, $L = 8$ µm). At $\delta/\Delta = 1$ the plane is the long-pulse face of 3b.
    """)
    return


@app.cell(hide_code=True)
def _(gpa_box, np, pgse_box):
    import functools

    in_x = np.logspace(np.log10(0.3), np.log10(14.0), 34)     # L / ℓ_g
    in_y = np.logspace(-1.25, 0.75, 40)                        # ℓ_Δ / L = √(DΔ)/L

    @functools.lru_cache(maxsize=8)
    def interior_error(r):
        """|E_GPA/E − 1| and E on the (in_y, in_x) grid at δ/Δ = r."""
        Dt = in_y**2
        dt = r * Dt
        E = np.empty((in_y.size, in_x.size))
        F = np.empty_like(E)
        for j, xv in enumerate(in_x):
            N = int(min(7 * xv + 30, 130))
            E[:, j] = pgse_box(xv**3, dt, Dt, N=N).real
            F[:, j] = np.exp(gpa_box(xv**3, dt, Dt))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(F / E - 1), E

    return in_x, in_y, interior_error


@app.cell(hide_code=True)
def _(mo):
    in_r = mo.ui.slider(steps=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], value=0.5, label="δ / Δ", show_value=True)
    in_r
    return (in_r,)


@app.cell(hide_code=True)
def _(C_ALT, C_MAIN, GAMMA, in_r, in_x, in_y, interior_error, mo, np, plt):
    _r = in_r.value
    _err, _E = interior_error(_r)
    _valid = np.abs(_E) > 1e-8
    _X, _Y = np.meshgrid(in_x, in_y)
    _G = _X**3
    _dt = _r * _Y**2
    _H = _G * np.minimum(_dt, 1.0)
    _ell = np.minimum(_Y, 1.0)
    _H2 = _G * _ell * np.minimum(_dt, _ell**2)

    _fig, _a = plt.subplots(figsize=(7.4, 5.0))
    _pc = _a.pcolormesh(in_x, in_y, np.where(_valid, np.log10(np.clip(_err, 1e-5, 10)), np.nan), cmap="Purples", vmin=-4, vmax=0.5, shading="nearest", rasterized=True)
    _a.pcolormesh(in_x, in_y, np.where(_valid, np.nan, 1.0), cmap=ListedColormap(["#b9bec5"]), shading="nearest", rasterized=True)
    _cb = _fig.colorbar(_pc, ax=_a, pad=0.02)
    _cb.set_label("log₁₀ |E_GPA / E − 1|")
    _a.contour(in_x, in_y, np.where(_valid, _err, np.nan), levels=[0.05], colors="k", linewidths=2.2)
    _a.contour(in_x, in_y, _H, levels=[1, 3], colors=C_ALT, linewidths=1.5, linestyles="--")
    _a.contour(in_x, in_y, _H2, levels=[1, 3], colors=C_MAIN, linewidths=1.5, linestyles="-.")
    _a.plot([], [], color="k", lw=2.2, label="exact: GPA 5% off")
    _a.plot([], [], color=C_ALT, lw=1.5, ls="--", label="H = γgL·min(δ, L²/D) = 1, 3")
    _a.plot([], [], color=C_MAIN, lw=1.5, ls="-.", label="H₂ = γgℓ·min(δ, ℓ²/D) = 1, 3")
    _a.axhline(1, color="0.4", lw=0.7, ls=":")
    _a.axvline(1, color="0.4", lw=0.7, ls=":")
    # running example
    _Lr, _Dr = 8.0, 2.0
    _xr = _Lr / (_Dr / (GAMMA * 205)) ** (1 / 3)
    _yr = np.sqrt(_Dr * 20) / _Lr
    if _r == 0.5:
        _a.plot(_xr, _yr, "o", ms=10, mfc="none", mec="k", mew=1.8)
        _a.text(_xr * 1.15, _yr * 1.35, "running example", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    _a.set(xscale="log", yscale="log", xlim=(in_x[0], in_x[-1]), ylim=(in_y[0], in_y[-1]), xlabel="L / ℓ_g", ylabel="ℓ_Δ / L = √(DΔ)/L", title=f"Error of the Gaussian phase approximation, δ/Δ = {_r}")
    _a.legend(frameon=False, fontsize=7.5, loc="lower left")
    _a.grid(False)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Grey: E < 10⁻⁸. Left of the black curve the GPA is within 5%. A good rule draws its line just inside the black curve everywhere.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The test.** For each combination of $\delta/\Delta$ and $\Delta$ below, the gradient was raised until the GPA was 5% off the exact MCF signal, and both rules were evaluated there. A rule with a constant critical value would be perfect. Entries are $H^*\,/\,H_2^*$ (in the long-$\Delta$ rows $\ell = L$, so $H = H_2$):

    | $D\Delta/L^2$ | $\delta/\Delta = 0.05$ | $\delta/\Delta = 0.2$ | $\delta/\Delta = 1$ |
    | --- | --- | --- | --- |
    | 0.003 | — | — | 27.4 / 1.50 |
    | 0.01 | — | 12.3 / 1.23 | 13.9 / 1.39 |
    | 0.03 | 7.1 / 1.22 | 7.2 / 1.25 | 8.3 / 1.44 |
    | 0.1 | 5.6 / 1.76 | 5.5 / 1.75 | 7.2 / 2.28 |
    | 0.3 | 3.1 / 1.69 | 3.3 / 1.81 | 5.2 / 2.86 |
    | 1 | 3.0 | 3.8 | 8.8 |
    | 3 | 3.5 | 6.1 | 6.2 |
    | 10 | 5.6 | 6.9 | 4.5 |

    (— : the GPA stayed within 5% up to $\gamma gL^3/D = 2\times10^4$.) Over 34 such cases ($\delta/\Delta$ from 0.01 to 1, $D\Delta/L^2$ from 0.003 to 10):

    - **$H$** fails at values between 2.8 and 27. At long $\Delta$ its shape is right and $H\lesssim3$ is a safe 5% rule. At short $\Delta$ it is too strict, by up to an order of magnitude.
    - **$H_2$** fails between 1.2 and 8.8. **$H_2\lesssim1$ is safe everywhere**; at $\Delta\gtrsim L^2/D$, $H_2\lesssim3$ is enough.

    → Running example: $H = 4.4$, $H_2 = 3.5$, and the GPA is 4.9% off. As $\delta/\Delta$ decreases, the error contour moves to the right (larger $g$), because only $q = \gamma g\delta$ then matters: at fixed $g$ a shorter pulse is a weaker experiment.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Compute $H$ and $H_2$ for the running example at 205 mT/m. Which one is closer to its critical value?

    **(b)** Set $\delta/\Delta = 1$. In which part of the map is the rule $H\lesssim1$ furthest from the exact contour? Explain physically why the GPA works there at much larger $g$ than $H$ predicts.

    **(c)** Show that $H_2$ reduces to the narrow-pulse boundary $q\min(\ell_D, L)$ for $\delta\ll\ell^2/D$, and to $(\min(\ell_D, L)/\ell_g)^3$ for $\delta\gg\ell^2/D$.

    **(d)** Turn the result into a recipe of three steps that you could give a student who wants to fit a GPA-based model (for example an axon-diameter model) to data.
    """, "8 · Testing a rule of thumb")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\gamma GL = 2.6752\times10^{-4}\times205\times8 = 0.439$ /ms. $H = 0.439\times\min(10, 32) = 4.39$. $\ell = \min(\sqrt{40}, 8) = 6.32$ µm, so $\gamma G\ell = 0.347$ /ms and $H_2 = 0.347\times\min(10, 20) = 3.47$. The GPA is 4.9% off here (Chapter 2: 5% at 205 mT/m). Both rules are conservative: "$\lesssim1$" would stop at about 47 and 59 mT/m. $H_2 = 3.5$ is closer to the typical critical value at this $\Delta$.

    **(b)** ⇒ The rule $H$ fails worst at **short $\Delta$** (bottom of the map), up to 27 times too strict at $D\Delta/L^2 = 0.003$. There $\ell_D\ll L$: most spins never meet a wall in the time of the experiment, and free diffusion has an exactly Gaussian phase at any $g$. The GPA breaks only when the few wall spins start to dominate the signal, i.e. at the onset of localisation, $\ell_D\approx1.5\,\ell_g$. $H$ uses $L$ and so counts a pore size the spins have not explored.

    **(c)** ⇒ If $\delta < \ell^2/D$: $H_2 = \gamma g\delta\,\ell = q\ell = q\min(\ell_D, L)$. ✓

    ⇒ If $\delta > \ell^2/D$: $H_2 = \gamma g\ell^3/D = (\ell/\ell_g)^3$. With $\ell = \ell_\Delta$ (short $\Delta$) this is the free–localisation line; with $\ell = L$ it is the motional narrowing–localisation line. ✓

    → On the long-pulse face $\ell_\Delta = \sqrt{D\Delta} = \sqrt{DT/2}$, so $\ell_\Delta/\ell_D = 1/\sqrt2$: same lines up to a constant.

    **(d)** ① Compute $\ell = \min(\sqrt{D\Delta}, L)$ with the largest pore size you want to fit, and $H_2 = \gamma g\ell\min(\delta, \ell^2/D)$ for your strongest gradient.
    ② $H_2\le1$: the GPA formulas are safe to 5%. $1 < H_2\le3$ and $\Delta\gtrsim L^2/D$: probably fine, check one point with an exact method. $H_2 > 3$: do not use the GPA; use the matrix method (Chapter 2) or simulations (Chapters 4–5).
    ③ Check one protocol point against the exact MCF, as in this chapter, before trusting the fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3d · Two extra axes

    The box assumed reflecting, impermeable walls. Real walls can also **remove magnetization** (surface relaxation) or **let it through** (permeability). Each adds one dimensionless number and one more axis to the box.

    ## ① Relaxing walls: Brownstein and Tarr

    If the wall relaxes magnetization with relaxivity $\rho$ (µm/ms), the reflecting condition of Chapter 1 becomes the Robin condition

    $$D\,\frac{\partial m}{\partial n} + \rho\,m = 0\quad\text{on the wall}.$$

    The eigenmodes change, and with them the relaxation of the whole pore ([Brownstein & Tarr 1979](https://doi.org/10.1103/PhysRevA.19.2446)). For a slab, the slowest symmetric mode is $\cos\big(k(x - L/2)\big)$ with

    $$\frac{kL}2\,\tan\frac{kL}2 = \frac{\rho L}{2D},\qquad \frac1{T_{2,\text{surf}}} = Dk^2 .$$

    The new axis is $\rho L/D$, which compares how fast the wall removes spins with how fast diffusion brings them there:

    | $\rho L/D$ | Regime | Rate of the slowest mode | Decay |
    | --- | --- | --- | --- |
    | $\ll1$ | **fast diffusion** (surface-limited) | $\rho\,S/V$ ($= 2\rho/L$ for a slab) | one exponential |
    | $\gg1$ | **slow diffusion** (diffusion-limited) | $D\pi^2/L^2$ | several exponentials |

    The fast-diffusion limit is why NMR relaxometry measures $S/V$ in rocks and why relaxation and pore size are linked ([Brownstein & Tarr 1979](https://doi.org/10.1103/PhysRevA.19.2446)). For diffusion MRI, relaxing walls take spins out of the layer next to the wall, the same layer that carries the short-time correction ① and the localisation signal of 3b. [Mitra et al. (1993)](https://doi.org/10.1103/PhysRevB.47.8565) show that $\rho$ enters $D(t)$ at order $t$.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, mo, np, plt):
    _c = np.logspace(-3, 3, 300)                     # ρL/D
    _lo, _hi = np.zeros_like(_c), np.full_like(_c, np.pi / 2 - 1e-12)
    for _ in range(80):                              # vectorised bisection for y tan y = ρL/(2D)
        _mid = 0.5 * (_lo + _hi)
        _f = _mid * np.tan(_mid) - _c / 2
        _hi = np.where(_f > 0, _mid, _hi)
        _lo = np.where(_f > 0, _lo, _mid)
    _y = 0.5 * (_lo + _hi)                           # y = kL/2
    _rate = 4 * _y**2                                # (L²/D) / T2,surf
    _w0 = 2 * np.sin(_y) ** 2 / (_y**2 + _y * np.sin(_y) * np.cos(_y))

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    _a1.loglog(_c, _rate, color=C_MAIN, lw=2.2, label="slowest mode, exact")
    _a1.loglog(_c, 2 * _c, color=C_ALT, lw=1.3, ls="--", label="fast diffusion: ρS/V = 2ρ/L")
    _a1.axhline(np.pi**2, color=C_FREE, lw=1.3, ls="--", label="slow diffusion: Dπ²/L²")
    _a1.set(ylim=(1e-3, 30), xlabel="ρL / D", ylabel="(L²/D) · 1/T₂,surf", title="Brownstein–Tarr: two regimes")
    _a1.legend(frameon=False, fontsize=8, loc="upper left")
    _a2.semilogx(_c, _w0, color=C_MAIN, lw=2.2)
    _a2.axhline(8 / np.pi**2, color=C_FREE, lw=1.0, ls=":")
    _a2.text(2e2, 8 / np.pi**2 + 0.012, "8/π² = 0.81", fontsize=8, color="0.4")
    _a2.set(ylim=(0.78, 1.01), xlabel="ρL / D", ylabel="weight of the slowest mode", title="Uniform start: how mono-exponential?")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Running example: with ρ = 0.01 µm/ms, ρL/D = 0.04 (fast diffusion), and T₂,surf = L/(2ρ) = 400 ms.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive the eigenvalue condition $\frac{kL}2\tan\frac{kL}2 = \frac{\rho L}{2D}$ for the symmetric modes of a slab with the Robin condition at both walls. Then show that for $\rho L/D\ll1$ the slowest rate is $2\rho/L$, and for $\rho L/D\gg1$ it is $D\pi^2/L^2$.

    **(b)** In the fast-diffusion regime, show that $1/T_{2,\text{surf}} = \rho\,S/V$ for any pore shape, by integrating $\partial m/\partial t = D\nabla^2m$ over the pore with an almost uniform $m$.

    **(c)** Why does the fast-diffusion regime give a single exponential? Read the weight of the slowest mode off the right panel at $\rho L/D = 0.04$ and at $\rho L/D = 100$.
    """, "9 · Relaxing walls")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Centre the slab: $x\in[-L/2, L/2]$, symmetric mode $u = \cos kx$. At $x = L/2$ the outward normal is $+x$: $D u' + \rho u = 0$ ⇒ $-Dk\sin\frac{kL}2 + \rho\cos\frac{kL}2 = 0$ ⇒ $k\tan\frac{kL}2 = \rho/D$. Multiply by $L/2$. ✓

    ⇒ $\rho L/D\ll1$: $y = kL/2$ is small, $y\tan y\approx y^2 = \rho L/(2D)$, so $Dk^2 = 4Dy^2/L^2 = 2\rho/L$.

    ⇒ $\rho L/D\gg1$: $y\tan y$ is large only near the pole, $y\to\pi/2$, so $Dk^2\to D\pi^2/L^2$. The wall acts as a perfect sink (a Dirichlet condition).

    **(b)** ⇒ $\frac{d}{dt}\int_Vm = D\oint_S\partial_nm\,dS = -\rho\oint_Sm\,dS$, by the divergence theorem (*your notes:* it is pinned on several of your sheets) and the Robin condition. With $m\approx\bar m$ uniform: $V\dot{\bar m} = -\rho S\bar m$, so $\bar m\propto e^{-\rho(S/V)t}$. ✓

    **(c)** ⇒ When diffusion is fast compared with the wall, $m$ stays uniform, and a uniform $m$ is almost exactly the slowest mode. The weight is 0.99999 at $\rho L/D = 0.04$ and 0.83 at 100, approaching $8/\pi^2 = 0.81$ for a perfect sink. The rest is spread over faster modes: the decay becomes multi-exponential.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ② Exchange between compartments: Kärger

    Now let two compartments exchange spins across a membrane. If the membrane is the bottleneck, each compartment is well mixed and can be described by one number, its magnetization. With narrow pulses, and a Gaussian signal inside each compartment, the two magnetizations obey a linear ODE system ([Kärger 1985](<https://doi.org/10.1016/0001-8686(85)80018-X>)):

    $$\frac{d\mathbf m}{dt} = -\big(q^2\mathbf D + \mathbf K\big)\,\mathbf m,\qquad \mathbf D = \begin{pmatrix}D_1&0\\0&D_2\end{pmatrix},\quad \mathbf K = \begin{pmatrix}k_1&-k_2\\-k_1&k_2\end{pmatrix},$$

    with $\mathbf m(0) = (p_1, p_2)$ and detailed balance $k_1p_1 = k_2p_2$. The signal is $E = (1,1)\,e^{-(b\mathbf D + \mathbf K\Delta)}\,\mathbf p$, with $b = q^2\Delta$. *Your notes:* a linear system $\dot{\mathbf x} = A\mathbf x$, solved by $e^{tA}$ and its eigenvalues, as on your ODE sheets.

    The new axis is $\Delta/\tau_{\text{ex}}$, with $1/\tau_{\text{ex}} = k_1 + k_2$. From Chapter 1, a barrier-limited membrane with permeability $\kappa$ gives $k_1 = \kappa\,(S/V)_1$.

    → *Chapter 1 already contains this model as a set of slab modes* (the permeable-membrane figure and Exercise 9). With a membrane in the middle of a slab, the uniform mode stays, and the slowest odd mode, one half high and the other low, decays at $D\lambda\approx4\kappa/L$. That is $k_1 + k_2$: each half has $S/V = 2/L$, so $k_1 = k_2 = 2\kappa/L$. Kärger's two magnetizations are the projections on these two modes; the faster modes, which he neglects, are the ones that need $\kappa L/D\ll1$ to die out first. The model needs $\kappa L/D\ll1$ (the same condition as for tortuosity in ②), and each compartment must be inside its own cumulant region (3a ④). Filter-exchange imaging (FEXI) measures $\tau_{\text{ex}}$ this way ([Lasič et al. 2011](https://doi.org/10.1002/mrm.22782)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ex_k = mo.ui.slider(-1.5, 1.5, step=0.1, value=0.0, label="log₁₀(Δ / τ_ex) for the highlighted curve", show_value=True)
    ex_p = mo.ui.slider(0.1, 0.9, step=0.05, value=0.3, label="fraction p₁ in the fast compartment", show_value=True)
    mo.hstack([ex_k, ex_p], justify="start", gap=2)
    return ex_k, ex_p


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, ex_k, ex_p, expm, mo, np, plt):
    _D1, _D2 = 2.0, 0.3
    _p1 = ex_p.value
    _p = np.array([_p1, 1 - _p1])

    def _E(b, x):
        """Kärger signal for b (ms/µm²) and x = Δ/τ_ex."""
        k1, k2 = x * (1 - _p1), x * _p1                # k1 + k2 = 1/τ_ex, k1 p1 = k2 p2
        Kd = np.array([[k1, -k2], [-k1, k2]])
        return np.array([np.ones(2) @ expm(-(bb * np.diag([_D1, _D2]) + Kd)) @ _p for bb in b])

    _b = np.linspace(0, 5, 60)
    _fig, _a = plt.subplots(figsize=(6.6, 3.8))
    _ylab = []
    for _x, _al in ((0.0, 0.9), (1.0, 0.6), (3.0, 0.5), (10.0, 0.4), (100.0, 0.3)):
        _a.semilogy(_b, _E(_b, _x), color=C_FREE, lw=1.0, alpha=_al)
        _ylab.append((_E(np.array([5.0]), _x)[0], _x))
    _last = None
    for _yv, _x in sorted(_ylab, reverse=True):         # keep the grey labels apart
        _yv = _yv if _last is None else min(_yv, _last / 1.22)
        _a.text(5.05, _yv, f"{_x:g}", fontsize=7.5, color="0.4", va="center")
        _last = _yv
    _xs = 10**ex_k.value
    _a.semilogy(_b, _E(_b, _xs), color=C_MAIN, lw=2.4, label=f"Δ/τ_ex = {_xs:.2g}")
    _a.semilogy(_b, np.exp(-_b * (_p1 * _D1 + (1 - _p1) * _D2)), color=C_ALT, lw=1.2, ls="--", label="fast exchange: e^{−b D̄}")
    _a.set(xlabel="b (ms/µm²)", ylabel="E", ylim=(1e-3, 1.1), xlim=(0, 5.4), title=f"Kärger: D₁ = {_D1}, D₂ = {_D2} µm²/ms (grey labels: Δ/τ_ex)")
    _a.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Show that the Kärger signal is bi-exponential in $b$ with the true fractions when $\Delta\ll\tau_{\text{ex}}$, and mono-exponential with $\bar D = p_1D_1 + p_2D_2$ when $\Delta\gg\tau_{\text{ex}}$.

    **(b)** In between, fit two exponentials to the highlighted curve by eye. Do the "apparent fractions" equal $p_1$ and $p_2$? What does that mean for a two-compartment model fitted at one $\Delta$?

    **(c)** For a slab compartment of width $L = 8$ µm with membranes of permeability $\kappa = 0.02$ µm/ms on both sides, compute the exit rate $k = \kappa S/V$ and the residence time $1/k$. Is the Kärger model justified ($\kappa L/D\ll1$)?
    """, "10 · Exchange")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\Delta\ll\tau_{\text{ex}}$: $\mathbf K\Delta\to0$, so $E = (1,1)e^{-b\mathbf D}\mathbf p = p_1e^{-bD_1} + p_2e^{-bD_2}$.

    ⇒ $\Delta\gg\tau_{\text{ex}}$ (and also $\tau_{\text{ex}}\ll1/(q^2\lvert D_1-D_2\rvert)$, so that spins switch before they dephase): exchange mixes the compartments many times during $\Delta$. $\mathbf K$ has eigenvalues $0$ and $1/\tau_{\text{ex}}$; the second eigenvector dies, and the first is $\mathbf p$ itself. Projected on it, $b\mathbf D$ contributes $b\,(p_1D_1 + p_2D_2)$: $E\approx e^{-b\bar D}$. Every spin samples both compartments, so its displacement is Gaussian with the average diffusivity. The same averaging produced motional narrowing in 3b. In the figure, even $\Delta/\tau_{\text{ex}} = 10$ is not yet at $e^{-b\bar D}$ at $b = 5$: the second condition, $b\lvert D_1 - D_2\rvert\ll\Delta/\tau_{\text{ex}}$, fails there.

    **(b)** ⇒ No. At intermediate $\Delta$ the curve is still close to bi-exponential, but the slow "component" is smaller than $p_2$ and both apparent diffusivities move towards $\bar D$: spins that started slow have partly become fast. A two-compartment model fitted without exchange returns fractions and diffusivities that depend on $\Delta$. Kärger's point was that this $\Delta$-dependence measures the exchange rate.

    **(c)** ⇒ $S/V = 2/L = 0.25$ µm⁻¹, so $k = 0.02\times0.25 = 0.005$ /ms and $1/k = 200$ ms.

    ⇒ $\kappa L/D = 0.02\times8/2 = 0.08\ll1$: the membrane is the bottleneck, and the compartment is well mixed on the time scale of exchange ($L^2/D = 32$ ms against 200 ms). Kärger is justified. For a typical $\Delta$ of 20–50 ms, $\Delta/\tau_{\text{ex}}$ is about 0.1–0.25: slow exchange, but measurable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3e · Chapter 3 on one page

    *A crib sheet: every formula of this chapter with its region. The conditions are the ones tested above against the exact signal of Chapter 2.*

    | ★ Regime | Condition | Slab, width $L$ | Cylinder ($q\perp$ axis) / sphere, radius $R$ | First reference |
    | --- | --- | --- | --- | --- |
    | **Free**, any pulses | $\ell_D\ll L$, $\ell_D\lesssim\ell_g$ | $E = e^{-bD}$; PGSE $b = \gamma^2g^2\delta^2(\Delta-\delta/3)$ | same | Stejskal & Tanner 1965 |
    | Anisotropic free | Gaussian with tensor $\mathbf D$ | $E = e^{-\mathbf b:\mathbf D}$, $\ b_{ij} = q_iq_jt$ | same | Basser et al. 1994 |
    | **Short times** (narrow pulses) | $Dt\lesssim0.1L^2$, $q\ell_D\lesssim1$ | $D(t)/D = 1 - \frac{8}{3\sqrt\pi}\frac{\sqrt{Dt}}L$; $\ K\approx\frac{16}{5\sqrt\pi}\frac{\sqrt{Dt}}L$ | $1 - \frac4{3\sqrt\pi\,d}\frac SV\sqrt{Dt}$, $S/V = 2/R$ ($d = 2$), $3/R$ ($d = 3$) | Mitra et al. 1992 |
    | **Long times**, closed pore | $Dt\gtrsim0.3L^2$ | $D(t) = L^2/(12t)$, $K = -3/5$ | $D(t) = R^2/(4t)$ / $R^2/(5t)$ | Robertson 1966; Tanner & Stejskal 1968 |
    | **Form factor** | $Dt\gtrsim0.3L^2$ (dip $< 10^{-2}$ for $Dt > 0.36L^2$) | $\operatorname{sinc}^2(qL/2)$, zero at $qL = 2\pi$ | $[2J_1(qR)/qR]^2$, zero $qR = 3.832$ / $[3j_1(qR)/qR]^2$, zero $qR = 4.493$ | Tanner & Stejskal 1968; Callaghan et al. 1991 |
    | **Cumulant edge** (narrow pulses) | — | $qX_{\text{rms}}\lesssim1$; 5% at $(1.2/\lvert K\rvert)^{1/4}$; never past $qL = 2\pi$ | same rule with $X_{\text{rms}}$ of the pore | this chapter |
    | Tortuosity | connected medium | $1/D_\infty = 1/D + 1/(\kappa L)$ | — | — |
    | Disorder | $t\to\infty$ | $D(t) - D_\infty\propto t^{-(p+d)/2}$; 1D short-range: $t^{-1/2}$ | — | Novikov et al. 2014 |
    | **Free**, long pulses | $\ell_D < \ell_g, L$ | $\ln E = -\gamma^2g^2DT^3/12$ | same | Carr & Purcell 1954 |
    | **Motional narrowing** | $L < \ell_g, \ell_D$; $DT\gtrsim3L^2$ | $\ln E = -\frac{\gamma^2g^2L^4}{120D}\big(T - \frac{17}{56}\frac{L^2}D\big)$ | $-\frac7{96}\frac{\gamma^2g^2R^4T}D$ / $-\frac8{175}\frac{\gamma^2g^2R^4T}D$ | Robertson 1966; Neuman 1974 |
    | **Localisation** | $L/\ell_g\gtrsim4$, $\ell_D\gtrsim2\ell_g$ | $E\approx5.89\frac{\ell_g}Le^{-0.509\,DT/\ell_g^2}$ | rate $0.509\,D/\ell_g^2$ (locally flat wall) | Stoller et al. 1991; Hürlimann et al. 1995 |
    | **GPA inside the box** | — | $H_2 = \gamma g\ell\min(\delta, \ell^2/D)\lesssim1$, $\ \ell = \min(\sqrt{D\Delta}, L)$ | same, with $L\to2R$ (untested) | this chapter |
    | Relaxing walls | Robin condition | $\frac{kL}2\tan\frac{kL}2 = \frac{\rho L}{2D}$; fast diffusion $1/T_2 = \rho S/V$ | $\rho S/V$ in the fast-diffusion limit | Brownstein & Tarr 1979 |
    | Exchange | $\kappa L/D\ll1$, narrow pulses | $E = (1,1)\,e^{-(b\mathbf D + \mathbf K\Delta)}\,\mathbf p$, $\ k_1 = \kappa(S/V)_1$ | same | Kärger 1985 |

    → **Mnemonics.** The cylinder's 1.22 is the Airy disk of optics. The three long-pulse exponents are $6, 4{+}2, 2$ powers of $1/\ell_g$: free $(\ell_D/\ell_g)^6$, narrowing $(L/\ell_g)^4(\ell_D/\ell_g)^2$, localisation $(\ell_D/\ell_g)^2$.

    **Open questions (BEWIJS).** The kurtosis coefficient $16/(5\sqrt\pi)$ is derived here for a slab only; the general-$d$ version with $S/V$ is not. The refined rule $H_2$ is tested for the slab only, not yet for cylinders and spheres.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementations

    - **Closed-form compartment models.** Camino and dmipy implement the cylinder and sphere signals of this chapter in their narrow-pulse (Stejskal–Tanner), Callaghan and Gaussian-phase forms, for fitting axon-diameter and cell-size models. Before fitting one, evaluate $H_2$ for the protocol (Exercise 8).
    - **The exact references.** The MCF of Chapter 2 (for example MCFAL, MISST, the cylindrical-surface code) computes the signal everywhere in the box, including the white bands of the maps.
    - **Validating simulators.** The closed forms are the standard checks for numerical simulators. A recent example: the "replay packs" of [Fick (2026)](https://arxiv.org/abs/2611.01758) store Monte Carlo walker histories so that any sequence can be computed later as a cheap inner product. They are checked against analytical solutions in pores before realistic axon meshes. Chapter 5 returns to this paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary of Chapter 3

    | Idea | Formula |
    | --- | --- |
    | The PGSE box | $\tilde\Delta = D\Delta/L^2$, $\ \tilde\delta = D\delta/L^2$, $\ \Gamma = \gamma gL^3/D = (L/\ell_g)^3$ |
    | Narrow-pulse face | $E(q,t) = \langle e^{iqX}\rangle = \int\bar P(X,t)\,e^{iqX}dX = \sum_n e^{-D\lambda_nt}\lvert\tilde u_n(q)\rvert^2/L\ \ge0$ |
    | Its four regions | free; short times ($S/V$); form factor; Gaussian cumulant, edge $qX_{\text{rms}}\approx1$ |
    | Long-pulse face | free $g^2T^3$; motional narrowing $g^2T$; localisation $g^{2/3}T$; triple point $\ell_D = \ell_g = L$ |
    | Inside the box | GPA safe for $H_2 = \gamma g\ell\min(\delta, \ell^2/D)\lesssim1$, $\ell = \min(\sqrt{D\Delta}, L)$ |
    | Lines vs regions | the lines give the shape; the constants shift the edges by factors 0.5–3; around the lines are crossover bands |
    | Extra axes | $\rho L/D$ (relaxing walls), $\Delta/\tau_{\text{ex}}$ and $\kappa L/D$ (exchange) |

    ---
    ## Coming next

    **Chapter 4 · Solving the PDE on a grid or a mesh**

    - **Finite differences:** the domain as voxels; the Laplacian as jumps to neighbours; explicit time steps and their stability limit; periodic boundaries with a phase shift. *Implementation:* MATI.
    - **Finite elements:** the weak form of Chapter 1 with hat functions, mass and stiffness matrices, a mesh that follows the cell walls; permeable membranes as double nodes; the mesh-based modes used by the matrix formalism. *Implementation:* SpinDoctor.
    - Every solver is checked against the closed forms of this chapter: the form factor, motional narrowing and localisation.

    **Chapter 5 · Monte Carlo: simulating the random walk**

    - Walkers, reflections and membranes; choosing the time step; the $1/\sqrt N$ noise floor, against the exact signals of Chapters 2 and 3.
    - Walker histories as reusable data: replay packs ([Fick 2026](https://arxiv.org/abs/2611.01758)).
    - *Implementations:* Camino, MC-DC Simulator, Disimpy, RMS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## References

    Where possible, the first paper that derived or introduced each idea.

    | Year | Reference | What it introduced |
    | --- | --- | --- |
    | 1954 | [H. Y. Carr & E. M. Purcell, *Phys. Rev.* 94:630](https://doi.org/10.1103/PhysRev.94.630) | echo decay in a constant gradient, $e^{-\gamma^2g^2DT^3/12}$ |
    | 1965 | [E. O. Stejskal & J. E. Tanner, *J. Chem. Phys.* 42:288](https://doi.org/10.1063/1.1695690) | the PGSE sequence, its $b$-value, and the narrow-pulse limit |
    | 1966 | [B. Robertson, *Phys. Rev.* 151:273](https://doi.org/10.1103/PhysRev.151.273) | restricted diffusion in a slab; motional narrowing |
    | 1968 | [J. E. Tanner & E. O. Stejskal, *J. Chem. Phys.* 49:1768](https://doi.org/10.1063/1.1670306) | the narrow-pulse signal of a slab |
    | 1974 | [C. H. Neuman, *J. Chem. Phys.* 60:4508](https://doi.org/10.1063/1.1680931) | GPA formulas for slab, cylinder and sphere; the long-time constants |
    | 1979 | [K. R. Brownstein & C. E. Tarr, *Phys. Rev. A* 19:2446](https://doi.org/10.1103/PhysRevA.19.2446) | relaxing walls; fast- and slow-diffusion regimes |
    | 1985 | [J. Kärger, *Adv. Colloid Interface Sci.* 23:129](<https://doi.org/10.1016/0001-8686(85)80018-X>) | exchange between compartments |
    | 1990 | [D. G. Cory & A. N. Garroway, *Magn. Reson. Med.* 14:435](https://doi.org/10.1002/mrm.1910140303) | q-space imaging of the displacement |
    | 1991 | [P. T. Callaghan, A. Coy, D. MacGowan, K. J. Packer & F. O. Zelaya, *Nature* 351:467](https://doi.org/10.1038/351467a0) | diffusion–diffraction |
    | 1991 | W. B. Hyslop & P. C. Lauterbur, *J. Magn. Reson.* 94:501 | restricted diffusion in NMR microscopy; brighter edges |
    | 1991 | [S. D. Stoller, W. Happer & F. J. Dyson, *Phys. Rev. A* 44:7459](https://doi.org/10.1103/PhysRevA.44.7459) | the localisation regime |
    | 1992 | [P. P. Mitra, P. N. Sen, L. M. Schwartz & P. Le Doussal, *Phys. Rev. Lett.* 68:3555](https://doi.org/10.1103/PhysRevLett.68.3555) | short-time $D(t)$ and the surface-to-volume ratio |
    | 1992 | B. Pütz, D. Barsky & K. Schulten, *J. Magn. Reson.* 97:27 | theory of edge enhancement |
    | 1993 | [P. P. Mitra, P. N. Sen & L. M. Schwartz, *Phys. Rev. B* 47:8565](https://doi.org/10.1103/PhysRevB.47.8565) | the next order in $D(t)$: curvature and surface relaxation |
    | 1993 | [L. L. Latour, P. P. Mitra, R. L. Kleinberg & C. H. Sotak, *J. Magn. Reson. A* 101:342](https://doi.org/10.1006/jmra.1993.1056) | measuring $S/V$ from $D(t)$ |
    | 1994 | [P. J. Basser, J. Mattiello & D. LeBihan, *J. Magn. Reson. B* 103:247](https://doi.org/10.1006/jmrb.1994.1037) | the $b$-matrix and diffusion tensor estimation |
    | 1994 | [T. M. de Swiet & P. N. Sen, *J. Chem. Phys.* 100:5597](https://doi.org/10.1063/1.467127) | bounded diffusion in a constant gradient; regimes set by length scales |
    | 1995 | [M. D. Hürlimann, K. G. Helmer, T. M. de Swiet & P. N. Sen, *J. Magn. Reson. A* 113:260](https://doi.org/10.1006/jmra.1995.1091) | first observation of localisation |
    | 1995 | [T. M. de Swiet, *J. Magn. Reson. B* 109:12](https://doi.org/10.1006/jmrb.1995.1141) | diffusive edge enhancement in imaging |
    | 2007 | [D. S. Grebenkov, *Rev. Mod. Phys.* 79:1077](https://doi.org/10.1103/RevModPhys.79.1077) | review: length scales and regimes, MCF |
    | 2011 | [S. Lasič, M. Nilsson, J. Lätt, F. Ståhlberg & D. Topgaard, *Magn. Reson. Med.* 66:356](https://doi.org/10.1002/mrm.22782) | filter-exchange imaging (FEXI) |
    | 2014 | [D. S. Novikov, J. H. Jensen, J. A. Helpern & E. Fieremans, *PNAS* 111:5088](https://doi.org/10.1073/pnas.1316944111) | power-law tails of $D(t)$ and structural disorder |
    | 2015 | [A. Burcaw, E. Fieremans & D. S. Novikov, *NeuroImage* 114:18](https://doi.org/10.1016/j.neuroimage.2015.03.061) | $D(t)$ along axons, the $t^{-1/2}$ tail |
    | 2019 | [N. Moutal, K. Demberg, D. S. Grebenkov & T. A. Kuder, *J. Magn. Reson.* 305:162](https://doi.org/10.1016/j.jmr.2019.06.016) | localisation: modern theory and experiments |
    | 2026 | [R. H. J. Fick, arXiv 2611.01758](https://arxiv.org/abs/2611.01758) | replayable Monte Carlo: walker histories as reusable data, checked against pore solutions |
    """)
    return


if __name__ == "__main__":
    app.run()
