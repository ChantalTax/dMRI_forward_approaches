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
    app_title="dMRI simulation · Ch 2 · Series expansions",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

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
    # gamma in units that fit µm, ms and mT/m: phase (rad) = GAMMA * g[mT/m] * x[µm] * t[ms]
    GAMMA = 2.6752e-4
    D0 = 2.0             # µm²/ms, used throughout this chapter
    return C_ALT, C_FREE, C_MAIN, D0, GAMMA, expm, mo, np, plt


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

    def pgse_mcf(G, L, delta, Delta, N=24, D=D0):
        """Exact PGSE echo in a slab with Grebenkov's MCF: three matrix exponentials."""
        lam, B = slab_B(L, N)
        Lam = np.diag(lam)
        P1 = expm(-(D * Lam + 1j * GAMMA * G * B) * delta)
        P2 = expm(-D * Lam * (Delta - delta))
        P3 = expm(-(D * Lam - 1j * GAMMA * G * B) * delta)
        return (P3 @ P2 @ P1)[0, 0]

    def slab_weights(L, nmax=801):
        """Weights w_n = B_0n² and eigenvalues λ_n of the modes that couple to the uniform mode (odd n)."""
        n = np.arange(1, nmax, 2)
        return 8 * L**2 / (n * np.pi) ** 4, (n * np.pi / L) ** 2

    def gpa_phase_var(G, L, delta, Delta, D=D0):
        """⟨φ²⟩ for a PGSE in a slab in the Gaussian phase approximation (sum over modes)."""
        w, lam = slab_weights(L)
        a = D * lam
        cross = 0.5 * (np.exp(-a * (Delta - delta)) + np.exp(-a * (Delta + delta))) - np.exp(-a * Delta)
        I = (4 * (a * delta - 1 + np.exp(-a * delta)) - 4 * cross) / a**2
        return (GAMMA * np.asarray(G, float)) ** 2 * np.sum(w * I)

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

    return gpa_phase_var, pgse_mcf, slab_B, slab_cumulants, slab_weights


@app.cell(hide_code=True)
def _(mo):
    show_answers = mo.ui.switch(label="Teacher view: show all answers", value=False)
    mo.vstack(
        [
            mo.md(
                r"""
    # Chapter 2 · Series expansions: two ways to expand the signal

    *© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.*

    *Course: simulating the diffusion MRI signal. Chapter 1 set up the eigenmodes and the Bloch–Torrey equation. This chapter turns them into the signal.*

    **What you'll take away**

    1. The Bloch–Torrey equation as a **matrix equation** in the eigenmodes of Chapter 1: the modes stay the same, the gradient only couples them.
    2. How to compute the signal **exactly** for any gradient waveform: Grebenkov's MCF and the multiple propagator, and the error each makes.
    3. How many modes you need, and why.
    4. The **cumulant expansion** of the signal, with the Gaussian phase approximation as its first term, built from the mode weights of Chapter 1.
    5. Why finite gradient pulses change the meaning of the measured $D$ and $K$.
    6. How the two series are connected: every cumulant is a product of the same matrices.
    7. How far the cumulant series reaches (its radius of convergence, set by the nearest complex zero of the signal), and how to compute cumulants of any order in practice.

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
    | 1 | Foundations: notation, diffusion equation, interface and boundary conditions, eigenmodes, displacement cumulants; Bloch–Torrey; length scales; Feynman–Kac | done |
    | **2** | **Series expansions:** (a) the eigenmodes coupled by the gradient, i.e. matrix methods; (b) the cumulant expansion in powers of the gradient; (c) how the two connect | this notebook |
    | 3 | Analytical formulas: the PGSE box, the narrow-pulse $(q,t)$ map, the long-pulse $(g,T)$ map, the inside of the box, relaxing walls and exchange | next |
    | 4 | Solving the PDE on a grid or a mesh: finite differences and finite elements | coming |
    | 5 | Monte Carlo: simulating the random walk | coming |

    **Units:** lengths in µm, times in ms, gradients in mT/m. In this chapter $D = 2$ µm²/ms throughout. All examples use a **slab** of width $L$ with the gradient across it: its modes are known in closed form, so every number can be checked by hand. Cylinders and spheres work the same way with the modes from the Chapter 1 tables.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you need from Chapter 1

    | Idea | Formula |
    | --- | --- |
    | Eigenmodes of a pore | $-\nabla^2u_n = \lambda_nu_n$, $\ \partial u_n/\partial n = 0$ on the walls, $\ \int u_nu_m = \delta_{nm}$ |
    | Slab | $u_0 = 1/\sqrt L$, $\ u_n = \sqrt{2/L}\cos(n\pi x/L)$, $\ \lambda_n = (n\pi/L)^2$ |
    | Propagator | $P(x\mid x_0,t) = \sum_n u_n(x)u_n(x_0)e^{-D\lambda_nt}$ |
    | Averages $\langle\cdot\rangle$ | the mean over many spins, each starting at a uniformly random $x_0$ and following its own random path: $\langle f\rangle = \lim_{N\to\infty}\frac1N\sum_{j=1}^N f_j$. For start and end points only: $\langle f(x_0, x)\rangle = \frac1L\iint f(x_0, x)\,P(x\mid x_0,t)\,dx\,dx_0$ |
    | MSD weights in a slab | $\langle(x-x_0)^2\rangle = \frac1L\iint(x-x_0)^2P(x\mid x_0,t)\,dx\,dx_0 = \frac{L^2}{6} - 2\sum_{n\ \text{odd}} \frac{8L^2}{(n\pi)^4}e^{-D\lambda_nt}$ |
    | Bloch–Torrey | $\partial m/\partial t = D\nabla^2m - i\gamma\,g(t)\,x\,m$, $\ E = \int m\,dx\big/\int m(x,0)\,dx$ |
    | Path picture | $E = \langle e^{-i\varphi}\rangle = \lim_{N\to\infty}\frac1N\sum_{j=1}^N e^{-i\varphi_j}$, $\ \varphi_j = \gamma\int_0^T g(t)\,x_j(t)\,dt$ for spin $j$ with path $x_j(t)$ |
    | Gradient length | $\ell_g = (D/\gamma g)^{1/3}$ |

    ## Two kinds of series

    Both halves of this chapter expand the signal in a series. They are easy to confuse, and they do very different things.

    | | **2a · Eigenmode expansion (matrix methods)** | **2b · Cumulant expansion** |
    | --- | --- | --- |
    | What is expanded | the magnetization $m(x,t)$, in the eigenmodes $u_n$ | $\ln E$, in powers of the gradient |
    | Small parameter | none: keep enough modes | the phase spread, $\langle\varphi^2\rangle = \lim\frac1N\sum_j\varphi_j^2\lesssim 1$ |
    | Result | exact (to any precision) at any gradient | an approximation for weak gradients |
    | Output | numbers | closed formulas you can fit and interpret |
    | First term | — | the Gaussian phase approximation |

    Section 2c shows that they are built from the same pieces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 2a · Matrix methods

    ## The Bloch–Torrey equation in the eigenbasis

    The eigenmodes of a pore are a complete set: any magnetization that satisfies the interface conditions can be written as a sum of them. So write

    $$m(x,t) = \sum_n c_n(t)\,u_n(x).$$

    Putting this into Bloch–Torrey turns one partial differential equation into a set of ordinary ones, one per mode:

    $$\frac{d\mathbf c}{dt} = -\big(D\Lambda + i\gamma\,g(t)\,\mathbf B\big)\,\mathbf c,\qquad \Lambda = \mathrm{diag}(\lambda_0,\lambda_1,\dots),\qquad B_{nm} = \int u_n(x)\,x\,u_m(x)\,dx .$$

    - **The modes do not change.** They are the modes of the pore from Chapter 1, fixed by its shape and interface conditions. The gradient does not create new ones.
    - **Diffusion is diagonal.** On its own, each mode decays at its own rate $D\lambda_n$.
    - **The gradient couples the modes.** The field $x$ is not a mode, so multiplying a mode by $x$ spreads it over the others; $\mathbf B$ says how much goes where. All the geometry is in $\Lambda$ and $\mathbf B$.
    - **Start and finish.** Spins start uniform, $m(x,0) = u_0$, so $\mathbf c(0) = \mathbf e_0 = (1,0,0,\dots)$. Only the uniform mode has a nonzero integral, so the signal is just the first component at the echo: $E = c_0(T)$.

    $\mathbf B$ is the **position matrix** of Chapter 1 (Exercise 5): real, symmetric, nonzero only between modes of opposite parity, with couplings that fall off as $1/n^2$, and $B_{0n}^2$ equal to the MSD weights. Together, $\Lambda$ and $\mathbf B$ are everything the geometry contributes.

    The whole simulation problem is now: propagate $\mathbf c$ from $\mathbf e_0$ to the echo, and read off $c_0$. The methods below differ only in **how they step through time**.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ Insert $m = \sum_m c_m(t)u_m(x)$ in $\partial m/\partial t = D\nabla^2m - i\gamma g(t)\,x\,m$, and use $\nabla^2u_m = -\lambda_mu_m$:

    $$\sum_m \dot c_m\,u_m = -D\sum_m\lambda_m c_m\,u_m - i\gamma g\sum_m c_m\,x\,u_m .$$

    ⇒ Multiply by $u_n$ and integrate over the pore. Orthonormality, $\int u_nu_m = \delta_{nm}$, picks out one term on the left and in the first sum:

    $$\dot c_n = -D\lambda_n c_n - i\gamma g\sum_m\Big(\int u_n\,x\,u_m\,dx\Big)c_m = -D\lambda_n c_n - i\gamma g\sum_m B_{nm}c_m .$$

    → Every $u_m$ already satisfies the interface conditions, so the sum does too: nothing else to impose.

    → For a slab the uniform mode is $u_0 = 1/\sqrt L$. With $m(x,0) = u_0$, $\ \int m(x,0)\,dx = \sqrt L$, and at the echo $\int m\,dx = \sum_n c_n\int u_n = c_0\sqrt L$ (all other modes integrate to zero). So $E = c_0(T)$.

    → Measure $x$ from the centre of the pore. Then $B_{00} = \langle x\rangle = 0$, and the diagonal of $\mathbf B$ vanishes for a symmetric pore. A shift of origin would only add a constant to the diagonal, a common phase that a refocused echo removes.
    """, title="From Bloch–Torrey to dc/dt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The PGSE signal with Grebenkov's MCF

    When the gradient is **constant** over an interval of length $\tau$, $d\mathbf c/dt = -(D\Lambda + i\gamma g\mathbf B)\mathbf c$ has constant coefficients and is solved by one matrix exponential: $\mathbf c(t+\tau) = e^{-(D\Lambda + i\gamma g\mathbf B)\tau}\,\mathbf c(t)$.

    A PGSE sequence is piecewise constant: a lobe of $+G$ for $\delta$, no gradient for $\Delta - \delta$, and a second lobe. The refocusing pulse flips the phase, which is the same as flipping the sign of the second lobe (the "effective gradient"). So the exact signal is a product of three matrices, sandwiched between the uniform mode:

    $$E = \Big[\,e^{-(D\Lambda - i\gamma G\mathbf B)\,\delta}\;e^{-D\Lambda(\Delta-\delta)}\;e^{-(D\Lambda + i\gamma G\mathbf B)\,\delta}\,\Big]_{00}.$$

    This is the **MCF** method in the form of [Grebenkov (2007)](https://doi.org/10.1103/RevModPhys.79.1077), building on [Barzykin (1999)](https://doi.org/10.1006/jmre.1999.1778). It is exact for any piecewise-constant waveform; the only approximation is the number of modes $N$ kept. Read the product from right to left: that is the order in time.

    Move the sliders: the same sliders drive every figure in this chapter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    L_a = mo.ui.slider(4, 12, step=1, value=8, label="Slab width L (µm)", show_value=True)
    G_a = mo.ui.slider(0, 600, step=10, value=250, label="Gradient G (mT/m)", show_value=True)
    N_a = mo.ui.slider(1, 40, step=1, value=6, label="Modes kept N", show_value=True)
    delta_a = mo.ui.slider(1, 20, step=1, value=10, label="δ (ms)", show_value=True)
    Delta_a = mo.ui.slider(5, 60, step=1, value=20, label="Δ (ms)", show_value=True)
    mo.vstack([mo.hstack([L_a, G_a, N_a], justify="start", gap=2), mo.hstack([delta_a, Delta_a], justify="start", gap=2)])
    return Delta_a, G_a, L_a, N_a, delta_a


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, Delta_a, G_a, GAMMA, L_a, N_a, delta_a, mo, np, pgse_mcf, plt):
    _L, _G, _N, _de = L_a.value, G_a.value, N_a.value, delta_a.value
    _De = max(Delta_a.value, _de)
    _Gs = np.linspace(0, 600, 61)
    _ref = np.array([pgse_mcf(g, _L, _de, _De, N=40).real for g in _Gs])
    _approx = np.array([pgse_mcf(g, _L, _de, _De, N=_N).real for g in _Gs])
    _free = np.exp(-((GAMMA * _Gs * _de) ** 2) * (_De - _de / 3) * D0)
    _E_ref = pgse_mcf(_G, _L, _de, _De, N=40).real
    _E_N = pgse_mcf(_G, _L, _de, _De, N=_N).real

    _fig, _a = plt.subplots(figsize=(6.8, 3.7))
    _a.semilogy(_Gs, np.abs(_ref), color=C_MAIN, lw=2, label="exact (N = 40)")
    _a.semilogy(_Gs, np.abs(_approx), color=C_ALT, ls="--", lw=1.5, label=f"N = {_N} modes")
    _a.semilogy(_Gs, _free, color=C_FREE, lw=1.2, label="free diffusion, exp(−bD)")
    _a.axvline(_G, color="k", lw=0.8, ls=":")
    _a.set(xlabel="G (mT/m)", ylabel="|E|", ylim=(1e-3, 1.1), title=f"PGSE in a slab, L = {_L} µm, δ = {_de} ms, Δ = {_De} ms")
    _a.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"At G = {_G} mT/m: exact **{_E_ref:.6f}**, with {_N} modes **{_E_N:.6f}** (error {abs(_E_N - _E_ref):.1e}).")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Show that $E = 1$ at $G = 0$, whatever $N$.

    **(b)** Set $N = 1$. What does the signal do as you raise $G$, and why? Then try $N = 2, 3, 4, 6$ at $G = 250$ mT/m.

    **(c)** Why is it enough to know the first row and column of each matrix product at the very start and end, while the full matrices are needed in between?

    **(d)** Make $\Delta$ much longer than $\tau_1 = L^2/(\pi^2 D)$. Which factor in the product changes, and what does it do to the modes $n \ge 1$?
    """, "1 · PGSE in three matrix exponentials")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ At $G = 0$ every factor is $e^{-D\Lambda\tau}$, a diagonal matrix with $e^{-D\lambda_0\tau} = 1$ in the top-left corner (since $\lambda_0 = 0$). So $E = [e^{-D\Lambda T}]_{00} = 1$. Diffusion alone never changes the total magnetization.

    **(b)** ⇒ With one mode, $\mathbf B = (B_{00}) = (0)$: the gradient has nothing to couple to, so $E = 1$ at every $G$. Dephasing needs at least one non-uniform mode to move magnetization into.

    ⇒ At $G = 250$ mT/m, $L = 8$ µm: the error is about $9\times10^{-3}$ for $N = 2$, $4\times10^{-4}$ for $N = 3$, $3\times10^{-5}$ for $N = 4$ and $6\times10^{-7}$ for $N = 6$. (From $N = 4$ to $5$ nothing changes: mode 4 has even parity and only enters through mode 3 or 5.)

    **(c)** ⇒ $\mathbf c(0) = \mathbf e_0$ picks out the first column of the rightmost matrix, and $E = c_0(T)$ picks out the first row of the leftmost. In between, the gradient moves magnetization into all modes and back, so all entries matter.

    **(d)** ⇒ Only the middle factor $e^{-D\Lambda(\Delta-\delta)}$ changes. For $\Delta - \delta \gg \tau_1$ it kills every mode except $n = 0$: it becomes the projector $\mathbf e_0\mathbf e_0^{\mathsf T}$. Then $E = [e^{-(D\Lambda - i\gamma G\mathbf B)\delta}]_{00}\,[e^{-(D\Lambda + i\gamma G\mathbf B)\delta}]_{00} = |[e^{-(D\Lambda + i\gamma G\mathbf B)\delta}]_{00}|^2$: the two lobes act independently. With narrow pulses this becomes the diffraction pattern of Chapter 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Any waveform: a staircase, or diffusion and kicks in turn

    Real gradients have **ramps**: the lobes are trapezoids, and oscillating or free waveforms are not piecewise constant at all. There are two ways to step through such a waveform with the same $\Lambda$ and $\mathbf B$.

    **1 · Staircase MCF.** Replace each ramp by $K$ short constant steps (at the midpoint value) and multiply their exact exponentials ([Barzykin 1999](https://doi.org/10.1006/jmre.1999.1778); Grebenkov 2007). Each step costs a full $N\times N$ matrix exponential, because $D\Lambda$ and $\mathbf B$ do not commute.

    **2 · Multiple propagator.** Cut the whole sequence into $M$ steps of length $\tau$ and, within each step, **separate** diffusion from dephasing: half a step of diffusion, a phase kick, half a step of diffusion,

    $$\mathbf c \;\leftarrow\; e^{-D\Lambda\tau/2}\;e^{-i\gamma g\,\tau\,\mathbf B}\;e^{-D\Lambda\tau/2}\;\mathbf c .$$

    Diffusion is diagonal, so $e^{-D\Lambda\tau/2}$ is just a list of numbers. The kick is diagonal in the eigenbasis of $\mathbf B$, which you diagonalise once. So a step costs only two matrix–vector products. This is the multiple (narrow-pulse) propagator of [Caprihan et al. (1996)](https://doi.org/10.1006/jmra.1996.0013) and [Callaghan (1997)](https://doi.org/10.1006/jmre.1997.1233), in the symmetric splitting of [Strang (1968)](https://doi.org/10.1137/0705041). Callaghan writes the kick in the position basis as the matrix $A_{nm}(q) = \int u_n e^{-iqx} u_m$, with $q = \gamma g\tau$: the same thing.

    Both methods are **second order**: halving the step divides the error by four. The figure checks this on a trapezoidal PGSE.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ramp_w = mo.ui.slider(0.5, 4.5, step=0.5, value=2.0, label="Ramp time r (ms)", show_value=True)
    K_w = mo.ui.slider(1, 16, step=1, value=2, label="Steps per ramp K (left panel)", show_value=True)
    mo.hstack([ramp_w, K_w], justify="start", gap=2)
    return K_w, ramp_w


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, G_a, GAMMA, K_w, L_a, delta_a, Delta_a, expm, mo, np, plt, ramp_w, slab_B):
    _L, _G, _de = L_a.value, G_a.value, delta_a.value
    _De = max(Delta_a.value, _de)
    _r = min(ramp_w.value, _de / 2)
    _N = 24
    _lam, _B = slab_B(_L, _N)
    _T = _De + _de

    def _g(t):
        def lobe(s):
            return np.clip(np.minimum(s / _r, (_de - s) / _r), 0, 1) * ((s >= 0) & (s <= _de))
        return _G * (lobe(t) - lobe(t - _De))

    def _edges(K):
        e = [0.0, _de, _De, _De + _de]
        for t0 in (0.0, _De):
            e += list(t0 + np.linspace(0, _r, K + 1)) + list(t0 + _de - _r + np.linspace(0, _r, K + 1))
        return np.unique(np.round(e, 10))

    def _staircase(K):
        c = np.zeros(_N, complex)
        c[0] = 1
        ed = _edges(K)
        for a, b in zip(ed[:-1], ed[1:]):
            c = expm(-(D0 * np.diag(_lam) + 1j * GAMMA * _g(0.5 * (a + b)) * _B) * (b - a)) @ c
        return c[0], len(ed) - 1

    _beta, _V = np.linalg.eigh(_B)

    def _strang(M):
        tau = _T / M
        h = np.exp(-D0 * _lam * tau / 2)
        c = np.zeros(_N, complex)
        c[0] = 1
        for j in range(M):
            c = h * c
            c = _V @ (np.exp(-1j * GAMMA * _g((j + 0.5) * tau) * tau * _beta) * (_V.T @ c))
            c = h * c
        return c[0]

    _ref = _strang(6000)
    _Ks = [1, 2, 4, 8, 16]
    _st = [_staircase(k) for k in _Ks]
    _err_st = [max(abs(e - _ref), 1e-15) for e, _ in _st]
    _pieces = [p for _, p in _st]
    _Ms = [25, 50, 100, 200, 400, 800]
    _err_mp = [max(abs(_strang(m) - _ref), 1e-15) for m in _Ms]

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    _tt = np.linspace(0, _T, 2000)
    _a1.plot(_tt, _g(_tt), color=C_MAIN, lw=2, label="trapezoidal waveform")
    _ed = _edges(K_w.value)
    _mid = 0.5 * (_ed[:-1] + _ed[1:])
    _a1.stairs(_g(_mid), _ed, color=C_ALT, lw=1.3, label=f"staircase, K = {K_w.value}")
    _a1.set(xlabel="t (ms)", ylabel="effective g (mT/m)", title="What the staircase MCF sees")
    _a1.legend(frameon=False, fontsize=8, loc="center right")

    _a2.loglog(_pieces, _err_st, "o-", color=C_ALT, label="staircase MCF (x: constant pieces)")
    _a2.loglog(_Ms, _err_mp, "s-", color=C_MAIN, label="multiple propagator (x: time steps)")
    _x = np.array([10, 1000])
    _a2.loglog(_x, _err_mp[2] * (_x / _Ms[2]) ** -2.0, ":", color=C_FREE, label="∝ 1/steps²")
    _a2.set(xlabel="number of steps", ylabel="|E − E_ref|", title="Both are second order")
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Reference: multiple propagator with 6000 steps, E = **{_ref.real:.6f}**. Uses L, G, δ, Δ from the PGSE sliders; N = 24 modes. With r → 0 the staircase is exact with only 3 pieces.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** The multiple propagator would be exact if $D\Lambda$ and $\mathbf B$ commuted. Use $e^{X}e^{Y} = e^{X + Y + \frac12[X,Y] + \dots}$ to show that the symmetric (Strang) split has an error of order $\tau^3$ per step, so $\tau^2$ over the whole sequence. What would the plain split $e^{-D\Lambda\tau}e^{-i\gamma g\tau\mathbf B}$ give?

    **(b)** Set the ramp time to its smallest value. Why does the staircase MCF then need hardly any steps?

    **(c)** Per step, the staircase MCF costs a matrix exponential and the multiple propagator two matrix–vector products. For $N$ modes, how do the costs scale? When would you still choose the staircase?
    """, "2 · Two ways through a trapezoid")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - **(a)** Write $X = -D\Lambda\tau/2$ and $Y = -i\gamma g\tau\mathbf B$. Each is of order $\tau$. In the symmetric product $e^Xe^Ye^X$, check which commutator terms cancel.
    - **(c)** A dense $N\times N$ matrix product costs about $N^3$ operations; a matrix–vector product about $N^2$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ The exact step is $e^{2X + Y}$. The plain split gives $e^{2X}e^Y = e^{2X + Y + [X, Y] + O(\tau^3)}$: an error $[X, Y] \sim \tau^2$ per step, so $\tau$ over the $T/\tau$ steps. First order.

    ⇒ The symmetric product $e^Xe^Ye^X$ is its own reverse, so its expansion contains no even-order terms: the $\frac12[X,Y]$ from the left pair cancels the $\frac12[Y,X]$ from the right pair. The first error is a double commutator, of order $\tau^3$ per step, so $\tau^2$ over the sequence. Second order, as the figure shows.

    → The commutator is nonzero because $\mathbf B$ moves magnetization between modes that decay at different rates. Physically: the split pretends spins first diffuse, then dephase, while in reality they do both at once.

    **(b)** ⇒ Without ramps the PGSE is piecewise constant, and the MCF is exact on each piece: 3 pieces (lobe, gap, lobe) are enough. The staircase error comes only from the ramps, and shrinks with them.

    **(c)** ⇒ Staircase: about $N^3$ per piece (the exponential). Multiple propagator: about $N^2$ per step, after one diagonalisation of $\mathbf B$.

    → Choose the staircase when the waveform is mostly constant (PGSE with short ramps, or a few long constant pieces). There a handful of exact pieces beats thousands of small steps. For oscillating or free waveforms the multiple propagator is usually cheaper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many modes? Smoothness, and Jackson's theorem

    How large must $N$ be? Approximation theory gives the right way to think about it.

    **The idea.** The slab's modes are cosines. A sum of $N$ of them is a truncated Fourier series of the magnetization **mirrored at both walls** (its even, periodic extension). **Jackson's theorems** (Jackson 1911) bound how well $N$ terms can approximate a function by how smooth it is: if the mirrored function has $k$ continuous derivatives, the best error falls at least as fast as $N^{-k}$ (times a measure of how smoothly the $k$-th derivative varies). Bernstein's theorems (Bernstein 1912) give the converse, and for analytic functions the convergence is exponential. In short: **smoothness of the mirrored function sets the speed of convergence.**

    Mirroring is what makes the wall behaviour matter. A function with a nonzero slope at a wall gets a kink when mirrored, even if it is perfectly smooth inside. The left panel shows the cosine coefficients of four functions:

    - a **step** (a jump): coefficients fall as $1/n$;
    - the gradient field **$x$** itself: slope 1 at the walls, so the mirror has kinks, and the coefficients fall as $1/n^2$. These are the couplings $B_{0n}$ of Chapter 1, Exercise 5;
    - a **smooth step with zero slope at the walls**: the first jump is in the third derivative of the mirror, so $1/n^4$;
    - the **propagator** at $t > 0$: diffusion has smoothed it completely, and the coefficients $e^{-D\lambda_n t}u_n(x_0)$ fall faster than any power.

    **And the Bloch–Torrey equation?** Diffusion smooths, but the gradient keeps feeding in the field $x$, which does not fit the walls. Differentiate the equation once in $x$ and evaluate it at a wall, where $\partial m/\partial x = 0$ at all times: you find $\partial^3 m/\partial x^3 = i\gamma g\, m/D \ne 0$. So the mirrored magnetization has a jump in its third derivative, and the convergence in $N$ is algebraic, not exponential. The right panel shows it for the PGSE above: the error in $E$ falls roughly as $N^{-5}$. That is steep enough that 10–20 modes give 6–8 correct digits at moderate gradients.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, Delta_a, G_a, GAMMA, L_a, delta_a, mo, np, pgse_mcf, plt):
    _n = np.arange(1, 201)
    _k = _n * np.pi
    # cosine coefficients on [0, 1], in closed form (integrate by parts)
    _c_step = np.abs(np.sqrt(2) * np.sin(_k / 2) / _k)                  # step at x = 1/2
    _c_x = np.abs(np.sqrt(2) * ((-1.0) ** _n - 1) / _k**2)               # x
    _c_smooth = np.abs(np.sqrt(2) * 12 * ((-1.0) ** _n - 1) / _k**4)     # 3x² − 2x³
    _c_prop = np.sqrt(2) * np.exp(-(_n * np.pi) ** 2 * 0.002)            # envelope of e^{-Dλt} u_n(x0)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    _odd = _n % 2 == 1
    _a1.loglog(_n[_odd], _c_step[_odd], "o", ms=3, color=C_FREE, label="step: ∝ 1/n")
    _a1.loglog(_n[_odd], _c_x[_odd], "o", ms=3, color=C_MAIN, label="x (the gradient field): ∝ 1/n²")
    _a1.loglog(_n[_odd], _c_smooth[_odd], "o", ms=3, color=C_ALT, label="smooth step, zero slope at walls: ∝ 1/n⁴")
    _keep = _c_prop > 1e-14
    _a1.loglog(_n[_keep], _c_prop[_keep], "-", color="k", lw=1.2, label="propagator at t > 0 (envelope)")
    _a1.set(xlabel="mode number n", ylabel="|cosine coefficient|", ylim=(1e-12, 2), title="Smoothness sets the decay (odd n shown)")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower left")

    _L, _G, _de = L_a.value, G_a.value, delta_a.value
    _De = max(Delta_a.value, _de)
    _Ns = np.array([4, 6, 8, 12, 16, 24, 32, 48])
    _ref = pgse_mcf(_G, _L, _de, _De, N=96)
    _err = np.maximum([abs(pgse_mcf(_G, _L, _de, _De, N=int(k)) - _ref) for k in _Ns], 1e-16)
    _a2.loglog(_Ns, _err, "o-", color=C_MAIN, label=f"L = {_L} µm, G = {_G} mT/m")
    _a2.loglog(_Ns, _err[3] * (_Ns / _Ns[3]) ** -5.0, "--", color=C_FREE, label="guide ∝ N⁻⁵")
    _a2.set(xlabel="number of modes N", ylabel="|E(N) − E(96)|", title="PGSE signal error against N", xticks=[4, 8, 16, 32, 48], xticklabels=["4", "8", "16", "32", "48"])
    _a2.minorticks_off()
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()

    _ellg = (D0 / (GAMMA * max(_G, 1e-9))) ** (1 / 3)
    mo.vstack([_fig, mo.md(f"Uses the PGSE sliders. Gradient length ℓ_g = (D/γG)^(1/3) = **{_ellg:.2f} µm**; L/(π ℓ_g) = {_L / (np.pi * _ellg):.1f}.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practical rules for choosing $N$**

    1. **Resolve the finest length scale in $m$.** Mode $n$ has a wavelength of about $2L/n$, so you need $N \gtrsim$ a few $\times L/(\pi\ell)$, where $\ell$ is the smallest of $\sqrt{D\,t}$ for the shortest relevant time and the gradient length $\ell_g = (D/\gamma g)^{1/3}$.
    2. **Damping must beat dephasing in the last mode kept** (Grebenkov's criterion, [Grebenkov 2007](https://doi.org/10.1103/RevModPhys.79.1077), Sec. VI.A): $D\lambda_N \gg \gamma g L$.
    3. **Always check:** double $N$ and see whether $E$ changes in the digits you care about.

    In 2D and 3D the number of modes below a given $\lambda$ grows as $\lambda^{d/2}$ ([Weyl's law](https://doi.org/10.1007/BF01456804), 1912), so the same resolution costs about $(L/\ell)^d$ modes. A linear gradient couples only modes whose angular orders differ by one, which keeps cylinders and spheres cheap.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** In Chapter 1, Exercise 5 you found that $B_{0n} \propto 1/n^2$. Explain the $1/n^2$ with the mirrored function. Would measuring $x$ from a wall instead of the centre change anything?

    **(b)** Use rule 2 for $L = 8$ µm, $D = 2$ µm²/ms and $G = 300$ mT/m: from which $N$ on is the last mode damped more strongly than it is dephased? Compare with the right-hand panel.

    **(c)** Raise $G$ to 600 mT/m. What happens to the error at fixed $N$, and to $\ell_g$? Which regime of the Chapter 1 length-scale table are you moving towards?
    """, "3 · Why the couplings fall as 1/n²")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Mirroring $x$ on $[0, L]$ at both walls gives a triangle wave. It is continuous but has kinks (jumps in its first derivative) at $x = 0$ and $L$. One derivative is fine, the next jumps, so the coefficients fall as $1/n^2$.

    ⇒ Measuring from a wall adds the constant $L/2$, which only changes the $n = 0$ coefficient (and puts $L/2$ on the diagonal of $\mathbf B$, a common phase). The slope at the walls is still 1, so the $1/n^2$ stays.

    **(b)** ⇒ $\gamma g L = 2.6752\times10^{-4}\times300\times8 = 0.64$ /ms. Mode $N$ decays at $D(N\pi/L)^2 = 0.31\,N^2$ /ms.

    ⇒ Damping exceeds dephasing from $N \approx 1.5$ and is ten times larger from $N \approx 5$. The panel agrees: with 6 modes the error is already of order $10^{-6}$.

    **(c)** ⇒ The error at fixed $N$ grows, and $\ell_g$ shrinks to 2.3 µm. At $G = 300$ mT/m it is 2.9 µm, about $L/3$; as $\ell_g$ falls further below $L$ you approach the **localisation** regime. There $m$ develops thin layers of thickness $\sim\ell_g$ at the walls, and you need $N \gtrsim L/\ell_g$ modes to resolve them. Chapter 3 maps that regime.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementations

    | Toolbox | Method | Geometry | Notes |
    | --- | --- | --- | --- |
    | **MCFAL** (Grebenkov) | MCF, piecewise-constant gradients | slab, cylinder, sphere; multilayer structures | the reference implementation of Grebenkov (2007) |
    | **MISST** (Drobnjak, Ianuş et al.) | matrix method in the style of Callaghan, arbitrary waveforms | standard compartments | built for waveform optimisation |
    | **SpinDoctor** matrix formalism | eigenmodes computed by finite elements on a mesh, then the same $d\mathbf c/dt$ | realistic cells, with permeable membranes | the modes come from Chapter 4's mesh |
    | **ReMiDi** | differentiable matrix formalism on a mesh | realistic cells | gradients of the signal with respect to the geometry, for fitting |
    | Cylindrical-surface code ([Canales-Rodríguez, Tax et al. 2026](https://doi.org/10.1088/1361-6560/aeb2a0)) | exact PGSE on a cylindrical surface | concentric layers (myelin) | $\lambda_n = n^2/r^2$ from Chapter 1, closed-form $\mathbf B$ |

    The matrix method is only as general as the modes: closed-form modes for simple shapes, a mesh (Chapter 4) for anything else. Once the modes are known, the time stepping is the same everywhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 2b · The cumulant expansion

    ## Expanding the logarithm of the signal

    The path picture gives $E = \langle e^{-i\varphi}\rangle$: the **characteristic function** of the phase. As for the displacement in Chapter 1, its logarithm is a series of cumulants:

    $$\ln E = \sum_{n\ge1}\frac{(-i)^n}{n!}\,\kappa_n(\varphi) = -\frac12\langle\varphi^2\rangle + \frac1{24}\Big(\langle\varphi^4\rangle - 3\langle\varphi^2\rangle^2\Big) - \dots$$

    The odd cumulants vanish for a refocused echo in a symmetric pore (the phase is as likely to be $+\varphi$ as $-\varphi$). Since $\varphi \propto g$, this is a series in **powers of the gradient**: $\kappa_2 \propto g^2$, $\kappa_4 \propto g^4$, and so on.

    **The Gaussian phase approximation (GPA)** keeps only the first term: $E \approx e^{-\langle\varphi^2\rangle/2}$. It is exact when the phase is Gaussian (free diffusion, Exercise 4), and good whenever the higher cumulants are small compared with $\langle\varphi^2\rangle$.

    ## The second cumulant from the modes

    $\varphi = \gamma\int g(t)\,x(t)\,dt$ is linear in the path, so its variance only needs the **position autocorrelation**:

    $$\langle\varphi^2\rangle = \gamma^2\iint g(t_1)\,g(t_2)\,\langle x(t_1)\,x(t_2)\rangle\,dt_1\,dt_2 .$$

    In a pore, with spins starting uniformly and $x$ measured from the centre, the propagator of Chapter 1 gives

    $$\langle x(t_1)\,x(t_2)\rangle = \sum_{n\ge1} B_{0n}^2\; e^{-D\lambda_n|t_1-t_2|}.$$

    So the GPA needs only two lists per geometry: the weights $w_n = B_{0n}^2$ and the rates $D\lambda_n$. For the slab they are the MSD weights $8L^2/(n\pi)^4$ and $D(n\pi/L)^2$. Each mode contributes a double integral of the waveform against one exponential, which you can do once and for all for a PGSE (Murday & Cotts 1968; Neuman 1974; for any waveform, Stepišnik 1993 and van Gelderen et al. 1994).
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ For a uniform start in a pore of length $L$, $\langle f(x(t_1))\,h(x(t_2))\rangle = \frac1L\iint f(x_0)\,h(x)\,P(x\,|\,x_0,|t_2 - t_1|)\,dx\,dx_0$.

    ⇒ Insert the mode sum of the propagator: $\langle x(t_1)x(t_2)\rangle = \frac1L\sum_n e^{-D\lambda_n|t_1-t_2|}\Big(\int x\,u_n\,dx\Big)^2$.

    ⇒ With $u_0 = 1/\sqrt L$: $\frac1{\sqrt L}\int x\,u_n\,dx = \int u_0\,x\,u_n\,dx = B_{0n}$, so each term is $B_{0n}^2e^{-D\lambda_n|t_1-t_2|}$. The $n = 0$ term is $B_{00}^2 = \langle x\rangle^2 = 0$ for a centred $x$.

    → At $t_1 = t_2$: $\sum_n B_{0n}^2 = \langle x^2\rangle = L^2/12$ for the slab. Check: $\sum_{n\ \text{odd}} 8L^2/(n\pi)^4 = 8L^2/96 = L^2/12$. ✓
    """, title="The position autocorrelation from the modes")
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** For **free** diffusion, show that the phase is Gaussian, and that $\langle\varphi^2\rangle = 2bD$ with $b = \int_0^T q(t)^2\,dt$ and $q(t) = \gamma\int_0^t g(t')\,dt'$. Conclude that $E = e^{-bD}$.

    **(b)** For a PGSE (effective gradient $+G$ on $[0,\delta]$, $-G$ on $[\Delta,\Delta+\delta]$) and one exponential $e^{-a|t_1-t_2|}$, show that

    $$\iint g(t_1)g(t_2)\,e^{-a|t_1-t_2|}\,dt_1dt_2 = \frac{G^2}{a^2}\Big[4\big(a\delta - 1 + e^{-a\delta}\big) - 4e^{-a\Delta}\big(\cosh a\delta - 1\big)\Big].$$

    **(c)** Check two limits for the slab, using $w_n = 8L^2/(n\pi)^4$, $\sum_n w_n\lambda_n = 1$ and $\sum_{n\ \text{odd}} n^{-6} = \pi^6/960$.

    - Large pore (all $a\delta, a\Delta \ll 1$): you should recover free diffusion, $\langle\varphi^2\rangle = 2bD$ with $b = \gamma^2G^2\delta^2(\Delta - \delta/3)$.
    - Small pore (all $a\delta \gg 1$): you should recover motional narrowing, $\ln E = -\gamma^2G^2L^4(2\delta)/(120D)$, the formula in the Chapter 1 length-scale table with $T = 2\delta$ the time the gradient is on.
    """, "4 · The Gaussian phase approximation for a slab")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - **(a)** Integrate $\varphi = \gamma\int_0^T g\,x\,dt$ by parts, using $q(T) = 0$ (echo condition): $\varphi = -\int_0^T q(t)\,dx(t)$. The increments $dx$ of free diffusion are independent Gaussians with variance $2D\,dt$.
    - **(b)** Split into the two same-lobe terms and the two cross terms. Same lobe: $\int_0^\delta\!\int_0^\delta e^{-a|t_1-t_2|} = 2(a\delta - 1 + e^{-a\delta})/a^2$. Cross term: the integrals factorise, $\int_0^\delta e^{at_1}dt_1\int_\Delta^{\Delta+\delta}e^{-at_2}dt_2$.
    - **(c)** Expand in $a$ to first order for the large pore; keep only the term linear in $\delta$ for the small pore.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\varphi = \gamma\int_0^T g(t)\,x(t)\,dt = \big[q\,x\big]_0^T - \int_0^T q(t)\,dx(t) = -\int_0^T q(t)\,dx(t)$, since $q(0) = q(T) = 0$.

    ⇒ The increments $dx$ are independent and Gaussian, so $\varphi$ is a weighted sum of independent Gaussians: Gaussian itself, with mean zero.

    ⇒ Variances of independent terms add: $\langle\varphi^2\rangle = \int_0^T q(t)^2\cdot 2D\,dt = 2bD$, and $E = e^{-\langle\varphi^2\rangle/2} = e^{-bD}$. ✓ For a PGSE, $b = \gamma^2G^2\delta^2(\Delta - \delta/3)$ ([Stejskal & Tanner 1965](https://doi.org/10.1063/1.1695690)).

    **(b)** ⇒ Same lobe, twice: $2G^2\cdot\frac{2(a\delta - 1 + e^{-a\delta})}{a^2}$.

    ⇒ Cross terms, twice, with $g_1g_2 = -G^2$: $\int_0^\delta e^{at_1}dt_1 = \frac{e^{a\delta}-1}{a}$ and $\int_\Delta^{\Delta+\delta}e^{-at_2}dt_2 = e^{-a\Delta}\frac{1 - e^{-a\delta}}{a}$. Their product is $e^{-a\Delta}\frac{2\cosh a\delta - 2}{a^2}$, so the cross terms give $-\frac{4G^2}{a^2}e^{-a\Delta}(\cosh a\delta - 1)$.

    ⇒ Adding gives the formula. ✓

    **(c)** ⇒ **Large pore.** To first order in $a$: $4(a\delta - 1 + e^{-a\delta}) \approx 2a^2\delta^2 - \frac23a^3\delta^3$ and $4e^{-a\Delta}(\cosh a\delta - 1) \approx 2a^2\delta^2 - 2a^3\Delta\delta^2$. The bracket is $2a^3\delta^2(\Delta - \delta/3)$, so each mode gives $2aG^2\delta^2(\Delta - \delta/3)$.

    ⇒ Summed: $\langle\varphi^2\rangle = 2\gamma^2G^2\delta^2(\Delta - \delta/3)\,D\sum_n w_n\lambda_n = 2bD$, because $\sum_{n\ \text{odd}}\frac{8L^2}{(n\pi)^4}\frac{(n\pi)^2}{L^2} = \frac{8}{\pi^2}\cdot\frac{\pi^2}{8} = 1$. ✓ Before the walls are felt, the pore looks infinite.

    ⇒ **Small pore.** For $a\delta \gg 1$ the bracket is $\approx 4a\delta$, so each mode gives $4G^2\delta/a$ and $\langle\varphi^2\rangle = \frac{4\gamma^2G^2\delta}{D}\sum_n\frac{w_n}{\lambda_n} = \frac{4\gamma^2G^2\delta}{D}\cdot 8L^4\sum_{n\ \text{odd}}\frac{1}{(n\pi)^6} = \frac{4\gamma^2G^2\delta L^4}{D}\cdot\frac{8}{960} = \frac{\gamma^2G^2\delta L^4}{30D}$.

    ⇒ $\ln E = -\langle\varphi^2\rangle/2 = -\gamma^2G^2L^4(2\delta)/(120D)$. ✓ Motional narrowing ([Robertson 1966](https://doi.org/10.1103/PhysRev.151.273)): the signal no longer depends on $\Delta$, only on how long the gradient is on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See it: the GPA against the exact signal

    The exact curve is the MCF of 2a with 40 modes; the GPA uses the closed formula of Exercise 4 summed over 400 modes. Both use the PGSE sliders above.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, Delta_a, G_a, GAMMA, L_a, delta_a, gpa_phase_var, mo, np, pgse_mcf, plt):
    _L, _G, _de = L_a.value, G_a.value, delta_a.value
    _De = max(Delta_a.value, _de)
    _Gs = np.linspace(5, 600, 120)
    _ex = np.array([pgse_mcf(g, _L, _de, _De, N=40).real for g in _Gs])
    _gpa = np.array([np.exp(-gpa_phase_var(g, _L, _de, _De) / 2) for g in _Gs])
    _free = np.exp(-((GAMMA * _Gs * _de) ** 2) * (_De - _de / 3) * D0)
    _rel = (_gpa - _ex) / np.abs(_ex)
    _i5 = np.argmax(np.abs(_rel) > 0.05) if np.any(np.abs(_rel) > 0.05) else None

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    _a1.semilogy(_Gs, np.abs(_ex), color=C_MAIN, lw=2, label="exact (MCF)")
    _a1.semilogy(_Gs, _gpa, color=C_ALT, ls="--", lw=1.5, label="Gaussian phase")
    _a1.semilogy(_Gs, _free, color=C_FREE, lw=1.2, label="free diffusion")
    _a1.axvline(_G, color="k", lw=0.8, ls=":")
    _a1.set(xlabel="G (mT/m)", ylabel="|E|", ylim=(1e-3, 1.1), title="Signal against gradient")
    _a1.legend(frameon=False, fontsize=8, loc="lower left")
    _a2.plot(_Gs, 100 * _rel, color=C_ALT, lw=1.8)
    _a2.axhline(5, color=C_FREE, ls=":", lw=1)
    _a2.set(xlabel="G (mT/m)", ylabel="(GPA − exact) / exact  (%)", ylim=(-20, 100), title="Error of the Gaussian phase approximation")
    _fig.tight_layout()
    _at = np.exp(-gpa_phase_var(_G, _L, _de, _De) / 2)
    _exG = pgse_mcf(_G, _L, _de, _De, N=40).real
    _txt = f"At G = {_G} mT/m: exact **{_exG:.4f}**, GPA **{_at:.4f}** ({100 * (_at - _exG) / max(abs(_exG), 1e-12):+.1f}%). "
    _txt += f"The GPA is 5% off from **{_Gs[_i5]:.0f} mT/m**." if _i5 is not None else "The GPA stays within 5% over this range."
    mo.vstack([_fig, mo.md(_txt)])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    With the defaults ($L = 8$ µm, $\delta = 10$ ms, $\Delta = 20$ ms):

    **(a)** At which gradient is the GPA 5% off? Why does it work so well at low $G$?

    **(b)** The exact curve dips to $|E| \approx 0.001$ near $G \approx 464$ mT/m and comes back up; the GPA cannot. What would a Gaussian phase need to do to produce a dip?

    **(c)** For a finite number of modes, $E(G)$ is an **entire** function of $G$: a finite sum of exponentials, finite everywhere in the complex plane. It has a zero at $G^* \approx 463 + 20i$ mT/m. What does that imply for the cumulant series of $\ln E$ in powers of $G$?

    **(d)** Make the slab smaller ($L = 4$ µm). Where does the GPA fail now, and why?
    """, "5 · When is the phase Gaussian?")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ About 205–210 mT/m (1.3% at 150 mT/m, about 30% at 300 mT/m).

    → At low $G$ the signal is close to 1, so $\ln E \approx -\langle\varphi^2\rangle/2$ plus a fourth-cumulant term of order $G^4$ that is still small. The non-Gaussian shape of the phase only matters once its spread approaches about one radian.

    **(b)** ⇒ A Gaussian phase gives $e^{-\langle\varphi^2\rangle/2}$: positive and monotonic in $G$. A dip and a recovery mean that the phases of different spins interfere and nearly cancel at one $G$, then cancel less at a higher $G$. That needs a phase distribution with structure, here set by the walls: the restricted spins cannot spread their phase indefinitely.

    **(c)** ⇒ $\ln E$ is singular where $E = 0$. A Taylor series converges inside the largest disc around $G = 0$ that contains no singularity, so the cumulant series converges only for $|G| < |G^*| \approx 464$ mT/m (assuming no zero lies closer).

    → The dip on the real axis is the shadow of that nearby complex zero. Beyond $|G^*|$ no number of cumulants describes the signal. Below it, the series converges, but slowly near the edge: the GPA (one term) is already 5% off at less than half of $|G^*|$.

    → Chapter 3 shows that for narrow pulses at long times the zero lies on the real axis, at $qL = 2\pi$ for a slab.

    **(d)** ⇒ With $L = 4$ µm the GPA stays within 5% over the whole range. The pore is much smaller than $\ell_g$ (2.3–3.4 µm here), so spins sample the whole pore many times while the gradient is on: motional narrowing, where the phase is a sum of many nearly independent pieces and therefore close to Gaussian (central limit theorem).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The fourth cumulant: kurtosis, and why finite pulses matter

    Diffusion kurtosis imaging ([Jensen et al. 2005](https://doi.org/10.1002/mrm.20508)) fits the first two terms in powers of $b$:

    $$\ln E = -b\,D_{\text{app}} + \frac16\,b^2D_{\text{app}}^2K_{\text{app}} + O(b^3).$$

    **With narrow pulses** ($\delta \to 0$), the phase is $-q$ times the displacement $X$ (Chapter 3 derives this), and $b = q^2\Delta$. Then $D_{\text{app}}$ and $K_{\text{app}}$ are exactly the displacement cumulants of Chapter 1: $D(\Delta)$ and $K(\Delta)$.

    **With finite pulses** they are not. During a pulse the spin keeps moving, so the phase records a **time-averaged position** during each lobe, not the end points. The signal becomes the Fourier transform of a "centre-of-mass" propagator, and isolated pores look smaller than they are ([Mitra & Halperin 1995](https://doi.org/10.1006/jmra.1995.1060)). The measured cumulants are then time integrals of the multi-time correlation functions $\langle x(t_1)x(t_2)\rangle$, $\langle x(t_1)x(t_2)x(t_3)x(t_4)\rangle$, … weighted by the waveform.

    The figure extracts $D_{\text{app}}$ and $K_{\text{app}}$ from the exact signal (MCF, a polynomial fit of $\ln E$ in $b$ at small $b$) and compares them with the displacement values of Chapter 1.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, GAMMA, L_a, mo, np, pgse_mcf, plt, slab_cumulants):
    _L = L_a.value

    def _fit(de, De, npts=7, bDmax=0.3):
        bmax = bDmax / D0
        Gmax = np.sqrt(bmax / (De - de / 3)) / (GAMMA * de)
        Gs = np.linspace(Gmax / npts, Gmax, npts)
        y = np.array([np.log(pgse_mcf(g, _L, de, De, N=20).real) for g in Gs])
        b = (GAMMA * Gs * de) ** 2 * (De - de / 3)
        c = np.polyfit(b, y, 3)[::-1]
        Dapp = -c[1]
        return Dapp / D0, 6 * c[2] / Dapp**2

    _Dg = np.logspace(0, 2, 300)
    _dd, _KK = slab_cumulants(D0 * _Dg / _L**2)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    _a1.semilogx(_Dg, _dd, color="k", lw=1.2, label="displacement (Ch 1)")
    _a2.semilogx(_Dg, _KK, color="k", lw=1.2, label="displacement (Ch 1)")
    for _de, _col, _ls in [(0.05, C_MAIN, "o"), (2.0, C_FREE, "s"), (10.0, C_ALT, "^")]:
        _Ds = np.logspace(np.log10(max(_de, 1.0)), 2, 9)
        _res = np.array([_fit(_de, d) for d in _Ds])
        _a1.semilogx(_Ds, _res[:, 0], _ls, color=_col, ms=4, label=f"δ = {_de:g} ms")
        _a2.semilogx(_Ds, _res[:, 1], _ls, color=_col, ms=4, label=f"δ = {_de:g} ms")
    _a1.set(xlabel="Δ (ms)", ylabel="D_app / D", title="Apparent diffusivity")
    _a2.axhline(-0.6, color=C_FREE, lw=0.8, ls=":")
    _a2.set(xlabel="Δ (ms)", ylabel="K_app", ylim=(-0.75, 0.3), title="Apparent kurtosis")
    _a1.legend(frameon=False, fontsize=8)
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Slab of width L = {_L} µm (from the PGSE sliders), τ₁ = {_L**2 / (np.pi**2 * D0):.1f} ms. Dotted line: K = −3/5.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Which set of points follows the black curves, and why?

    **(b)** For $\delta = 10$ ms, compare $D_{\text{app}}$ with the displacement value at long $\Delta$. Is the apparent pore larger or smaller than the real one? Explain with the time-averaged position during a lobe.

    **(c)** Two groups report the kurtosis of the same tissue at the same $\Delta$, one with $\delta = 2$ ms, the other with $\delta = 10$ ms. Should they agree?
    """, "6 · Kurtosis with finite pulses")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ The $\delta = 0.05$ ms points. With pulses much shorter than $\tau_1$ (and than $\Delta$), the spins barely move during a lobe, the phase is $-qX$, and the signal cumulants are the displacement cumulants (Chapter 1, and Chapter 3 for the narrow-pulse limit). Even then the agreement is only as good as $\delta/\tau_1$ is small: the differences shrink linearly with $\delta$.

    **(b)** ⇒ $D_{\text{app}}$ is below the displacement value: the pore looks **smaller**. During a 10 ms lobe in an 8 µm slab ($\tau_1 \approx 3$ ms) each spin crosses the pore several times, so its time-averaged position sits closer to the centre than its instantaneous position. The phase measures that average, and the distribution of averages is narrower than the pore. This is the centre-of-mass effect of Mitra & Halperin (1995); Codd & Callaghan (1999) showed that long pulses therefore make pores look smaller if one assumes narrow pulses.

    **(c)** ⇒ Not in general. $K_{\text{app}}$ depends on $\delta$ as well as $\Delta$, because with finite pulses it is no longer a property of the displacement alone but of the displacement and the waveform. Comparisons need the same timing, or a model (like the matrix method) that accounts for it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same in frequency: the diffusion spectrum

    Write the variance of the phase in the frequency domain. With $Q(\omega) = \int q(t)\,e^{i\omega t}dt$,

    $$\langle\varphi^2\rangle = \frac2\pi\int_0^\infty |Q(\omega)|^2\,\mathcal D(\omega)\,d\omega ,$$

    where $\mathcal D(\omega)$ is the **diffusion spectrum**, the Fourier transform of the velocity autocorrelation (Stepišnik 1993). The gradient waveform acts as a filter $|Q(\omega)|^2$ that picks which frequencies of the motion are seen. For free diffusion $\mathcal D(\omega) = D$ at every frequency and Parseval gives back $2bD$. For a pore, the modes give

    $$\mathcal D(\omega) = \sum_{n\ge1} w_n\,D\lambda_n\,\frac{\omega^2}{(D\lambda_n)^2 + \omega^2},$$

    which rises from 0 (no long-range motion in a closed pore) to $D$ (at high frequency the walls are not felt). **Oscillating gradients** (OGSE) have $|Q|^2$ peaked at their frequency, so they sample $\mathcal D(\omega)$ there: high frequencies probe short times.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, L_a, mo, np, plt, slab_weights):
    _f = np.logspace(0, 4, 400)            # Hz
    _om = 2 * np.pi * _f / 1000            # rad/ms
    _fig, _a = plt.subplots(figsize=(6.8, 3.6))
    for _L, _col in [(4, C_FREE), (L_a.value, C_MAIN), (12, C_ALT)]:
        _w, _lam = slab_weights(_L)
        _aa = D0 * _lam
        _Dw = np.array([np.sum(_w * _aa * o**2 / (_aa**2 + o**2)) for o in _om])
        _a.semilogx(_f, _Dw / D0, color=_col, lw=2.2 if _L == L_a.value else 1.3, label=f"L = {_L} µm")
        _a.axvline(1000 * D0 * (np.pi / _L) ** 2 / (2 * np.pi), color=_col, lw=0.7, ls=":")
    _a.set(xlabel="frequency f = ω/2π (Hz)", ylabel="D(ω) / D (spectrum)", ylim=(0, 1.05), title="Diffusion spectrum of a slab")
    _a.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Dotted lines: the rate of the first mode, Dλ₁/2π. The spectrum rises around it.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 2c · The link: cumulants are matrix products

    The two halves of this chapter look different, but they are built from the same $\Lambda$ and $\mathbf B$. Expand the exact solution of 2a in powers of the gradient. Treating $i\gamma g\mathbf B$ as a perturbation of $D\Lambda$ (variation of constants, applied repeatedly):

    $$c_0(T) = 1 \;-\; i\gamma\int_0^T g(t)\,B_{00}\,dt \;-\; \gamma^2\int_0^T\!dt_2\int_0^{t_2}\!dt_1\;g(t_2)\,g(t_1)\,\big[\mathbf B\,e^{-D\Lambda(t_2-t_1)}\mathbf B\big]_{00} \;+\;\dots$$

    The $n$-th term holds $n$ factors of $\mathbf B$ separated by diffusion propagators. Each such sandwich is a **multi-time correlation function** of the position,

    $$\big[\mathbf B\,e^{-D\Lambda(t_2-t_1)}\mathbf B\big]_{00} = \langle x(t_1)\,x(t_2)\rangle,\qquad \big[\mathbf B e^{-D\Lambda\tau_3}\mathbf B e^{-D\Lambda\tau_2}\mathbf B e^{-D\Lambda\tau_1}\mathbf B\big]_{00} = \langle x(t_1)\,x(t_2)\,x(t_3)\,x(t_4)\rangle,\ \dots$$

    which is where the name of the method comes from: **MCF = multiple correlation function** (Grebenkov 2007). The time-ordered exponential of Axelrod & Sen (2001) is the same expansion written compactly.

    - The **moments** of the phase are these terms. Taking the logarithm turns them into **cumulants**: the second-order term is $-\langle\varphi^2\rangle/2$, the GPA.
    - The matrix methods of 2a sum the whole series at once, by computing the exponentials directly. That is why they work at any $G$.
    - The cumulant series of 2b truncates the logarithm. That is why it has a radius of convergence (Exercise 5c), while the series for $E$ itself does not (for a finite number of modes).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Why does the first-order term vanish?

    **(b)** Show that the second-order term equals $-\langle\varphi^2\rangle/2$ with $\langle\varphi^2\rangle$ from 2b. Use $\mathbf e_0^{\mathsf T}e^{-D\Lambda s} = \mathbf e_0^{\mathsf T}$ and $e^{-D\Lambda s}\mathbf e_0 = \mathbf e_0$.

    **(c)** For a finite number of modes, the series for $E = c_0(T)$ converges for every $G$, but the cumulant series for $\ln E$ does not. Explain the difference in one sentence. Which would you use to compute the signal at $G = 600$ mT/m in an 8 µm slab, and which to write down a model you can fit?
    """, "7 · The second-order term is the GPA")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - **(b)** Write $[\mathbf Be^{-D\Lambda\tau}\mathbf B]_{00} = \sum_n B_{0n}\,e^{-D\lambda_n\tau}\,B_{n0}$ and compare with the position autocorrelation in 2b. The integral over $t_1 < t_2$ is half of the integral over the whole square, because the integrand is symmetric.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ It is $-i\gamma B_{00}\int g\,dt$. With $x$ measured from the centre, $B_{00} = \langle x\rangle = 0$. (Even with another origin, $\int g\,dt = 0$ at the echo, so it vanishes anyway.)

    **(b)** ⇒ $[\mathbf Be^{-D\Lambda\tau}\mathbf B]_{00} = \sum_n B_{0n}^2e^{-D\lambda_n\tau} = \langle x(t_1)x(t_2)\rangle$ with $\tau = t_2 - t_1$ (2b, and $\mathbf B$ symmetric).

    ⇒ The second-order term is $-\gamma^2\int_{t_1<t_2}g(t_1)g(t_2)\langle x(t_1)x(t_2)\rangle = -\frac{\gamma^2}{2}\iint g(t_1)g(t_2)\langle x(t_1)x(t_2)\rangle = -\frac12\langle\varphi^2\rangle$. ✓

    ⇒ So to second order $E \approx 1 - \frac12\langle\varphi^2\rangle$, and $\ln E \approx -\frac12\langle\varphi^2\rangle$: the GPA is the first nontrivial term of the matrix expansion.

    **(c)** ⇒ $E$ is a finite sum of exponentials, finite everywhere, so its power series converges for all $G$; $\ln E$ blows up where $E = 0$, which caps the radius of its series.

    → To compute the signal at 600 mT/m, use the matrix method (or the full series, but its terms grow large and cancel, so computing the exponentials directly is far more accurate). To build a fittable model at low $b$, use the cumulants: $D_{\text{app}}$ and $K_{\text{app}}$, with the caveats of Exercise 6.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How far does the cumulant series reach? The radius of convergence, and computing it in practice

    Exercises 5c and 7c showed *that* the cumulant series stops somewhere. This section shows **where**, and how to compute the cumulants of any order in practice.

    **The radius of convergence.** $\ln E(G)$ is analytic wherever $E\ne0$, so its power series around $G = 0$ converges inside the largest disc that contains no zero of $E$. Two symmetries place the zeros:

    - $E$ is **even** in $G$: reversing the gradient is the same as mirroring the slab. So $\ln E$ is a series in $G^2$, i.e. in $b$.
    - $E$ is **real** for real $G$ (a symmetric pore), so $E(\bar G) = \overline{E(G)}$.

    → Zeros come in fours, $\pm G^*$ and $\pm\bar G^*$, and the radius is $R = \lvert G^*\rvert$, the modulus of the nearest one. In $b$-units, $b^* = \gamma^2\lvert G^*\rvert^2\delta^2(\Delta - \delta/3)$.

    **What the zero does to the coefficients.** Write $\ln E = \sum_k c_{2k}G^{2k}$. Close to its nearest zeros, $\ln E$ behaves like $\ln(1 - G^2/G^{*2}) + \ln(1 - G^2/\bar G^{*2})$ plus a part that is analytic in a larger disc. Expanding $\ln(1-u) = -\sum_k u^k/k$ gives, for large $k$ (Darboux 1878),

    $$c_{2k}\approx-\frac2k\,\lvert G^*\rvert^{-2k}\cos(2k\theta^*),\qquad \theta^* = \arg G^* .$$

    → The coefficients shrink like $R^{-2k}$: the cumulant series behaves like a geometric series in $(G/R)^2$. Truncating it after the $G^{2K}$ term leaves an error of order $(G/R)^{2K+2}/K$. That is tiny for $G\ll R$ and useless near $R$, however many terms you keep.

    **Three ways to get the cumulants**

    | Route | How | Strength | Weakness |
    | --- | --- | --- | --- |
    | Fit a polynomial in $b$ to $\ln E$ | DKI-style, on $[0, b_{\max}]$ | works on measured data | biased by the omitted higher terms; ill-conditioned beyond 2–3 terms |
    | Nested time integrals (2c) | $k$-fold integrals of $\mathbf Be^{-D\Lambda\tau}\mathbf B\cdots$ | exact, and each term has a meaning | the cost grows fast with the order; impractical beyond the 4th |
    | Cauchy integral on a circle | $E$ at $M$ complex $G$ on a circle, then an FFT of $\ln E$ | every order at once, to rounding error; also reveals $R$ | needs a model that accepts complex $G$ |

    **Recipe: cumulants from a circle in the complex plane** ([Lyness & Moler 1967](https://doi.org/10.1137/0704019))

    ① **Find the nearest zero $G^*$.** Look for a dip of $\lvert E\rvert$ along real $G$, or for dark spots of $\lvert E\rvert$ in the complex plane, and refine with Newton's method in complex $G$. The MCF needs no change for a complex $G$: it is the same matrix exponential.

    ② **Choose a circle inside the disc**, $r < R$ (here $r = 0.7R$), with $M$ points $G_j = r\,e^{2\pi ij/M}$ ($M = 64$).

    ③ **Compute $\ln E(G_j)$** with its phase unwrapped along the circle.

    ④ **FFT:** $c_k = \mathrm{FFT}(\ln E)_k\big/(M\,r^k)$. This is the trapezoidal rule for Cauchy's integral $c_k = \frac1{2\pi i}\oint\ln E(G)\,G^{-k-1}dG$. On a circle it converges geometrically, with an error of order $(r/R)^M$ ([Trefethen & Weideman 2014](https://doi.org/10.1137/130932132)).

    ⑤ **Check:** $c_0\approx0$ (since $E(0) = 1$) and all odd $c_k\approx0$. If the circle encloses a zero, the unwrapped phase does not return to its start: it gains $2\pi$ per zero inside (the argument principle), and both checks fail.

    ⑥ **Convert.** With $b = \beta G^2$ and $\beta = \gamma^2\delta^2(\Delta-\delta/3)$: $\ D_{\text{app}} = -c_2/\beta$ and $K_{\text{app}} = 6c_4/(\beta^2D_{\text{app}}^2)$, and so on for the higher orders.

    The figures use the PGSE sliders at the top of 2a.
    """)
    return


@app.cell(hide_code=True)
def _(D0, GAMMA, np, slab_B):
    def pgse_complex(Gs, L, delta, Delta, N=20, D=D0):
        """Exact PGSE echo (MCF) for an array of real or complex gradients G (mT/m).

        Each lobe is exponentiated with an eigendecomposition, for all G at once."""
        Gs = np.atleast_1d(np.asarray(Gs, complex))
        lam, B = slab_B(L, N)

        def lobe(sign):
            A = D * np.diag(lam)[None] + sign * 1j * GAMMA * Gs[:, None, None] * B[None]
            mu, V = np.linalg.eig(A)
            return V, mu, np.linalg.inv(V)

        Vp, mp, Wp = lobe(+1.0)
        Vm, mm, Wm = lobe(-1.0)
        c = np.einsum("mij,mj->mi", Vp, np.exp(-mp * delta) * Wp[:, :, 0])    # first lobe acting on e0
        c = np.exp(-D * lam * (Delta - delta)) * c                              # gap
        c = np.einsum("mij,mj->mi", Wm, c)                                      # second lobe
        return np.einsum("mj,mj->m", Vm[:, 0, :], np.exp(-mm * delta) * c)

    def nearest_zero(L, delta, Delta, Gmax=2000.0, N=20):
        """Nearest zero of E(G) in the first quadrant, or None if there is none with |G| < Gmax."""
        re = np.linspace(Gmax / 60, Gmax, 60)
        im = np.linspace(0.0, Gmax / 2, 31)
        RE, IM = np.meshgrid(re, im)
        A = np.abs(pgse_complex((RE + 1j * IM).ravel(), L, delta, Delta, N)).reshape(RE.shape)
        P = np.pad(A, 1, constant_values=np.inf)
        is_min = np.ones(A.shape, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    is_min &= A <= P[1 + di : 1 + di + A.shape[0], 1 + dj : 1 + dj + A.shape[1]]
        cands = (RE + 1j * IM)[is_min]
        cands = cands[np.argsort(np.abs(cands))][:6]
        best = None
        for g in cands:
            g = complex(g) + 1e-3j * Gmax                                       # start just off the real axis
            ok = False
            for _ in range(50):                                                 # Newton in complex G
                h = 1e-4 * abs(g)
                f, fp, fm = pgse_complex([g, g + h, g - h], L, delta, Delta, N)
                step = f / ((fp - fm) / (2 * h))
                g = g - step
                if abs(step) < 1e-10 * abs(g):
                    ok = True
                    break
            g = complex(abs(g.real), abs(g.imag))
            if ok and abs(g) < Gmax and (best is None or abs(g) < abs(best) - 1e-6):
                best = g
        return best

    def cauchy_coeffs(L, delta, Delta, r, M=64, N=20):
        """Taylor coefficients c_k of ln E(G) from M points on the circle |G| = r, and the winding number."""
        Gc = r * np.exp(2j * np.pi * np.arange(M) / M)
        E = pgse_complex(Gc, L, delta, Delta, N)
        phase = np.unwrap(np.angle(E))
        wind = int(np.round((phase[-1] + np.angle(E[0] / E[-1]) - phase[0]) / (2 * np.pi)))
        lnE = np.log(np.abs(E)) + 1j * phase
        c = np.fft.fft(lnE) / M / r ** np.arange(M, dtype=float)
        return c, wind

    return cauchy_coeffs, nearest_zero, pgse_complex


@app.cell(hide_code=True)
def _(Delta_a, L_a, cauchy_coeffs, delta_a, nearest_zero):
    rc_L, rc_de = L_a.value, delta_a.value
    rc_De = max(Delta_a.value, rc_de)
    rc_zero = nearest_zero(rc_L, rc_de, rc_De)
    rc_R = abs(rc_zero) if rc_zero is not None else 2000.0
    rc_r = 0.7 * rc_R
    rc_c, rc_wind = cauchy_coeffs(rc_L, rc_de, rc_De, rc_r)
    return rc_De, rc_L, rc_R, rc_c, rc_de, rc_r, rc_wind, rc_zero


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_FREE,
    C_MAIN,
    D0,
    GAMMA,
    mo,
    np,
    pgse_complex,
    plt,
    rc_De,
    rc_L,
    rc_R,
    rc_c,
    rc_de,
    rc_r,
    rc_wind,
    rc_zero,
):
    _re = np.linspace(0.0, 1.5 * rc_R, 64)
    _im = np.linspace(-0.5 * rc_R, 0.5 * rc_R, 41)
    _RE, _IM = np.meshgrid(_re, _im)
    _A = np.abs(pgse_complex((_RE + 1j * _IM).ravel(), rc_L, rc_de, rc_De)).reshape(_RE.shape)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    _pc = _a1.pcolormesh(_re, _im, np.log10(np.clip(_A, 1e-4, 1e2)), cmap="Blues_r", vmin=-4, vmax=2, shading="gouraud", rasterized=True)
    _cb = _fig.colorbar(_pc, ax=_a1, pad=0.02)
    _cb.set_label("log₁₀ |E(G)|")
    _th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    _a1.plot(rc_R * np.cos(_th), rc_R * np.sin(_th), color=C_ALT, lw=1.6, ls="--", label="|G| = R (radius of convergence)")
    _a1.plot(rc_r * np.cos(_th), rc_r * np.sin(_th), color="w", lw=1.2, ls=":", label="Cauchy circle, r = 0.7R")
    _tj = 2 * np.pi * np.arange(64) / 64
    _keep = np.cos(_tj) >= 0
    _a1.plot(rc_r * np.cos(_tj[_keep]), rc_r * np.sin(_tj[_keep]), ".", color="w", ms=3)
    _a1.axhline(0, color="k", lw=0.8)
    if rc_zero is not None:
        _a1.plot([rc_zero.real] * 2, [rc_zero.imag, -rc_zero.imag], "x", color=C_ALT, ms=9, mew=2.2, label="zeros G*, Ḡ*")
    _a1.set(xlim=(0, 1.5 * rc_R), ylim=(-0.5 * rc_R, 0.5 * rc_R), xlabel="Re G (mT/m)", ylabel="Im G (mT/m)", title="|E| in the complex gradient plane")
    _a1.set_aspect("equal")
    _a1.grid(False)
    _a1.legend(frameon=True, framealpha=0.85, fontsize=7.5, loc="upper left")

    _k = np.arange(1, 21)
    _c2k = rc_c[2 * _k].real
    _a2.semilogy(2 * _k, np.abs(_c2k) * rc_R ** (2 * _k), "o", color=C_MAIN, ms=5, label="|c₂ₖ| R²ᵏ from the circle")
    if rc_zero is not None:
        _thz = np.angle(rc_zero)
        _a2.semilogy(2 * _k, (2 / _k) * np.abs(np.cos(2 * _k * _thz)), "-", color=C_ALT, lw=1.4, label="Darboux: (2/k)·|cos 2kθ*|")
    _a2.set(xlabel="order 2k", ylabel="|c₂ₖ| · R²ᵏ", title="The nearest zero sets the coefficients", ylim=(1e-3, 10))
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()

    _beta = GAMMA**2 * rc_de**2 * (rc_De - rc_de / 3)
    _Dapp = -rc_c[2].real / _beta
    _Kapp = 6 * rc_c[4].real / (_beta**2 * _Dapp**2)
    _odd = np.abs(rc_c[1:40:2]).max() * rc_r
    if rc_zero is None:
        _ztxt = "No zero of E(G) with |G| < 2000 mT/m: the series reaches at least that far (motional narrowing keeps the phase nearly Gaussian)."
    else:
        _ztxt = (f"Nearest zero **G\\* = {rc_zero.real:.1f} + {rc_zero.imag:.1f}i mT/m**, so R = **{rc_R:.1f} mT/m**; "
                 f"in b-units b\\* = **{_beta * rc_R**2:.2f} ms/µm²**, and b\\*·D_app = **{_beta * rc_R**2 * _Dapp:.2f}**.")
    mo.vstack([_fig, mo.md(
        _ztxt + f" From the circle: D_app/D = **{_Dapp / D0:.4f}**, K_app = **{_Kapp:+.3f}**. "
        f"Checks: |c₀| = {abs(rc_c[0]):.1e}, largest odd |cₖ|·r = {_odd:.1e}, winding number = {rc_wind}."
    )])
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, GAMMA, mo, np, pgse_complex, plt, rc_De, rc_L, rc_R, rc_c, rc_de):
    _G = np.linspace(1.0, 1.3 * rc_R, 220)
    _E = pgse_complex(_G, rc_L, rc_de, rc_De).real
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    _a1.semilogy(_G, np.abs(_E), color="k", lw=2.4, label="exact (MCF)")
    _cols = [C_FREE, "#8fb3ea", C_MAIN, "#1c4fa3", C_ALT]
    for _K, _col in zip((1, 2, 4, 8, 16), _cols):
        _s = np.zeros_like(_G)
        for _j in range(1, _K + 1):
            _s = _s + rc_c[2 * _j].real * _G ** (2 * _j)
        _lab = "Gaussian phase (G²)" if _K == 1 else f"up to G^{2 * _K}"
        _a1.semilogy(_G, np.exp(np.clip(_s, -50, 50)), color=_col, lw=1.3, ls="--", label=_lab)
    _a1.axvline(rc_R, color=C_ALT, lw=1.0, ls=":")
    _a1.text(rc_R * 1.01, 0.4, "R", color=C_ALT, fontsize=9)
    _a1.set(ylim=(1e-4, 1.5), xlabel="G (mT/m)", ylabel="|E|", title="Partial sums of the cumulant series")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower left")

    _beta = GAMMA**2 * rc_de**2 * (rc_De - rc_de / 3)
    _Dapp = -rc_c[2].real / _beta
    _Kapp = 6 * rc_c[4].real / (_beta**2 * _Dapp**2)
    _bstarD = _beta * rc_R**2 * _Dapp
    _xs = np.logspace(-2, np.log10(0.6 * _bstarD), 12)            # b_max · D_app
    _Kfit = []
    for _x in _xs:
        _Gm = np.sqrt(_x / _Dapp / _beta)
        _Gs = np.linspace(_Gm / 12, _Gm, 12)
        _y = np.log(pgse_complex(_Gs, rc_L, rc_de, rc_De).real)
        _p = np.polyfit(_beta * _Gs**2, _y, 2)
        _Kfit.append(6 * _p[0] / _p[1] ** 2)
    _a2.semilogx(_xs, _Kfit, "o-", color=C_MAIN, ms=4, label="K from a quadratic fit on [0, b_max]")
    _a2.axhline(_Kapp, color="k", lw=1.2, ls="--", label=f"exact K_app (circle) = {_Kapp:+.3f}")
    _a2.set(xlabel="b_max · D_app", ylabel="fitted K_app", title="Fitting a truncated series biases K")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _i = int(np.argmin(np.abs(_xs - 0.3)))
    mo.vstack([_fig, mo.md(
        f"Inside R the partial sums converge to the exact curve, slowly near R; beyond R they diverge. "
        f"Right: at b_max·D_app = {_xs[_i]:.2f} the quadratic fit gives K = {_Kfit[_i]:+.3f}, "
        f"{100 * (_Kfit[_i] / _Kapp - 1):+.0f}% from the exact value. The bias grows with b_max/b\\* (here b\\*·D_app = {_bstarD:.2f})."
    )])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Show that the zeros of $E(G)$ come in fours, $\pm G^*$ and $\pm\bar G^*$, and conclude that the cumulant series has only even powers of $G$ and radius $\lvert G^*\rvert$.

    **(b)** Derive the Darboux estimate $c_{2k}\approx-\frac2k\lvert G^*\rvert^{-2k}\cos(2k\theta^*)$. Check it against the right-hand panel of the first figure. Why does the "root test" $\lvert c_{2k}\rvert^{-1/2k}$ approach $R$ only slowly?

    **(c)** With the defaults ($L = 8$ µm, $\delta = 10$, $\Delta = 20$ ms), how many terms of the series do you need for $\lvert E\rvert$ to 1% at $G = 300$ mT/m, and at $450$ mT/m? Use the partial sums.

    **(d)** A DKI protocol on this sample goes up to $b = 2.5$ ms/µm². Is it inside the radius of convergence? How biased is the fitted kurtosis, and why is it biased even though the series converges there?

    **(e)** In the recipe, what goes wrong if the Cauchy circle encloses a zero? How could you turn that failure into a way of counting zeros?
    """, "8 · The radius of convergence in practice")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - **(a)** Mirroring the slab, $x\to-x$, flips the sign of $\mathbf B$. And for real $G$ the echo of a symmetric pore is real.
    - **(b)** Near $G^*$: $\ln E = \ln(G - G^*) + $ (analytic). Pair the four zeros as $(1 - G^2/G^{*2})(1 - G^2/\bar G^{*2})$. For the root test, take the $2k$-th root of $2/k$.
    - **(d)** The quadratic fit has no $b^3$ term, so the $b^3$ coefficient leaks into the $b^2$ one.
    - **(e)** The argument principle: $\frac1{2\pi}\oint d\arg E$ = number of zeros inside.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Mirroring $x\to-x$ maps the slab onto itself and $\mathbf B\to-\mathbf B$, which is the same as $G\to-G$. So $E(-G) = E(G)$: if $G^*$ is a zero, so is $-G^*$.

    ⇒ For real $G$, $E$ is real. The power series of $E$ then has real coefficients, so $E(\bar G) = \overline{E(G)}$: if $G^*$ is a zero, so is $\bar G^*$.

    → $E$ even ⇒ $\ln E$ even ⇒ only even powers, a series in $G^2$ (in $b$). The four zeros all have modulus $\lvert G^*\rvert$, which is the radius.

    **(b)** ⇒ Near the zeros, $\ln E \approx \ln(1 - G^2/G^{*2}) + \ln(1 - G^2/\bar G^{*2})$. With $\ln(1-u) = -\sum_k u^k/k$, the coefficient of $G^{2k}$ is $-\frac1k\big(G^{*-2k} + \bar G^{*-2k}\big) = -\frac2k\lvert G^*\rvert^{-2k}\cos(2k\theta^*)$. ✓ The other singularities are further away, so their contributions die faster with $k$. In the figure the estimate matches the exact coefficients to four digits from order 16 on.

    ⇒ Root test: $\lvert c_{2k}\rvert^{-1/2k} = R\,\big(k/2\lvert\cos2k\theta^*\rvert\big)^{1/2k}\approx R\,(1 + \frac{\ln k}{2k})$. The factor $k^{1/2k}$ tends to 1 only logarithmically slowly: at $2k = 30$ it is still about 10% above $R$. Fitting the Darboux form, as in the figure, is much more accurate.

    **(c)** ⇒ At 300 mT/m, $(G/R)^2 = 0.42$: the terms fall by a factor of 2.4 per order, and up to $G^8$ (4 terms) gives $\lvert E\rvert$ to 1%. At 450 mT/m, $(G/R)^2 = 0.94$: the terms hardly fall, and even 16 terms (up to $G^{32}$) are about 15% off. Close to $R$ the series is correct but useless.

    **(d)** ⇒ $b = 2.5$ against $b^* = 25.6$ ms/µm²: well inside, $b/b^*\approx0.1$. The quadratic fit still returns $K\approx-0.56$ against the exact $-0.51$, about 10% too negative (right-hand panel).

    → Convergence of the full series does not make a two-term fit unbiased. The fit has no $b^3$ term, and the omitted $c_6$ term, of relative size about $b/b^*$, is absorbed into the $b^2$ coefficient. Keeping $b_{\max}\ll b^*$ makes the bias small. Adding a cubic term reduces it, at the cost of more noise.

    **(e)** ⇒ If the circle encloses zeros, $\ln E$ is not single-valued on it: the unwrapped phase gains $2\pi$ per zero inside, so the samples are not periodic. The FFT then returns nonsense, $c_0\ne0$ and the odd coefficients are no longer zero.

    → The same quantity is a zero counter: the winding number $\frac1{2\pi}\oint d\arg E$ (printed under the first figure) is the number of zeros inside the circle. Increase $r$ until it jumps from 0 to 4 (the four symmetric zeros): the jump brackets $R$, without any Newton iteration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary of Chapter 2

    | Idea | Formula |
    | --- | --- |
    | Bloch–Torrey in the eigenbasis | $d\mathbf c/dt = -(D\Lambda + i\gamma g(t)\mathbf B)\,\mathbf c$, $\ \mathbf c(0) = \mathbf e_0$, $\ E = c_0(T)$ |
    | Gradient matrix | $B_{nm} = \int u_n\,x\,u_m$; slab: parity rule, $B_{0n}^2 = 8L^2/(n\pi)^4$ |
    | Grebenkov's MCF (PGSE) | $E = [e^{-(D\Lambda - i\gamma G\mathbf B)\delta}e^{-D\Lambda(\Delta-\delta)}e^{-(D\Lambda + i\gamma G\mathbf B)\delta}]_{00}$ |
    | Multiple propagator | $e^{-D\Lambda\tau/2}e^{-i\gamma g\tau\mathbf B}e^{-D\Lambda\tau/2}$ per step; error $\propto\tau^2$ |
    | Number of modes | mirrored $m$ has a jump in $\partial_x^3$: error $\sim N^{-5}$; $D\lambda_N \gg \gamma gL$; double $N$ to check |
    | Cumulant expansion | $\ln E = -\frac12\langle\varphi^2\rangle + \frac1{24}\kappa_4 - \dots$, $\ \kappa_4 = \langle\varphi^4\rangle - 3\langle\varphi^2\rangle^2$, averages over spins (their random paths); GPA keeps the first term |
    | GPA from the modes | $\langle\varphi^2\rangle = \gamma^2\iint g(t_1)g(t_2)\langle x(t_1)x(t_2)\rangle dt_1dt_2$, $\ \langle x(t_1)x(t_2)\rangle = \frac1L\iint x_0\,x\,P(x\mid x_0,\lvert t_2-t_1\rvert)\,dx\,dx_0 = \sum_n B_{0n}^2e^{-D\lambda_n\lvert t_1-t_2\rvert}$ |
    | Limits of the GPA | large pore: $e^{-bD}$; small pore: $\ln E = -\gamma^2G^2L^4T/(120D)$ |
    | Radius of the cumulant series | $R = \lvert G^*\rvert$, the nearest complex zero of $E(G)$; zeros in fours $\pm G^*, \pm\bar G^*$; $c_{2k}\approx-\frac2k R^{-2k}\cos2k\theta^*$ |
    | Cumulants in practice | $c_k = \mathrm{FFT}(\ln E)_k/(Mr^k)$ on a circle $r < R$; check $c_0\approx0$ and the winding number; polynomial fits in $b$ are biased by the omitted terms |
    | Finite pulses | $D_{\text{app}}$, $K_{\text{app}}$ depend on $\delta$: the centre-of-mass effect |
    | Diffusion spectrum | $\langle\varphi^2\rangle = \frac2\pi\int_0^\infty\lvert Q\rvert^2\mathcal D(\omega)d\omega$, $\ \mathcal D = \sum w_nD\lambda_n\,\omega^2/((D\lambda_n)^2 + \omega^2)$ |
    | The link | Dyson series of $c_0(T)$: products $[\mathbf B e^{-D\Lambda\tau}\mathbf B\cdots]_{00}$ = multi-time correlation functions |

    ---
    ## Coming next

    **Chapter 3 · Analytical formulas: a map of the PGSE parameter space**

    - **The PGSE box:** three dimensionless numbers, $D\Delta/L^2$, $D\delta/L^2$ and $\gamma gL^3/D$, with a narrow-pulse face and a long-pulse face.
    - **Narrow pulses, the $(q,t)$ map:** $E = \langle e^{iqX}\rangle$ (Tanner & Stejskal 1968); free diffusion and the $b$-matrix; the surface-to-volume ratio and the short-time kurtosis (Mitra et al. 1992); closed pores, tortuosity and disorder (Novikov et al. 2014); diffraction and the form factors of slab, cylinder and sphere (Callaghan et al. 1991); where the cumulant series stops, $qX_{\text{rms}}\approx1$.
    - **Long pulses, the $(g,T)$ map:** free diffusion, motional narrowing (Robertson 1966; Neuman 1974) and localisation (Stoller, Happer & Dyson 1991; Hürlimann et al. 1995), with the triple point.
    - **Inside the box:** when the Gaussian phase approximation of 2b holds for a real PGSE, tested against the exact MCF of 2a.
    - **Extra axes:** relaxing walls (Brownstein & Tarr 1979) and exchange (Kärger 1985).

    **Chapter 4 · Solving the PDE on a grid or a mesh:** finite differences (MATI) and finite elements (SpinDoctor), including the mesh-based modes used by the matrix formalism.

    **Chapter 5 · Monte Carlo:** walkers, reflections and membranes; the $1/\sqrt N$ noise floor, checked against the exact signals of this chapter; walker histories stored as reusable "replay packs" ([Fick 2026](https://arxiv.org/abs/2611.01758)).
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
    | 1878 | G. Darboux, *J. Math. Pures Appl.* (3) 4:5 | large-order coefficients of a series from its nearest singularities |
    | 1911 | D. Jackson, dissertation, Göttingen | error of the best trigonometric approximation against smoothness |
    | 1912 | S. Bernstein, *Mém. Acad. Roy. Belgique* | the converse: fast approximation implies smoothness |
    | 1912 | [H. Weyl, *Math. Ann.* 71:441](https://doi.org/10.1007/BF01456804) | how the number of eigenvalues grows |
    | 1965 | [E. O. Stejskal & J. E. Tanner, *J. Chem. Phys.* 42:288](https://doi.org/10.1063/1.1695690) | the PGSE sequence and its $b$-value |
    | 1966 | [B. Robertson, *Phys. Rev.* 151:273](https://doi.org/10.1103/PhysRev.151.273) | eigenmode solution for a slab; motional narrowing |
    | 1967 | [J. N. Lyness & C. B. Moler, *SIAM J. Numer. Anal.* 4:202](https://doi.org/10.1137/0704019) | Taylor coefficients of analytic functions from points on a circle in the complex plane |
    | 1968 | J. S. Murday & R. M. Cotts, *J. Chem. Phys.* 48:4938 | the Gaussian phase approximation for a sphere |
    | 1968 | [G. Strang, *SIAM J. Numer. Anal.* 5:506](https://doi.org/10.1137/0705041) | symmetric operator splitting |
    | 1974 | C. H. Neuman, *J. Chem. Phys.* 60:4508 | GPA formulas for slab, cylinder and sphere |
    | 1993 | J. Stepišnik, *Physica B* 183:343 | the signal in terms of the diffusion spectrum $\mathcal D(\omega)$ |
    | 1994 | P. van Gelderen, D. DesPres, P. C. M. van Zijl & C. T. W. Moonen, *J. Magn. Reson. B* 103:255 | GPA for arbitrary waveforms |
    | 1995 | [P. P. Mitra & B. I. Halperin, *J. Magn. Reson. A* 113:94](https://doi.org/10.1006/jmra.1995.1060) | finite pulses: the centre-of-mass propagator |
    | 1996 | [A. Caprihan, L. Z. Wang & E. Fukushima, *J. Magn. Reson. A* 118:94](https://doi.org/10.1006/jmra.1996.0013) | the multiple narrow-pulse (propagator) approximation |
    | 1997 | [P. T. Callaghan, *J. Magn. Reson.* 129:74](https://doi.org/10.1006/jmre.1997.1233) | the matrix formalism for general waveforms |
    | 1999 | [A. V. Barzykin, *J. Magn. Reson.* 139:342](https://doi.org/10.1006/jmre.1999.1778) | exact eigenmode solution for step-wise gradients |
    | 1999 | S. L. Codd & P. T. Callaghan, *J. Magn. Reson.* 137:358 | long pulses make pores look smaller; wall relaxation in the matrix formalism |
    | 2001 | S. Axelrod & P. N. Sen, *J. Chem. Phys.* 114:6878 | systematic expansions and asymptotic regimes for restricted diffusion |
    | 2005 | [J. H. Jensen et al., *Magn. Reson. Med.* 53:1432](https://doi.org/10.1002/mrm.20508) | diffusion kurtosis imaging |
    | 2007 | [D. S. Grebenkov, *Rev. Mod. Phys.* 79:1077](https://doi.org/10.1103/RevModPhys.79.1077) | review; the multiple correlation function (MCF) approach |
    | 2014 | [L. N. Trefethen & J. A. C. Weideman, *SIAM Rev.* 56:385](https://doi.org/10.1137/130932132) | why the trapezoidal rule on a circle converges geometrically |
    | 2026 | [E. J. Canales-Rodríguez, C. M. W. Tax et al., *Phys. Med. Biol.*](https://doi.org/10.1088/1361-6560/aeb2a0) | exact PGSE signal on a cylindrical surface |
    """)
    return


if __name__ == "__main__":
    app.run()
