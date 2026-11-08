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
    app_title="dMRI simulation · Ch 1 · Foundations",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    # Small NumPy versions of the few SciPy routines we need. Leaving SciPy out keeps the
    # in-browser (WebAssembly) version light enough for tablets and phones.
    def expm(A):
        """Matrix exponential by scaling and squaring with a Taylor series."""
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

    _tau_q = np.linspace(0, np.pi, 257)
    _w_q = np.full(_tau_q.size, _tau_q[1]); _w_q[[0, -1]] /= 2

    def jv(m, x):
        """Bessel function J_m(x), integer m, from its integral representation."""
        x = np.asarray(x, float)
        integrand = np.cos(m * _tau_q - x[..., None] * np.sin(_tau_q))
        return (integrand @ _w_q) / np.pi

    def brentq(f, a, b, tol=1e-13):
        """Root of f in [a, b] by bisection (f(a) and f(b) must differ in sign)."""
        fa = f(a)
        for _ in range(200):
            c = 0.5 * (a + b)
            fc = f(c)
            if fa * fc <= 0:
                b = c
            else:
                a, fa = c, fc
            if b - a < tol:
                break
        return 0.5 * (a + b)

    def _roots(fvec, k, xmax=40.0):
        """First k positive roots of a vectorised function, by sign changes then bisection."""
        grid = np.linspace(1e-6, xmax, 8000)
        vals = fvec(grid)
        out = []
        for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if va * vb < 0:
                out.append(brentq(lambda x: float(fvec(np.array([x]))[0]), a, b))
                if len(out) == k:
                    break
        return np.array(out)

    def jn_zeros(m, k):
        """First k positive zeros of J_m."""
        return _roots(lambda x: jv(m, x), k)

    def jnp_zeros(m, k):
        """First k positive zeros of J_m' (m >= 1)."""
        return _roots(lambda x: 0.5 * (jv(m - 1, x) - jv(m + 1, x)), k)

    def spherical_jn(l, x, derivative=False):
        """Spherical Bessel function j_l(x) for l = 0..4, or its derivative."""
        x = np.asarray(x, float)
        s_, c_ = np.sin(x), np.cos(x)
        j = [s_ / x, s_ / x**2 - c_ / x]
        for n in range(1, 4):
            j.append((2 * n + 1) / x * j[n] - j[n - 1])
        if not derivative:
            return j[l]
        return -j[1] if l == 0 else j[l - 1] - (l + 1) / x * j[l]

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
    C_ALT = "#d9622b"    # approximation / walkers
    # gamma in units that fit µm, ms and mT/m: phase (rad) = GAMMA * g[mT/m] * x[µm] * t[ms]
    GAMMA = 2.6752e-4
    return (
        C_ALT,
        C_FREE,
        C_MAIN,
        GAMMA,
        brentq,
        expm,
        jn_zeros,
        jnp_zeros,
        jv,
        mo,
        np,
        plt,
        spherical_jn,
    )


@app.cell(hide_code=True)
def _(mo):
    show_answers = mo.ui.switch(label="Teacher view: show all answers", value=False)
    mo.vstack(
        [
            mo.md(
                r"""
    # Chapter 1 · Foundations: diffusion, eigenmodes and the Bloch–Torrey equation

    *© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.*

    *Course: simulating the diffusion MRI signal. This chapter sets up the equations that every later method solves.*

    **What you'll take away**

    1. How a pore's shape enters diffusion: through the **eigenmodes** of the Laplacian with the right interface conditions.
    2. How to derive those modes yourself for a slab, and where to look them up for cylinders and spheres.
    3. The **Bloch–Torrey equation** for the magnetization, and the signal as the **average phase factor** of random walkers.
    4. Why those two descriptions give the same signal (the **Feynman–Kac** relation), and what that means for simulation.

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
    | **1** | **Foundations:** notation, diffusion equation, interface and boundary conditions, eigenmodes; Bloch–Torrey; phases and Feynman–Kac | this notebook |
    | 2 | Analytical formulas: the regimes where the signal has a closed form | coming |
    | 3 | Matrix methods: the eigenmodes of 1a, coupled by the gradient | coming |
    | 4 | Solving the PDE on a grid or a mesh: finite differences and finite elements | coming |
    | 5 | Monte Carlo: simulating the random walk | coming |

    Chapters 2–5 each follow the same pattern: the **principle** (with its assumptions), then how different **toolboxes implement it**, then exercises.

    **Units used throughout:** lengths in µm, times in ms, $D$ in µm²/ms (free water at 37 °C ≈ 3), gradients in mT/m.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notation and the maths you will need

    **Symbols**

    | Symbol | Meaning | Units |
    | --- | --- | --- |
    | $t$, $T$ | time; echo time | ms |
    | $\mathbf r = (x,y,z)^\mathsf T$ | position | µm |
    | $\Omega$, $\partial\Omega$ | the domain (a pore, or the whole voxel) and its outer boundary | |
    | $\Omega_n$, $\Gamma_{nm}$ | compartment $n$; the interface (membrane) between compartments $n$ and $m$ | |
    | $\mathbf n$ | outward unit normal on a boundary or interface | |
    | $C(\mathbf r,t)$ | concentration of tagged molecules | 1/µm³ |
    | $m(\mathbf r,t) = M_x + iM_y$ | transverse magnetization as one complex number | |
    | $\rho$ | spin density at $t=0$ | |
    | $D$, $\mathbf D$ | diffusion coefficient; diffusion tensor (3×3) | µm²/ms |
    | $\kappa$ | membrane permeability | µm/ms |
    | $\varrho$ | surface relaxivity | µm/ms |
    | $\gamma$ | gyromagnetic ratio of ¹H, $2.675\times10^{8}$ rad s⁻¹ T⁻¹ | |
    | $\mathbf g(t) = \mathbf g\,f(t)$ | effective diffusion gradient; $f(t)$ its time profile | mT/m |
    | $\mathbf q(t) = \gamma\int_0^t\mathbf g\,dt'$, $b = \int_0^T q^2dt$ | phase-winding wave vector; $b$-value | rad/µm; ms/µm² |
    | $\delta$, $\Delta$ | gradient pulse duration and separation (PGSE) | ms |
    | $\varphi$ | phase gathered by one spin | rad |
    | $S$, $S_0$, $E = S/S_0$ | signal with and without diffusion weighting; normalised signal | |
    | $\lambda_n$, $u_n(\mathbf r)$, $\tau_n$ | eigenvalue, eigenmode, decay time $1/(D\lambda_n)$ | 1/µm², –, ms |
    | $P(\mathbf r\,|\,\mathbf r_0,t)$ | propagator: probability density to be at $\mathbf r$ at $t$ after starting at $\mathbf r_0$ | 1/µm³ |

    **Notation**

    | Notation | Meaning |
    | --- | --- |
    | $\mathbf r$, $\mathbf g$, $\mathbf J$ (bold lower case) | vectors; $\mathbf a\cdot\mathbf b = a_xb_x + a_yb_y + a_zb_z$ |
    | $\mathbf D$ (bold capital) | a tensor, here a symmetric 3×3 matrix: $\mathbf D\mathbf v$ is a vector |
    | $\partial C/\partial t$, $\partial C/\partial x$ | partial derivatives: change in one variable with the others held fixed |
    | $\nabla = (\partial_x, \partial_y, \partial_z)$ | "nabla", the vector of partial derivatives |
    | $\nabla C$ | gradient of a scalar: a vector pointing uphill, length = steepness |
    | $\nabla\cdot\mathbf J = \partial_xJ_x + \partial_yJ_y + \partial_zJ_z$ | divergence of a vector field: net outflow per unit volume |
    | $\nabla^2 C = \nabla\cdot\nabla C = \partial_x^2C + \partial_y^2C + \partial_z^2C$ | Laplacian: how much $C$ differs from its local average |
    | $\partial C/\partial n = \mathbf n\cdot\nabla C$ | normal derivative on a surface |
    | $\int_\Omega\cdots d\mathbf r$, $\oint_S\cdots dS$ | volume integral; integral over a closed surface |
    | $\delta(x)$ | Dirac delta: zero except at 0, $\int\delta(x)\,dx = 1$, $\int f(x)\delta(x-a)\,dx = f(a)$ |
    | $\delta_{mn}$ | Kronecker delta: 1 if $m=n$, else 0 |
    | $\langle X\rangle$ | expectation (average over many spins or many random paths) |
    | $i$ | imaginary unit, $i^2=-1$ |

    **Relationships used in the derivations**

    | Relationship | Statement | Used for |
    | --- | --- | --- |
    | Divergence theorem | $\int_V\nabla\cdot\mathbf J\,dV = \oint_S\mathbf J\cdot\mathbf n\,dS$ | the diffusion equation from conservation |
    | Integration by parts (1D) | $\int_a^b u\,v'\,dx = [u\,v]_a^b - \int_a^b u'\,v\,dx$ | eigenvalues are $\ge 0$; free-diffusion phase |
    | Green's identity (3D version) | $\int_\Omega u\,\nabla^2v\,d\mathbf r = \oint_{\partial\Omega} u\,\dfrac{\partial v}{\partial n}dS - \int_\Omega\nabla u\cdot\nabla v\,d\mathbf r$ | the same, in a pore of any shape |
    | Product rule | $\nabla\cdot(D\nabla C) = D\nabla^2C$ when $D$ is constant | uniform diffusivity |
    | Taylor expansion | $f(x-s) = f(x) - s f'(x) + \tfrac{s^2}{2}f''(x) - \dots$; $\ e^{\varepsilon}\approx 1+\varepsilon$ | small time steps (Feynman–Kac) |
    | Euler's formula | $e^{i\theta} = \cos\theta + i\sin\theta$ | phases as rotating arrows |
    | Product of cosines | $\cos a\cos b = \tfrac12[\cos(a-b)+\cos(a+b)]$ | orthogonality of slab modes |
    | Orthonormality | $\int_\Omega u_m u_n\,d\mathbf r = \delta_{mn}$ | projecting onto modes |
    | Gaussian moments | for $s\sim\mathcal N(0,\sigma^2)$: $\langle s\rangle = 0$, $\langle s^2\rangle = \sigma^2$, $\langle e^{-is}\rangle = e^{-\sigma^2/2}$ | random-walk steps; free-diffusion signal |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 1a · Diffusion and the eigenmodes of a pore

    ## The diffusion equation

    Let $C(\mathbf r,t)$ be the concentration of tagged water molecules (for us, later: magnetization). Two ingredients give the diffusion equation:

    1. **Conservation.** Molecules are not created or destroyed. What leaves a small volume must flow out through its surface.
    2. **Fick's law.** The flux points downhill, from high to low concentration: $\mathbf J = -D\nabla C$.

    Together they give

    $$\frac{\partial C}{\partial t} = D\,\nabla^2 C .$$

    [Fick (1855)](https://doi.org/10.1002/andp.18551700105) proposed the flux law by analogy with Fourier's law of heat conduction (Fourier 1822): the same equation describes heat, with temperature for $C$ and conductivity for $D$.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ Take any fixed volume $V$ with surface $S$ and outward normal $\mathbf n$. The amount inside changes only by flow through the surface:

    $$\frac{d}{dt}\int_V C\,dV = -\oint_S \mathbf J\cdot\mathbf n\,dS .$$

    ⇒ The divergence theorem turns the surface integral into a volume integral: $\oint_S \mathbf J\cdot\mathbf n\,dS = \int_V \nabla\cdot\mathbf J\,dV$.

    ⇒ This holds for *every* volume $V$, so the integrands must be equal: $\partial C/\partial t = -\nabla\cdot\mathbf J$.

    ⇒ Insert Fick's law: $\partial C/\partial t = \nabla\cdot(D\nabla C)$. For uniform $D$ this is $D\nabla^2 C$.

    → Keep the form $\nabla\cdot(D\nabla C)$ in mind: it is the one to use when $D$ differs between compartments.
    """, title="Derivation in four steps")
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    In free space (1D) a molecule released at $x=0$ spreads as

    $$G(x,t)=\frac{1}{\sqrt{4\pi Dt}}\exp\!\left(-\frac{x^2}{4Dt}\right).$$

    **(a)** Check that $G$ solves $\partial G/\partial t = D\,\partial^2 G/\partial x^2$.
    **(b)** Show that its variance is $\langle x^2\rangle = 2Dt$.
    """, "1 · The free Gaussian")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    (a) Write $G = (4\pi D t)^{-1/2} e^{-u}$ with $u = x^2/(4Dt)$, and differentiate the prefactor and the exponent separately.
    (b) Compare $G$ with a normal distribution $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-x^2/2\sigma^2}$ and read off $\sigma^2$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Time derivative: $\dfrac{\partial G}{\partial t} = G\left(-\dfrac{1}{2t} + \dfrac{x^2}{4Dt^2}\right)$.

    ⇒ Space derivatives: $\dfrac{\partial G}{\partial x} = -\dfrac{x}{2Dt}G$, and $\dfrac{\partial^2 G}{\partial x^2} = G\left(\dfrac{x^2}{4D^2t^2} - \dfrac{1}{2Dt}\right)$.

    ⇒ Multiply the second by $D$: $D\,\partial^2 G/\partial x^2 = G\left(\dfrac{x^2}{4Dt^2} - \dfrac{1}{2t}\right)$, which is $\partial G/\partial t$. ✓

    **(b)** $G$ is a normal distribution with $2\sigma^2 = 4Dt$, so $\langle x^2\rangle = \sigma^2 = 2Dt$ (per axis; $6Dt$ in 3D).

    → This is the same $\sqrt{2Dt}$ spread you get from a random walk. The equation for the *concentration* and the random walk of *one molecule* describe the same thing. [Einstein (1905)](https://doi.org/10.1002/andp.19053220806) made exactly this link, deriving $\langle x^2\rangle = 2Dt$ from the random motion of suspended particles. Part 1b builds on exactly that.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The domain: pores, compartments and the voxel

    The equation alone does not fix the solution. We also need where the molecules start, and what happens where they meet a wall. **This is where the microstructure enters the problem.** First some names (panel b follows the usual multi-compartment notation):

    - The **domain** $\Omega$ is the region the molecules live in: one closed pore (panel a), or the whole imaging voxel (panel b).
    - The voxel is split into **compartments** $\Omega_n$: the cells and the extracellular space $\Omega_e$.
    - Two compartments meet at an **interface** $\Gamma_{nm}$: a cell membrane.
    - The **boundary** $\partial\Omega$ is the outer edge of the voxel. Where a cell is cut by the voxel edge, that piece of its surface is called $\Gamma_n$.

    So we have **interface conditions** (on membranes and cell walls: physics) and **boundary conditions** (on the voxel edge: a modelling choice about what lies outside the voxel). For a single closed pore the two coincide: its wall is an impermeable interface and also the edge of the domain.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_MAIN, np, plt):
    from matplotlib.patches import Circle, Ellipse, Rectangle

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) one closed pore with a reflected random walk
    _th = np.linspace(0, 2 * np.pi, 400)
    _r = 1 + 0.16 * np.cos(3 * _th) + 0.08 * np.sin(2 * _th + 0.5)
    _X, _Y = _r * np.cos(_th), _r * np.sin(_th)
    _a1.fill(_X, _Y, color=C_MAIN, alpha=0.12)
    _a1.plot(_X, _Y, color=C_MAIN, lw=2.2)

    def _inside(q):
        return np.hypot(*q) < np.interp(np.arctan2(q[1], q[0]) % (2 * np.pi), _th, _r) - 0.02

    _rng = np.random.default_rng(3)
    _p = np.array([0.1, -0.1])
    _path = [_p.copy()]
    for _ in range(700):
        _s = _rng.normal(0, 0.045, 2)
        if _inside(_p + _s):
            _p = _p + _s
        elif _inside(_p - _s):
            _p = _p - _s
        _path.append(_p.copy())
    _path = np.array(_path)
    _a1.plot(_path[:, 0], _path[:, 1], color=C_ALT, lw=0.7, alpha=0.8)
    _a1.text(-0.25, 0.35, r"$\Omega$", fontsize=16)
    _k = 60
    _tx, _ty = np.gradient(_X)[_k], np.gradient(_Y)[_k]
    _nx, _ny = _ty / np.hypot(_tx, _ty), -_tx / np.hypot(_tx, _ty)
    _a1.annotate("", xy=(_X[_k] + 0.35 * _nx, _Y[_k] + 0.35 * _ny), xytext=(_X[_k], _Y[_k]),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    _a1.text(_X[_k] + 0.38 * _nx, _Y[_k] + 0.38 * _ny, r"$\mathbf{n}$", fontsize=12)
    _a1.text(0.55, -1.32, "wall: impermeable interface\n" r"$\partial C/\partial n = 0$ (the walker is reflected)",
             fontsize=9, ha="center")
    _a1.set(xlim=(-1.45, 1.55), ylim=(-1.6, 1.4), aspect="equal", title="(a) One closed pore")
    _a1.axis("off")

    # (b) a voxel split into compartments
    _a2.add_patch(Rectangle((0, 0), 10, 10, facecolor="0.93", edgecolor="k", lw=2.2, ls="--"))
    _clip = Rectangle((0, 0), 10, 10, transform=_a2.transData)
    _cells = [Circle((3, 3), 1.9), Ellipse((7.0, 6.6), 4.4, 3.4, angle=25), Circle((2.6, 7.6), 1.5), Circle((9.6, 2.3), 1.6)]
    for _c, _name in zip(_cells, [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]):
        _c.set(facecolor=(0.165, 0.435, 0.859, 0.18), edgecolor=C_MAIN, lw=2)
        _a2.add_patch(_c)
        _c.set_clip_path(_clip)
        _a2.text(min(_c.center[0], 9.1), _c.center[1], _name, fontsize=13, ha="center", va="center")
    _a2.text(5.4, 1.0, r"$\Omega_e$ (extracellular)", fontsize=11, ha="center")
    _a2.annotate(r"$\Gamma_{1e}$: membrane", xy=(3 + 1.9 * np.cos(0.6), 3 + 1.9 * np.sin(0.6)), xytext=(5.4, 4.4),
                 fontsize=9.5, arrowprops=dict(arrowstyle="-", lw=0.8))
    _a2.annotate(r"$\Gamma_4$: part of $\partial\Omega_4$ on the voxel edge", xy=(10, 2.3), xytext=(10.4, 0.4),
                 fontsize=9.5, arrowprops=dict(arrowstyle="-", lw=0.8))
    _a2.annotate(r"$\partial\Omega$: voxel boundary", xy=(10, 8.5), xytext=(10.4, 9.2), fontsize=9.5,
                 arrowprops=dict(arrowstyle="-", lw=0.8))
    _ang = 2.3
    _px, _py = 3 + 1.9 * np.cos(_ang), 3 + 1.9 * np.sin(_ang)
    _a2.annotate("", xy=(_px + 0.8 * np.cos(_ang), _py + 0.8 * np.sin(_ang)), xytext=(_px, _py),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))
    _a2.text(_px + 0.85 * np.cos(_ang) - 0.35, _py + 0.85 * np.sin(_ang), r"$\mathbf{n}_1$", fontsize=11)
    _a2.set(xlim=(-0.3, 15.5), ylim=(-0.4, 10.4), aspect="equal", title=r"(b) A voxel $\Omega$ split into compartments")
    _a2.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial, interface and boundary conditions

    **Initial condition.** Before the experiment the spins are in equilibrium and spread evenly: $C(\mathbf r,0)=\rho$, a constant (or one constant per compartment).

    **Interface conditions** on cell walls and membranes $\Gamma_{nm}$ ($\partial/\partial n$: derivative along the outward normal):

    | Interface | Condition | What it means | In a random walk |
    | --- | --- | --- | --- |
    | Impermeable, inert | $\dfrac{\partial C}{\partial n}=0$ (Neumann type) | no flux through the wall | the walker is reflected |
    | Permeable membrane | $D_n\dfrac{\partial C_n}{\partial n_n} = -D_m\dfrac{\partial C_m}{\partial n_m} = \kappa\,(C_m - C_n)$ (transmission) | the flux is continuous; the jump in $C$ across the membrane is set by the permeability $\kappa$ | the walker crosses with a small probability, else it is reflected |
    | Relaxing wall | $D\dfrac{\partial C}{\partial n} + \varrho\, C = 0$ (Robin type) | the wall destroys magnetization at a rate set by the relaxivity $\varrho$ | the walker is removed with a small probability; rarely needed in soft tissue |
    | Perfect sink | $C = 0$ (Dirichlet type) | every molecule that touches the wall is lost | not realistic for dMRI |

    Two limits of the permeable membrane are worth remembering: $\kappa = 0$ gives an impermeable wall on both sides, and $\kappa\to\infty$ forces $C_n = C_m$, as if the membrane were not there.

    **Boundary conditions** on the voxel edge $\partial\Omega$:

    | Boundary | Condition | When to use it |
    | --- | --- | --- |
    | Reflecting | $\partial C/\partial n = 0$ | the voxel is closed, or what lies outside does not matter |
    | Periodic | values and fluxes on opposite faces match; with a gradient on, up to a phase $e^{i\theta(t)}$, $\theta = \gamma g L\int_0^t f\,dt'$ | the voxel is one tile of a repeating tissue; avoids artificial walls at the voxel edge |

    The names Neumann, Robin and Dirichlet refer to the *type* of condition (on the derivative, on a mix, on the value), wherever it is applied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bc_check = mo.ui.radio(
        options=[
            "a reflecting wall",
            "no membrane at all",
            "a perfect sink",
        ],
        label="**Quick check:** a membrane with permeability κ → ∞ behaves like…",
    )
    bc_check
    return (bc_check,)


@app.cell(hide_code=True)
def _(bc_check, mo):
    if bc_check.value is None:
        _out = mo.md("")
    elif bc_check.value == "no membrane at all":
        _out = mo.callout(mo.md("**Yes.** A finite flux with κ → ∞ is only possible if the jump $C_- - C_+$ goes to zero, so $C$ is continuous across the membrane, as in a single compartment."), kind="success")
    else:
        _out = mo.callout(mo.md("**Not quite.** Look at the second equality: flux = κ (C₋ − C₊). For the flux to stay finite as κ grows, the jump C₋ − C₊ must shrink to zero."), kind="warn")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eigenmodes: the natural patterns of a pore

    Put the diffusion equation in a closed pore $\Omega$ with reflecting walls. Some concentration patterns keep their **shape** and only fade in amplitude. These are the **eigenmodes** $u_n(\mathbf r)$ of the pore. Any initial concentration can be written as a sum of eigenmodes, and each one fades at its own rate:

    $$C(\mathbf r,t) = \sum_n c_n\, u_n(\mathbf r)\, e^{-D\lambda_n t}, \qquad -\nabla^2 u_n = \lambda_n u_n \ \text{in } \Omega, \qquad \frac{\partial u_n}{\partial n}=0 \ \text{on the walls.}$$

    - $\lambda_0 = 0$ with $u_0$ constant: the uniform state. It never fades, because reflecting walls conserve the total amount.
    - Every other mode fades with time constant $\tau_n = 1/(D\lambda_n)$. Finer patterns have larger $\lambda_n$ and fade faster.
    - The geometry enters **only** through $\lambda_n$ and $u_n$. Once you have them, you have solved diffusion in that pore for every starting condition and every time.
    - Separation of variables and expansion in modes go back to Fourier (1822); the general theory of such eigenproblems is Sturm–Liouville theory (Sturm 1836; Liouville 1836). In diffusion NMR, [Robertson (1966)](https://doi.org/10.1103/PhysRev.151.273) was the first to expand the magnetization in the modes of a slab.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ **Try a product.** Guess $C(\mathbf r,t)=u(\mathbf r)\,T(t)$ and put it in $\partial C/\partial t = D\nabla^2 C$:

    $$u\,T' = D\,T\,\nabla^2 u .$$

    ⇒ **Separate.** Divide by $D\,u\,T$:

    $$\frac{T'}{D\,T} = \frac{\nabla^2 u}{u}.$$

    The left side depends only on $t$, the right only on $\mathbf r$. Both must equal the same constant; call it $-\lambda$.

    ⇒ **Time part:** $T' = -D\lambda T$, so $T(t) = e^{-D\lambda t}$.

    ⇒ **Space part:** $-\nabla^2 u = \lambda u$, with the wall condition $\partial u/\partial n = 0$ (it is a condition on $C$, so on $u$). Only special values $\lambda_n$ allow a solution: the eigenvalues.

    ⇒ **Superpose.** The equation is linear, so any sum of such products is again a solution. The eigenmodes are orthonormal, $\int_\Omega u_m u_n\,d\mathbf r = \delta_{mn}$, so the coefficients follow from the initial condition by projection:

    $$c_n = \int_\Omega u_n(\mathbf r)\,C(\mathbf r,0)\,d\mathbf r .$$

    ⇒ **Point source.** Start with one molecule at $\mathbf r_0$: $C(\mathbf r,0) = \delta(\mathbf r - \mathbf r_0)$. Then $c_n = u_n(\mathbf r_0)$, and the solution is the **propagator**, the probability density of being at $\mathbf r$ at time $t$ after starting at $\mathbf r_0$:

    $$P(\mathbf r\,|\,\mathbf r_0,t) = \sum_n u_n(\mathbf r)\,u_n(\mathbf r_0)\,e^{-D\lambda_n t}.$$

    → Why a point source if spins start everywhere? Because every other start is a sum of point sources: $C(\mathbf r,t) = \int P(\mathbf r|\mathbf r_0,t)\,C(\mathbf r_0,0)\,d\mathbf r_0$. One propagator serves all initial conditions.
    """, title="Derivation: separation of variables")
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Take a slab $0 \le x \le L$ (diffusion between two parallel reflecting planes).

    **(a)** Solve $-u'' = \lambda u$ with $u'(0)=u'(L)=0$. Consider $\lambda<0$, $\lambda=0$ and $\lambda>0$ separately.
    **(b)** Normalise the modes so that $\int_0^L u_n^2\,dx = 1$.
    **(c)** Show that two different modes are orthogonal.
    **(d)** Write down the propagator $P(x\,|\,x_0,t)$ for the slab.
    """, "2 · Eigenmodes of a slab")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - For $\lambda = -k^2 < 0$ the general solution is $A\cosh kx + B\sinh kx$; for $\lambda = k^2 > 0$ it is $A\cos kx + B\sin kx$.
    - Apply $u'(0)=0$ first: it removes one of the two functions. Then $u'(L)=0$ restricts $k$.
    - For (b) and (c): $\cos a\cos b = \tfrac12[\cos(a-b)+\cos(a+b)]$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\lambda = -k^2 < 0$: $u = A\cosh kx + B\sinh kx$. $u'(0) = Bk = 0$ gives $B = 0$; $u'(L) = Ak\sinh kL = 0$ gives $A = 0$. No negative eigenvalues. (Quicker: multiply $-u''=\lambda u$ by $u$ and integrate by parts with the wall conditions: $\lambda\int u^2 = \int u'^2 \ge 0$.)

    ⇒ $\lambda = 0$: $u = A + Bx$, and $u' = B = 0$. So $u_0$ is a constant: the uniform mode.

    ⇒ $\lambda = k^2 > 0$: $u = A\cos kx + B\sin kx$. $u'(0) = Bk = 0$ gives $B = 0$. $u'(L) = -Ak\sin kL = 0$ with $A \ne 0$ needs $\sin kL = 0$, so $kL = n\pi$.

    $$\lambda_n = \left(\frac{n\pi}{L}\right)^2,\qquad u_n(x) \propto \cos\frac{n\pi x}{L},\qquad n = 0,1,2,\dots$$

    **(b)** ⇒ $\int_0^L A^2\cos^2(n\pi x/L)\,dx = A^2 L/2$ for $n \ge 1$, and $A^2 L$ for $n=0$. So

    $$u_0 = \frac{1}{\sqrt L},\qquad u_n = \sqrt{\frac 2L}\cos\frac{n\pi x}{L}\ \ (n\ge1).$$

    **(c)** ⇒ For $n \ne m$: $\int_0^L \cos\frac{n\pi x}{L}\cos\frac{m\pi x}{L}dx = \tfrac12\int_0^L\left[\cos\frac{(n-m)\pi x}{L} + \cos\frac{(n+m)\pi x}{L}\right]dx = 0$, because each cosine runs over a whole number of half periods and integrates to $\sin(\text{integer}\cdot\pi)=0$.

    **(d)** ⇒ Insert into the general propagator:

    $$P(x\,|\,x_0,t) = \frac1L + \frac2L\sum_{n=1}^{\infty}\cos\frac{n\pi x}{L}\cos\frac{n\pi x_0}{L}\,e^{-D(n\pi/L)^2 t}.$$

    → The first term is the uniform long-time state, $1/L$. The slowest mode that still carries information fades with $\tau_1 = L^2/(\pi^2 D)$: 3.2 ms for $L = 8$ µm and $D = 2$ µm²/ms.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def slab_modes(x, L, N):
        """First N Neumann modes of a slab [0, L] on grid x: returns (U [N, len(x)], lam [N])."""
        n = np.arange(N)[:, None]
        U = np.where(n == 0, 1.0 / np.sqrt(L), np.sqrt(2.0 / L) * np.cos(n * np.pi * x[None, :] / L))
        lam = (np.arange(N) * np.pi / L) ** 2
        return U, lam

    def slab_propagator(x, x0, t, L, D, N):
        U, lam = slab_modes(x, L, N)
        u0 = slab_modes(np.array([x0]), L, N)[0][:, 0]
        return (u0 * np.exp(-D * lam * t)) @ U

    return slab_modes, slab_propagator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See it: modes and the propagator in a slab

    Left: the first four modes and how fast each fades. Right: a molecule released at $x_0$, built from $N$ modes. Make $t$ short and $N$ small, and watch the sum fail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    L1 = mo.ui.slider(4, 20, step=1, value=10, label="Slab width L (µm)", show_value=True)
    D1 = mo.ui.slider(0.5, 3.0, step=0.1, value=2.0, label="D (µm²/ms)", show_value=True)
    logt1 = mo.ui.slider(-2, 2, step=0.05, value=-1.0, label="log₁₀ t (t in ms)", show_value=True)
    N1 = mo.ui.slider(1, 80, step=1, value=6, label="Number of modes N", show_value=True)
    x01 = mo.ui.slider(0.0, 1.0, step=0.01, value=0.3, label="Start x₀ / L", show_value=True)
    mo.vstack(
        [
            mo.hstack([L1, D1], justify="start", gap=2),
            mo.hstack([logt1, N1, x01], justify="start", gap=2, wrap=True),
        ]
    )
    return D1, L1, N1, logt1, x01


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_FREE,
    C_MAIN,
    D1,
    L1,
    N1,
    logt1,
    mo,
    np,
    plt,
    slab_modes,
    slab_propagator,
    x01,
):
    _L, _D, _N = L1.value, D1.value, N1.value
    _t = 10 ** logt1.value
    _x0 = x01.value * _L
    _x = np.linspace(0, _L, 600)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    _U, _lam = slab_modes(_x, _L, 4)
    for _n in range(4):
        _tau = "∞" if _n == 0 else f"{1 / (_D * _lam[_n]):.2f} ms"
        _a1.plot(_x, _U[_n] * np.sqrt(_L), lw=2 if _n == 1 else 1.3, label=f"n = {_n}, τ = {_tau}")
    _a1.set(xlabel="x (µm)", ylabel="u_n(x) · √L", title="Slab modes: finer pattern, faster fade", ylim=(-1.7, 1.7))
    _a1.legend(frameon=False, fontsize=8, loc="lower left")

    _P_N = slab_propagator(_x, _x0, _t, _L, _D, _N)
    _P_ref = slab_propagator(_x, _x0, _t, _L, _D, 600)
    _P_free = np.exp(-(_x - _x0) ** 2 / (4 * _D * _t)) / np.sqrt(4 * np.pi * _D * _t)
    _a2.plot(_x, _P_free, color=C_FREE, ls="--", lw=1.3, label="free Gaussian (no walls)")
    _a2.plot(_x, _P_ref, color=C_MAIN, lw=2.2, label="converged (600 modes)")
    _a2.plot(_x, _P_N, color=C_ALT, lw=1.3, label=f"{_N} modes")
    _a2.axhline(1 / _L, color="k", lw=0.8, ls=":")
    _a2.text(_L * 0.99, 1 / _L, "1/L", ha="right", va="bottom", fontsize=8)
    _ymax = max(_P_ref.max(), 1.3 / _L) * 1.15
    _a2.set(xlabel="x (µm)", ylabel="P(x | x₀, t) (1/µm)", ylim=(-0.25 * _ymax, _ymax),
            title=f"Propagator at t = {_t:.3g} ms")
    _a2.legend(frameon=False, fontsize=8, loc="upper right")
    _fig.tight_layout()

    _n_needed = int(np.ceil(_L / np.pi * np.sqrt(5 / (_D * _t))))
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Spread so far √(2Dt) = **{np.sqrt(2 * _D * _t):.2f} µm** against L = {_L} µm. "
                f"Modes with Dλₙt < 5 still matter at this time: about **{_n_needed}** of them."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Predict first: for $L = 10$ µm and $D = 2$ µm²/ms, about how many modes do you need for a smooth propagator at $t = 0.1$ ms? And at $t = 10$ ms? Check with the sliders.
    **(b)** At short times the walls have not been felt yet. Which curve should the propagator then match?
    **(c)** What does this tell you about when eigenmode methods are cheap, and when they are expensive?
    """, "3 · How many modes?")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Mode $n$ still matters while $e^{-D\lambda_n t}$ is not negligible, say $D\lambda_n t \lesssim 5$ ($e^{-5} < 1\%$).
    ⇒ With $\lambda_n = (n\pi/L)^2$: $n \lesssim \dfrac{L}{\pi}\sqrt{\dfrac{5}{Dt}}$.
    ⇒ $t = 0.1$ ms: $n \approx \frac{10}{\pi}\sqrt{25} \approx 16$. $\ t = 10$ ms: $n \approx \frac{10}{\pi}\sqrt{0.25} \approx 2$.

    **(b)** The free Gaussian of Exercise 1, centred at $x_0$, as long as $\sqrt{2Dt}$ is small compared with the distance from $x_0$ to the nearest wall. Move $x_0$ close to a wall and you see the reflected part appear first there.

    **(c)** At long times only a handful of modes survive, so eigenmode methods are cheap exactly where diffusion has explored the whole pore. At short times you need many modes to build a narrow peak. There, a free Gaussian (or a random walk) is the natural description. You will see this trade-off again in Chapters 3 and 5.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Spins start **uniformly** in the slab. Using the propagator, show that the mean-squared displacement is

    $$\langle (x - x_0)^2\rangle(t) = \frac{L^2}{6} - \frac{16L^2}{\pi^4}\sum_{n\ \text{odd}}\frac{e^{-D(n\pi/L)^2 t}}{n^4},$$

    and that it tends to $L^2/6$ at long times. (In the random-walk notebook the walkers between two walls levelled off at $L^2/6$. Here is where that number comes from.)
    """, "4 · Mean-squared displacement in a slab")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - Expand the square: $\langle(x-x_0)^2\rangle = \langle x^2\rangle + \langle x_0^2\rangle - 2\langle x\,x_0\rangle$.
    - Uniform start and reflecting walls mean the walkers stay uniform at all times, so $\langle x^2\rangle = \langle x_0^2\rangle = L^2/3$.
    - $\langle x\,x_0\rangle = \frac1L\iint x\,x_0\,P(x|x_0,t)\,dx\,dx_0$. Insert the mode sum: each mode contributes $\frac1L\big(\int_0^L x\,u_n\,dx\big)^2 e^{-D\lambda_n t}$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    ⇒ $\langle x^2\rangle = \langle x_0^2\rangle = \frac1L\int_0^L x^2dx = \frac{L^2}{3}$.

    ⇒ $\langle x\,x_0\rangle = \frac1L\sum_n e^{-D\lambda_n t}\left(\int_0^L x\,u_n(x)\,dx\right)^2$.

    ⇒ $n=0$: $\int_0^L x/\sqrt L\,dx = L^{3/2}/2$, giving $\frac1L\cdot\frac{L^3}{4} = \frac{L^2}{4}$.

    ⇒ $n\ge1$: integrate by parts, $\int_0^L x\cos\frac{n\pi x}{L}dx = \frac{L^2}{(n\pi)^2}\big[(-1)^n - 1\big]$. This is zero for even $n$ and $-2L^2/(n\pi)^2$ for odd $n$. With the factor $\sqrt{2/L}$ squared: $\frac1L\cdot\frac2L\cdot\frac{4L^4}{(n\pi)^4} = \frac{8L^2}{(n\pi)^4}$.

    ⇒ Put together:

    $$\langle(x-x_0)^2\rangle = \frac{2L^2}{3} - 2\left[\frac{L^2}{4} + \sum_{n\ \text{odd}}\frac{8L^2}{(n\pi)^4}e^{-D\lambda_n t}\right] = \frac{L^2}{6} - \frac{16L^2}{\pi^4}\sum_{n\ \text{odd}}\frac{e^{-D\lambda_n t}}{n^4}.$$

    → Check $t=0$: $\sum_{n\ \text{odd}} n^{-4} = \pi^4/96$, so the MSD is $L^2/6 - L^2/6 = 0$. ✓
    → $t\to\infty$: all exponentials vanish and the MSD is $L^2/6$: start and end positions have become independent, each uniform in the slab.
    → The weights $8L^2/(n\pi)^4$ fall as $n^{-4}$: the first mode carries 98.6% of the total. Remember this. In Chapter 2 these same weights set the signal in the Gaussian phase approximation.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D1, L1, mo, np, plt):
    _L, _D = L1.value, D1.value
    _t = np.logspace(-2, 2.3, 400)
    _n = np.arange(1, 400, 2)[:, None]
    _msd = _L**2 / 6 - 16 * _L**2 / np.pi**4 * np.sum(np.exp(-_D * (_n * np.pi / _L) ** 2 * _t) / _n**4, axis=0)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a1.loglog(_t, 2 * _D * _t, color=C_FREE, ls="--", label="free: 2Dt")
    _a1.loglog(_t, _msd, color=C_MAIN, lw=2, label="slab, from the modes")
    _a1.axhline(_L**2 / 6, color=C_ALT, lw=1, ls=":", label="L²/6")
    _a1.set(xlabel="t (ms)", ylabel="⟨(x − x₀)²⟩ (µm²)", title="Mean-squared displacement")
    _a1.legend(frameon=False, fontsize=8)

    _a2.semilogx(_t, _msd / (2 * _t), color=C_MAIN, lw=2, label="⟨(x − x₀)²⟩ / 2t")
    _a2.axhline(_D, color=C_FREE, ls="--", label="free D")
    _a2.axvline(_L**2 / (np.pi**2 * _D), color="k", lw=0.8, ls=":")
    _a2.text(_L**2 / (np.pi**2 * _D), _D * 0.05, " τ₁", fontsize=9)
    _a2.set(xlabel="t (ms)", ylabel="apparent D (µm²/ms)", ylim=(0, _D * 1.1), title="The apparent diffusivity falls once walls are felt")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Uses L and D from the sliders above. τ₁ = L²/(π²D) = **{_L**2 / (np.pi**2 * _D):.2f} ms**.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other geometries

    The recipe is always the same: solve $-\nabla^2 u = \lambda u$ with $\partial u/\partial n = 0$ on the walls. In a disk or sphere, separate in polar or spherical coordinates. The radial part becomes a Bessel function, and the wall condition picks out the roots $\alpha$ of its **derivative**.

    | Geometry | Eigenmodes | Wall condition fixes | Eigenvalues | Radial normalisation | First non-zero root | Slowest time $\tau_1$ |
    | --- | --- | --- | --- | --- | --- | --- |
    | Slab $0\le x\le L$ | $\cos\dfrac{n\pi x}{L}$ | $\sin(n\pi)=0$ | $\left(\dfrac{n\pi}{L}\right)^2$ | $\int_0^L\cos^2 = L/2$ | $\pi$ (with $L$) | $\dfrac{L^2}{\pi^2 D}=0.101\,\dfrac{L^2}{D}$ |
    | Cylinder, radius $R$ (cross-section) | $J_m\!\left(\dfrac{\alpha_{mk}r}{R}\right)\cos m\theta,\ \sin m\theta$ | $J_m'(\alpha_{mk})=0$ | $\left(\dfrac{\alpha_{mk}}{R}\right)^2$ | $\int_0^R J_m^2\,r\,dr = \dfrac{R^2}{2}\left(1-\dfrac{m^2}{\alpha^2}\right)J_m(\alpha)^2$ | $\alpha_{11} = 1.8412$ | $\dfrac{R^2}{1.8412^2 D}=0.295\,\dfrac{R^2}{D}$ |
    | Cylindrical surface, radius $r$ (the circle across the axis) | $\cos n\phi,\ \sin n\phi$ | periodic in $\phi$: there is no wall | $\dfrac{n^2}{r^2}$ | $\int_0^{2\pi r}\cos^2 n\phi\,ds = \pi r$ | $n = 1$ | $\dfrac{r^2}{D}$ |
    | Sphere, radius $R$ | $j_l\!\left(\dfrac{\alpha_{lk}r}{R}\right)Y_{lm}(\theta,\phi)$ | $j_l'(\alpha_{lk})=0$ | $\left(\dfrac{\alpha_{lk}}{R}\right)^2$ | $\int_0^R j_l^2\,r^2dr = \dfrac{R^3}{2}\left(1-\dfrac{l(l+1)}{\alpha^2}\right)j_l(\alpha)^2$ | $\alpha_{11} = 2.0816$ | $\dfrac{R^2}{2.0816^2 D}=0.231\,\dfrac{R^2}{D}$ |

    $J_m$ are Bessel functions, $j_l$ spherical Bessel functions, $Y_{lm}$ spherical harmonics. Along a cylinder's axis diffusion is free, so a cylinder is a disk in the cross-section times free diffusion along $z$.

    → The slowest non-uniform mode of a disk is not the radially symmetric one ($m = 0$, $\alpha = 3.83$) but the one with one sign change across the disk ($m = 1$, $\alpha = 1.84$). The same holds for the sphere ($l=1$). These modes look like the slab's $\cos(\pi x/L)$: high on one side, low on the other.

    The roots below are computed in this notebook; the table quotes them.
    """)
    return


@app.cell(hide_code=True)
def _(brentq, jn_zeros, jnp_zeros, mo, np, spherical_jn):
    def _sph_roots(l, k_max=3):
        f = lambda x: spherical_jn(l, x, derivative=True)
        grid = np.linspace(0.1, 20, 4000)
        vals = f(grid)
        roots = [brentq(f, a, b) for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]) if va * vb < 0]
        return roots[:k_max]

    _cyl = {0: [0.0] + list(jn_zeros(1, 2))}  # J0' = -J1, so the roots of J0' are the zeros of J1 (plus 0)
    for _m in (1, 2, 3):
        _cyl[_m] = list(jnp_zeros(_m, 3))
    _sph = {0: [0.0] + _sph_roots(0, 2), 1: _sph_roots(1), 2: _sph_roots(2), 3: _sph_roots(3)}

    _rows = "\n".join(
        f"| {_m} | " + ", ".join(f"{a:.4f}" for a in _cyl[_m][:3]) + " | " + ", ".join(f"{a:.4f}" for a in _sph[_m][:3]) + " |"
        for _m in (0, 1, 2, 3)
    )
    mo.md(
        "**Roots α of the derivative** (eigenvalue = (α/R)²)\n\n"
        "| m or l | cylinder: J′ₘ(α) = 0 | sphere: j′ₗ(α) = 0 |\n| --- | --- | --- |\n" + _rows
    )
    return


@app.cell(hide_code=True)
def _(jn_zeros, jnp_zeros, jv, np, plt):
    _r = np.linspace(0, 1, 160)
    _th = np.linspace(0, 2 * np.pi, 240)
    _RR, _TT = np.meshgrid(_r, _th)
    _modes = [(0, 0.0), (1, jnp_zeros(1, 1)[0]), (2, jnp_zeros(2, 1)[0]),
              (0, jn_zeros(1, 1)[0]), (3, jnp_zeros(3, 1)[0]), (1, jnp_zeros(1, 2)[1])]
    _fig, _axes = plt.subplots(1, 6, figsize=(11, 2.2), subplot_kw={"projection": "polar"})
    for _ax, (_m, _a) in zip(_axes, _modes):
        _Z = jv(_m, _a * _RR) * np.cos(_m * _TT) if _a > 0 else np.ones_like(_RR)
        _v = np.abs(_Z).max()
        _ax.pcolormesh(_TT, _RR, _Z, cmap="RdBu_r", vmin=-_v, vmax=_v, shading="auto")
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.grid(False)
        _ax.set_title(f"m = {_m}, α = {_a:.2f}", fontsize=8.5)
    _fig.suptitle("Disk eigenmodes (cylinder cross-section), sorted by α: red and blue have opposite sign", fontsize=9.5, y=1.04)
    _fig
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Take $D = 2$ µm²/ms. Compute the slowest decay time $\tau_1$ for
    (i) a slab with $L = 8$ µm, (ii) a cylinder with **radius** $R = 5$ µm, (iii) a sphere with radius $R = 5$ µm.

    A PGSE experiment uses $\Delta = 20$ ms. In which pores has the magnetization explored the whole pore by then? What does that mean for what the signal can tell you about pore *size*?
    """, "5 · Time scales of three pores")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    ⇒ Slab: $\tau_1 = L^2/(\pi^2 D) = 64/(9.87\times 2) = $ **3.2 ms**.
    ⇒ Cylinder: $\tau_1 = R^2/(\alpha_{11}^2 D) = 25/(3.390\times 2) = $ **3.7 ms**.
    ⇒ Sphere: $\tau_1 = R^2/(\alpha_{11}^2 D) = 25/(4.333\times 2) = $ **2.9 ms**.

    → All three are much shorter than 20 ms: after a few $\tau_1$ every non-uniform mode has faded and the spins have explored the whole pore. The signal is then in the long-time (**fully restricted**, or motional-narrowing) regime. It still depends on size, but only through a few numbers such as $\langle x^2\rangle$ in the pore, not through details of the shape.

    → Watch out for radius versus diameter: a "10 µm cell" has $R = 5$ µm. Using the diameter in $R^2$ overestimates $\tau_1$ fourfold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    t_check = mo.ui.radio(
        options=["It doubles", "It grows 4×", "It stays the same", "It grows √2×"],
        label="**Predict:** you double the size of a pore. What happens to its slowest decay time τ₁?",
    )
    t_check
    return (t_check,)


@app.cell(hide_code=True)
def _(mo, t_check):
    if t_check.value is None:
        _out = mo.md("")
    elif t_check.value == "It grows 4×":
        _out = mo.callout(mo.md("**Yes.** τ₁ ∝ L²/D: diffusion covers a distance ℓ in a time ∝ ℓ². Twice the size needs four times as long."), kind="success")
    else:
        _out = mo.callout(mo.md("**Not quite.** Look at the table: every τ₁ is (size)²/D times a number. Check it with the L slider above."), kind="warn")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A surface instead of a volume: cylindrical surfaces and concentric layers

    Water trapped between the wraps of myelin moves along thin sheets. [Canales-Rodríguez, Tax et al. (2025)](https://doi.org/10.3389/fphy.2025.1516630) model each sheet as the **surface of a cylinder** of radius $r$, and the myelin sheath as a stack of **impermeable concentric cylinders** separated by very thin layers of water. Along the axis the motion is free. Across it, a spin moves on a circle.

    **Modes on the circle.** With arc length $s = r\phi$, the eigenproblem is $-u'' = \lambda u$ with $u$ periodic: a circle has no wall, so periodicity replaces the wall condition.

    $$u_0 = \frac{1}{\sqrt{2\pi r}},\qquad u_n = \frac{\cos n\phi}{\sqrt{\pi r}},\ \frac{\sin n\phi}{\sqrt{\pi r}},\qquad \lambda_n = \frac{n^2}{r^2}\quad(n\ge1,\ \text{each twice}).$$

    The slowest decay time is $\tau_1 = r^2/D$, about 3.4 times longer than for the inside of a disk of the same radius ($0.295\,R^2/D$): a spin on the rim cannot cut across, it has to go around.

    **Concentric layers.** The walls between layers are impermeable, so the modes of the whole sheath are simply the modes of each circle, and every layer keeps its own uniform mode (the $\kappa = 0$ case of the next section). A uniform start puts in layer $i$ a share of spins proportional to its circumference:

    $$E = \sum_i w_i\,E(r_i),\qquad w_i = \frac{r_i}{\sum_j r_j}.$$

    The layers are evenly spaced between the inner radius (the axon) and the outer radius; in the 2025 simulations they are 7.5 nm apart, roughly a 4–5 nm lipid bilayer plus about 3 nm of water. Outer layers count more, so a single-cylinder fit returns an effective radius biased towards the outer radius. The 2025 paper approximates it by the radius-weighted mean $\sum_i w_i r_i = \sum_i r_i^2/\sum_i r_i$. [Canales-Rodríguez, Tax et al. (2026)](https://doi.org/10.1088/1361-6560/aeb2a0) ([arXiv 2603.23421](https://arxiv.org/abs/2603.23421)) then derived the exact PGSE signal for one cylindrical surface from exactly these modes, with the matrix method of Chapter 3. They name distributions of concentric radii as a natural extension.

    → A layer of finite thickness is an annulus between two radii. Its modes combine the Bessel functions $J_m$ and $Y_m$, with the wall condition at both radii; the cylindrical surface is the limit of a very thin annulus.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rout5 = mo.ui.slider(0.5, 5.0, step=0.1, value=2.0, label="Outer radius r_out (µm)", show_value=True)
    gratio5 = mo.ui.slider(0.5, 0.9, step=0.01, value=0.7, label="g-ratio r_in / r_out", show_value=True)
    nlay5 = mo.ui.slider(2, 120, step=1, value=10, label="Number of layers", show_value=True)
    mo.hstack([rout5, gratio5, nlay5], justify="start", gap=2, wrap=True)
    return gratio5, nlay5, rout5


@app.cell(hide_code=True)
def _(C_ALT, C_MAIN, gratio5, mo, nlay5, np, plt, rout5):
    _D = 0.8  # µm²/ms, an illustrative myelin-water diffusivity
    _rout, _g, _N = rout5.value, gratio5.value, nlay5.value
    _rin = _g * _rout
    _r = np.linspace(_rin, _rout, _N)
    _w = _r / _r.sum()
    _tau = _r**2 / _D

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    _phi = np.linspace(0, 2 * np.pi, 300)
    _a1.fill(_rin * np.cos(_phi), _rin * np.sin(_phi), color="0.9")
    for _ri in _r:
        _a1.plot(_ri * np.cos(_phi), _ri * np.sin(_phi), color=C_MAIN, lw=1)
    _a1.text(0, 0.4 * _rin, "axon", ha="center", va="center", fontsize=9)
    _a1.annotate("", xy=(_rout, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1))
    _a1.text(_rout / 2, -0.08 * _rout, "r_out", ha="center", va="top", fontsize=8)
    _a1.set(aspect="equal", xlim=(-1.1 * _rout, 1.1 * _rout), ylim=(-1.1 * _rout, 1.1 * _rout),
            title=f"{_N} myelin layers, r_in = {_rin:.2f} µm")
    _a1.axis("off")

    _a2.bar(_r, _w, width=0.7 * (_rout - _rin) / max(_N - 1, 1), color=C_MAIN, alpha=0.7, label="weight w_i = r_i / Σ r_j")
    _a2.set(xlabel="layer radius r_i (µm)", ylabel="weight in the signal", title="Outer layers weigh more, and are slower")
    _t2 = _a2.twinx()
    _t2.plot(_r, _tau, "o-", color=C_ALT, ms=3, label="τ₁ = r²/D")
    _t2.set_ylabel("slowest decay time τ₁ (ms)", color=C_ALT)
    _t2.grid(False)
    _h1, _l1 = _a2.get_legend_handles_labels()
    _h2, _l2 = _t2.get_legend_handles_labels()
    _a2.legend(_h1 + _h2, _l1 + _l2, frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()

    _rw = np.sum(_w * _r)
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Radius-weighted mean Σwᵢrᵢ (the 2025 effective-radius approximation) = **{_rw:.3f} µm**, against the plain mean (r_in + r_out)/2 = "
                f"{(_rin + _rout) / 2:.3f} µm. τ₁ runs from {_tau[0]:.2f} to {_tau[-1]:.2f} ms with D = {_D} µm²/ms. "
                f"With 7.5 nm between layers, this sheath would hold about **{int(round((_rout - _rin) / 0.0075)) + 1} layers**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
        **(a)** Derive the modes of a circle of radius $r$: solve $-u''(s) = \lambda u(s)$ on $0 \le s < 2\pi r$ with $u$ periodic, and normalise.
        **(b)** Spins are spread uniformly over a stack of concentric cylindrical surfaces of equal length. Show that layer $i$ contributes with weight $w_i = r_i/\sum_j r_j$.
        **(c)** For $r = 1$ µm and $D = 0.8$ µm²/ms, compare $\tau_1$ on the surface with $\tau_1$ inside a solid cylinder of the same radius.
        """, "6 · Modes on a cylindrical surface")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ $\lambda<0$: $\cosh$ and $\sinh$ are not periodic, so no solutions.
        ⇒ $\lambda = 0$: $u = A + Bs$ is periodic only if $B = 0$: the uniform mode.
        ⇒ $\lambda = k^2 > 0$: $u = A\cos ks + B\sin ks$. Periodicity, $u(s+2\pi r) = u(s)$, needs $2\pi r k = 2\pi n$, so $k = n/r$ and $\lambda_n = n^2/r^2$. Both $\cos$ and $\sin$ survive: each eigenvalue appears twice (unlike the slab, where the wall condition removed the sine).
        ⇒ Normalise: $\int_0^{2\pi r}\cos^2(ns/r)\,ds = \pi r$ and $\int_0^{2\pi r}1\,ds = 2\pi r$, so $u_0 = 1/\sqrt{2\pi r}$ and $u_n = \cos(n\phi)/\sqrt{\pi r}$, $\sin(n\phi)/\sqrt{\pi r}$.

        **(b)** ⇒ The number of spins in a layer is density × area, and the area of a cylindrical surface of length $\ell$ is $2\pi r_i\ell$. With equal density and length, layer $i$ holds a fraction $2\pi r_i\ell/\sum_j 2\pi r_j\ell = r_i/\sum_j r_j$.
        ⇒ The layers do not exchange spins, so the total signal is the weighted sum of the layer signals.

        **(c)** ⇒ Surface: $\tau_1 = r^2/D = 1/0.8 = 1.25$ ms. Solid cylinder: $\tau_1 = R^2/(1.8412^2 D) = 1/(3.39\times0.8) = 0.37$ ms.
        → The surface is 3.4 times slower to "forget" where a spin started. In the long-time limit the variance of $x$ is $r^2/2$ on the surface and $R^2/4$ inside the cylinder, so the surface also looks larger. Both effects make myelin water distinguishable from intra-axonal water, in principle.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When the interfaces are not reflecting

    Everything so far assumed impermeable walls. The recipe for the eigenmodes stays the same, $-\nabla^2u = \lambda u$, but the condition at the interfaces changes, and with it the modes:

    - **Permeable membranes.** The compartments are no longer separate problems. The eigenmodes belong to the **whole domain**: each mode spans both sides of a membrane, with a continuous flux and a jump in value set by $\kappa$. With $\kappa = 0$ every compartment has its own uniform mode ($\lambda = 0$ once per compartment). Opening the membrane turns all but one of these into slowly decaying **exchange modes**. Their rate is the exchange rate between compartments, the basis of the [Kärger (1985)](<https://doi.org/10.1016/0001-8686(85)80018-X>) model (Chapter 2).
    - **Relaxing walls.** The lowest eigenvalue is no longer zero. Even the most uniform mode fades: magnetization is lost at the walls. For weak relaxivity its rate is $D\lambda_0 \approx \varrho\,S/V$, the surface-to-volume ratio times the relaxivity ([Brownstein & Tarr 1979](https://doi.org/10.1103/PhysRevA.19.2446)).
    - **Perfect sinks.** All modes vanish at the wall, and the slowest one fades with $\lambda_0 = (\pi/L)^2$ in a slab: nothing survives.
    - **Periodic boundaries.** The modes become $\cos(2\pi n x/L)$ and $\sin(2\pi n x/L)$, each eigenvalue appearing twice. The uniform mode survives: nothing is lost, it just leaves on one side and re-enters on the other.

    The modes remain orthonormal in all these cases, so everything in the previous sections (projection, propagator, time scales) carries over unchanged.

    Try it in a slab of $L = 8$ µm with $D = 2$ µm²/ms, a membrane in the middle, and a choice of outer walls. The modes are computed numerically (a fine grid with the interface rules built in), so you can compare them with the formulas of the next exercise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    outer3 = mo.ui.dropdown(
        options=["reflecting", "relaxing (Robin)", "perfect sink (Dirichlet)", "periodic"],
        value="reflecting",
        label="Outer walls",
    )
    logrho3 = mo.ui.slider(-3, 1, step=0.1, value=-1, label="log₁₀ ϱ (ϱ in µm/ms, relaxing walls only)", show_value=True)
    membrane3 = mo.ui.switch(label="Membrane in the middle", value=True)
    logkappa3 = mo.ui.slider(-3, 2, step=0.1, value=-1.5, label="log₁₀ κ (κ in µm/ms)", show_value=True)
    mo.vstack(
        [
            mo.hstack([outer3, logrho3], justify="start", gap=2, wrap=True),
            mo.hstack([membrane3, logkappa3], justify="start", gap=2, wrap=True),
        ]
    )
    return logkappa3, logrho3, membrane3, outer3


@app.cell(hide_code=True)
def _(np):
    def fv_modes(L, D, outer="reflecting", rho=0.0, kappa=np.inf, n=400, k=6):
        """Lowest k modes of -d/dx(D d/dx) on [0, L] with n cells, an optional membrane at L/2
        (permeability kappa) and the chosen outer walls. Returns cell centres, eigenvalues lambda (1/µm²)
        and modes normalised to unit L2 norm."""
        h = L / n
        A = np.zeros((n, n))
        c = np.full(n - 1, D / h**2)
        if np.isfinite(kappa):
            # membrane face: the two half cells and the membrane act as resistances in series
            c[n // 2 - 1] = 1.0 / (h * (h / D + 1.0 / kappa)) if kappa > 0 else 0.0
        for i, ci in enumerate(c):
            A[i, i] += ci
            A[i + 1, i + 1] += ci
            A[i, i + 1] -= ci
            A[i + 1, i] -= ci
        if outer == "relaxing (Robin)" and rho > 0:
            w = 1.0 / (h * (h / (2 * D) + 1.0 / rho))
            A[0, 0] += w
            A[-1, -1] += w
        elif outer == "perfect sink (Dirichlet)":
            A[0, 0] += 2 * D / h**2
            A[-1, -1] += 2 * D / h**2
        elif outer == "periodic":
            A[0, 0] += D / h**2
            A[-1, -1] += D / h**2
            A[0, -1] -= D / h**2
            A[-1, 0] -= D / h**2
        w, V = np.linalg.eigh(A)
        V = V[:, :k] / np.sqrt(h)
        V = V * np.sign(V[np.argmax(np.abs(V) > 1e-6 * np.abs(V).max(axis=0), axis=0), np.arange(k)])
        return (np.arange(n) + 0.5) * h, w[:k] / D, V

    return (fv_modes,)


@app.cell(hide_code=True)
def _(C_FREE, fv_modes, logkappa3, logrho3, membrane3, mo, np, outer3, plt):
    _L, _D = 8.0, 2.0
    _kappa = 10 ** logkappa3.value if membrane3.value else np.inf
    _rho = 10 ** logrho3.value
    _x, _lam, _V = fv_modes(_L, _D, outer3.value, _rho, _kappa)
    _ref = (np.arange(6) * np.pi / _L) ** 2

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for _j in range(4):
        _a1.plot(_x, _V[:, _j] * np.sqrt(_L), lw=2 if _j == 1 else 1.3, label=f"mode {_j}: Dλ = {abs(_D * _lam[_j]):.4f} /ms")
    if membrane3.value:
        _a1.axvline(_L / 2, color="k", lw=2, alpha=0.6)
        _a1.text(_L / 2, 1.85, f" membrane, κ = {_kappa:.3g} µm/ms", fontsize=8)
    _a1.set(xlabel="x (µm)", ylabel="mode × √L", ylim=(-2.1, 2.1), title=f"Lowest modes · outer walls: {outer3.value}")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower left")

    _idx = np.arange(6)
    _a2.bar(_idx, _D * _lam, color="#2a6fdb", alpha=0.7, label="this case")
    _a2.plot(_idx, _D * _ref, "o", color=C_FREE, ms=6, label="reflecting slab, no membrane")
    _a2.set(xlabel="mode number", ylabel="decay rate Dλ (1/ms)", title="Decay rates")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()

    _notes = []
    if _lam[0] * _D < 1e-8:
        _notes.append("The lowest mode does not decay: no magnetization is lost.")
    else:
        _notes.append(f"Even the lowest mode decays, at **{_D * _lam[0]:.4f} /ms** (decay time {1 / (_D * _lam[0]):.1f} ms): magnetization is lost at the walls.")
        if outer3.value == "relaxing (Robin)":
            _notes.append(f"Compare ϱ·S/V = 2ϱ/L = **{2 * _rho / _L:.4f} /ms**, the weak-relaxivity estimate.")
    if membrane3.value and outer3.value == "reflecting":
        _notes.append(f"Mode 1 is the **exchange mode**: one half high, the other low. Its rate {_D * _lam[1]:.4f} /ms "
                      f"compares with 4κ/L = {4 * _kappa / _L:.4f} /ms, the exchange rate between the halves (valid for small κ).")
    mo.vstack([_fig, mo.md(" ".join(_notes))])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
        A slab $-a \le x \le a$ (so $L = 2a$) with reflecting outer walls and a permeable membrane at $x = 0$, the same $D$ on both sides.

        **(a)** Explain why every mode is either even or odd in $x$, and show that the even modes do not notice the membrane.
        **(b)** For the odd modes, try $u = \cos k(a - x)$ for $x>0$ and $u = -\cos k(a + x)$ for $x<0$. Use the membrane condition to show that

        $$k\tan(ka) = \frac{2\kappa}{D}.$$

        **(c)** Check the limits $\kappa\to 0$ and $\kappa\to\infty$ against what you expect.
        **(d)** For small $\kappa$, show that the slowest odd mode decays at the rate $D\lambda \approx 4\kappa/L$, and interpret it.
        """, "7 · Eigenmodes with a permeable membrane")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
        - (a) The problem looks the same after $x\to -x$. An even mode has $u'(0) = 0$ from both sides and no jump: the membrane carries no flux and sees no jump.
        - (b) Both trial functions already satisfy the outer wall conditions $u'(\pm a) = 0$. The flux at the membrane is $-D\,u'(0)$; the jump is $u(0^+) - u(0^-)$.
        - (d) For small $ka$, $\tan(ka)\approx ka$.
        """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ Reflecting $x\to-x$ maps the problem onto itself, so (for non-degenerate eigenvalues) every mode is even or odd. An even mode has $u'(0^+) = -u'(0^-)$; flux continuity then forces $u'(0) = 0$, and the membrane condition with zero flux forces no jump. Even modes are the reflecting-slab modes $\cos(kx)$ with $ka = n\pi$, whatever $\kappa$ is.

        **(b)** ⇒ For $x > 0$: $u = \cos k(a-x)$, $u' = k\sin k(a-x)$, so $u'(a) = 0$ ✓ and $u'(0^+) = k\sin ka$. The odd partner has the same slope at 0, so the flux is continuous ✓.
        ⇒ The jump is $u(0^+) - u(0^-) = 2\cos ka$.
        ⇒ The flux from left to right, $-D\,u'(0)$, must equal $\kappa\,[u(0^-) - u(0^+)]$:

        $$-Dk\sin ka = -2\kappa\cos ka \quad\Longrightarrow\quad k\tan(ka) = \frac{2\kappa}{D}.$$

        **(c)** ⇒ $\kappa\to0$: $\tan ka = 0$, so $ka = n\pi$, including $k = 0$. The odd mode with $k=0$ is "$+1$ on the right, $-1$ on the left": together with the even uniform mode, each half has its own uniform mode. Two separate compartments. ✓
        ⇒ $\kappa\to\infty$: $\cos ka = 0$, so $ka = (n+\tfrac12)\pi$, i.e. $k = (2n+1)\pi/L$: the odd modes of a single slab without membrane. ✓

        **(d)** ⇒ For small $\kappa$ the lowest odd root has small $ka$: $k\cdot ka \approx 2\kappa/D$, so $\lambda = k^2 \approx 2\kappa/(Da)$ and

        $$D\lambda \approx \frac{2\kappa}{a} = \frac{4\kappa}{L}.$$

        → Interpretation: a spin in one half crosses at a rate $\kappa\,S/V = \kappa/a$. The difference between the two halves relaxes at the sum of the two crossing rates, $2\kappa/a$. That is exactly the exchange rate of a two-compartment (Kärger) model. With $\kappa = 0.03$ µm/ms and $L = 8$ µm it is 0.015 /ms: an exchange time of about 70 ms. Check it with the sliders.
        → The same steps with a relaxing wall instead of a membrane give $k\tan(kL/2) = \varrho/D$ for the lowest mode, and $D\lambda_0 \approx 2\varrho/L = \varrho\,S/V$ for weak relaxivity.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 1b · The Bloch–Torrey equation and the expected value of the phase

    ## From Bloch to Bloch–Torrey

    After the 90° pulse the magnetization lies in the transverse plane. Write it as one complex number, $m = M_x + iM_y$. In the frame rotating at the Larmor frequency, the only field left is the diffusion gradient's, $\mathbf g(t)\cdot\mathbf r$. A spin at $\mathbf r$ therefore precesses at angular frequency $\gamma\,\mathbf g(t)\cdot\mathbf r$, which on its own gives $\partial m/\partial t = -i\gamma\,\mathbf g\cdot\mathbf r\, m$.

    The spins also diffuse, and magnetization is carried with them exactly like the concentration in 1a. [Torrey (1956)](https://doi.org/10.1103/PhysRev.104.563) added the two effects to [Bloch's (1946)](https://doi.org/10.1103/PhysRev.70.460) equations:

    $$\boxed{\ \frac{\partial m}{\partial t} = D\,\nabla^2 m \;-\; i\gamma\,\mathbf g(t)\cdot\mathbf r\; m\ }\qquad m(\mathbf r,0) = \rho,\qquad \frac{\partial m}{\partial n}=0 \text{ on impermeable interfaces}.$$

    - **Diffusion** smooths phase differences out; the **gradient** winds them up. Every regime in this course is a contest between those two.
    - The **180° pulse** of the spin echo ([Hahn 1950](https://doi.org/10.1103/PhysRev.80.580)) flips the sign of all phase gathered so far. We account for it by using the **effective gradient**, whose sign flips at each 180° pulse.
    - The scanner measures the sum over the voxel: $S(T) = \int_\Omega m(\mathbf r,T)\,d\mathbf r$, and we report $E = S/S_0$.
    - The domain, interface and boundary conditions are the same as in 1a. So the **geometry enters exactly as before**: through the eigenmodes. The gradient adds a coupling between them (Chapter 3).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    With transverse relaxation the equation reads $\partial m/\partial t = D\nabla^2 m - i\gamma\,\mathbf g\cdot\mathbf r\,m - m/T_2$.
    Show that, for a uniform $T_2$, relaxation only multiplies the signal by $e^{-T/T_2}$, so that it cancels in $E = S/S_0$ when $S_0$ is measured at the same echo time.
    """, "8 · Where did T₂ go?")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    ⇒ Substitute $m = e^{-t/T_2}\,\tilde m$. Then $\partial m/\partial t = e^{-t/T_2}\left(\partial\tilde m/\partial t - \tilde m/T_2\right)$.

    ⇒ The $-\tilde m/T_2$ terms on both sides cancel, leaving $\partial\tilde m/\partial t = D\nabla^2\tilde m - i\gamma\,\mathbf g\cdot\mathbf r\,\tilde m$: the equation without relaxation.

    ⇒ So $S(T) = e^{-T/T_2}\tilde S(T)$, and $S_0$ (no gradient, same $T$) carries the same factor. It cancels in the ratio.

    → This fails when $T_2$ differs between compartments: then each compartment is weighted by its own $e^{-T/T_{2,n}}$, which is how echo time changes the apparent volume fractions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Length scales: who wins, diffusion, the gradient or the walls?

    Bloch–Torrey contains a handful of lengths. Which regime you are in, and so which approximation or simulation method works, depends on which length is the **shortest**.

    | Length | Formula | What it measures |
    | --- | --- | --- |
    | Pore size | $L$ (slab width) or $R$ (radius) | where the walls are |
    | Diffusion length | $\ell_D = \sqrt{DT}$ | how far a spin wanders during the experiment |
    | Phase length | $\ell_q = 1/(\gamma g T)$ (or $1/q$ for pulses) | over which distance the gradient has wound one radian of phase |
    | Gradient (dephasing) length | $\ell_g = \left(D/\gamma g\right)^{1/3}$ | the distance over which diffusion and dephasing balance |
    | Surface-relaxation length | $\ell_\varrho = D/\varrho$ | how far a spin travels along a relaxing wall before it loses its magnetization |
    | $T_2$ diffusion length | $\sqrt{D\,T_2}$ | how far a spin travels before bulk $T_2$ relaxation has removed it |

    [Grebenkov (2007)](https://doi.org/10.1103/RevModPhys.79.1077) lists the first four as $L$, $L_p = \sqrt{DT}$, $L_q$ and the relaxation length $L_h = D/\varrho$. The gradient length $\ell_g$ is the one that sets the strong-gradient regime ([Stoller, Happer & Dyson 1991](https://doi.org/10.1103/PhysRevA.44.7459); [de Swiet & Sen 1994](https://doi.org/10.1063/1.467127)). The three are not independent: $\ell_g^3 = \ell_D^2\,\ell_q$.

    | Shortest length | Regime | What the signal does |
    | --- | --- | --- |
    | $\ell_D$ | free diffusion, or short times | $E \approx e^{-bD}$; walls only add a correction $\propto (S/V)\sqrt{DT}$ |
    | $L$ | motional narrowing (fully restricted) | spins average over the pore; slab: $\ln E \approx -\gamma^2g^2L^4T/(120D)$ ([Robertson 1966](https://doi.org/10.1103/PhysRev.151.273)) |
    | $\ell_g$ | localisation | magnetization survives only in layers of thickness $\sim\ell_g$ at the walls; $\ln E \propto -DT/\ell_g^2 \propto -g^{2/3}T$, not $-g^2$ |

    For relaxing walls compare $\ell_\varrho$ with $L$. If $\ell_\varrho \gg L$ (the usual case in tissue), spins visit the walls many times before relaxing: the loss is uniform, at rate $\varrho\,S/V$ (the **fast-diffusion** regime, [Brownstein & Tarr 1979](https://doi.org/10.1103/PhysRevA.19.2446)). If $\ell_\varrho \ll L$, relaxation is limited by how fast spins reach the wall.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ In a time $t$ a spin diffuses a distance $\ell \approx \sqrt{Dt}$.

    ⇒ Two spins a distance $\ell$ apart precess at frequencies that differ by $\gamma g\ell$, so in time $t$ their phases differ by $\gamma g\ell t$.

    ⇒ Diffusion and dephasing balance when the spin's own wandering spreads it over the distance that carries one radian of phase difference: $\ell = \sqrt{Dt}$ and $\gamma g\ell t = 1$.

    ⇒ From the second, $t = 1/(\gamma g\ell)$. Put it in the first: $\ell^2 = D/(\gamma g\ell)$, so

    $$\ell_g = \left(\frac{D}{\gamma g}\right)^{1/3},\qquad \tau_g = \frac{\ell_g^2}{D}.$$

    → Closer to a wall than $\ell_g$, a spin is held back by the wall before it can dephase much: that is why magnetization survives there in the localisation regime.
    """, title="Where ℓ_g comes from")
    return


@app.cell(hide_code=True)
def _(mo):
    D4 = mo.ui.slider(0.1, 3.0, step=0.1, value=2.0, label="D (µm²/ms)", show_value=True)
    logG4 = mo.ui.slider(0, 3, step=0.05, value=1.9, label="log₁₀ g (g in mT/m)", show_value=True)
    T4 = mo.ui.slider(5, 150, step=1, value=60, label="encoding time T (ms)", show_value=True)
    L4 = mo.ui.slider(0.5, 30, step=0.5, value=10, label="pore size L (µm)", show_value=True)
    logrho4 = mo.ui.slider(-5, -1, step=0.1, value=-3, label="log₁₀ ϱ (ϱ in µm/ms)", show_value=True)
    T2_4 = mo.ui.slider(10, 150, step=5, value=60, label="T₂ (ms)", show_value=True)
    mo.vstack(
        [
            mo.hstack([D4, logG4, T4], justify="start", gap=2, wrap=True),
            mo.hstack([L4, logrho4, T2_4], justify="start", gap=2, wrap=True),
        ]
    )
    return D4, L4, T2_4, T4, logG4, logrho4


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_FREE,
    C_MAIN,
    D4,
    GAMMA,
    L4,
    T2_4,
    T4,
    logG4,
    logrho4,
    mo,
    np,
    plt,
):
    _D, _g, _T, _L = D4.value, 10 ** logG4.value, T4.value, L4.value
    _rho, _T2 = 10 ** logrho4.value, T2_4.value
    _lD = np.sqrt(_D * _T)
    _lq = 1 / (GAMMA * _g * _T)
    _lg = (_D / (GAMMA * _g)) ** (1 / 3)
    _lrho = _D / _rho
    _lT2 = np.sqrt(_D * _T2)
    _names = ["pore size L", "diffusion ℓ_D", "phase ℓ_q", "gradient ℓ_g", "surface relaxation ℓ_ϱ", "T₂ diffusion √(DT₂)"]
    _vals = np.array([_L, _lD, _lq, _lg, _lrho, _lT2])
    _cols = ["k", C_MAIN, C_FREE, C_ALT, C_FREE, C_FREE]

    _fig, _ax = plt.subplots(figsize=(9, 3.0))
    _ax.barh(np.arange(6), _vals, color=_cols, alpha=0.75)
    _ax.axvline(_L, color="k", lw=0.8, ls=":")
    _ax.set_xscale("log")
    _ax.set_yticks(np.arange(6), _names)
    _ax.invert_yaxis()
    for _i, _v in enumerate(_vals):
        _ax.text(_v * 1.08, _i, (f"{_v:,.0f} µm" if _v >= 100 else f"{_v:.3g} µm"), va="center", fontsize=8)
    _ax.set(xlabel="length (µm), log scale", xlim=(min(_vals.min(), 0.05) / 2, max(_vals.max(), 50) * 4),
            title="The shortest of L, ℓ_D and ℓ_g decides the regime")
    _fig.tight_layout()

    _three = {"ℓ_D": _lD, "L": _L, "ℓ_g": _lg}
    _short = min(_three, key=_three.get)
    _regime = {
        "ℓ_D": "**free diffusion / short times**: walls are barely felt, E ≈ exp(−bD) with a small surface-to-volume correction.",
        "L": "**motional narrowing**: spins explore the whole pore and average the gradient; the signal depends on size through L⁴/D.",
        "ℓ_g": "**localisation**: dephasing beats diffusion; magnetization survives near the walls and ln E scales as g^(2/3).",
    }[_short]
    _relax = ("fast-diffusion: the wall loss is uniform, at rate ϱS/V" if _lrho > 10 * _L
              else "diffusion-limited: spins relax as soon as they reach the wall" if _lrho < _L / 10
              else "intermediate between fast-diffusion and diffusion-limited")
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Shortest of L, ℓ_D, ℓ_g: **{_short}**, so {_regime}  \n"
                f"Surface relaxation: ℓ_ϱ/L = {_lrho / _L:.3g}, {_relax}.  \n"
                f"Check: ℓ_g³ = {_lg**3:.3g} µm³ and ℓ_D²·ℓ_q = {_lD**2 * _lq:.3g} µm³."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
        **(a)** Show that $\ell_g^3 = \ell_D^2\,\ell_q$.
        **(b)** Take $D = 2$ µm²/ms and $T = 60$ ms. Compute $\ell_D$, and $\ell_g$ for $g = 80$ mT/m (clinical) and $g = 300$ mT/m (a strong research scanner). Which regime applies to a 10 µm cell, and to a 1 µm axon?
        **(c)** For the 1 µm case, estimate the attenuation from the motional-narrowing formula for a slab at 300 mT/m. What does it tell you about measuring axon diameters?
        **(d)** With an illustrative relaxivity $\varrho = 10^{-3}$ µm/ms, are 10 µm cells in the fast-diffusion regime?
        """, "9 · Length scales on real scanners")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ $\ell_D^2\,\ell_q = DT\cdot\dfrac{1}{\gamma gT} = \dfrac{D}{\gamma g} = \ell_g^3$. ✓ The gradient length is a geometric mean of the diffusion and phase lengths.

        **(b)** ⇒ $\ell_D = \sqrt{2\times60} = 11.0$ µm.
        ⇒ $g = 80$ mT/m: $\ell_g = \big(2/(2.6752\times10^{-4}\times80)\big)^{1/3} = 4.5$ µm. $\ g = 300$ mT/m: $\ell_g = 2.9$ µm.
        ⇒ **10 µm cell:** $\ell_g < L \approx \ell_D$. The gradient length is the shortest: at strong weighting the signal from such cells enters the localisation regime, and the Gaussian phase picture breaks down (Exercise 11 shows the start of this).
        ⇒ **1 µm axon:** $L \ll \ell_g < \ell_D$: motional narrowing.

        **(c)** ⇒ $\gamma^2g^2L^4T/(120D) = (2.6752\times10^{-4}\times300)^2\times1\times60/(120\times2) = 0.0016$: the signal drops by 0.16%.
        → The size information sits in a tiny attenuation that scales as $L^4$. This is why axon-diameter estimation needs the strongest gradients and very high SNR, and why small axons are hard to tell apart.

        **(d)** ⇒ $\ell_\varrho = 2/10^{-3} = 2000$ µm $\gg 10$ µm: yes. Spins hit the walls many times before relaxing, the loss is uniform at rate $\varrho S/V$, and it simply looks like a shorter $T_2$ for that compartment.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two pictures of the same signal

    **Field picture.** Solve the Bloch–Torrey equation for $m(\mathbf r,t)$ everywhere, then integrate at the echo. Chapters 2 to 4 work this way.

    **Path picture.** Follow single spins. Each spin random-walks along its own path $\mathbf r(t)$ (reflected at walls) and gathers its own phase

    $$\varphi = \gamma\int_0^T \mathbf g(t)\cdot\mathbf r(t)\,dt .$$

    The voxel contains so many spins that the signal is the **average phase factor** over all possible paths, starting from uniformly spread positions:

    $$E = \big\langle e^{-i\varphi}\big\rangle_{\text{paths}} .$$

    In the language of probability: the signal is the **characteristic function** of the random phase. Chapter 5 (Monte Carlo) works this way.

    *Sign convention: some texts write $e^{+i\varphi}$. For a refocused echo the two give the same magnitude.*

    The two pictures give exactly the same $E$. This is an example of the **Feynman–Kac relation** between parabolic PDEs and averages over random paths. [Carr & Purcell (1954)](https://doi.org/10.1103/PhysRev.94.630) already computed echo attenuation this way, as an average phase over random walks; the general relation is due to [Kac (1949)](https://doi.org/10.1090/S0002-9947-1949-0027960-X), inspired by [Feynman's (1948)](https://doi.org/10.1103/RevModPhys.20.367) path integrals. You can derive it yourself in a few lines.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Work in 1D, in a slab with a constant-in-space gradient $g(t)$ along $x$. Define the **phase-weighted density of walkers**

    $$w(x,t) = \Big\langle e^{-i\varphi_t}\;\delta\big(x - X_t\big)\Big\rangle ,\qquad \varphi_t = \gamma\int_0^t g(t')X_{t'}\,dt' ,$$

    where $X_t$ is the position of a walker at time $t$. In words: count the walkers that are at $x$ at time $t$, each one weighted by its current phase factor.

    **(a)** Show that $\int w(x,t)\,dx = \langle e^{-i\varphi_t}\rangle$: integrating $w$ gives the path-picture signal.
    **(b)** Write $w(x,t+\delta t)$ in terms of $w(\cdot,t)$ for one short step $\delta t$. In that step a walker moves from $y$ to $x$ with the Gaussian step density $p(x-y)$ of variance $2D\,\delta t$, and gains the phase $\gamma g(t)\,y\,\delta t$.
    **(c)** Expand for small $\delta t$ and show that $w$ obeys the Bloch–Torrey equation.
    **(d)** What are the initial and interface conditions for $w$? Conclude that $\int m\,dx = \langle e^{-i\varphi}\rangle$.
    """, "10 · Derive the Feynman–Kac relation")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) $\int\delta(x - X_t)\,dx = 1$ for every walker.
    - (b) Sum over where the walker was at time $t$: $w(x,t+\delta t) = \int p(x-y)\,e^{-i\gamma g(t) y\,\delta t}\,w(y,t)\,dy$. The walk has no memory, so only the position at $t$ matters.
    - (c) Two small things: the phase factor, $e^{-i\gamma g y\delta t}\approx 1 - i\gamma g y\,\delta t$, and the step. For any smooth $f$, write $y = x - s$, Taylor-expand $f(x-s)$ in $s$, and use $\langle s\rangle = 0$ and $\langle s^2\rangle = 2D\,\delta t$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\int w\,dx = \big\langle e^{-i\varphi_t}\int\delta(x - X_t)\,dx\big\rangle = \langle e^{-i\varphi_t}\rangle$. ✓

    **(b)** ⇒ A walker at $x$ at $t+\delta t$ was at some $y$ at time $t$, carried the weight $e^{-i\varphi_t}$ there, and multiplied it by $e^{-i\gamma g(t)y\,\delta t}$ during the step:

    $$w(x,t+\delta t) = \int p(x-y)\; e^{-i\gamma g(t)\,y\,\delta t}\; w(y,t)\,dy .$$

    **(c)** ⇒ **The step.** For a smooth $f$, put $y = x-s$:

    $$\int p(x-y) f(y)\,dy = \int p(s)\Big[f(x) - s f'(x) + \tfrac{s^2}{2}f''(x) - \dots\Big]ds = f(x) + D\,\delta t\, f''(x) + O(\delta t^2),$$

    because $p$ integrates to 1, has mean 0 and variance $2D\,\delta t$ (higher moments are of order $\delta t^2$).

    ⇒ **The phase.** $e^{-i\gamma g y\delta t} = 1 - i\gamma g y\,\delta t + O(\delta t^2)$.

    ⇒ Apply the step rule to $f(y) = (1 - i\gamma g y\,\delta t)\,w(y,t)$ and keep terms up to first order in $\delta t$:

    $$w(x,t+\delta t) = w(x,t) + D\,\delta t\,\frac{\partial^2 w}{\partial x^2} - i\gamma g(t)\,x\,\delta t\; w(x,t) + O(\delta t^2).$$

    ⇒ Move $w(x,t)$ to the left, divide by $\delta t$ and let $\delta t\to 0$:

    $$\frac{\partial w}{\partial t} = D\,\frac{\partial^2 w}{\partial x^2} - i\gamma\, g(t)\,x\, w .$$

    This is the Bloch–Torrey equation. ✓

    **(d)** ⇒ **Initial:** at $t=0$ every walker has zero phase and the walkers are spread uniformly, so $w(x,0) = \rho = 1/L$, the same as $m(x,0)$.
    ⇒ **Walls:** reflected walkers are never lost, so no weighted walkers flow through a wall: $\partial w/\partial x = 0$, the Neumann condition on $m$.
    ⇒ Same equation, same initial and interface conditions, so the same solution: $w = m$. Then (a) gives

    $$\int m(x,T)\,dx = \big\langle e^{-i\varphi}\big\rangle .\qquad\blacksquare$$

    → **What it means.** The magnetization at $x$ *is* the phase-weighted density of spins at $x$. The PDE tracks the density; the random walk tracks the spins.
    → **What it means for simulation.** Step (b), repeated many times, is a recipe: move each walker with a Gaussian step, then add its phase. That is a Monte Carlo simulation (Chapter 5). Written in the eigenbasis of 1a, the same alternation of "diffuse, then add phase" is the multiple propagator method (Chapter 3). Both converge to the Bloch–Torrey solution as $\delta t\to0$, for exactly the reason shown in (c).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See it: walkers and modes give the same signal

    A slab of width $L$, PGSE with $\delta = 10$ ms and $\Delta = 20$ ms, $D = 2$ µm²/ms. The **walkers** (Monte Carlo, 40 000 of them, step 0.05 ms) compute $\langle e^{-i\varphi}\rangle$ directly. The **exact** curve solves the Bloch–Torrey equation in the slab's eigenmodes (the matrix method of Chapter 3). The dashed curve keeps only the variance of the phase: $E \approx e^{-\langle\varphi^2\rangle/2}$, the Gaussian phase approximation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    L2 = mo.ui.slider(4, 12, step=1, value=8, label="Slab width L (µm)", show_value=True)
    G2 = mo.ui.slider(0, 600, step=10, value=150, label="Gradient G (mT/m)", show_value=True)
    mo.hstack([L2, G2], justify="start", gap=2)
    return G2, L2


@app.cell(hide_code=True)
def _(np):
    def slab_B(L, N):
        """Matrix of x (centred on the slab) in the slab's eigenmodes."""
        n = np.arange(N)
        B = np.zeros((N, N))
        for m in range(N):
            for k in range(N):
                if m == k:
                    continue
                if m == 0 or k == 0:
                    j = max(m, k)
                    B[m, k] = np.sqrt(2) * L * ((-1) ** j - 1) / (j * np.pi) ** 2
                else:
                    B[m, k] = (L / np.pi**2) * (((-1) ** (m - k) - 1) / (m - k) ** 2 + ((-1) ** (m + k) - 1) / (m + k) ** 2)
        return (n * np.pi / L) ** 2, B

    return (slab_B,)


@app.cell(hide_code=True)
def _(GAMMA, expm, np, slab_B):
    def pgse_exact(G, L, D=2.0, delta=10.0, Delta=20.0, N=40):
        """Exact PGSE echo in a slab: three matrix exponentials in the eigenbasis."""
        lam, B = slab_B(L, N)
        Lam = np.diag(lam)
        E1 = expm(-(D * Lam + 1j * GAMMA * G * B) * delta)
        E2 = expm(-D * Lam * (Delta - delta))
        E3 = expm(-(D * Lam - 1j * GAMMA * G * B) * delta)
        return (E3 @ E2 @ E1)[0, 0]

    return (pgse_exact,)


@app.cell(hide_code=True)
def _(L2, np):
    def _walk(L, D=2.0, delta=10.0, Delta=20.0, dt=0.05, n=40000, seed=7):
        rng = np.random.default_rng(seed)
        nsteps = int(round((Delta + delta) / dt))
        t = (np.arange(nsteps) + 0.5) * dt
        f = np.where(t < delta, 1.0, 0.0) - np.where((t >= Delta) & (t < Delta + delta), 1.0, 0.0)
        x = rng.uniform(0, L, n)
        I = np.zeros(n)              # ∫ f(t) x(t) dt for each walker; phase = GAMMA * G * I
        sd = np.sqrt(2 * D * dt)
        paths = np.empty((nsteps, 5))
        for k in range(nsteps):
            x = x + rng.normal(0, sd, n)
            x = np.abs(x)            # reflect at 0
            x = L - np.abs(L - x)    # reflect at L
            I += f[k] * (x - L / 2) * dt
            paths[k] = x[:5]
        return t, f, paths, I

    mc_t, mc_f, mc_paths, mc_I = _walk(L2.value)
    return mc_I, mc_f, mc_paths, mc_t


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_FREE,
    C_MAIN,
    G2,
    GAMMA,
    L2,
    mc_I,
    mc_f,
    mc_paths,
    mc_t,
    mo,
    np,
    pgse_exact,
    plt,
):
    _L, _G = L2.value, G2.value
    _Gs = np.linspace(0, 600, 61)
    _exact = np.array([abs(pgse_exact(g, _L)) for g in _Gs])
    _mc = np.array([abs(np.mean(np.exp(-1j * GAMMA * _gv * mc_I))) for _gv in _Gs])
    _var_I = np.mean(mc_I**2)
    _gpa = np.exp(-0.5 * (GAMMA * _Gs) ** 2 * _var_I)

    _phi = GAMMA * _G * mc_I
    _E_mc = np.mean(np.exp(-1j * _phi))
    _E_ex = abs(pgse_exact(_G, _L))
    _E_gpa = np.exp(-0.5 * np.mean(_phi**2))
    _se = np.sqrt((1 - abs(_E_mc) ** 2) / (2 * len(_phi)))

    _fig, (_a0, _a1, _a2) = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    _a0.fill_between(mc_t, 0, _L, where=mc_f != 0, color=C_FREE, alpha=0.18, step="mid")
    _a0.plot(mc_t, mc_paths, lw=0.8)
    _a0.set(xlabel="t (ms)", ylabel="x (µm)", ylim=(0, _L), title="5 walkers (grey: gradient on)")

    _w = np.angle(np.exp(1j * _phi))
    _a1.hist(_w, bins=60, range=(-np.pi, np.pi), density=True, color=C_ALT, alpha=0.45, label="walkers")
    _s = np.sqrt(np.mean(_phi**2))
    if _s > 0:
        _xx = np.linspace(-np.pi, np.pi, 300)
        _gw = sum(np.exp(-(_xx + 2 * np.pi * k) ** 2 / (2 * _s**2)) for k in range(-6, 7)) / np.sqrt(2 * np.pi * _s**2)
        _a1.plot(_xx, _gw, color="k", lw=1.2, label="Gaussian, same variance")
        _a1.set_ylim(0, 1.3 * _gw.max())
    _a1.set(xlabel="phase φ, wrapped to (−π, π]", ylabel="density", title=f"Phase at G = {_G} mT/m")
    _a1.legend(frameon=False, fontsize=7.5, loc="upper left")

    _a2.semilogy(_Gs, _exact, color=C_MAIN, lw=2, label="exact (eigenmodes)")
    _a2.semilogy(_Gs, _mc, "o", color=C_ALT, ms=3, label="walkers ⟨exp(−iφ)⟩")
    _a2.axhline(1 / np.sqrt(len(mc_I)), color=C_ALT, lw=0.8, ls=":", label="walker noise floor 1/√N")
    _a2.semilogy(_Gs, _gpa, color="k", ls="--", lw=1.2, label="Gaussian phase")
    _a2.axvline(_G, color="k", lw=0.8, ls=":")
    _a2.set(xlabel="G (mT/m)", ylabel="|E| (log scale)", ylim=(1e-3, 1.1), title="Signal against gradient")
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"At G = {_G} mT/m: exact **{_E_ex:.4f}**, walkers **{abs(_E_mc):.4f} ± {_se:.4f}**, "
                f"Gaussian phase **{_E_gpa:.4f}** ({100 * (_E_gpa - _E_ex) / max(_E_ex, 1e-12):+.1f}% against exact)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gpa_check = mo.ui.radio(
        options=[
            "Yes, the phase stays Gaussian at every G",
            "No: at high G the histogram is no longer a Gaussian",
        ],
        label="**Predict first, then raise G:** is the phase still Gaussian at high gradient?",
    )
    gpa_check
    return (gpa_check,)


@app.cell(hide_code=True)
def _(gpa_check, mo):
    if gpa_check.value is None:
        _out = mo.md("")
    elif gpa_check.value.startswith("No"):
        _out = mo.callout(mo.md("**Right.** Walls bound how far a spin can go, so the phase cannot spread like a Gaussian. Small G hides this; large G exposes it. The exact curve then drops below the Gaussian one and can even pass through zero and come back, which no Gaussian can do."), kind="success")
    else:
        _out = mo.callout(mo.md("**Look again** at G = 300 mT/m and above, and compare the histogram with the black curve. Then compare the dashed and solid curves on the right."), kind="warn")
    _out
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
        **(a)** For **free** diffusion, show that the phase is Gaussian, and that $\langle\varphi^2\rangle = 2bD$ with $b = \int_0^T q(t)^2\,dt$ and $q(t) = \gamma\int_0^t g(t')\,dt'$. Conclude that $E = e^{-bD}$.
        **(b)** In the slab above ($L = 8$ µm), use the plot to find the gradient at which the Gaussian phase approximation is 5% off. Why does it work well at low $G$?
        **(c)** Near the dip of the exact curve the walkers stop following it. Why? How many walkers would you need to resolve $|E| = 0.001$?
        """, "11 · When is the phase Gaussian?")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) Integrate $\varphi = \gamma\int_0^T g\,x\,dt$ by parts, using $q(T) = 0$ (echo condition): $\varphi = -\int_0^T q(t)\,dx(t)$. The increments $dx$ of free diffusion are independent Gaussians with variance $2D\,dt$.
    - For a Gaussian random variable with mean zero, $\langle e^{-i\varphi}\rangle = e^{-\langle\varphi^2\rangle/2}$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ $\varphi = \gamma\int_0^T g(t)\,x(t)\,dt = \big[q\,x\big]_0^T - \int_0^T q(t)\,dx(t) = -\int_0^T q(t)\,dx(t)$, since $q(0) = q(T) = 0$.

        ⇒ The increments $dx$ are independent and Gaussian, so $\varphi$ is a weighted sum of independent Gaussians: Gaussian itself, with mean zero.

        ⇒ Variances of independent terms add: $\langle\varphi^2\rangle = \int_0^T q(t)^2\cdot 2D\,dt = 2bD$.

        ⇒ Hence $E = \langle e^{-i\varphi}\rangle = e^{-\langle\varphi^2\rangle/2} = e^{-bD}$. ✓ (For a PGSE this $b$ is $\gamma^2G^2\delta^2(\Delta - \delta/3)$, the result of [Stejskal & Tanner (1965)](https://doi.org/10.1063/1.1695690).)

        **(b)** ⇒ For $L = 8$ µm the dashed curve is 1.3% above the exact one at 150 mT/m and about 5% above at 205 mT/m (about 30% at 300 mT/m).

        → At low $G$ the signal is close to 1, so $\ln E \approx -\langle\varphi^2\rangle/2$ plus terms in $\langle\varphi^4\rangle$ that are of order $G^4$ and still small. The non-Gaussian shape only matters once the phase spread approaches about one radian. That expansion, and what you can compute with it, is the start of Chapter 2.

        **(c)** ⇒ The walker estimate is an average of $N$ unit phasors. Where the true $E$ is tiny, the phasors point almost uniformly in all directions, and their average does not vanish but has a random length of about $1/\sqrt N$ (a 2D random walk of $N$ unit steps, divided by $N$). With $N = 40\,000$ that floor is 0.005: the dotted line.
        ⇒ To see $|E| = 0.001$ you need $1/\sqrt N \ll 0.001$, so $N \gg 10^6$ walkers. Halving the noise costs four times the walkers. Chapter 5 comes back to this; the eigenmode solution has no such floor.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many modes? Smoothness, and Jackson's theorem

    When you solve the Bloch–Torrey equation in the eigenmodes (Chapter 3), you keep $N$ modes. How large must $N$ be? Approximation theory gives the right way to think about it.

    **The idea.** The slab's modes are cosines. A sum of $N$ of them is a truncated Fourier series of the function **mirrored at both walls** (its even, periodic extension). **Jackson's theorems** (Jackson 1911) bound how well $N$ terms can approximate a function by how smooth it is: if the mirrored function has $k$ continuous derivatives, the best error falls at least as fast as $N^{-k}$ (times a measure of how smoothly the $k$-th derivative varies). Bernstein's theorems (Bernstein 1912) give the converse, and for analytic functions the convergence is exponential. In short: **smoothness of the mirrored function sets the speed of convergence.**

    Mirroring is what makes the wall behaviour matter. A function with a non-zero slope at a wall gets a kink when mirrored, even if it is perfectly smooth inside. The left panel shows the cosine coefficients of four functions:

    - a **step** (a jump): coefficients fall as $1/n$;
    - the gradient field **$x$** itself: slope 1 at the walls, so the mirror has kinks, and the coefficients fall as $1/n^2$. These coefficients are exactly the couplings $B_{0n}$ between the uniform mode and mode $n$ that the gradient creates;
    - a **smooth step with zero slope at the walls**: the first jump is in the third derivative of the mirror, so $1/n^4$;
    - the **propagator** at $t > 0$: diffusion has smoothed it completely, and the coefficients $e^{-D\lambda_n t}u_n(x_0)$ fall faster than any power.

    **And the Bloch–Torrey equation?** Diffusion smooths, but the gradient keeps feeding in the field $x$, which does not fit the walls. Differentiate the equation once in $x$ and evaluate it at a wall, where $\partial m/\partial x = 0$ at all times: you find $\partial^3 m/\partial x^3 = i\gamma g\, m/D \ne 0$. So the mirrored magnetization has a jump in its third derivative, and the convergence in $N$ is algebraic, not exponential. The right panel shows it for the PGSE example of the previous section: the error in $E$ falls roughly as $N^{-5}$. That is steep enough that 10–20 modes give 6–8 correct digits at moderate gradients.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, G2, L2, mo, np, pgse_exact, plt):
    _L = 1.0
    _n = np.arange(1, 201)
    _k = _n * np.pi
    # cosine coefficients on [0, 1], in closed form (integrate by parts)
    _c_step = np.abs(np.sqrt(2) * np.sin(_k / 2) / _k)                  # step at x = 1/2
    _c_x = np.abs(np.sqrt(2) * ((-1.0) ** _n - 1) / _k**2)               # x
    _c_smooth = np.abs(np.sqrt(2) * 12 * ((-1.0) ** _n - 1) / _k**4)     # 3x² − 2x³
    _c_prop = np.sqrt(2) * np.exp(-(_n * np.pi) ** 2 * 0.002)  # envelope of e^{-Dλt} u_n(x0)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    _odd = _n % 2 == 1
    _a1.loglog(_n[_odd], _c_step[_odd], "o", ms=3, color=C_FREE, label="step: ∝ 1/n")
    _a1.loglog(_n[_odd], _c_x[_odd], "o", ms=3, color=C_MAIN, label="x (the gradient field): ∝ 1/n²")
    _a1.loglog(_n[_odd], _c_smooth[_odd], "o", ms=3, color=C_ALT, label="smooth step, zero slope at walls: ∝ 1/n⁴")
    _keep = _c_prop > 1e-14
    _a1.loglog(_n[_keep], _c_prop[_keep], "-", color="k", lw=1.2, label="propagator at t > 0 (envelope): faster than any power")
    _a1.set(xlabel="mode number n", ylabel="|cosine coefficient|", ylim=(1e-12, 2), title="Smoothness sets the decay (odd n shown)")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower left")

    _Ns = np.array([4, 6, 8, 12, 16, 24, 32, 48])
    _ref = pgse_exact(G2.value, L2.value, N=128)
    _err = np.array([abs(pgse_exact(G2.value, L2.value, N=int(k)) - _ref) for k in _Ns])
    _err = np.maximum(_err, 1e-16)
    _a2.loglog(_Ns, _err, "o-", color=C_MAIN, label=f"L = {L2.value} µm, G = {G2.value} mT/m")
    _a2.loglog(_Ns, _err[3] * (_Ns / _Ns[3]) ** -5.0, "--", color=C_FREE, label="guide ∝ N⁻⁵")
    _a2.set(xlabel="number of modes N", ylabel="|E(N) − E(128)|", title="PGSE signal error against N", xticks=[4, 8, 16, 32, 48], xticklabels=["4", "8", "16", "32", "48"])
    _a2.minorticks_off()
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()

    _ellg = (2.0 / (2.6752e-4 * max(G2.value, 1e-9))) ** (1 / 3)
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Uses L and G from the walker section. Gradient length ℓ_g = (D/γG)^(1/3) = **{_ellg:.2f} µm**; "
                f"L/(π ℓ_g) = {L2.value / (np.pi * _ellg):.1f}."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practical rules for choosing $N$**

    1. **Resolve the finest length scale in $m$.** Mode $n$ has a wavelength of about $2L/n$, so you need $N \gtrsim$ a few $\times L/(\pi\ell)$, where $\ell$ is the smallest of $\sqrt{D\,t}$ for the shortest relevant time and the gradient length $\ell_g = (D/\gamma g)^{1/3}$.
    2. **Damping must beat dephasing in the last mode kept** (Grebenkov's criterion, [Grebenkov 2007](https://doi.org/10.1103/RevModPhys.79.1077), Sec. VI.A): $D\lambda_N \gg \gamma g L$.
    3. **Always check:** double $N$ and see whether $E$ changes in the digits you care about.

    In 2D and 3D the number of modes below a given $\lambda$ grows as $\lambda^{d/2}$ ([Weyl's law](https://doi.org/10.1007/BF01456804), 1912), so the same resolution costs about $(L/\ell)^d$ modes. A linear gradient couples only some of them, which keeps cylinders and spheres cheap (Chapter 3).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
        **(a)** Compute the cosine coefficients $c_n = \int_0^L x\,u_n(x)\,dx$ of the gradient field on a slab, and show that they fall as $1/n^2$. Relate them to $B_{0n} = \int_0^L u_0\,x\,u_n\,dx$.
        **(b)** Explain the $1/n^2$ with the mirrored function. Would measuring $x$ from the slab's centre change anything?
        **(c)** Use rule 2 above for $L = 8$ µm, $D = 2$ µm²/ms and $G = 300$ mT/m: from which $N$ on is the last mode damped more strongly than it is dephased? Compare with the right-hand panel.
        """, "12 · Why the couplings fall as 1/n²")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ From Exercise 4: $\int_0^L x\cos\frac{n\pi x}{L}dx = \frac{L^2}{(n\pi)^2}\big[(-1)^n - 1\big]$, so with $u_n = \sqrt{2/L}\cos(n\pi x/L)$:

        $$c_n = \sqrt{\frac 2L}\,\frac{L^2\,[(-1)^n-1]}{(n\pi)^2} = -\frac{2\sqrt2\,L^{3/2}}{(n\pi)^2}\ \ (n\ \text{odd}),\qquad c_n = 0\ \ (n\ \text{even}).$$

        ⇒ $B_{0n} = c_n/\sqrt L = \sqrt2\,L\,[(-1)^n-1]/(n\pi)^2$: the same $1/n^2$.

        **(b)** ⇒ Mirroring $x$ on $[0,L]$ at both walls gives a triangle wave. It is continuous but has kinks (jumps in its first derivative) at $x = 0$ and $L$. One derivative is fine, the next jumps, so the coefficients fall as $1/n^2$.
        ⇒ Measuring $x$ from the centre subtracts a constant, which only changes $c_0$. The slope at the walls is still 1, so nothing else changes.

        **(c)** ⇒ $\gamma g L = 2.6752\times10^{-4}\times300\times8 = 0.64$ /ms. Mode $N$ decays at $D(N\pi/L)^2 = 0.31\,N^2$ /ms.
        ⇒ Damping exceeds dephasing from $N \approx 1.5$, and is ten times larger from $N \approx 5$. The panel agrees: with 6 modes the error is already about $2\times10^{-6}$.
        → At $G = 300$ mT/m the gradient length is 2.9 µm, about $L/3$, so a handful of modes resolves $m$. In the localisation regime (Chapter 2) $\ell_g$ becomes much smaller than $L$, and the count grows quickly.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary of Chapter 1

    | Idea | Formula |
    | --- | --- |
    | Diffusion equation | $\partial C/\partial t = \nabla\cdot(D\nabla C)$ |
    | Interface conditions | impermeable $\partial C/\partial n = 0$; permeable: continuous flux $= \kappa\times$ jump |
    | Eigenmodes of a pore | $-\nabla^2 u_n = \lambda_n u_n$, $\ \partial u_n/\partial n = 0$; time scales $\tau_n = 1/(D\lambda_n)$ |
    | Propagator | $P(\mathbf r\,|\,\mathbf r_0,t) = \sum_n u_n(\mathbf r)u_n(\mathbf r_0)e^{-D\lambda_n t}$ |
    | Slab | $u_n = \sqrt{2/L}\cos(n\pi x/L)$, $\ \lambda_n = (n\pi/L)^2$, MSD $\to L^2/6$ |
    | Permeable membrane, relaxing walls | modes of the whole domain; exchange rate $\approx 4\kappa/L$; wall loss $\approx\varrho\,S/V$ |
    | Number of modes | set by the smoothness of the mirrored magnetization; check by doubling $N$ |
    | Cylindrical surface | $\lambda_n = n^2/r^2$ (periodic, no wall); concentric layers weighted by $r_i/\sum r_j$ |
    | Length scales | $\ell_D = \sqrt{DT}$, $\ell_g = (D/\gamma g)^{1/3}$, $\ell_\varrho = D/\varrho$; the shortest of $L$, $\ell_D$, $\ell_g$ sets the regime |
    | Bloch–Torrey | $\partial m/\partial t = D\nabla^2 m - i\gamma\,\mathbf g(t)\cdot\mathbf r\,m$, $\ E = \frac{1}{S_0}\int m\,d\mathbf r$ |
    | Path picture (Feynman–Kac) | $E = \langle e^{-i\varphi}\rangle$, $\ \varphi = \gamma\int \mathbf g\cdot\mathbf r\,dt$ |
    | Free diffusion | $\varphi$ Gaussian, $\langle\varphi^2\rangle = 2bD$, $\ E = e^{-bD}$ |

    ---
    ## Coming next

    **Chapter 2 · Analytical formulas: the regimes where the signal has a closed form**

    Organised by the regime map of the length-scale section: each formula is placed where it is valid, with its assumptions, and checked against an exact solution.

    - **Free diffusion, any waveform:** $E = e^{-bD}$ with $b = \int q^2dt$; the $b$-matrix for anisotropic diffusion (Stejskal & Tanner 1965; Basser et al. 1994).
    - **Narrow pulses** ($\delta \ll \Delta$ and $\delta \ll \tau_1$): the signal as the Fourier transform of the averaged propagator; closed forms for slab, cylinder and sphere; $q$-space, diffusion–diffraction and the long-time pore form factor (Tanner & Stejskal 1968; Callaghan et al. 1991).
    - **Gaussian phase approximation, any waveform:** closed formulas from the mode weights and time scales of 1a (Murday & Cotts 1968; Neuman 1974; van Gelderen et al. 1994); the same in frequency, the diffusion spectrum $D(\omega)$ and OGSE (Stepišnik 1993); exact in a harmonic well. Cumulants and kurtosis as the next terms.
    - **Short times:** the surface-to-volume ratio from $D(t)$ (Mitra et al. 1992).
    - **Long times:** motional narrowing (Robertson 1966), tortuosity and homogenisation, and structural disorder from how $D(t)$ approaches $D_\infty$ (Novikov et al. 2014).
    - **Strong gradients:** the localisation regime, where none of the above holds (Stoller, Happer & Dyson 1991; Hürlimann et al. 1995).
    - **Exchange between compartments:** the Kärger model (Kärger 1985).
    - **Relaxing walls:** fast-diffusion and diffusion-limited regimes (Brownstein & Tarr 1979).
    - *Implementations:* closed-form compartment models, for example the cylinder and sphere models in Camino and dmipy.

    **Chapter 3 · Matrix methods: the eigenmodes of 1a, coupled by the gradient**

    - The gradient as a matrix $\mathbf B$ in the eigenbasis; the Bloch–Torrey equation becomes $d\mathbf c/dt = -(D\Lambda + i\gamma g(t)\mathbf B)\,\mathbf c$.
    - **Grebenkov's MCF:** one matrix exponential per piece of constant gradient, exact for piecewise-constant waveforms (Barzykin 1999; Grebenkov 2007).
    - **Multiple propagator:** diffusion and phase kicks in turn; Strang splitting and its $\tau^2$ error (Caprihan et al. 1996; Callaghan 1997).
    - How the methods relate: the time-ordered exponential (Axelrod & Sen 2001) and the MCF moment series, whose first term is the Gaussian phase approximation of Chapter 2; trapezoids and other waveforms; how many modes.
    - Beyond simple shapes: modes computed on a mesh (SpinDoctor's matrix formalism), a differentiable version for fitting (ReMiDi), and cylindrical surfaces (Canales-Rodríguez, Tax et al. 2026).
    - *Implementations:* MCFAL, MISST, SpinDoctor, the cylindrical-surface code.

    **Chapter 4 · Solving the PDE on a grid or a mesh**

    - **Finite differences:** the domain as voxels; the Laplacian as jumps to neighbours; explicit time steps and their stability limit; periodic boundaries with a phase shift. *Implementation:* MATI.
    - **Finite elements:** the weak form, mass and stiffness matrices, a mesh that follows the cell walls; permeable membranes as interface conditions. *Implementation:* SpinDoctor.
    - Physics-informed neural networks: the PDE as a loss function.

    **Chapter 5 · Monte Carlo: simulating the random walk**

    - The algorithm from Exercise 10: Gaussian steps, reflection at walls, phase accumulation, and the average $\langle e^{-i\varphi}\rangle$.
    - Permeable membranes as a crossing probability; choosing the time step; the $1/\sqrt N$ noise.
    - *Implementations:* Camino, MC-DC Simulator, Disimpy, RMS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## References

    Where possible, the first paper that derived or introduced each idea. Chapter 2 onwards add the papers for the methods.

    | Year | Reference | What it introduced |
    | --- | --- | --- |
    | 1822 | J. Fourier, *Théorie analytique de la chaleur* (Paris) | the heat equation; separation of variables; expansion in Fourier series |
    | 1836 | C. Sturm; J. Liouville, *J. Math. Pures Appl.* 1 | eigenvalue problems for second-order equations (Sturm–Liouville theory) |
    | 1855 | [A. Fick, *Ann. Phys.* 170:59](https://doi.org/10.1002/andp.18551700105) | flux proportional to the concentration gradient; the diffusion equation |
    | 1905 | [A. Einstein, *Ann. Phys.* 322:549](https://doi.org/10.1002/andp.19053220806) | Brownian motion as a random walk; $\langle x^2\rangle = 2Dt$ |
    | 1911 | D. Jackson, dissertation, Göttingen | error of the best polynomial and trigonometric approximation against smoothness |
    | 1912 | S. Bernstein, *Mém. Acad. Roy. Belgique* | the converse: fast approximation implies smoothness |
    | 1912 | [H. Weyl, *Math. Ann.* 71:441](https://doi.org/10.1007/BF01456804) | how the number of eigenvalues grows (Weyl's law) |
    | 1946 | [F. Bloch, *Phys. Rev.* 70:460](https://doi.org/10.1103/PhysRev.70.460) | nuclear induction; the Bloch equations |
    | 1948 | [R. P. Feynman, *Rev. Mod. Phys.* 20:367](https://doi.org/10.1103/RevModPhys.20.367) | path integrals |
    | 1949 | [M. Kac, *Trans. Am. Math. Soc.* 65:1](https://doi.org/10.1090/S0002-9947-1949-0027960-X) | the Feynman–Kac relation between parabolic PDEs and averages over random paths |
    | 1950 | [E. L. Hahn, *Phys. Rev.* 80:580](https://doi.org/10.1103/PhysRev.80.580) | the spin echo |
    | 1954 | [H. Y. Carr & E. M. Purcell, *Phys. Rev.* 94:630](https://doi.org/10.1103/PhysRev.94.630) | diffusion in spin echoes as a random-walk phase average |
    | 1956 | [H. C. Torrey, *Phys. Rev.* 104:563](https://doi.org/10.1103/PhysRev.104.563) | the Bloch–Torrey equation |
    | 1965 | [E. O. Stejskal & J. E. Tanner, *J. Chem. Phys.* 42:288](https://doi.org/10.1063/1.1695690) | the pulsed-gradient spin echo and its $b$-value |
    | 1966 | [B. Robertson, *Phys. Rev.* 151:273](https://doi.org/10.1103/PhysRev.151.273) | eigenmode expansion for restricted diffusion in a slab; motional narrowing |
    | 1979 | [K. R. Brownstein & C. E. Tarr, *Phys. Rev. A* 19:2446](https://doi.org/10.1103/PhysRevA.19.2446) | eigenmodes with relaxing walls; fast-diffusion and diffusion-limited regimes |
    | 1985 | [J. Kärger, *Adv. Colloid Interface Sci.* 23:129](<https://doi.org/10.1016/0001-8686(85)80018-X>) | exchange between compartments |
    | 1991 | [S. D. Stoller, W. Happer & F. J. Dyson, *Phys. Rev. A* 44:7459](https://doi.org/10.1103/PhysRevA.44.7459) | the localisation regime and its length scale |
    | 1994 | [T. M. de Swiet & P. N. Sen, *J. Chem. Phys.* 100:5597](https://doi.org/10.1063/1.467127) | bounded diffusion in a constant gradient; regimes set by length scales |
    | 2007 | [D. S. Grebenkov, *Rev. Mod. Phys.* 79:1077](https://doi.org/10.1103/RevModPhys.79.1077) | review: length scales, eigenmode (MCF) description, numerical methods |
    | 2025 | [E. J. Canales-Rodríguez, C. M. W. Tax et al., *Front. Phys.* 13:1516630](https://doi.org/10.3389/fphy.2025.1516630) | diffusion on cylindrical surfaces; concentric cylinders for myelin |
    | 2026 | [E. J. Canales-Rodríguez, C. M. W. Tax et al., *Phys. Med. Biol.*](https://doi.org/10.1088/1361-6560/aeb2a0) | exact PGSE signal on a cylindrical surface with the spectral matrix formalism ([arXiv 2603.23421](https://arxiv.org/abs/2603.23421)) |
    """)
    return


if __name__ == "__main__":
    app.run()
