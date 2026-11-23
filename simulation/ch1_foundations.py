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
    3. What the **distribution of displacements** looks like in a pore: its time-dependent diffusivity $D(t)$ and kurtosis $K(t)$.
    4. What a **diffusion gradient** does, and the two routes to the signal: the **Bloch–Torrey equation** for the magnetization, and the **average phase factor** of random walkers.
    5. The **full problem** that every simulator solves (compartments, membranes, relaxation), and the simplifications behind the exact methods.
    6. The **weak form** of the same problem, and why choosing its test functions amounts to choosing a numerical method.
    7. Why the PDE and the random walk give the same signal (the **Feynman–Kac** relation), and what that means for simulation.

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
    | **1** | **Foundations:** notation, diffusion equation, interface and boundary conditions, eigenmodes; Bloch–Torrey; the full multi-compartment problem and its weak form; phases and Feynman–Kac | this notebook |
    | 2 | Series expansions: (a) the eigenmodes of 1a coupled by the gradient, i.e. matrix methods; (b) the cumulant expansion in powers of the gradient; (c) how the two connect | next notebook |
    | 3 | Analytical formulas: the PGSE box, the narrow-pulse $(q,t)$ map, the long-pulse $(g,T)$ map, the inside of the box, relaxing walls and exchange | done |
    | 4 | Grids and meshes: the method of lines, finite differences, time stepping and stability, periodic cells, finite elements, curved walls | done |
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
    | $\mathbf x = (x,y,z)^\mathsf T$ | position (in 1D: $x$) | µm |
    | $\mathbf x_0$ | start position of a molecule, at $t = 0$ (in 1D: $x_0$) | µm |
    | $\mathbf r = \mathbf x - \mathbf x_0$, $\ r = \lvert\mathbf r\rvert$ | displacement, and the distance travelled; in 1D $r = x - x_0$, with its sign | µm |
    | $s$, $R$ | radial coordinate (distance from the axis or centre of a cylinder or sphere); radius of a pore | µm |
    | $\Omega$, $\partial\Omega$ | the domain (a pore, or the whole voxel) and its outer boundary | |
    | $\Omega_n$, $\Gamma_{nm}$ | compartment $n$; the interface (membrane) between compartments $n$ and $m$ | |
    | $\mathbf n$ | outward unit normal on a boundary or interface | |
    | $C(\mathbf x,t)$ | concentration of tagged molecules | 1/µm³ |
    | $m(\mathbf x,t) = M_x + iM_y$ | transverse magnetization as one complex number | |
    | $\rho$, $\rho_n$ | spin density at $t=0$ (in compartment $n$) | |
    | $T_{2,n}$ | transverse relaxation time in compartment $n$ | ms |
    | $D$, $\mathbf D$ | diffusion coefficient; diffusion tensor (3×3) | µm²/ms |
    | $\kappa$ | membrane permeability | µm/ms |
    | $\varrho$ | surface absorption rate (for magnetization: surface relaxivity) | µm/ms |
    | $\gamma$ | gyromagnetic ratio of ¹H, $2.675\times10^{8}$ rad s⁻¹ T⁻¹ | |
    | $\mathbf g(t) = \mathbf g\,f(t)$ | effective diffusion gradient; $f(t)$ its time profile | mT/m |
    | $\mathbf q(t) = \gamma\int_0^t\mathbf g\,dt'$, $b = \int_0^T q^2dt$ | phase-winding wave vector; $b$-value | rad/µm; ms/µm² |
    | $\delta$, $\Delta$ | gradient pulse duration and separation (PGSE) | ms |
    | $\varphi$ | phase gathered by one spin | rad |
    | $\xi$ | Fourier variable of a characteristic function $\langle e^{i\xi r}\rangle$ (1a); in 1b it becomes $q$ | 1/µm |
    | $S$, $S_0$, $E = S/S_0$ | signal with and without diffusion weighting; normalised signal | |
    | $\lambda_n$, $u_n(\mathbf x)$, $\tau_n$ | eigenvalue, eigenmode, decay time $1/(D\lambda_n)$ | 1/µm², –, ms |
    | $P(\mathbf x\mid\mathbf x_0,t)$ | propagator of one molecule: probability density to be at $\mathbf x$ at $t$ after starting at $\mathbf x_0$ | 1/µm³ |
    | $\bar P(\mathbf r,t)$ | ensemble-average propagator: probability density of the displacement $\mathbf r$, over all molecules starting uniformly | 1/µm³ |
    | $G(r,t)$ | propagator without walls (free diffusion); it depends only on the displacement, $P(\mathbf x\mid\mathbf x_0,t) = \bar P(\mathbf r,t) = G(r,t)$ | 1/µm³ (1/µm in 1D) |

    **Notation**

    | Notation | Meaning |
    | --- | --- |
    | $\mathbf x$, $\mathbf g$, $\mathbf J$ (bold lower case) | vectors; $\mathbf a\cdot\mathbf b = a_xb_x + a_yb_y + a_zb_z$ |
    | $\mathbf D$ (bold capital) | a tensor, here a symmetric 3×3 matrix: $\mathbf D\mathbf v$ is a vector |
    | $\partial C/\partial t$, $\partial C/\partial x$ | partial derivatives: change in one variable with the others held fixed |
    | $\nabla = (\partial_x, \partial_y, \partial_z)$ | "nabla", the vector of partial derivatives |
    | $\nabla C$ | gradient of a scalar: a vector pointing uphill, length = steepness |
    | $\nabla\cdot\mathbf J = \partial_xJ_x + \partial_yJ_y + \partial_zJ_z$ | divergence of a vector field: net outflow per unit volume |
    | $\nabla^2 C = \nabla\cdot\nabla C = \partial_x^2C + \partial_y^2C + \partial_z^2C$ | Laplacian: how much $C$ differs from its local average |
    | $\partial C/\partial n = \mathbf n\cdot\nabla C$ | normal derivative on a surface |
    | $\nabla^2$ in cylindrical coordinates $(s,\theta,z)$, $s$ = distance from the axis | $\dfrac{1}{s}\dfrac{\partial}{\partial s}\Big(s\dfrac{\partial}{\partial s}\Big) + \dfrac{1}{s^2}\dfrac{\partial^2}{\partial\theta^2} + \dfrac{\partial^2}{\partial z^2}$ |
    | $\nabla^2$ in spherical coordinates $(s,\theta,\phi)$, $s$ = distance from the centre | $\dfrac{1}{s^2}\dfrac{\partial}{\partial s}\Big(s^2\dfrac{\partial}{\partial s}\Big) + \dfrac{1}{s^2}\Big[\dfrac{1}{\sin\theta}\dfrac{\partial}{\partial\theta}\Big(\sin\theta\dfrac{\partial}{\partial\theta}\Big) + \dfrac{1}{\sin^2\theta}\dfrac{\partial^2}{\partial\phi^2}\Big]$ |
    | $\int_\Omega\cdots d\mathbf x$, $\oint_S\cdots dS$ | volume integral; integral over a closed surface |
    | $\delta(x)$ | Dirac delta: zero except at 0, $\int\delta(x)\,dx = 1$, $\int f(x)\delta(x-a)\,dx = f(a)$ |
    | $\delta_{mn}$ | Kronecker delta: 1 if $m=n$, else 0 |
    | $i$ | imaginary unit, $i^2=-1$ |

    **Averages: always say over what.** An average $\langle\cdot\rangle$ in this course always carries a subscript that says what it averages over. That is where most confusion comes from: the same symbol can mean one molecule's possible end points, or all molecules, or all random paths.

    | Notation | Average over | Definition |
    | --- | --- | --- |
    | $\langle f\rangle_{x\mid x_0}$ | the end point $x$ of a molecule that started at a fixed $x_0$ | $\int f(x)\,P(x\mid x_0,t)\,dx$ |
    | $\langle f\rangle_{x}$, $\langle f\rangle_{x_0}$ | one position, spread uniformly over the pore (volume $V$) | $\frac1V\int f(x)\,dx$ |
    | $\langle f\rangle_{x,x_0}$ | start **and** end: every molecule, starting uniformly | $\frac1V\iint f(x, x_0)\,P(x\mid x_0,t)\,dx\,dx_0$ |
    | $\langle f\rangle_{\text{paths}}$ | whole random paths $x(t')$, $0\le t'\le t$, starting uniformly | $\lim_{N\to\infty}\frac1N\sum_{j=1}^N f_j$ over $N$ walkers |
    | $\langle f\rangle_\zeta$ | a random variable $\zeta$ with probability density $p$ | $\int f(\zeta)\,p(\zeta)\,d\zeta$ |

    → With a uniform start and reflecting walls the molecules stay uniform, so for a function of the end point alone $\langle f(x)\rangle_{x,x_0} = \langle f(x)\rangle_x$. In 3D, read $\mathbf x$, $\mathbf x_0$ for $x$, $x_0$.

    **Relationships used in the derivations**

    | Relationship | Statement | Used for |
    | --- | --- | --- |
    | Divergence theorem | $\int_V\nabla\cdot\mathbf J\,dV = \oint_S\mathbf J\cdot\mathbf n\,dS$ | the diffusion equation from conservation |
    | Integration by parts (1D) | $\int_a^b u\,v'\,dx = [u\,v]_a^b - \int_a^b u'\,v\,dx$ | eigenvalues are $\ge 0$; free-diffusion phase |
    | Green's identity (3D version) | $\int_\Omega u\,\nabla^2v\,d\mathbf x = \oint_{\partial\Omega} u\,\dfrac{\partial v}{\partial n}dS - \int_\Omega\nabla u\cdot\nabla v\,d\mathbf x$ | the same, in a pore of any shape |
    | Product rule | $\nabla\cdot(D\nabla C) = D\nabla^2C$ when $D$ is constant | uniform diffusivity |
    | Taylor expansion | $f(x-\zeta) = f(x) - \zeta f'(x) + \tfrac{\zeta^2}{2}f''(x) - \dots$; $\ e^{\varepsilon}\approx 1+\varepsilon$ | small time steps (Feynman–Kac) |
    | Euler's formula | $e^{i\theta} = \cos\theta + i\sin\theta$ | phases as rotating arrows |
    | Product of cosines | $\cos a\cos b = \tfrac12[\cos(a-b)+\cos(a+b)]$ | orthogonality of slab modes |
    | Orthonormality | $\int_\Omega u_m u_n\,d\mathbf x = \delta_{mn}$ | projecting onto modes |
    | Gaussian moments | for $\zeta\sim\mathcal N(0,\sigma^2)$: $\langle \zeta\rangle_\zeta = 0$, $\langle \zeta^2\rangle_\zeta = \sigma^2$, $\langle e^{-is}\rangle_\zeta = e^{-\sigma^2/2}$ | random-walk steps; free-diffusion signal |
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    Eigenmodes, projections and propagators are linear algebra with functions in place of vectors. Every step in this course has a finite-dimensional twin:

    | Vectors and matrices | Functions and operators |
    | --- | --- |
    | vector $\mathbf v$, components $v_i$ | function $f(x)$, "components" $f$ at each $x$ |
    | dot product $\mathbf u\cdot\mathbf v = \sum_iu_iv_i$ | inner product $\int f(x)\,g(x)\,dx$ |
    | matrix times vector $\mathbf A\mathbf v$ | operator on a function: $\nabla^2f$, or $\int K(x,y)\,f(y)\,dy$ |
    | eigenvectors $\mathbf A\mathbf u_n = \lambda_n\mathbf u_n$ | eigenmodes $-\nabla^2u_n = \lambda_nu_n$, *with* the wall conditions |
    | symmetric matrix: real $\lambda_n$, orthonormal $\mathbf u_n$ | self-adjoint operator (here: $\nabla^2$ with reflecting walls): real $\lambda_n$, orthonormal $u_n$ |
    | components in the eigenbasis $c_n = \mathbf u_n\cdot\mathbf v$ | projection $c_n = \int u_n\,f\,dx$ |
    | inverse, $\mathbf v = \mathbf A^{-1}\mathbf b$ | Green's function: solve once for a point source $\delta(x - x_0)$, then superpose |
    | $e^{-t\mathbf A}$ via eigenvalues | $e^{tD\nabla^2}$: the propagator $\sum_nu_n(x)\,u_n(x_0)\,e^{-D\lambda_nt}$ |
    | a matrix in a new basis, $\mathbf U^{\mathsf T}\mathbf A\mathbf U$ | $\int u_n\,x\,u_m\,dx$: the position in the mode basis (Exercise 5) |
    """, title="Functions as vectors: the linear algebra behind eigenmodes")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 1a · Diffusion and the eigenmodes of a pore

    ## The diffusion equation

    A water molecule is knocked about by its neighbours roughly every picosecond. Over milliseconds, the times that matter in this course, each molecule therefore performs a random walk: **Brownian motion**. Molecules that start near the same point $\mathbf x_0$ spread out in random directions, with a probability density $P(\mathbf x\mid\mathbf x_0,t)$ to be at $\mathbf x$ at time $t$: the **propagator**. The concentration is what very many such walkers add up to.

    Let $C(\mathbf x,t)$ be the concentration of tagged water molecules. Two ingredients give the diffusion equation:

    1. **Conservation.** Molecules are not created or destroyed. What leaves a small volume must flow out through its surface.
    2. **Fick's law.** The flux points downhill, from high to low concentration: $\mathbf J = -D\nabla C$.

    Together they give

    $$\boxed{\ \frac{\partial C}{\partial t} = D\,\nabla^2 C\ }.$$

    **Two propagators: one molecule, and all molecules.** The **propagator** $P(\mathbf x\mid\mathbf x_0,t)$ describes one molecule: the probability density of finding it at $\mathbf x$ at time $t$, given that it started at $\mathbf x_0$. It is one particular solution of the diffusion equation. Put a single molecule at $\mathbf x_0$, so $C(\mathbf x, 0) = \delta(\mathbf x - \mathbf x_0)$, and solve with the wall conditions of the domain:

    $$\boxed{\ P(\mathbf x\mid\mathbf x_0,t) = C(\mathbf x,t)\quad\text{for}\quad C(\mathbf x,0) = \delta(\mathbf x - \mathbf x_0)\ }$$

    → Every other start is a sum of point sources, so one propagator solves the equation for every initial condition: $C(\mathbf x,t) = \int P(\mathbf x\mid\mathbf x_0,t)\,C(\mathbf x_0,0)\,d\mathbf x_0$.

    Often we follow all molecules at once, starting spread uniformly over the domain (volume $V$), and ask only how far each one moved: the **displacement** $\mathbf r = \mathbf x - \mathbf x_0$. Its probability density is the **ensemble-average propagator**, the propagator for a move by $\mathbf r$, averaged over all starting points:

    $$\boxed{\ \bar P(\mathbf r,t) = \frac1V\int_V P(\mathbf x_0 + \mathbf r\mid\mathbf x_0,t)\,d\mathbf x_0 = \frac1V\iint\delta\big(\mathbf r - (\mathbf x - \mathbf x_0)\big)\,P(\mathbf x\mid\mathbf x_0,t)\,d\mathbf x\,d\mathbf x_0\ }$$

    The two forms are the same: the second counts every pair of start and end points whose difference is $\mathbf r$, and integrating out $\mathbf x$ against the delta function sets $\mathbf x = \mathbf x_0 + \mathbf r$, which gives the first.

    | | $P(\mathbf x\mid\mathbf x_0,t)$ | $\bar P(\mathbf r,t)$ |
    | --- | --- | --- |
    | describes | one molecule, from a known start $\mathbf x_0$ | all molecules, starting uniformly |
    | variable | where the molecule ends up, $\mathbf x$ | how far each molecule moved, $\mathbf r = \mathbf x - \mathbf x_0$ |
    | averages taken with it | $\langle\cdot\rangle_{\mathbf x\mid\mathbf x_0}$ | $\langle\cdot\rangle_{\mathbf x,\mathbf x_0}$ |
    | without walls | $G(r,t)$, the same for every start | $G(r,t)$ as well |

    → Without walls every start is equivalent, so the propagator depends only on the displacement, and the two are the same function: the free Gaussian $G(r,t)$ of Exercise 1. With walls, $P$ depends on where the molecule starts, and $\bar P$ averages that dependence away. Both are built from the eigenmodes of the pore (next sections, Exercise 2).

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
    Without walls (1D), a molecule that starts at $x_0$ is found at time $t$ at $x$ with probability density

    $$\boxed{\ P(x\mid x_0,t) = G(r,t) = \frac{1}{\sqrt{4\pi Dt}}\exp\!\left(-\frac{r^2}{4Dt}\right),\qquad r = x - x_0\ }$$

    It depends only on the displacement $r$, not on where the molecule started: without walls every start is equivalent. So $G$ is also the ensemble-average propagator, $\bar P(r,t) = G(r,t)$.

    **(a)** Check that $G$ solves the diffusion equation, $\partial G/\partial t = D\,\partial^2 G/\partial r^2$ (the same as $D\,\partial^2/\partial x^2$ for fixed $x_0$), and that it starts as a point: $G(r,t)\to\delta(r)$ as $t\to0$ (it keeps unit area while its width shrinks to zero).

    **(b)** Show that the variance of the displacement is $\langle r^2\rangle_{x\mid x_0} = \langle (x - x_0)^2\rangle_{x\mid x_0} = 2Dt$.
    """, "1 · The free Gaussian")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    (a) Write $G = (4\pi D t)^{-1/2} e^{-u}$ with $u = r^2/(4Dt)$, and differentiate the prefactor and the exponent separately.

    (b) Compare $G$ with a normal distribution $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-r^2/2\sigma^2}$ and read off $\sigma^2$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Time derivative: $\dfrac{\partial G}{\partial t} = G\left(-\dfrac{1}{2t} + \dfrac{r^2}{4Dt^2}\right)$.

    ⇒ Space derivatives: $\dfrac{\partial G}{\partial r} = -\dfrac{r}{2Dt}G$, and $\dfrac{\partial^2 G}{\partial r^2} = G\left(\dfrac{r^2}{4D^2t^2} - \dfrac{1}{2Dt}\right)$.

    ⇒ Multiply the second by $D$: $D\,\partial^2 G/\partial r^2 = G\left(\dfrac{r^2}{4Dt^2} - \dfrac{1}{2t}\right)$, which is $\partial G/\partial t$. ✓ For fixed $x_0$, $\partial/\partial r = \partial/\partial x$, so this is the diffusion equation in $x$.

    **(b)** $G$ is a normal distribution in $r$ with $2\sigma^2 = 4Dt$, so $\langle r^2\rangle_{x\mid x_0} = \sigma^2 = 2Dt$ (per axis; $\langle\lvert\mathbf r\rvert^2\rangle = 6Dt$ in 3D).

    → This is the same $\sqrt{2Dt}$ spread you get from a random walk. The equation for the *concentration* and the random walk of *one molecule* describe the same thing. [Einstein (1905)](https://doi.org/10.1002/andp.19053220806) made exactly this link, deriving $\langle r^2\rangle = 2Dt$ from the random motion of suspended particles. Part 1b builds on exactly that.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **See it: from coin flips to a Gaussian.** Each of 4000 walkers starts at $x_0$ and takes $n$ equal steps of $\pm a$ in a time $t$, with $a = \sqrt{2D\,t/n}$, so every walk has the same variance $2Dt$. With one step the displacements $r = x - x_0$ sit at two points; with more steps their histogram approaches $G(r,t)$ (left), and their variance grows as $2Dt$ (right): the two results of Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rw_t = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="t (ms)", show_value=True)
    rw_D = mo.ui.slider(0.5, 3.0, step=0.1, value=2.0, label="D (µm²/ms)", show_value=True)
    rw_n = mo.ui.dropdown(options=["1", "2", "5", "20", "200"], value="20", label="steps per walk n")
    mo.hstack([rw_t, rw_D, rw_n], justify="start", gap=2, wrap=True)
    return rw_D, rw_n, rw_t


@app.cell(hide_code=True)
def _(C_ALT, C_MAIN, mo, np, plt, rw_D, rw_n, rw_t):
    _t, _D, _n = rw_t.value, rw_D.value, int(rw_n.value)
    _N = 4000
    _a = np.sqrt(2 * _D * _t / _n)                  # step length: n steps give variance 2Dt
    _rng = np.random.default_rng(1)
    _steps = _a * (2 * _rng.integers(0, 2, size=(_N, _n)) - 1)
    _paths = np.cumsum(_steps, axis=1)
    _x = _paths[:, -1]
    _sig = np.sqrt(2 * _D * _t)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.7))
    # left: histogram against the Gaussian
    _xx = np.linspace(-4.5 * _sig, 4.5 * _sig, 400)
    _G = np.exp(-_xx**2 / (4 * _D * _t)) / np.sqrt(4 * np.pi * _D * _t)
    if _n <= 5:
        _vals, _cnt = np.unique(np.round(_x / _a).astype(int), return_counts=True)
        _a1.vlines(_vals * _a, 0, _cnt / _N / (2 * _a), color=C_MAIN, lw=4, label="walkers (mass per site / 2a)")
    else:
        _edges = np.arange(-(_n + 1) * _a, (_n + 1.01) * _a, 2 * _a)   # one bin per lattice site (sites are 2a apart)
        _a1.hist(_x, bins=_edges, density=True, color=C_MAIN, alpha=0.55, label="walkers")
    _a1.plot(_xx, _G, color=C_ALT, lw=2, label="G(r, t)")
    _a1.set(xlabel="displacement r = x − x₀ (µm)", ylabel="density (1/µm)", xlim=(-4.5 * _sig, 4.5 * _sig), title=f"Displacements at t = {_t:g} ms, n = {_n}")
    _a1.legend(frameon=False, fontsize=8, loc="upper right")

    # middle: variance against time
    _tk = _t * np.arange(1, _n + 1) / _n
    _var = (_paths**2).mean(axis=0)
    _tl = np.linspace(0, 20, 50)
    _a2.plot(_tl, 2 * _D * _tl, color=C_ALT, lw=2, label="2Dt")
    _a2.plot(_tk, _var, "o", color=C_MAIN, ms=4 if _n > 20 else 6, label="walkers ⟨r²⟩")
    _a2.set(xlabel="t (ms)", ylabel="⟨r²⟩ (µm²)", xlim=(0, 20), ylim=(0, 2 * _D * 20 * 1.05),
            title="Variance grows linearly")
    _a2.legend(frameon=False, fontsize=8, loc="upper left")

    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"Step length a = {_a:.2f} µm. Walkers: ⟨r²⟩ = **{(_x**2).mean():.2f}** µm², against 2Dt = **{2 * _D * _t:.2f}** µm²."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The domain: pores, compartments and a block of tissue

    The equation alone does not fix the solution. We also need where the molecules start, and what happens where they meet a wall. **This is where the microstructure enters the problem.** First some names (panel b follows the usual multi-compartment notation):

    - The **domain** $\Omega$ is the region the molecules live in: one closed pore (panel a), or a whole block of tissue (panel b).
    - The block is split into **compartments** $\Omega_n$: the cells and the extracellular space $\Omega_e$.
    - Two compartments meet at an **interface** $\Gamma_{nm}$: a cell membrane.
    - The **boundary** $\partial\Omega$ is the outer edge of the block. Where a cell is cut by that edge, that piece of its surface is called $\Gamma_n$.

    So we have **interface conditions** (on membranes and cell walls: physics) and **boundary conditions** (on the outer edge: a modelling choice about what lies outside the block). For a single closed pore the two coincide: its wall is an impermeable interface and also the edge of the domain.
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

    # (b) a block of tissue split into compartments
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
    _a2.annotate(r"$\Gamma_4$: part of $\partial\Omega_4$ on the outer edge", xy=(10, 2.3), xytext=(10.4, 0.4),
                 fontsize=9.5, arrowprops=dict(arrowstyle="-", lw=0.8))
    _a2.annotate(r"$\partial\Omega$: outer boundary", xy=(10, 8.5), xytext=(10.4, 9.2), fontsize=9.5,
                 arrowprops=dict(arrowstyle="-", lw=0.8))
    _ang = 2.3
    _px, _py = 3 + 1.9 * np.cos(_ang), 3 + 1.9 * np.sin(_ang)
    _a2.annotate("", xy=(_px + 0.8 * np.cos(_ang), _py + 0.8 * np.sin(_ang)), xytext=(_px, _py),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))
    _a2.text(_px + 0.85 * np.cos(_ang) - 0.35, _py + 0.85 * np.sin(_ang), r"$\mathbf{n}_1$", fontsize=11)
    _a2.set(xlim=(-0.3, 15.5), ylim=(-0.4, 10.4), aspect="equal", title=r"(b) A block of tissue $\Omega$ split into compartments")
    _a2.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial, interface and boundary conditions

    **Initial condition.** At the start the molecules are in equilibrium and spread evenly: $C(\mathbf x,0)=\rho$, a constant (or one constant per compartment).

    **Interface conditions** on cell walls and membranes $\Gamma_{nm}$ ($\partial/\partial n$: derivative along the outward normal). Each one is a statement about the flux into the wall, which by Fick's law is $\mathbf J\cdot\mathbf n = -D\,\partial C/\partial n$: zero, set by the jump across a membrane, or proportional to $C$ at the wall.

    | Interface | Condition | What it means | In a random walk |
    | --- | --- | --- | --- |
    | Impermeable, inert | $\dfrac{\partial C}{\partial n}=0$ (Neumann type) | no flux through the wall | the walker is reflected |
    | Permeable membrane | $D_n\dfrac{\partial C_n}{\partial n_n} = -D_m\dfrac{\partial C_m}{\partial n_m} = \kappa\,(C_m - C_n)$ (transmission) | the flux is continuous; the jump in $C$ across the membrane is set by the permeability $\kappa$ | the walker crosses with a small probability, else it is reflected |
    | Partially absorbing wall | $D\dfrac{\partial C}{\partial n} + \varrho\, C = 0$ (Robin type) | molecules that touch the wall bind or react there with a small probability, so the flux into the wall is proportional to $C$ at the wall, with rate $\varrho$ | the walker is removed with a small probability |
    | Perfect sink | $C = 0$ (Dirichlet type) | every molecule that touches the wall is lost: the limit $\varrho\to\infty$ | the walker is removed at its first touch |

    Two limits of the permeable membrane are worth remembering: $\kappa = 0$ gives an impermeable wall on both sides, and $\kappa\to\infty$ forces $C_n = C_m$, as if the membrane were not there.

    **Boundary conditions** on the outer edge $\partial\Omega$:

    | Boundary | Condition | When to use it |
    | --- | --- | --- |
    | Reflecting | $\partial C/\partial n = 0$ | the block is closed, or what lies outside does not matter |
    | Periodic | values and fluxes on opposite faces match | the block is one tile of a repeating tissue; avoids artificial walls at its edge |

    The names Neumann, Robin and Dirichlet refer to the *type* of condition (on the derivative, on a mix, on the value), wherever it is applied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The full problem, and the step to one compartment

    Put the pieces together. In every compartment the molecules diffuse with their own diffusivity, starting from their own equilibrium density:

    $$\boxed{\ \frac{\partial C_n}{\partial t} = \nabla\cdot(D_n\nabla C_n)\ \text{ in }\Omega_n,\qquad C_n(\mathbf x,0) = \rho_n\ },$$

    with a reflecting outer edge, $D_n\,\partial C_n/\partial n = 0$ on $\Gamma_n$, and on every membrane $\Gamma_{nm}$

    $$D_n\frac{\partial C_n}{\partial n_n} = -D_m\frac{\partial C_m}{\partial n_m} = \kappa_{nm}\big(c_{nm}C_m - c_{mn}C_n\big),\qquad c_{nm} = \frac{2\rho_n}{\rho_n + \rho_m} .$$

    → **Why the factors $c_{nm}$?** Compartments can hold different amounts of water at equilibrium, $\rho_n\ne\rho_m$. The plain condition $\kappa(C_m - C_n)$ would then drive a flux even at equilibrium and slowly even out the two densities, which real tissue does not do. The factors make the equilibrium state, $C_n = \rho_n$ in every compartment, a state without flux: $c_{nm}\rho_m - c_{mn}\rho_n = 0$ (check it). For equal densities $c_{nm} = 1$, and the condition is the one in the table.

    → **The step to one compartment.** Most of this chapter, and Chapters 2 and 3, work with a single pore: uniform $D$, reflecting walls, $\rho = 1$. That is the full problem with every interface made impermeable: each compartment then evolves on its own, and the block is a collection of single pores. Membranes come back where they are the point: the exchange mode below (Exercises 10 and 11), exchange in Chapter 3, and the simulators of Chapters 4 and 5, which solve the full problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eigenmodes: the natural patterns of a pore

    Put the diffusion equation in a bounded domain $\Omega$: one pore, or several compartments joined by membranes, with any of the interface and boundary conditions of the tables above. Some concentration patterns keep their **shape** and only fade in amplitude. These are the **eigenmodes** $u_n(\mathbf x)$ of the domain. Any initial concentration can be written as a sum of eigenmodes, and each one fades at its own rate:

    $$\boxed{\ C(\mathbf x,t) = \sum_n c_n\, u_n(\mathbf x)\, e^{-D\lambda_n t}, \qquad -\nabla^2 u_n = \lambda_n u_n \ \text{in } \Omega, \qquad \text{with the same wall and interface conditions as } C\ }.$$

    The derivation below shows why the conditions on $C$ carry over to every $u_n$ unchanged. For the reflecting walls of most of this chapter, that is $\partial u_n/\partial n = 0$.

    - With reflecting or periodic walls (membranes allowed): $\lambda_0 = 0$ with $u_0$ constant, the uniform state. It never fades, because no molecules are lost. With absorbing walls or sinks even the slowest mode fades: $\lambda_0 > 0$ (see "When the interfaces are not reflecting").
    - Every other mode fades with time constant $\tau_n = 1/(D\lambda_n)$. Finer patterns have larger $\lambda_n$ and fade faster.
    - The geometry enters **only** through $\lambda_n$ and $u_n$. Once you have them, you have solved diffusion in that pore for every starting condition and every time.
    - Separation of variables and expansion in modes go back to Fourier (1822); the general theory of such eigenproblems is Sturm–Liouville theory (Sturm 1836; Liouville 1836).
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ **Try a product.** Guess $C(\mathbf x,t)=u(\mathbf x)\,T(t)$ and put it in $\partial C/\partial t = D\nabla^2 C$:

    $$u\,T' = D\,T\,\nabla^2 u .$$

    ⇒ **Separate.** Divide by $D\,u\,T$:

    $$\frac{T'}{D\,T} = \frac{\nabla^2 u}{u}.$$

    The left side depends only on $t$, the right only on $\mathbf x$. Both must equal the same constant; call it $-\lambda$.

    ⇒ **Time part:** $T' = -D\lambda T$, so $T(t) = e^{-D\lambda t}$.

    ⇒ **Space part, and this is where the boundary conditions go in:** $-\nabla^2 u = \lambda u$. Every wall and interface condition in the tables above is **linear and homogeneous**: a combination of $C$ and its normal derivative set equal to zero. Put $C = u\,T$ into one of them, for example the absorbing wall:

    $$D\frac{\partial C}{\partial n} + \varrho\,C = T(t)\Big[D\frac{\partial u}{\partial n} + \varrho\,u\Big] = 0\quad\text{for all } t\quad\Longrightarrow\quad D\frac{\partial u}{\partial n} + \varrho\,u = 0 .$$

    $T(t)$ drops out, and $u$ inherits exactly the condition of $C$:

    | Condition on $C$ | Condition on every mode $u_n$ |
    | --- | --- |
    | reflecting, $\partial C/\partial n = 0$ | $\partial u_n/\partial n = 0$ |
    | partially absorbing, $D\,\partial C/\partial n + \varrho C = 0$ | $D\,\partial u_n/\partial n + \varrho u_n = 0$ |
    | perfect sink, $C = 0$ | $u_n = 0$ |
    | membrane: continuous flux $= \kappa\times$ jump | the same two conditions for $u_n$ on both sides |
    | periodic | $u_n$ and its derivative match on opposite faces |

    → The initial condition is the one condition that does **not** go into $u$: it is used last, to find the coefficients $c_n$.

    → Homogeneity is what makes this work. A wall held at a fixed nonzero value, $C = C_w$, would not separate; you would first subtract the steady state and solve for the rest.

    ⇒ **The eigenvalues.** The equation $-\nabla^2u = \lambda u$ has a nonzero solution that obeys these conditions only for special values $\lambda_n$: the eigenvalues. The conditions decide which values (Exercise 2 for a reflecting slab, the recipe before Exercise 10 for membranes and absorbing walls). They are never negative: multiply by $u$, integrate, and use Green's identity,

    $$\lambda\int_\Omega u^2\,d\mathbf x = \int_\Omega\lvert\nabla u\rvert^2\,d\mathbf x + \frac\varrho D\oint_{\text{absorbing walls}}u^2\,dS + \frac\kappa D\oint_{\text{membranes}}(u^- - u^+)^2\,dS\ \ge0 .$$

    Each term is zero exactly when its effect is absent: no gradient inside, no absorbing wall, no jump across a membrane. So $\lambda_0 = 0$ needs a constant mode and walls that lose nothing.

    ⇒ **Superpose.** The equation is linear, so any sum of such products is again a solution. The eigenmodes are orthonormal, $\int_\Omega u_m u_n\,d\mathbf x = \delta_{mn}$. (Green's identity again: $(\lambda_m - \lambda_n)\int u_mu_n$ is a sum of boundary terms, and they cancel because both modes obey the same homogeneous conditions.) So the coefficients follow from the initial condition by projection:

    $$c_n = \int_\Omega u_n(\mathbf x)\,C(\mathbf x,0)\,d\mathbf x .$$

    ⇒ **Point source.** Start with one molecule at $\mathbf x_0$: $C(\mathbf x,0) = \delta(\mathbf x - \mathbf x_0)$. Then $c_n = u_n(\mathbf x_0)$, and the solution is the **propagator**, the probability density of being at $\mathbf x$ at time $t$ after starting at $\mathbf x_0$:

    $$\boxed{\ P(\mathbf x\mid\mathbf x_0,t) = \sum_n u_n(\mathbf x)\,u_n(\mathbf x_0)\,e^{-D\lambda_n t}\ }.$$

    → Why a point source if molecules start everywhere? Because every other start is a sum of point sources: $C(\mathbf x,t) = \int P(\mathbf x|\mathbf x_0,t)\,C(\mathbf x_0,0)\,d\mathbf x_0$. One propagator serves all initial conditions.
    """, title="Derivation: separation of variables")
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Take a slab $0 \le x \le L$ (diffusion between two parallel reflecting planes).

    **(a)** Solve $-u'' = \lambda u$ with $u'(0)=u'(L)=0$. Consider $\lambda<0$, $\lambda=0$ and $\lambda>0$ separately.

    **(b)** Normalise the modes so that $\int_0^L u_n^2\,dx = 1$.

    **(c)** Show that two different modes are orthogonal.

    **(d)** A molecule starts at $x_0$: the initial condition is $C(x,0) = \delta(x - x_0)$. Expand it in the modes and write down the propagator $P(x\,|\,x_0,t)$ for the slab.
    """, "2 · Eigenmodes of a slab")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - For $\lambda = -k^2 < 0$ the general solution is $A\cosh kx + B\sinh kx$; for $\lambda = k^2 > 0$ it is $A\cos kx + B\sin kx$.
    - Apply $u'(0)=0$ first: it removes one of the two functions. Then $u'(L)=0$ restricts $k$.
    - For (b) and (c): $\cos a\cos b = \tfrac12[\cos(a-b)+\cos(a+b)]$.
    - For (d): expand the start, $\delta(x - x_0) = \sum_nc_nu_n(x)$, multiply by $u_m$ and integrate. Orthonormality leaves $c_m = u_m(x_0)$ (the separation-of-variables derivation above).
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

    $$\boxed{\ P(x\,|\,x_0,t) = \frac1L + \frac2L\sum_{n=1}^{\infty}\cos\frac{n\pi x}{L}\cos\frac{n\pi x_0}{L}\,e^{-D(n\pi/L)^2 t}\ }.$$

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

    Top: how much each mode has faded, $e^{-D\lambda_nt}$, against time; the dots mark the time chosen with the slider. Bottom left: the $N$ terms of the mode sum, $u_n(x_0)\,e^{-D\lambda_nt}\,u_n(x)$: each mode, weighted by where the molecule started and by how much it has faded. Bottom right: their sum, the propagator of a molecule released at $x_0$. Make $t$ short and $N$ small, and watch the sum fail; move $t$ to the right, and watch the dots of the higher modes drop to zero one by one.
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

    _fig = plt.figure(figsize=(10, 8.2))
    _gs = _fig.add_gridspec(2, 2, height_ratios=[0.75, 1])
    _a0 = _fig.add_subplot(_gs[0, :])
    _a1 = _fig.add_subplot(_gs[1, 0])
    _a2 = _fig.add_subplot(_gs[1, 1])
    _U, _lam = slab_modes(_x, _L, _N)

    # top: e^{-D λ_n t} for every mode, with the current time marked
    _tt = np.logspace(-2, 2, 300)
    for _n in range(_N):
        _shade = plt.cm.Blues(0.95 - 0.6 * _n / max(_N - 1, 1))
        _a0.plot(_tt, np.exp(-_D * _lam[_n] * _tt), color=_shade, lw=1.6 if _n < 3 else 0.9)
        _a0.plot(_t, np.exp(-_D * _lam[_n] * _t), "o", color=_shade, ms=5, mec="k", mew=0.4, zorder=3)
    _a0.axvline(_t, color=C_ALT, lw=1.2)
    _a0.text(_t * 1.08, 1.02, f"t = {_t:.3g} ms", color=C_ALT, fontsize=8.5, va="bottom")
    _a0.axhline(np.exp(-5), color="k", lw=0.7, ls=":")
    _a0.text(1.1e-2, np.exp(-5) + 0.015, "e⁻⁵ < 1%: faded", fontsize=8, va="bottom")
    _a0.set_xscale("log")
    _a0.set(xlim=(1e-2, 1e2), ylim=(-0.03, 1.12), xlabel="t (ms)", ylabel=r"$e^{-D\lambda_n t}$",
            title=f"How much each mode has faded (n = 0 on top, never fades; lighter = higher n)")
    _u0 = slab_modes(np.array([_x0]), _L, _N)[0][:, 0]
    _terms = (_u0 * np.exp(-_D * _lam * _t))[:, None] * _U
    for _n in range(_N):
        _shade = plt.cm.Blues(0.95 - 0.6 * _n / max(_N - 1, 1))
        _tau = "∞" if _n == 0 else f"{1 / (_D * _lam[_n]):.3g} ms"
        _lab = f"n = {_n}, τₙ = {_tau}" if (_n < 3 or _n == _N - 1) else None
        _a1.plot(_x, _terms[_n], color=_shade, lw=1.6 if _n < 3 else 1.0, label=_lab)
    _a1.axvline(_x0, color="k", lw=0.8, ls=":")
    _ylim1 = 1.15 * max(np.abs(_terms).max(), 1.0 / _L)
    _a1.set(xlabel="x (µm)", ylabel=r"$u_n(x_0)\,e^{-D\lambda_n t}\,u_n(x)$  (1/µm)", ylim=(-_ylim1, _ylim1),
            title=f"The {_N} terms of the sum (dark: n = 0, light: n = {_N - 1})")
    _a1.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
               title=None if _N <= 4 else f"(+ {_N - 4} more terms, not labelled)", title_fontsize=7.5)

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

    **(c)** At long times only a handful of modes survive, so eigenmode methods are cheap exactly where diffusion has explored the whole pore. At short times you need many modes to build a narrow peak. There, a free Gaussian (or a random walk) is the natural description. You will see this trade-off again in Chapters 2 and 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean-squared displacement in a slab

    With the propagator in hand, every average over the molecules is an integral. The simplest one is the mean-squared displacement: how far, on average, the molecules have spread after a time $t$. Without walls it grows as $2Dt$ for ever (Exercise 1); between two walls it cannot.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Molecules start **uniformly** in the slab. The mean-squared displacement averages over where each molecule starts *and* where it ends,

    $$\langle (x - x_0)^2\rangle_{x,x_0} = \frac1L\int_0^L\!\!\int_0^L (x - x_0)^2\,P(x\mid x_0,t)\,dx\,dx_0 .$$

    Using the propagator, show that

    $$\langle (x - x_0)^2\rangle_{x,x_0}(t) = \frac{L^2}{6} - \frac{16L^2}{\pi^4}\sum_{n\ \text{odd}}\frac{e^{-D(n\pi/L)^2 t}}{n^4},$$

    and that it tends to $L^2/6$ at long times. (In the random-walk notebook the walkers between two walls levelled off at $L^2/6$. Here is where that number comes from.)
    """, "4 · Mean-squared displacement in a slab")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - Expand the square: $\langle(x-x_0)^2\rangle_{x,x_0} = \langle x^2\rangle_{x,x_0} + \langle x_0^2\rangle_{x,x_0} - 2\langle x\,x_0\rangle_{x,x_0}$.
    - Uniform start and reflecting walls mean the walkers stay uniform at all times, so $\langle x^2\rangle_{x,x_0} = \langle x_0^2\rangle_{x,x_0} = L^2/3$.
    - $\langle x\,x_0\rangle_{x,x_0} = \frac1L\iint x\,x_0\,P(x|x_0,t)\,dx\,dx_0$. Insert the mode sum: each mode contributes $\frac1L\big(\int_0^L x\,u_n\,dx\big)^2 e^{-D\lambda_n t}$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    ⇒ $\langle x^2\rangle_{x,x_0} = \langle x_0^2\rangle_{x,x_0} = \frac1L\int_0^L x^2dx = \frac{L^2}{3}$.

    ⇒ $\langle x\,x_0\rangle_{x,x_0} = \frac1L\sum_n e^{-D\lambda_n t}\left(\int_0^L x\,u_n(x)\,dx\right)^2$.

    ⇒ $n=0$: $\int_0^L x/\sqrt L\,dx = L^{3/2}/2$, giving $\frac1L\cdot\frac{L^3}{4} = \frac{L^2}{4}$.

    ⇒ $n\ge1$: integrate by parts, $\int_0^L x\cos\frac{n\pi x}{L}dx = \frac{L^2}{(n\pi)^2}\big[(-1)^n - 1\big]$. This is zero for even $n$ and $-2L^2/(n\pi)^2$ for odd $n$. With the factor $\sqrt{2/L}$ squared: $\frac1L\cdot\frac2L\cdot\frac{4L^4}{(n\pi)^4} = \frac{8L^2}{(n\pi)^4}$.

    ⇒ Put together:

    $$\langle(x-x_0)^2\rangle_{x,x_0} = \frac{2L^2}{3} - 2\left[\frac{L^2}{4} + \sum_{n\ \text{odd}}\frac{8L^2}{(n\pi)^4}e^{-D\lambda_n t}\right] = \frac{L^2}{6} - \frac{16L^2}{\pi^4}\sum_{n\ \text{odd}}\frac{e^{-D\lambda_n t}}{n^4}.$$

    → Check $t=0$: $\sum_{n\ \text{odd}} n^{-4} = \pi^4/96$, so the MSD is $L^2/6 - L^2/6 = 0$. ✓
    → $t\to\infty$: all exponentials vanish and the MSD is $L^2/6$: start and end positions have become independent, each uniform in the slab.
    → The weights $8L^2/(n\pi)^4$ fall as $n^{-4}$: the first mode carries 98.6% of the total. Remember this. Exercise 5 shows where they come from, and Chapter 2 meets the same weights again.
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
    ## The position in the mode basis

    The MSD needed the integrals $\int x\,u_n\,dx$. They are the first row of a matrix that every later chapter uses: the **position written in the mode basis**,

    $$\boxed{\ B_{nm} = \int u_n(x)\;x\;u_m(x)\,dx\ },$$

    with $x$ measured from the centre of the pore. Like the eigenvalues $\lambda_n$, it depends only on the geometry. Some of its properties hold for any pore, others only for special shapes:

    | Property | Holds for |
    | --- | --- |
    | real and symmetric, $B_{nm} = B_{mn}$ | every pore, any direction of $x$ |
    | $B_{00} = \langle x\rangle_x$ = the centre of the pore | every pore |
    | $B_{nm} = 0$ between modes of the same parity | pores symmetric about their centre (slab, disk, sphere) |
    | $B_{0n}\propto1/n^2$ | the slab; Chapter 2 explains the decay rate from the shape |

    It answers: if a pattern sits in mode $m$ and is multiplied by $x$, how much ends up in mode $n$? Together with the eigenvalues it is everything the matrix methods of Chapter 2 need to know about a pore; 1b shows why.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Take the slab, with $x$ measured from its centre.

    **(a)** Show that $\mathbf B$ is real and symmetric.

    **(b)** Show that $B_{nm} = 0$ whenever $n + m$ is even. (Think of parity about the centre.)

    **(c)** Compute the couplings to the uniform mode, $B_{0n}$, and show that $B_{0n}^2 = 8L^2/(n\pi)^4$ for odd $n$. Where did you see these numbers before?

    **(d)** Show that for $n, m \ge 1$

    $$B_{nm} = \frac{L}{\pi^2}\left[\frac{(-1)^{n-m}-1}{(n-m)^2} + \frac{(-1)^{n+m}-1}{(n+m)^2}\right].$$
    """, "5 · The position in the mode basis")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - **(b)** About the centre, $x - L/2$ is odd and $\cos(n\pi x/L)$ has parity $(-1)^n$. The integral of an odd function over a symmetric interval is zero.
    - **(c, d)** $\cos a\cos b = \frac12[\cos(a-b) + \cos(a+b)]$, and for $k \neq 0$, $\int_0^L (x - \tfrac L2)\cos\frac{k\pi x}{L}\,dx = \frac{L^2}{(k\pi)^2}\big[(-1)^k - 1\big]$ (integrate by parts).
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ The modes are real and $B_{nm} = \int u_n\,x\,u_m\,dx$ is unchanged when $n$ and $m$ swap.

    **(b)** ⇒ Put $y = x - L/2$. Then $u_n$ has parity $(-1)^n$ in $y$, and $y$ is odd, so $u_n\,y\,u_m$ has parity $(-1)^{n+m+1}$. For $n + m$ even it is odd, and its integral over $[-L/2, L/2]$ is zero.

    → In particular the diagonal vanishes: a mode is never coupled to itself.

    **(c)** ⇒ $B_{0n} = \frac{1}{\sqrt L}\sqrt{\frac 2L}\int_0^L(x - \tfrac L2)\cos\frac{n\pi x}{L}\,dx = \frac{\sqrt2}{L}\cdot\frac{L^2[(-1)^n-1]}{(n\pi)^2} = -\frac{2\sqrt2\,L}{(n\pi)^2}$ for odd $n$.

    ⇒ $B_{0n}^2 = \dfrac{8L^2}{(n\pi)^4}$: exactly the weights in the mean-squared displacement of Exercise 4 (there you computed $\int x\,u_n$ from a wall; centring only changes $n = 0$). Chapter 2 uses them again.

    **(d)** ⇒ $B_{nm} = \frac2L\int_0^L y\cos\frac{n\pi x}{L}\cos\frac{m\pi x}{L}dx = \frac1L\int_0^L y\Big[\cos\frac{(n-m)\pi x}{L} + \cos\frac{(n+m)\pi x}{L}\Big]dx$.

    ⇒ Apply the integral from the hint with $k = n - m$ and $k = n + m$ and simplify: $\frac1L\cdot\frac{L^2}{\pi^2}[\dots] = \frac{L}{\pi^2}[\dots]$. ✓
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_MAIN, mo, np, plt):
    def _slab_B(L, N):
        n = np.arange(N)
        m, k = n[:, None], n[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            B = (L / np.pi**2) * ((((-1.0) ** (m - k)) - 1) / (m - k) ** 2 + (((-1.0) ** (m + k)) - 1) / (m + k) ** 2)
            j = np.maximum(m, k)
            B0 = np.sqrt(2) * L * (((-1.0) ** j) - 1) / (j * np.pi) ** 2
        B = np.where((m == 0) | (k == 0), B0, B)
        return np.where(m == k, 0.0, B)

    _B = _slab_B(8.0, 16)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    _im = _a1.imshow(np.log10(np.abs(_B) + 1e-12), cmap="Blues", vmin=-3, vmax=1)
    _a1.set(xlabel="m", ylabel="n", title="log₁₀ |B_nm|, L = 8 µm", xticks=range(0, 16, 3), yticks=range(0, 16, 3))
    _a1.grid(False)
    _fig.colorbar(_im, ax=_a1, fraction=0.046)
    _n = np.arange(1, 16, 2)
    _a2.loglog(_n, np.abs(_B[0, 1::2]), "o-", color=C_MAIN, label="|B₀ₙ| (coupling to the uniform mode)")
    _ne = np.arange(2, 16, 2)
    _a2.loglog(_ne, np.abs(_B[3, _ne]), "s-", color=C_ALT, ms=4, label="|B₃ₙ| (even n; odd n give zero)")
    _a2.loglog(_n, np.abs(_B[0, 1]) * _n**-2.0, ":", color="k", lw=1, label="∝ 1/n²")
    _a2.set(xlabel="n", ylabel="|B| (µm)", title="Couplings fall off as 1/n²")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("The checkerboard is the parity rule of Exercise 5(b): $x$ only connects modes of opposite parity, and each mode mostly to its neighbours.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other geometries

    The recipe is always the same: solve $-\nabla^2 u = \lambda u$ with $\partial u/\partial n = 0$ on the walls. In a disk or sphere, separate in polar or spherical coordinates. The radial part becomes a Bessel function, and the wall condition picks out the roots $\alpha$ of its **derivative**. On a closed surface, such as a circle, there is no wall at all: the mode only has to come back to itself, and **periodicity replaces the wall condition**.

    | Geometry | Eigenmodes | Wall condition fixes | Eigenvalues | Radial normalisation | First non-zero root | Slowest time $\tau_1$ |
    | --- | --- | --- | --- | --- | --- | --- |
    | Slab $0\le x\le L$ | $\cos\dfrac{n\pi x}{L}$ | $\sin(n\pi)=0$ | $\left(\dfrac{n\pi}{L}\right)^2$ | $\int_0^L\cos^2 = L/2$ | $\pi$ (with $L$) | $\dfrac{L^2}{\pi^2 D}=0.101\,\dfrac{L^2}{D}$ |
    | Cylinder, radius $R$ (cross-section) | $J_m\!\left(\dfrac{\alpha_{mk}s}{R}\right)\cos m\theta,\ \sin m\theta$ | $J_m'(\alpha_{mk})=0$ | $\left(\dfrac{\alpha_{mk}}{R}\right)^2$ | $\int_0^R J_m^2\,s\,ds = \dfrac{R^2}{2}\left(1-\dfrac{m^2}{\alpha^2}\right)J_m(\alpha)^2$ | $\alpha_{11} = 1.8412$ | $\dfrac{R^2}{1.8412^2 D}=0.295\,\dfrac{R^2}{D}$ |
    | Cylindrical surface, radius $R$ (the circle across the axis) | $\cos n\phi,\ \sin n\phi$ (each $\lambda$ twice) | periodic in $\phi$: there is no wall | $\dfrac{n^2}{R^2}$ | $\int_0^{2\pi R}\cos^2 n\phi\,d\ell = \pi R$ | $n = 1$ | $\dfrac{R^2}{D}$ |
    | Concentric cylindrical surfaces $R_1 < \dots < R_N$, impermeable | the modes of each circle separately | periodic on each circle | $\dfrac{n^2}{R_i^2}$; $\lambda = 0$ once per layer | share of molecules in layer $i$: $w_i = R_i/\sum_jR_j$ | $n = 1$ on the outer layer | $\dfrac{R_N^2}{D}$ |
    | Annulus $R_{\text{in}}\le s\le R_{\text{out}}$ (a layer of finite thickness) | $\big[J_m(ks)Y_m'(kR_{\text{in}}) - Y_m(ks)J_m'(kR_{\text{in}})\big]\cos m\theta$ | $\partial_su = 0$ at both radii | $k^2$ | numerical | thin layer: $k\to1/R$ | thin layer: $\to R^2/D$ |
    | Sphere, radius $R$ | $j_l\!\left(\dfrac{\alpha_{lk}s}{R}\right)Y_{lm}(\theta,\phi)$ | $j_l'(\alpha_{lk})=0$ | $\left(\dfrac{\alpha_{lk}}{R}\right)^2$ | $\int_0^R j_l^2\,s^2ds = \dfrac{R^3}{2}\left(1-\dfrac{l(l+1)}{\alpha^2}\right)j_l(\alpha)^2$ | $\alpha_{11} = 2.0816$ | $\dfrac{R^2}{2.0816^2 D}=0.231\,\dfrac{R^2}{D}$ |

    $J_m$, $Y_m$ are Bessel functions of the first and second kind, $j_l$ spherical Bessel functions, $Y_{lm}$ spherical harmonics. Along a cylinder's axis diffusion is free, so a cylinder is a disk in the cross-section times free diffusion along $z$. The surface rows are explained at the end of this section.

    → The slowest non-uniform mode of a disk is not the radially symmetric one ($m = 0$, $\alpha = 3.83$) but the one with one sign change across the disk ($m = 1$, $\alpha = 1.84$). The same holds for the sphere ($l=1$). These modes look like the slab's $\cos(\pi x/L)$: high on one side, low on the other.

    The roots below are computed in this notebook; the table quotes them.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    In a disk, cylinder or sphere we write $s$ for the distance from the axis or centre (the radial coordinate) and $R$ for the radius of the pore.

    **Cylinder.** Write $u = F(s)\,\Theta(\theta)\,Z(z)$ in $-\nabla^2u = \lambda u$, with the Laplacian in cylindrical coordinates (notation section). For the modes of the cross-section $Z$ is constant. $\Theta$ must come back to itself after one turn, so $\Theta = \cos m\theta$ or $\sin m\theta$ with $m = 0, 1, 2, \dots$ What is left for the radial part $F$:

    $$s^2F'' + s\,F' + \big(\lambda s^2 - m^2\big)F = 0 .$$

    ⇒ With $y = \sqrt\lambda\,s$ this is **Bessel's equation**. Its solution that stays finite at the centre is $J_m(\sqrt\lambda\,s)$. (The second solution, $Y_m$, blows up at $s = 0$; it returns in an annulus, where the centre is not part of the domain.)

    ⇒ **The wall picks the scale.** The reflecting wall needs $F'(R) = 0$, so $\sqrt\lambda\,R$ must be a root $\alpha_{mk}$ of $J_m'$. The function $J_m$ is the same for every pore; the wall condition stretches it so that one of its flat points lands exactly on the wall: $\lambda = (\alpha_{mk}/R)^2$.

    **Sphere.** With $u = F(s)\,Y_{lm}(\theta,\phi)$ the radial equation becomes $s^2F'' + 2s\,F' + \big(\lambda s^2 - l(l+1)\big)F = 0$, solved by the spherical Bessel function $j_l(\sqrt\lambda\,s)$, with the wall at a root of $j_l'$.

    **Orthogonality.** Modes with different $m$ are orthogonal through the angle: $\int_0^{2\pi}\cos m\theta\cos m'\theta\,d\theta = 0$. Modes with the same $m$ and different roots are orthogonal through the radial coordinate, **with weight $s$**:

    $$\int_0^R J_m\Big(\frac{\alpha_{mk}s}{R}\Big)\,J_m\Big(\frac{\alpha_{mk'}s}{R}\Big)\,s\,ds = 0\qquad (k\ne k').$$

    → The weight is the $s\,ds$ of the area element, so this is ordinary orthogonality, $\int u\,u'\,dA = 0$. It follows from the same integration by parts as for the slab (Exercise 2c), because the radial equation can be written $(sF')' - \frac{m^2}sF = -\lambda\,s\,F$. For the sphere the weight is $s^2$.
    """, title="Where the Bessel functions come from")
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, jv, np, plt, spherical_jn):
    _a = np.linspace(0.2, 12, 1200)          # spherical recursion is unstable very close to 0
    _cols = [C_MAIN, C_ALT, "#2e9e6a", "#8a5cd1"]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for _m, _c in zip(range(4), _cols):
        _d = 0.5 * (jv(_m - 1, _a) - jv(_m + 1, _a))
        _a1.plot(_a, _d, color=_c, lw=1.6, label=f"m = {_m}")
        _z = _a[:-1][np.sign(_d[:-1]) != np.sign(_d[1:])]
        _a1.plot(_z, np.zeros_like(_z), "o", color=_c, ms=5, mec="k", mew=0.4)
        _dl = spherical_jn(_m, _a, derivative=True)
        _a2.plot(_a, _dl, color=_c, lw=1.6, label=f"l = {_m}")
        _z = _a[:-1][np.sign(_dl[:-1]) != np.sign(_dl[1:])]
        _a2.plot(_z, np.zeros_like(_z), "o", color=_c, ms=5, mec="k", mew=0.4)
    for _ax in (_a1, _a2):
        _ax.plot([0], [0], "o", color=_cols[0], ms=5, mec="k", mew=0.4)   # α = 0: the uniform mode
    for _ax, _t in ((_a1, r"Cylinder: $J_m'(\alpha)$"), (_a2, r"Sphere: $j_l'(\alpha)$")):
        _ax.axhline(0, color=C_FREE, lw=0.8)
        _ax.set(xlabel=r"$\alpha = \sqrt{\lambda}\,R$", xlim=(0, 12), title=_t)
        _ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _a1.set_ylim(-0.65, 0.6)
    _a2.set_ylim(-0.48, 0.4)
    _fig.suptitle("The dots are the allowed α: the wall sits where the derivative is zero", fontsize=10)
    _fig.tight_layout()
    _fig
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

    In which pores have the molecules explored the whole pore after 20 ms? What does that mean for what any observation at 20 ms can tell you about pore *size*?
    """, "6 · Time scales of three pores")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    ⇒ Slab: $\tau_1 = L^2/(\pi^2 D) = 64/(9.87\times 2) = $ **3.2 ms**.
    ⇒ Cylinder: $\tau_1 = R^2/(\alpha_{11}^2 D) = 25/(3.390\times 2) = $ **3.7 ms**.
    ⇒ Sphere: $\tau_1 = R^2/(\alpha_{11}^2 D) = 25/(4.333\times 2) = $ **2.9 ms**.

    → All three are much shorter than 20 ms: after a few $\tau_1$ every non-uniform mode has faded and the molecules have explored the whole pore: the long-time (**fully restricted**) regime. Their positions are then uniform and independent of where they started, so an observation depends on size only through a few numbers such as $\langle x^2\rangle_x$ in the pore, not through details of the shape.

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
    ### Surfaces instead of volumes: the myelin sheath

    Water trapped between the wraps of myelin moves along thin sheets. [Canales-Rodríguez, Tax et al. (2025)](https://doi.org/10.3389/fphy.2025.1516630) model each sheet as the **surface of a cylinder** of radius $R$, and the myelin sheath as a stack of **impermeable concentric cylinders** separated by very thin layers of water. Along the axis the motion is free. Across it, a molecule moves on a circle.

    **Modes on the circle.** With arc length $\ell = R\phi$, the eigenproblem is $-u'' = \lambda u$ with $u$ periodic:

    $$u_0 = \frac{1}{\sqrt{2\pi R}},\qquad u_n = \frac{\cos n\phi}{\sqrt{\pi R}},\ \frac{\sin n\phi}{\sqrt{\pi R}},\qquad \lambda_n = \frac{n^2}{R^2}\quad(n\ge1,\ \text{each twice}).$$

    The slowest decay time is $\tau_1 = R^2/D$, about 3.4 times longer than for the inside of a disk of the same radius ($0.295\,R^2/D$): a molecule on the rim cannot cut across, it has to go around.

    **Concentric layers.** The walls between layers are impermeable, so the modes of the whole sheath are simply the modes of each circle, and every layer keeps its own uniform mode (the $\kappa = 0$ case of the next section). A uniform start puts in layer $i$ a share of the molecules proportional to its circumference, and anything summed over the sheath is a weighted sum over the layers:

    $$w_i = \frac{R_i}{\sum_j R_j}.$$

    The layers are evenly spaced between the inner radius (the axon) and the outer radius; in the 2025 simulations they are 7.5 nm apart, roughly a 4–5 nm lipid bilayer plus about 3 nm of water. Outer layers hold more water, so a sheath-averaged quantity leans towards the outer radius. The 2025 paper summarises the sheath by the radius-weighted mean $\sum_i w_i R_i = \sum_i R_i^2/\sum_i R_i$. Chapter 2 uses the modes of one cylindrical surface again ([Canales-Rodríguez, Tax et al. 2026](https://doi.org/10.1088/1361-6560/aeb2a0)).

    → A layer of finite thickness is an annulus between two radii (last row of the table). Its modes combine $J_m$ and $Y_m$, with the wall condition at both radii; the cylindrical surface is the limit of a very thin annulus.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rout5 = mo.ui.slider(0.5, 5.0, step=0.1, value=2.0, label="Outer radius R_out (µm)", show_value=True)
    gratio5 = mo.ui.slider(0.5, 0.9, step=0.01, value=0.7, label="g-ratio R_in / R_out", show_value=True)
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
            title=f"{_N} myelin layers, R_in = {_rin:.2f} µm")
    _a1.axis("off")

    _a2.bar(_r, _w, width=0.7 * (_rout - _rin) / max(_N - 1, 1), color=C_MAIN, alpha=0.7, label="weight w_i = R_i / Σ R_j")
    _a2.set(xlabel="layer radius R_i (µm)", ylabel="share of the molecules", title="Outer layers weigh more, and are slower")
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
        **(a)** Derive the modes of a circle of radius $R$: solve $-u''(\ell) = \lambda u(\ell)$ on $0 \le \ell < 2\pi R$ with $u$ periodic, and normalise.

        **(b)** Molecules are spread uniformly over a stack of concentric cylindrical surfaces of equal length. Show that layer $i$ holds a share $w_i = R_i/\sum_j R_j$ of them.

        **(c)** For $R = 1$ µm and $D = 0.8$ µm²/ms, compare $\tau_1$ on the surface with $\tau_1$ inside a solid cylinder of the same radius.
        """, "7 · Modes on a cylindrical surface")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ $\lambda<0$: $\cosh$ and $\sinh$ are not periodic, so no solutions.
        ⇒ $\lambda = 0$: $u = A + B\ell$ is periodic only if $B = 0$: the uniform mode.
        ⇒ $\lambda = k^2 > 0$: $u = A\cos k\ell + B\sin k\ell$. Periodicity, $u(\ell+2\pi R) = u(\ell)$, needs $2\pi R k = 2\pi n$, so $k = n/R$ and $\lambda_n = n^2/R^2$. Both $\cos$ and $\sin$ survive: each eigenvalue appears twice (unlike the slab, where the wall condition removed the sine).
        ⇒ Normalise: $\int_0^{2\pi R}\cos^2(n\ell/R)\,d\ell = \pi R$ and $\int_0^{2\pi R}1\,d\ell = 2\pi R$, so $u_0 = 1/\sqrt{2\pi R}$ and $u_n = \cos(n\phi)/\sqrt{\pi R}$, $\sin(n\phi)/\sqrt{\pi R}$.

        **(b)** ⇒ The number of molecules in a layer is density × area, and the area of a cylindrical surface of length $H$ is $2\pi R_iH$. With equal density and length, layer $i$ holds a fraction $2\pi R_iH/\sum_j 2\pi R_jH = R_i/\sum_j R_j$.
        ⇒ The layers do not exchange molecules, so anything summed over the sheath is the weighted sum of the layer values.

        **(c)** ⇒ Surface: $\tau_1 = R^2/D = 1/0.8 = 1.25$ ms. Solid cylinder: $\tau_1 = R^2/(1.8412^2 D) = 1/(3.39\times0.8) = 0.37$ ms.
        → The surface is 3.4 times slower to "forget" where a molecule started. In the long-time limit the variance of $x$ is $R^2/2$ on the surface and $R^2/4$ inside the cylinder, so the surface also looks larger. Water on the sheath and water inside the axon diffuse differently, even at the same radius.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Beyond the mean-squared displacement: cumulants of the displacement

    The MSD is one number per time. The ensemble-average propagator $\bar P(r,t)$, defined in the diffusion-equation section, holds the whole **distribution of displacements** $r = x - x_0$ over all molecules.

    A compact way to describe a distribution is by its **cumulants** $\kappa_n$: the mean ($\kappa_1$, zero here by symmetry), the variance ($\kappa_2 = \langle r^2\rangle_{x,x_0}$), and then the shape ($\kappa_4 = \langle r^4\rangle_{x,x_0} - 3\langle r^2\rangle_{x,x_0}^2$ for a symmetric distribution). In 1D define

    $$\boxed{\ D(t) = \frac{\langle r^2\rangle_{x,x_0}}{2t},\qquad K(t) = \frac{\kappa_4}{\kappa_2^2} = \frac{\langle r^4\rangle_{x,x_0}}{\langle r^2\rangle_{x,x_0}^2} - 3\ }.$$

    $D(t)$ is the apparent diffusivity of the mean-squared-displacement figure. $K(t)$, the **excess kurtosis**, measures how far the shape is from a Gaussian. Free diffusion is Gaussian at every time, so $K = 0$ (Exercise 8a): **any nonzero $K$ comes from the geometry.**

    **The characteristic function** holds all moments and cumulants at once. For a probability density $p(r)$ it is the Fourier transform,

    $$\boxed{\ \phi(\xi) = \langle e^{i\xi r}\rangle = \int e^{i\xi r}\,p(r)\,dr\ }$$

    with $\xi$ the Fourier variable, a wavenumber in 1/µm.

    ⇒ $\phi(0) = 1$: the total probability.

    ⇒ Expand $e^{i\xi r} = 1 + i\xi r - \frac{\xi^2r^2}{2} - \dots$ and average term by term: $\phi(\xi) = \sum_n\frac{(i\xi)^n}{n!}\langle r^n\rangle$. Every moment sits in one function.

    ⇒ Its logarithm holds the cumulants. For the displacements of all molecules, $p = \bar P(r,t)$ and the average is over start and end:

    $$\boxed{\ \ln\big\langle e^{i\xi r}\big\rangle_{x,x_0} = \sum_{n\ge1}\frac{(i\xi)^n}{n!}\,\kappa_n = -\frac{\xi^2}{2}\langle r^2\rangle_{x,x_0} + \frac{\xi^4}{24}\,\kappa_4 - \dots\ }$$

    (For one molecule from a known start, $p = P(x_0 + r\mid x_0, t)$ and the average is $\langle\cdot\rangle_{x\mid x_0}$.) A Gaussian stops at $\xi^2$, so all its cumulants beyond the variance vanish (Exercise 8a).

    Keep this line in mind: 1b shows that a measurement can give you exactly this characteristic function.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    ⇒ Shift the coordinate to the slab centre, $y = x - L/2$, so that odd powers average to zero. The displacement is $r = y - y_0$.

    ⇒ Expand: $\langle r^2\rangle_{x,x_0} = 2\langle y^2\rangle_{x,x_0} - 2\langle y\,y_0\rangle_{x,x_0}$ and $\langle r^4\rangle_{x,x_0} = 2\langle y^4\rangle_{x,x_0} - 8\langle y^3y_0\rangle_{x,x_0} + 6\langle y^2y_0^2\rangle_{x,x_0}$ (using $\langle y^3y_0\rangle_{x,x_0} = \langle y\,y_0^3\rangle_{x,x_0}$, because $P$ is symmetric in $x$ and $x_0$).

    ⇒ Each mixed average is a mode sum, as in Exercise 4: $\langle f(y)\,g(y_0)\rangle_{x,x_0} = \frac1L\sum_n e^{-D\lambda_n t}\,f_n\,g_n$, with $f_n = \int f\,u_n\,dx$.

    ⇒ The coefficients of $y$, $y^2$, $y^3$ follow from integrating by parts. With $k_n = n\pi/L$:

    $$y_n = -\frac{2\sqrt2\,L^{3/2}}{(n\pi)^2}\ (n\ \text{odd}),\qquad (y^2)_n = \frac{2\sqrt2\,L^{5/2}}{(n\pi)^2}\ (n\ \text{even} \ge 2),\qquad (y^3)_n = \sqrt2\,L^{7/2}\Big[-\frac{3}{2(n\pi)^2} + \frac{12}{(n\pi)^4}\Big]\ (n\ \text{odd}),$$

    and zero otherwise; for $n=0$ only $(y^2)_0 = L^{5/2}/12$ survives.

    → $D(t)/D$ and $K(t)$ depend on $t$ only through $Dt/L^2$: one curve for all slabs.
    """, title="Computing D(t) and K(t) from the modes")
    return


@app.cell(hide_code=True)
def _(np):
    def slab_cumulants(tau, N=1500):
        """D(t)/D0 and excess kurtosis K(t) of the displacement in a slab, uniform start.

        tau = D t / L^2 (array). Closed-form cosine coefficients of (x - L/2)^k, in units L = 1.
        """
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

    return (slab_cumulants,)


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D1, L1, mo, np, plt, slab_cumulants):
    _L, _D = L1.value, D1.value
    _t = np.logspace(-2, 2.7, 300)
    _d, _K = slab_cumulants(_D * _t / _L**2)

    _fig, _a1 = plt.subplots(figsize=(6.5, 4.6))
    _a1.semilogx(_t, _d, color=C_MAIN, lw=2, label="D(t) / D")
    _a1.semilogx(_t, _K, color=C_ALT, lw=2, label="K(t)")
    _tau = _D * _t / _L**2
    _sh = _tau < 0.1
    _a1.semilogx(_t[_sh], 1 - 8 / (3 * np.sqrt(np.pi)) * np.sqrt(_tau[_sh]), color="k", lw=1.0, ls=(0, (4, 3)),
                 zorder=4, label="short times: 1 − (8/3√π)√(Dt)/L (on top of D)")
    _a1.semilogx(_t[_sh], 16 / (5 * np.sqrt(np.pi)) * np.sqrt(_tau[_sh]), color=C_ALT, lw=1.1, ls="--",
                 label="short times: (16/5√π)√(Dt)/L")
    _a1.axhline(0, color=C_FREE, lw=0.8)
    _a1.axhline(-0.6, color=C_ALT, lw=0.8, ls=":", label="−3/5")
    _a1.axvline(_L**2 / (np.pi**2 * _D), color="k", lw=0.8, ls=":")
    _a1.text(_L**2 / (np.pi**2 * _D), 0.9, " τ₁", fontsize=9)
    _a1.set(xlabel="t (ms)", ylim=(-0.7, 1.05), title="Diffusivity and kurtosis of the displacement in a slab")
    _a1.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Uses L and D from the sliders above the propagator plot.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** *Free diffusion first.* Show that the characteristic function of the free Gaussian $G(r,t)$ is $\langle e^{i\xi r}\rangle_{x\mid x_0} = \int e^{i\xi r}\,G(r,t)\,dr = e^{-\xi^2Dt}$. What does this tell you about all cumulants beyond the variance, and about $K$?

    **(b)** At long times the start and end positions are independent, and each is uniform in the slab. Show that $r = x - x_0$ then has a **triangular** distribution on $[-L, L]$, and that its excess kurtosis is $K_\infty = -3/5$.

    **(c)** Show that its characteristic function is $\langle e^{i\xi r}\rangle_{x,x_0} = \operatorname{sinc}^2(\xi L/2)$, with $\operatorname{sinc}z = \sin z/z$. Expand $\ln\operatorname{sinc}^2(\xi L/2)$ to order $\xi^4$ and read off $\kappa_2$ and $\kappa_4$. Do they agree with (a)?

    **(d)** In the figure $K(t)$ is **positive** at short times before it turns negative. Explain both signs.
    """, "8 · The kurtosis of a slab")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - **(a)** Complete the square in the exponent: $i\xi r - \frac{r^2}{4Dt} = -\frac{(r - 2i\xi Dt)^2}{4Dt} - \xi^2Dt$. The remaining Gaussian integrates to 1.
    - **(b)** The density of a sum of two independent variables is the convolution of their densities: two boxes give a triangle. For the moments use symmetry, $\langle r^4\rangle_{x,x_0} = 2\int_0^L r^4\,\frac{L - r}{L^2}\,dr$.
    - **(c)** The characteristic function of a sum of independent variables is the product of theirs. You need $\ln\operatorname{sinc}z = -z^2/6 - z^4/180 - \dots$
    - **(d)** Split the molecules into two groups: those far from both walls, and those within about $\sqrt{Dt}$ of one.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Complete the square: $\langle e^{i\xi r}\rangle_{x\mid x_0} = e^{-\xi^2Dt}\int\frac{1}{\sqrt{4\pi Dt}}e^{-(r - 2i\xi Dt)^2/4Dt}\,dr = e^{-\xi^2Dt}$.

    ⇒ Its logarithm, $-\xi^2Dt = -\frac{\xi^2}{2}\,(2Dt)$, stops at $\xi^2$. Only the second cumulant, the variance $2Dt$, is nonzero; every higher cumulant is zero, so $K = 0$ at all times.

    → Without walls every start is equivalent, so the same holds for all molecules together, $\langle e^{i\xi r}\rangle_{x,x_0} = e^{-\xi^2Dt}$. Any nonzero kurtosis must come from the walls and the structure around the molecules.

    **(b)** ⇒ $x$ and $x_0$ are independent and uniform on $[0, L]$, so $r$ has the convolution of two boxes of height $1/L$:

    $$\bar P(r) = \frac{L - |r|}{L^2},\qquad |r|\le L .$$

    ⇒ $\langle r^2\rangle_{x,x_0} = \frac{2}{L^2}\int_0^L r^2(L - r)\,dr = \frac{2}{L^2}\Big(\frac{L^4}{3} - \frac{L^4}{4}\Big) = \frac{L^2}{6}$, the long-time MSD of Exercise 4. ✓

    ⇒ $\langle r^4\rangle_{x,x_0} = \frac{2}{L^2}\Big(\frac{L^6}{5} - \frac{L^6}{6}\Big) = \frac{L^4}{15}$.

    ⇒ $K_\infty = \dfrac{L^4/15}{L^4/36} - 3 = \dfrac{12}{5} - 3 = -\dfrac35$.

    → Negative: compared with a Gaussian of the same variance, the triangle has a broad shoulder and no tails at all. Every bounded distribution leans this way.

    **(c)** ⇒ $\langle e^{i\xi r}\rangle_{x,x_0} = \langle e^{i\xi x}\rangle_x\,\langle e^{-i\xi x_0}\rangle_{x_0} = \big|\langle e^{i\xi x}\rangle_x\big|^2$, and $\langle e^{i\xi x}\rangle_x = \frac1L\int_0^L e^{i\xi x}dx = e^{i\xi L/2}\operatorname{sinc}(\xi L/2)$. So $\langle e^{i\xi r}\rangle_{x,x_0} = \operatorname{sinc}^2(\xi L/2)$.

    ⇒ With $z = \xi L/2$: $\ln\operatorname{sinc}^2z = -\frac{z^2}{3} - \frac{z^4}{90} - \dots = -\frac{\xi^2L^2}{12} - \frac{\xi^4L^4}{1440} - \dots$

    ⇒ Compare with $-\frac{\xi^2}{2}\kappa_2 + \frac{\xi^4}{24}\kappa_4$: $\kappa_2 = L^2/6$ and $\kappa_4 = -L^4/60$, so $K = \kappa_4/\kappa_2^2 = -36/60 = -3/5$. ✓

    **(d)** ⇒ **Short times.** Most molecules are far from the walls and diffuse freely, with variance $2Dt$. Molecules within about $\sqrt{Dt}$ of a wall are held back and spread less. A mixture of zero-mean Gaussians with different variances $\sigma^2$ has $K = 3\,\mathrm{Var}(\sigma^2)/\langle\sigma^2\rangle^2 > 0$: a sharper peak and heavier tails than one Gaussian. The hindered fraction is about $(S/V)\sqrt{Dt}$, so $K$ grows like $\sqrt t$ at first, just as $D(t)$ first drops like $\sqrt t$ ([Mitra et al. 1992](https://doi.org/10.1103/PhysRevLett.68.3555)).

    ⇒ **Long times.** Every molecule feels both walls, and the bounded triangle of (b) takes over: $K\to-3/5$.

    → In between, $K$ peaks at about $+0.2$ near $Dt \approx 0.03\,L^2$ and crosses zero near $Dt \approx 0.08\,L^2$. The same geometry gives both signs: a kurtosis measured at one diffusion time cannot tell heterogeneity from restriction on its own.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Short times: the walls are felt through $S/V$

    At short times a molecule notices a wall only if it starts within a few $\sqrt{Dt}$ of it. Seen from that close, any smooth wall looks flat, and the share of molecules that close to a wall is proportional to the **surface-to-volume ratio** $S/V$ of the pore space. The result holds for every pore shape ([Mitra, Sen, Schwartz & Le Doussal 1992](https://doi.org/10.1103/PhysRevLett.68.3555)):

    $$\boxed{\ \frac{D(t)}{D} = 1 - \frac{4}{3\sqrt\pi\,d}\,\frac SV\,\sqrt{Dt} + O(t)\qquad\text{in } d \text{ dimensions}\ }.$$

    | Pore | $S/V$ | $d$ | $D(t)/D$ at short times |
    | --- | --- | --- | --- |
    | slab, width $L$ | $2/L$ | 1 | $1 - \frac{8}{3\sqrt\pi}\frac{\sqrt{Dt}}L$ |
    | cylinder, radius $R$ (cross-section) | $2/R$ | 2 | $1 - \frac{4}{3\sqrt\pi}\frac{\sqrt{Dt}}R$ |
    | sphere, radius $R$ | $3/R$ | 3 | $1 - \frac{4}{3\sqrt\pi}\frac{\sqrt{Dt}}R$ |

    → The first correction knows only $S/V$, nothing else about the shape. Curvature and absorbing walls appear one order later, in the $O(t)$ term ([Mitra, Sen & Schwartz 1993](https://doi.org/10.1103/PhysRevB.47.8565)). Measuring $D(t)$ at short times therefore gives $S/V$ directly; this was first done in porous rocks ([Latour et al. 1993](https://doi.org/10.1006/jmra.1993.1056)).

    → The kurtosis grows at the same order: for a slab $K(t)\approx\frac{16}{5\sqrt\pi}\frac{\sqrt{Dt}}L$, the positive short-time kurtosis of Exercise 8(d). Both slab laws are the dashed lines in the figure above. Exercise 9 derives them with the method of images.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Take one flat reflecting wall at $x = 0$, a molecule starting at height $x_0 > 0$, and a free Brownian increment $W\sim\mathcal N(0, 2Dt)$. By the method of images the reflected walker ends at $\lvert x_0 + W\rvert$, so its displacement is $r = \lvert x_0 + W\rvert - x_0$.

    **(a)** Show that $r^2 - W^2 = 4x_0^2 + 4x_0W$ when $x_0 + W < 0$, and $0$ otherwise. Average over $W$ and over $x_0$ (uniform, per unit wall area) to find the deficit $\int_0^\infty\langle r^2 - W^2\rangle_W\,dx_0$. Then put two such walls on a slab of width $L$, with the molecules spread uniformly, and recover $D(t)/D = 1 - \frac{8}{3\sqrt\pi}\frac{\sqrt{Dt}}{L}$.

    **(b)** *Any shape, any dimension.* Take a pore in $d$ dimensions with a smooth wall, and define $D(t) = \langle\lvert\mathbf r\rvert^2\rangle_{\mathbf x,\mathbf x_0}/(2dt)$. Argue that at short times (i) only molecules within a few $\sqrt{Dt}$ of the wall are affected, (ii) each of them sees a flat wall, and (iii) only the displacement component normal to that wall changes. Use the deficit of (a) to derive Mitra's law.

    **(c)** Do the same as in (a) for $r^4 - W^4$, and show that the slab kurtosis grows as $K(t) \approx \frac{16}{5\sqrt\pi}\frac{\sqrt{Dt}}{L}$.

    **(d)** In the figure above, up to which $Dt/L^2$ does each dashed line stay within 5% of the exact curve? (Read $Dt/L^2$ off with the sliders: for $L = 10$ µm and $D = 2$ µm²/ms, $Dt/L^2 = 0.01$ is $t = 0.5$ ms.) Why does the kurtosis line fail first?
    """, "9 · The surface-to-volume law")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - Scale lengths by $\sigma = \sqrt{2Dt}$: $x_0 = \sigma u$, $W = \sigma w$. The deficits become $\sigma^3 I_2$ and $\sigma^5 I_4$, with pure numbers $I_2$ and $I_4$.
    - Swap the order of integration: for each $w < 0$, integrate over $u$ from 0 to $-w$ first. You need $\int_{-\infty}^0 \lvert w\rvert^k\,\phi(w)\,dw$ with $\phi$ the standard normal density: $\frac12, \frac1{\sqrt{2\pi}}, \frac12, \frac{2}{\sqrt{2\pi}}, \frac32, \frac{8}{\sqrt{2\pi}}$ for $k = 0..5$.
    - (b) With uniform density $1/V$, the number of molecules per unit wall area, within reach of the wall, is the same as in (a). How much wall is there per unit volume?
    - (c) Expand $\langle r^4\rangle_{x,x_0}/\langle r^2\rangle_{x,x_0}^2 - 3$ to first order in the small deficits.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ If $x_0 + W\ge0$ nothing is reflected and $r = W$. If $x_0 + W < 0$, $r = -2x_0 - W$, so $r^2 - W^2 = 4x_0^2 + 4x_0W$.

    ⇒ Scale with $\sigma$: deficit $= \sigma^3 I_2$, $\ I_2 = \int_{-\infty}^0\phi(w)\int_0^{-w}(4u^2 + 4uw)\,du\,dw$. The inner integral is $\frac43\lvert w\rvert^3 - 2\lvert w\rvert^3 = -\frac23\lvert w\rvert^3$, so $I_2 = -\frac23\cdot\frac{2}{\sqrt{2\pi}} = -\frac{4}{3\sqrt{2\pi}}$.

    ⇒ Two walls, molecules spread over $L$ with density $1/L$: $\langle r^2\rangle_{x,x_0} = \sigma^2 + 2\sigma^3I_2/L$, so

    $$\frac{D(t)}{D} = \frac{\langle r^2\rangle_{x,x_0}}{\sigma^2} = 1 + \frac{2\sigma I_2}{L} = 1 - \frac{8}{3\sqrt{2\pi}}\frac{\sqrt{2Dt}}{L} = 1 - \frac{8}{3\sqrt\pi}\frac{\sqrt{Dt}}{L}. \checkmark$$

    → The deficit per unit wall area, $\sigma^3I_2 = -\frac{8}{3\sqrt\pi}(Dt)^{3/2}$, is all that (b) needs.

    **(b)** ⇒ (i) A molecule further than a few $\sigma$ from the wall does not reach it: the Gaussian tail beyond is exponentially small. (ii) On the scale $\sigma$ a smooth wall with radius of curvature $\gg\sigma$ is flat. (iii) Split the step into a normal part and $d - 1$ tangential parts. They are independent, and a flat wall reflects only the normal part, so each tangential part keeps its free $2Dt$ and the normal part gives $2Dt$ plus the deficit of (a).

    ⇒ With density $1/V$, the molecules near a piece of wall of area $dS$ give the deficit $\sigma^3I_2\,dS/V$. Summed over the wall:

    $$\langle\lvert\mathbf r\rvert^2\rangle_{\mathbf x,\mathbf x_0} = 2dDt + \frac SV\,\sigma^3I_2 = 2dDt - \frac{8}{3\sqrt\pi}\,\frac SV\,(Dt)^{3/2}.$$

    ⇒ Divide by $2dDt$:

    $$\frac{D(t)}{D} = 1 - \frac{4}{3\sqrt\pi\,d}\,\frac SV\,\sqrt{Dt}. \checkmark$$

    ⇒ Slab: $S/V = 2/L$ and $d = 1$ give (a) back. ✓

    → The steps (i)–(iii) are exactly what fails one order later: curvature on the scale $\sigma$, and corners, give the $O(t)$ term. For a flat wall that term vanishes, and the slab line is exact until molecules from one wall reach the other.

    **(c)** ⇒ For $x_0 + W < 0$: $r^4 - W^4 = (2x_0 + W)^4 - W^4$. The inner integral over $u\in[0,\lvert w\rvert]$ of $(2u + w)^4 - w^4$ is $\frac{(2u+w)^5}{10}\Big|_0^{\lvert w\rvert} - \lvert w\rvert^5 = \frac{\lvert w\rvert^5}{10} + \frac{\lvert w\rvert^5}{10} - \lvert w\rvert^5 = -\frac45\lvert w\rvert^5$, so $I_4 = -\frac45\cdot\frac{8}{\sqrt{2\pi}} = -\frac{32}{5\sqrt{2\pi}}$.

    ⇒ $\langle r^4\rangle_{x,x_0} = 3\sigma^4 + 2\sigma^5I_4/L$ and $\langle r^2\rangle_{x,x_0}^2 = \sigma^4(1 + 4\sigma I_2/L + \dots)$, so

    $$K = \frac{3 + 2\sigma I_4/L}{1 + 4\sigma I_2/L} - 3 \approx \frac{2\sigma}{L}\big(I_4 - 6I_2\big) = \frac{2\sigma}{L}\cdot\frac{8/5}{\sqrt{2\pi}} = \frac{16}{5\sqrt\pi}\frac{\sqrt{Dt}}{L} = 1.805\,\frac{\sqrt{Dt}}{L}.$$

    The mode sums give 1.79 at $Dt = 10^{-4}L^2$ and 1.75 at $10^{-3}L^2$, approaching 1.805.

    **(d)** ⇒ $D(t)$: the line is exact (to $10^{-6}$) up to $Dt\approx0.03\,L^2$ and still within 0.2% at $0.08\,L^2$. The two walls act independently until the molecules from one wall reach the other. $K(t)$: within 5% only up to $Dt\approx 2\times10^{-3}L^2$; it peaks at $+0.20$ near $0.03\,L^2$ and changes sign near $0.08\,L^2$, on its way to $-3/5$.

    → The kurtosis is a small excess divided by $\langle r^2\rangle_{x,x_0}^2$. Its next correction is already of relative order $\sqrt{Dt}/L$, while for $D(t)$ next to a flat wall the next correction is exponentially small.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When the interfaces are not reflecting

    Everything so far assumed impermeable walls. The recipe for the eigenmodes stays the same, $-\nabla^2u = \lambda u$, but the condition at the interfaces changes, and with it the modes:

    - **Permeable membranes.** The compartments are no longer separate problems. The eigenmodes belong to the **whole domain**: each mode spans both sides of a membrane, with a continuous flux and a jump in value set by $\kappa$. With $\kappa = 0$ every compartment has its own uniform mode ($\lambda = 0$ once per compartment). Opening the membrane turns all but one of these into slowly decaying **exchange modes**. Their rate is the exchange rate between compartments, the basis of the [Kärger (1985)](<https://doi.org/10.1016/0001-8686(85)80018-X>) model (Chapter 3).
    - **Partially absorbing walls.** The lowest eigenvalue is no longer zero. Even the most uniform mode fades: molecules are lost at the walls. For weak absorption its rate is $D\lambda_0 \approx \varrho\,S/V$, the surface-to-volume ratio times the absorption rate ([Brownstein & Tarr 1979](https://doi.org/10.1103/PhysRevA.19.2446)).
    - **Perfect sinks.** All modes vanish at the wall, and the slowest one fades with $\lambda_0 = (\pi/L)^2$ in a slab: nothing survives.
    - **Periodic boundaries.** The modes become $\cos(2\pi n x/L)$ and $\sin(2\pi n x/L)$, each eigenvalue appearing twice. The uniform mode survives: nothing is lost, it just leaves on one side and re-enters on the other.

    The modes remain orthonormal in all these cases, so everything in the previous sections (projection, propagator, time scales) carries over unchanged.

    Try it in a slab of $L = 8$ µm with $D = 2$ µm²/ms, a membrane in the middle, and a choice of outer walls. The modes are computed numerically (a fine grid with the interface rules built in), so you can compare them with the formulas of the next exercise.
    """)
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
    The slab modes of Exercise 2 came from one smooth function on one interval. With a membrane, the domain is cut into pieces, and the recipe gains one step: gluing the pieces together. In 1D, for a slab with a membrane at $x = 0$ between $[-a, 0]$ and $[0, a]$:

    ① **Solve piece by piece.** On each piece $D$ is constant and $-u'' = \lambda u$, so with $k = \sqrt\lambda$

    $$u = A_1\cos kx + B_1\sin kx\ \ (x < 0),\qquad u = A_2\cos kx + B_2\sin kx\ \ (x > 0).$$

    Four unknowns, $A_1, B_1, A_2, B_2$, and one eigenvalue to find.

    ② **Outer walls: one condition each.**

    | Wall | Condition at $x = \pm a$ |
    | --- | --- |
    | reflecting | $u' = 0$ |
    | partially absorbing | $D\,\partial u/\partial n = -\varrho\,u$, i.e. $Du'(a) = -\varrho u(a)$ and $Du'(-a) = +\varrho u(-a)$ |
    | perfect sink | $u = 0$ |
    | periodic | $u(-a) = u(a)$ and $u'(-a) = u'(a)$ (two conditions, no membrane needed) |

    ③ **The membrane: two conditions.** The flux is continuous, $Du'(0^-) = Du'(0^+)$, and it is driven by the jump, $-Du'(0) = \kappa\,\big(u(0^-) - u(0^+)\big)$.

    ④ **The eigenvalue condition.** Four conditions on four unknowns, all linear and with no right-hand side: $\mathbf M(k)\,\mathbf c = 0$. A mode exists only if the system has a nonzero solution, $\det\mathbf M(k) = 0$. That is a **transcendental equation** for $k$, with infinitely many roots $k_0 < k_1 < \dots$, and $\lambda_n = k_n^2$. Symmetry halves the work: for a membrane in the middle, the even modes do not see it and the odd ones give $k\tan(ka) = 2\kappa/D$ (Exercise 10). One absorbing wall on each side of a plain slab gives $k\tan(ka) = \varrho/D$ for the even modes.

    ⑤ **Normalise** over all pieces together: $\int_{-a}^0u^2\,dx + \int_0^au^2\,dx = 1$.

    ⑥ **Orthogonality comes for free.** Integrate $u_mu_n'' - u_nu_m''$ over each piece: what is left are boundary terms $[u_mu_n' - u_nu_m']$. At a reflecting, absorbing or periodic wall they cancel, because both modes obey the same condition. At the membrane write $f_n = -Du_n'(0)$ for the flux of mode $n$. The two sides together give $-\frac1D\big[f_n(u_m^- - u_m^+) - f_m(u_n^- - u_n^+)\big] = -\frac1D\big[\frac{f_nf_m}{\kappa} - \frac{f_mf_n}{\kappa}\big] = 0$. So $(\lambda_m - \lambda_n)\int u_mu_n = 0$, and modes with different eigenvalues are orthogonal.

    ⑦ **Propagator, as before:** $P(x\mid x_0,t) = \sum_nu_n(x)\,u_n(x_0)\,e^{-D\lambda_nt}$.

    → With a different $D$ on each side, solve $-(Du')' = \mu u$ instead; $\mu_n$ are then the decay rates themselves, and the same steps go through.

    → **What the figure does instead.** Cut $[0, L]$ into small cells and let each face between two cells carry a conductance: $D/h$ in the bulk, $1/(h/D + 1/\kappa)$ at the membrane (the two half cells and the membrane as resistances in series), $1/(h/2D + 1/\varrho)$ at an absorbing wall. The eigenvectors of the resulting matrix are the modes, without solving any transcendental equation. Chapter 4 explains this method in full.
    """, title="Recipe: modes with membranes and absorbing walls")
    return


@app.cell(hide_code=True)
def _(mo):
    outer3 = mo.ui.dropdown(
        options=["reflecting", "partially absorbing (Robin)", "perfect sink (Dirichlet)", "periodic"],
        value="reflecting",
        label="Outer walls",
    )
    logrho3 = mo.ui.slider(-3, 1, step=0.1, value=-1, label="log₁₀ ϱ (ϱ in µm/ms, absorbing walls only)", show_value=True)
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
        if outer == "partially absorbing (Robin)" and rho > 0:
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
        _notes.append("The lowest mode does not decay: no molecules are lost.")
    else:
        _notes.append(f"Even the lowest mode decays, at **{_D * _lam[0]:.4f} /ms** (decay time {1 / (_D * _lam[0]):.1f} ms): molecules are lost at the walls.")
        if outer3.value == "partially absorbing (Robin)":
            _notes.append(f"Compare ϱ·S/V = 2ϱ/L = **{2 * _rho / _L:.4f} /ms**, the weak-absorption estimate.")
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
        """, "10 · Eigenmodes with a permeable membrane")
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

        → Interpretation: a molecule in one half crosses at a rate $\kappa\,S/V = \kappa/a$. The difference between the two halves decays at the sum of the two crossing rates, $2\kappa/a$. That is exactly the exchange rate of a two-compartment (Kärger) model. With $\kappa = 0.03$ µm/ms and $L = 8$ µm it is 0.015 /ms: an exchange time of about 70 ms. Check it with the sliders.
        → The same steps with a partially absorbing wall instead of a membrane give $k\tan(kL/2) = \varrho/D$ for the lowest mode, and $D\lambda_0 \approx 2\varrho/L = \varrho\,S/V$ for weak absorption.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two compartments that exchange: the Kärger model

    ### From the modes to two numbers

    Go back to the slab with a membrane in the middle (Exercise 10). Its modes are exact, and the figure below shows how fast each of them fades as the membrane becomes more permeable.

    ⇒ **One mode never fades:** the uniform mode, $\lambda = 0$.

    ⇒ **One mode fades slowly:** the exchange mode, one half high and the other low, at the rate $\approx4\kappa/L$. It is the only mode that needs molecules to cross the membrane.

    ⇒ **All other modes fade fast:** they are patterns *inside* a half, and the slowest of them fades at $D(2\pi/L)^2$. When the membrane is the bottleneck, $\kappa L/D\ll1$, there is a wide gap between $4\kappa/L$ and these rates.

    → After a time of order $L^2/(4\pi^2D)$ only the first two modes are left. The concentration is then flat inside each half, and two numbers describe everything: the amounts in the two halves. Their equation is the mode equation cut down to those two modes. **The two slowest modes of the full problem become a 2×2 rate matrix** (Exercise 11a).

    ### From closed halves to Gaussian compartments

    In the slab each half is closed, so a molecule's displacement stays bounded and is not Gaussian. Kärger's model is meant for a medium with many units of each kind, such as a tissue of many cells. Watched over distances much larger than one unit, a molecule that stays within one kind of compartment adds up many independent steps, and its displacement is Gaussian again, with an effective diffusivity $D_i$ (the argument of Exercise 1). Molecules shut inside closed units have an effective $D_i$ near zero on that scale.

    [Kärger (1985)](<https://doi.org/10.1016/0001-8686(85)80018-X>) combines the two limits. *Which* compartment a molecule is in follows the two-mode rate equation, with rates $k_1$ (out of 1) and $k_2$ (out of 2). *How far* it gets while in compartment $i$ is free diffusion with $D_i$.

    ### The model

    ⇒ At equilibrium the fractions $p_1$, $p_2$ do not change: as many molecules leave 1 as enter it, $k_1p_1 = k_2p_2$ (**detailed balance**). One time scale is left, the exchange time $\tau_{\text{ex}} = 1/(k_1 + k_2)$.

    ⇒ Given the sequence of compartments it visited, a molecule's displacement is Gaussian, with variance $2\int_0^tD(t')\,dt'$; $D(t')$ jumps between $D_1$ and $D_2$. The distribution of displacements is a **mixture of Gaussians**, with widths set by how long each molecule spent where.

    ⇒ Keep track of the characteristic function separately for the molecules that are in compartment $i$ at time $t$, $f_i(\xi,t)$. Diffusion inside and hopping between give a linear system (Exercise 11b):

    $$\boxed{\ \frac{d\mathbf f}{dt} = -\big(\xi^2\mathbf D + \mathbf K\big)\,\mathbf f,\qquad \mathbf D = \begin{pmatrix}D_1&0\\0&D_2\end{pmatrix},\quad \mathbf K = \begin{pmatrix}k_1&-k_2\\-k_1&k_2\end{pmatrix},\quad \mathbf f(0) = \mathbf p\ },$$

    so that $\langle e^{i\xi r}\rangle_{x,x_0} = (1,1)\,e^{-(\xi^2\mathbf D + \mathbf K)t}\,\mathbf p$. (Bold $\mathbf K$ is the matrix of rates; italic $K$ stays the kurtosis.)

    → **Short times**, $t\ll\tau_{\text{ex}}$: nobody has switched yet. Two Gaussians side by side, with positive kurtosis $K_0 = 3p_1p_2(D_1 - D_2)^2/\bar D^2$.

    → **Long times**, $t\gg\tau_{\text{ex}}$: every molecule has sampled both compartments many times. One Gaussian, with $\bar D = p_1D_1 + p_2D_2$.

    → **The mean-squared displacement does not see exchange at all**: $\langle r^2\rangle_{x,x_0} = 2\bar Dt$ at every time. Only the kurtosis does: it falls from $K_0$ to zero, as $1/t$ at long times. Compare the slab of Exercise 8, where $D(t)$ falls and $K$ goes to $-3/5$. Restriction and exchange leave different fingerprints in $D(t)$ and $K(t)$.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, brentq, np, plt):
    _L, _D = 8.0, 2.0
    _a = _L / 2
    _kap = np.logspace(-4, 1.5, 120)

    def _odd(kappa, n):
        """n-th odd mode of the slab [-a, a] with a membrane at 0: k tan(ka) = 2κ/D."""
        f = lambda z: z * np.tan(z) - 2 * kappa * _a / _D      # z = k a
        return brentq(f, n * np.pi + 1e-12, n * np.pi + np.pi / 2 - 1e-12) / _a

    _ex = np.array([_D * _odd(k, 0) ** 2 for k in _kap])
    _odd1 = np.array([_D * _odd(k, 1) ** 2 for k in _kap])
    _even1 = np.full(_kap.size, _D * (np.pi / _a) ** 2)

    _fig, _ax = plt.subplots(figsize=(6.6, 4.4))
    _ax.loglog(_kap, _ex, color=C_MAIN, lw=2.4, label="exchange mode (one half high, one low)")
    _ax.loglog(_kap, _even1, color=C_FREE, lw=1.6, label="first pattern inside a half")
    _ax.loglog(_kap, _odd1, color=C_FREE, lw=1.6, ls="--", label="second odd mode")
    _ax.loglog(_kap, 4 * _kap / _L, color=C_ALT, lw=1.1, ls=":", label="4κ/L")
    _ax.axvline(_D / _L, color="k", lw=0.8, ls=":")
    _ax.text(_D / _L * 1.15, 6, "κL/D = 1", fontsize=8.5)
    _ax.fill_between(_kap, _ex, _even1, where=_kap < 0.1 * _D / _L, color=C_MAIN, alpha=0.08, lw=0)
    _ax.text(3e-4, 0.03, "the gap", fontsize=9, color=C_MAIN)
    _ax.set(xlabel="membrane permeability κ (µm/ms)", ylabel="decay rate Dλ (1/ms)", ylim=(1e-4, 20),
            title=f"Mode rates of a slab with a membrane (L = {_L:g} µm, D = {_D:g} µm²/ms)")
    _ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    kx_t = mo.ui.slider(-0.5, 3.0, step=0.05, value=1.0, label="log₁₀ t (ms)", show_value=True)
    kx_tau = mo.ui.slider(0.0, 2.5, step=0.05, value=1.3, label="log₁₀ τ_ex (ms)", show_value=True)
    kx_p = mo.ui.slider(0.1, 0.9, step=0.05, value=0.4, label="fraction p₁ in the fast compartment", show_value=True)
    mo.vstack([kx_t, kx_tau, kx_p])
    return kx_p, kx_t, kx_tau


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, kx_p, kx_t, kx_tau, mo, np, plt):
    _D1, _D2 = 2.0, 0.5
    _t, _tau, _p1 = 10**kx_t.value, 10**kx_tau.value, kx_p.value
    _p = np.array([_p1, 1 - _p1])
    _k1, _k2 = _p[1] / _tau, _p[0] / _tau            # k1 + k2 = 1/τ_ex and k1 p1 = k2 p2
    _Db = _p1 * _D1 + (1 - _p1) * _D2
    _K0 = 3 * _p1 * (1 - _p1) * (_D1 - _D2) ** 2 / _Db**2

    # characteristic function on a q grid, then back to the distribution of displacements
    _q = np.linspace(0, np.sqrt(40 / (_D2 * _t)), 1500)
    _A = np.zeros((_q.size, 2, 2))
    _A[:, 0, 0] = _q**2 * _D1 + _k1
    _A[:, 0, 1] = -_k2
    _A[:, 1, 0] = -_k1
    _A[:, 1, 1] = _q**2 * _D2 + _k2
    _w, _V = np.linalg.eig(-_A * _t)
    _M = np.einsum("qij,qj,qjk->qik", _V, np.exp(_w), np.linalg.inv(_V)).real
    _f = _M.sum(1) @ _p
    _X = np.linspace(-4 * np.sqrt(2 * _D1 * _t), 4 * np.sqrt(2 * _D1 * _t), 301)
    _wq = np.full(_q.size, _q[1]); _wq[[0, -1]] /= 2
    _P = (np.cos(np.outer(_X, _q)) @ (_wq * _f)) / np.pi
    _gauss = lambda d: np.exp(-_X**2 / (4 * d * _t)) / np.sqrt(4 * np.pi * d * _t)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    _a1.semilogy(_X, np.clip(_P, 1e-12, None), color=C_MAIN, lw=2.2, label="with exchange")
    _a1.semilogy(_X, _p1 * _gauss(_D1) + (1 - _p1) * _gauss(_D2), color=C_ALT, lw=1.2, ls="--", label="no exchange: two Gaussians")
    _a1.semilogy(_X, _gauss(_Db), color=C_FREE, lw=1.2, ls=":", label="fast exchange: one Gaussian, D̄")
    _a1.set(xlabel="displacement r = x − x₀ (µm)", ylabel="density", ylim=(_P.max() * 1e-4, _P.max() * 2),
            title=f"Distribution of displacements at t = {_t:.3g} ms")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower center")

    _xx = np.logspace(-2, 3, 300)
    _shape = 2 * (_xx - 1 + np.exp(-_xx)) / _xx**2
    _a2.semilogx(_xx, _K0 * _shape, color=C_MAIN, lw=2.2, label="K(t)")
    _a2.semilogx(_xx[_xx > 3], 2 * _K0 / _xx[_xx > 3], color=C_ALT, lw=1.1, ls="--", label="long times: 2K₀ τ_ex / t")
    _a2.axhline(_K0, color=C_FREE, lw=0.8, ls=":")
    _xt = _t / _tau
    _Kt = _K0 * 2 * (_xt - 1 + np.exp(-_xt)) / _xt**2
    _a2.plot(_xt, _Kt, "o", color=C_MAIN, ms=7, mec="k", mew=0.5)
    _a2.set(xlabel="t / τ_ex", ylabel="excess kurtosis K", ylim=(0, _K0 * 1.12), title="The kurtosis is what exchange changes")
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"D₁ = {_D1}, D₂ = {_D2} µm²/ms, D̄ = **{_Db:.2f}** µm²/ms; τ_ex = **{_tau:.3g} ms**, t/τ_ex = **{_xt:.3g}**. "
        f"K₀ = **{_K0:.3f}**, K(t) = **{_Kt:.3f}**. The mean-squared displacement is 2D̄t = {2 * _Db * _t:.3g} µm² at every exchange time."
    )])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** For the slab with a membrane, the amounts in the two halves obey $d\mathbf n/dt = -\mathbf K\mathbf n$ with $k_1 = k_2 = 2\kappa/L$. Find the eigenvalues and eigenvectors of $\mathbf K$ and match them with the two slowest modes of Exercise 10.

    **(b)** Over a short time $dt$, a molecule in compartment 1 either stays (probability $1 - k_1dt$) and takes a Gaussian step of variance $2D_1dt$, or hops to compartment 2 (probability $k_1dt$). Use this to derive $d\mathbf f/dt = -(\xi^2\mathbf D + \mathbf K)\mathbf f$ with $\mathbf f(0) = \mathbf p$.

    **(c)** Show that $\langle r^2\rangle_{x,x_0} = 2\bar Dt$ at every time, whatever the exchange rate.

    **(d)** Given its history, a molecule's displacement is Gaussian with variance $2Y$, where $Y = \int_0^tD(t')\,dt'$. Show that $K(t) = 3\,\mathrm{Var}(Y)/(\bar Dt)^2$. Then use the correlation of a two-state switch, $\langle(D(t') - \bar D)(D(t'') - \bar D)\rangle_{\text{paths}} = p_1p_2(D_1 - D_2)^2e^{-\lvert t' - t''\rvert/\tau_{\text{ex}}}$, to derive

    $$K(t) = K_0\,\frac{2\big(x - 1 + e^{-x}\big)}{x^2},\qquad x = \frac{t}{\tau_{\text{ex}}},\qquad K_0 = \frac{3p_1p_2(D_1 - D_2)^2}{\bar D^2}.$$

    **(e)** With the sliders: at which $t/\tau_{\text{ex}}$ has $K$ dropped to half of $K_0$? Does it depend on $p_1$? How does $K$ fall at long times, and what would you measure over a range of times to tell exchange from restriction (Exercise 8)?
    """, "11 · Two compartments that exchange")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) $\mathbf K = k\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$: try the vectors $(1,1)$ and $(1,-1)$.
    - (b) $f_1(\xi, t + dt) = (1 - k_1dt)\,e^{-\xi^2D_1dt}f_1(\xi,t) + k_2dt\,f_2(\xi,t)$. Expand to first order in $dt$.
    - (c) Two facts: $\mathbf K\mathbf p = 0$ (detailed balance) and $(1,1)\,\mathbf K = 0$ (hops lose no molecules). Expand $(1,1)\,e^{-(\xi^2\mathbf D + \mathbf K)t}\,\mathbf p$ to first order in $\xi^2$. Or think: at every moment the population is at equilibrium, so its average diffusivity is $\bar D$.
    - (d) For a Gaussian of variance $\sigma^2$, $\langle r^4\rangle = 3\sigma^4$. And $\int_0^t\!\int_0^te^{-\lvert t' - t''\rvert/\tau}\,dt'\,dt'' = 2\tau^2\big(x - 1 + e^{-x}\big)$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\mathbf K(1,1)^{\mathsf T} = 0$: eigenvalue 0, both halves equal, the **uniform mode**. $\mathbf K(1,-1)^{\mathsf T} = 2k\,(1,-1)^{\mathsf T}$: eigenvalue $2k = 4\kappa/L$, one half high and the other low, the **exchange mode** with the rate of Exercise 10(d). ✓

    → The 2×2 matrix keeps exactly the two slowest modes and drops all patterns inside the halves. With unequal halves the same holds with $k_1p_1 = k_2p_2$, and the eigenvalues are $0$ and $k_1 + k_2$.

    **(b)** ⇒ To first order in $dt$: $f_1(t + dt) = (1 - k_1dt)(1 - \xi^2D_1dt)f_1 + k_2dt\,f_2$, so $\dot f_1 = -(\xi^2D_1 + k_1)f_1 + k_2f_2$. In the same way $\dot f_2 = k_1f_1 - (\xi^2D_2 + k_2)f_2$. ✓ At $t = 0$ nobody has moved, $r = 0$, so $f_i(\xi, 0) = p_i$.

    → The same structure as the mode equations of the eigenmodes section: something that happens inside each compartment ($\xi^2\mathbf D$, diagonal) and a coupling between them ($\mathbf K$).

    **(c)** ⇒ Derivative with respect to $\xi^2$ at $\xi = 0$: $-\int_0^t(1,1)\,e^{-\mathbf K(t - t')}\,\mathbf D\,e^{-\mathbf Kt'}\,\mathbf p\,dt'$. With $e^{-\mathbf Kt'}\mathbf p = \mathbf p$ and $(1,1)e^{-\mathbf K(t-t')} = (1,1)$, this is $-t\,(1,1)\mathbf D\mathbf p = -\bar Dt$.

    ⇒ So $\ln\langle e^{i\xi r}\rangle_{x,x_0} = -\xi^2\bar Dt + O(\xi^4)$, and $\langle r^2\rangle_{x,x_0} = 2\bar Dt$. ✓

    → Each molecule's diffusivity changes, but the population as a whole stays at equilibrium, so its average diffusivity is $\bar D$ at every moment.

    **(d)** ⇒ $\langle r^2\rangle = 2\langle Y\rangle = 2\bar Dt$ and $\langle r^4\rangle = 3\langle(2Y)^2\rangle = 12\langle Y^2\rangle$, so

    $$K = \frac{12\langle Y^2\rangle}{4\bar D^2t^2} - 3 = \frac{3\big(\langle Y^2\rangle - \langle Y\rangle^2\big)}{(\bar Dt)^2} = \frac{3\,\mathrm{Var}(Y)}{(\bar Dt)^2} .$$

    ⇒ $\mathrm{Var}(Y) = \int_0^t\!\int_0^t\langle(D(t') - \bar D)(D(t'') - \bar D)\rangle\,dt'\,dt'' = p_1p_2(D_1 - D_2)^2\cdot2\tau_{\text{ex}}^2\big(x - 1 + e^{-x}\big)$.

    ⇒ Divide by $(\bar Dt)^2 = \bar D^2\tau_{\text{ex}}^2x^2$: $K = K_0\cdot2(x - 1 + e^{-x})/x^2$. ✓

    → Limits: for $x\to0$, $x - 1 + e^{-x}\approx x^2/2$, so $K\to K_0$: the two Gaussians of a mixture (Exercise 8d). For $x\to\infty$, $K\approx2K_0/x = 2K_0\tau_{\text{ex}}/t$.

    → Where the correlation comes from: at equal times $D(t') - \bar D$ has variance $p_1p_2(D_1 - D_2)^2$, and a two-state switch forgets its state at the total rate $k_1 + k_2 = 1/\tau_{\text{ex}}$.

    **(e)** ⇒ $K = K_0/2$ at $x\approx2.6$, for every $p_1$: the shape $2(x - 1 + e^{-x})/x^2$ does not depend on $p_1$, $D_1$ or $D_2$; they only set $K_0$.

    ⇒ At long times $K$ falls as $1/t$ towards zero, and $D(t)$ stays at $\bar D$ throughout. In a closed pore $D(t)$ falls and $K$ goes negative. Measure both over a range of times: a constant $D(t)$ with a kurtosis that decays to zero points to exchange; a falling $D(t)$ with a kurtosis that turns negative points to restriction. Chapter 3 shows the same model in the measured signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 1b · The diffusion MRI signal: what a gradient does, and two routes to it

    ## What a gradient does

    After the 90° pulse the magnetization lies in the transverse plane. Write it as one complex number, $m = M_x + iM_y$: an arrow with length $\lvert m\rvert$ and angle $\arg m$. In the frame rotating at the Larmor frequency all arrows stand still, until a **diffusion gradient** is switched on. The gradient adds a magnetic field that grows linearly with position, $\mathbf g\cdot\mathbf x$, so a spin at $\mathbf x$ precesses at the angular frequency $\gamma\,\mathbf g\cdot\mathbf x$.

    ⇒ **The gradient labels position with phase.** During a pulse each spin gathers a phase proportional to where it is. After a lobe with area $\int\mathbf g\,dt$ the phase winds along the gradient direction like a corkscrew, $\varphi = \mathbf q\cdot\mathbf x$, with wave vector $\mathbf q = \gamma\int\mathbf g\,dt$: one full turn every $2\pi/q$ (panel b).

    ⇒ **The 180° pulse and the second lobe unwind it.** The 180° pulse of the spin echo ([Hahn 1950](https://doi.org/10.1103/PhysRev.80.580)) flips the sign of all phase gathered so far, and the second lobe then adds the same phase again. We fold the 180° pulse into an **effective gradient** $\mathbf g(t) = f(t)\,\mathbf g$ that changes sign at the pulse: one positive lobe, one negative lobe (panel a).

    ⇒ **Only motion leaves a phase behind.** A spin that has not moved returns to exactly zero phase at the echo. A spin that moved keeps $\varphi = \gamma\int_0^T\mathbf g(t)\cdot\mathbf x(t)\,dt\ne0$. The scanner records the sum of all arrows in the voxel. Arrows that point in different directions partly cancel, and the signal drops (panel c).

    → The attenuation $E = S/S_0$ measures how the spins moved while the gradient was on. The geometry of 1a enters because the walls limit that motion.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, np, plt):
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.15, 1.2, 1.2]})

    # (a) the effective gradient of a PGSE sequence
    _dl, _DL = 10.0, 30.0
    _t = np.linspace(-2, _DL + _dl + 4, 800)
    _lobe1 = ((_t >= 0) & (_t <= _dl)).astype(float)
    _lobe2 = ((_t >= _DL) & (_t <= _DL + _dl)).astype(float)
    _a1.plot(_t, _lobe1 + _lobe2 + 0.04, color=C_FREE, lw=1.3, ls="--", label="gradient as played")
    _a1.plot(_t, _lobe1 - _lobe2, color=C_MAIN, lw=2.2, label="effective gradient f(t)")
    _t180 = (_dl + _DL) / 2
    _a1.axvline(_t180, color="k", lw=0.8, ls=":")
    _a1.text(_t180 + 0.8, -0.55, "180° pulse", ha="left", fontsize=8.5)
    _a1.annotate("", xy=(0, -1.25), xytext=(_dl, -1.25), arrowprops=dict(arrowstyle="<->", lw=0.9))
    _a1.text(_dl / 2, -1.45, "δ", ha="center", va="top", fontsize=10)
    _a1.annotate("", xy=(0, 1.15), xytext=(_DL, 1.15), arrowprops=dict(arrowstyle="<->", lw=0.9))
    _a1.text(_DL / 2, 1.2, "Δ", ha="center", va="bottom", fontsize=10)
    _a1.set(xlabel="t (ms)", yticks=[-1, 0, 1], ylim=(-1.8, 1.6), title="(a) Effective gradient")
    _a1.legend(frameon=False, fontsize=7.5, loc="lower right")

    def _arrows(ax, x, y, phi, color, scale=0.55):
        ax.quiver(x, y, scale * np.sin(phi), scale * np.cos(phi), angles="xy", scale_units="xy", scale=1,
                  color=color, width=0.007, headwidth=4, headlength=4.5)

    # (b) after the first lobe: phase winds with position
    _x = np.linspace(-4, 4, 17)
    _q = 2 * np.pi / 5.0                       # one turn every 5 µm
    _a2.axhline(0, color="0.8", lw=0.8)
    _arrows(_a2, _x, np.zeros_like(_x), _q * _x, C_MAIN)
    _a2.annotate("", xy=(-2.5, -1.0), xytext=(2.5, -1.0), arrowprops=dict(arrowstyle="<->", lw=0.9))
    _a2.text(0, -1.12, "one turn: 2π/q", ha="center", va="top", fontsize=9)
    _a2.text(0, 1.05, r"phase $\varphi = q\,x$", ha="center", fontsize=10)
    _a2.set(xlim=(-4.8, 4.8), ylim=(-1.6, 1.4), xlabel="position x along the gradient (µm)", yticks=[],
            title="(b) After the first lobe")

    # (c) at the echo: static spins refocus, moving spins keep a phase
    _rng = np.random.default_rng(7)
    _xs = np.linspace(-4, 2.0, 11)
    _phi_move = _rng.normal(0, 1.0, _xs.size)
    _a3.axhline(1.0, color="0.85", lw=0.8)
    _a3.axhline(-0.4, color="0.85", lw=0.8)
    _arrows(_a3, _xs, np.full_like(_xs, 1.0), np.zeros_like(_xs), C_MAIN)
    _arrows(_a3, _xs, np.full_like(_xs, -0.4), _phi_move, C_ALT)
    _E_move = np.mean(np.exp(-1j * _phi_move))
    _a3.text(-4.3, 1.75, "spins that did not move", fontsize=8.5)
    _a3.text(-4.3, 0.35, "spins that diffused", fontsize=8.5, color=C_ALT)
    _a3.plot([2.75, 2.75], [-1.3, 2.0], color="0.7", lw=0.8)
    _arrows(_a3, np.array([3.6]), np.array([1.0]), np.array([0.0]), C_MAIN, scale=0.9)
    _a3.quiver([3.6], [-0.4], [0.9 * _E_move.imag * -1], [0.9 * _E_move.real], angles="xy", scale_units="xy",
               scale=1, color=C_ALT, width=0.012, headwidth=4, headlength=4.5)
    _a3.text(3.6, 2.05, "average", ha="center", fontsize=8.5)
    _a3.text(3.6, 0.85, "E = 1", ha="center", va="top", fontsize=8.5)
    _a3.text(3.6, -0.55, f"E = {abs(_E_move):.2f}", ha="center", va="top", fontsize=8.5, color=C_ALT)
    _a3.set(xlim=(-4.6, 4.6), ylim=(-1.3, 2.3), xticks=[], yticks=[], title="(c) At the echo")
    for _ax in (_a2, _a3):
        for _sp in ("left", "right", "top"):
            _ax.spines[_sp].set_visible(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two routes to the signal

    The picture above suggests two ways to compute $E$:

    | | **Field picture** | **Path picture** |
    | --- | --- | --- |
    | Follows | the magnetization $m(\mathbf x,t)$ everywhere in the voxel | single spins, each on its own random path $\mathbf x(t)$ |
    | Computes | a PDE, the Bloch–Torrey equation; then $S = \int m\,d\mathbf x$ at the echo | each spin's phase $\varphi = \gamma\int\mathbf g\cdot\mathbf x\,dt$; then the average $E = \langle e^{-i\varphi}\rangle_{\text{paths}}$ |
    | The geometry enters as | interface conditions, through the eigenmodes of 1a | walls that reflect the walkers, or let them through |
    | Methods in this course | eigenmodes and matrices (Ch 2a), formulas (Ch 3), grids and meshes (Ch 4) | the cumulant expansion (Ch 2b), Monte Carlo (Ch 5) |

    The two routes give exactly the same $E$. That is the **Feynman–Kac relation**, which you derive at the end of the path picture (Exercise 14). The next two sections take the routes one at a time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The field picture: the Bloch–Torrey equation

    ### From Bloch to Bloch–Torrey

    Precession alone turns each arrow at its own rate: $\partial m/\partial t = -i\gamma\,\mathbf g(t)\cdot\mathbf x\,m$. The spins also diffuse, and magnetization is carried with them exactly like the concentration in 1a. [Torrey (1956)](https://doi.org/10.1103/PhysRev.104.563) added the two effects to [Bloch's (1946)](https://doi.org/10.1103/PhysRev.70.460) equations:

    $$\boxed{\ \frac{\partial m}{\partial t} = D\,\nabla^2 m \;-\; i\gamma\,\mathbf g(t)\cdot\mathbf x\; m\ }\qquad m(\mathbf x,0) = \rho,\qquad \frac{\partial m}{\partial n}=0 \text{ on impermeable interfaces}.$$

    - **Diffusion** smooths phase differences out; the **gradient** winds them up. Every regime in this course is a contest between those two.
    - $\mathbf g(t)$ is the effective gradient, so the 180° pulse is already in the equation.
    - The scanner measures the sum over the voxel: $S(T) = \int_\Omega m(\mathbf x,T)\,d\mathbf x$, and we report $E = S/S_0$.
    - The conditions are those of 1a, so the **geometry enters exactly as before**: through the eigenmodes. The gradient term multiplies $m$ by $\mathbf g\cdot\mathbf x$, which mixes the modes through the position matrix $\mathbf B$ of 1a (Chapter 2). [Robertson (1966)](https://doi.org/10.1103/PhysRev.151.273) was the first to expand the magnetization in the modes of a slab.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The full problem every simulator solves

    The magnetization lives in the same compartments, with the same conditions, as the concentration in 1a; the block of tissue $\Omega$ is now the imaging voxel, or a piece of it. Two things are added: the gradient, and relaxation, which can differ between compartments. With the gradient written as $\mathbf g(t) = f(t)\,\mathbf g$ (for PGSE, $f = +1$ during the first lobe and $-1$ during the second, once the 180° pulse is folded in), each compartment obeys

    $$\boxed{\ \frac{\partial m_n}{\partial t} = \nabla\cdot(D_n\nabla m_n) - \frac{m_n}{T_{2,n}} - i\gamma f(t)\,(\mathbf g\cdot\mathbf x)\,m_n\ \text{ in }\Omega_n,\qquad m_n(\mathbf x,0) = \rho_n\ },$$

    | Where | Condition | Meaning |
    | --- | --- | --- |
    | outer boundary $\Gamma_n$ | $D_n\,\partial m_n/\partial n = 0$ (or periodic, see below) | no spins leave the simulated block |
    | membrane $\Gamma_{nm}$ | $D_n\,\partial m_n/\partial n_n = -D_m\,\partial m_m/\partial n_m$ | what leaves one side enters the other |
    | membrane $\Gamma_{nm}$ | $D_n\,\partial m_n/\partial n_n = \kappa_{nm}\big(c_{nm}m_m - c_{mn}m_n\big)$ | the flux is driven by the jump across the membrane |

    and the signal is $S(T) = \int_\Omega m(\mathbf x, T)\,d\mathbf x$, $E = S/S_0$.

    Two conditions of 1a change meaning for magnetization:

    - The partially absorbing wall becomes a **relaxing wall**: $\varrho$ is the surface relaxivity, the rate at which a wall destroys magnetization, for example through paramagnetic sites on the wall ([Brownstein & Tarr 1979](https://doi.org/10.1103/PhysRevA.19.2446)). It matters in rocks and porous media, and is rarely needed in soft tissue.
    - A **periodic** outer boundary must remember the gradient. A spin that leaves through one face and re-enters through the opposite face has crossed one period $\mathbf L$ of the tissue, and the gradient has wound a phase across that distance. So $m$ matches on opposite faces only up to that phase: $m(\mathbf x + \mathbf L, t) = e^{-i\mathbf q(t)\cdot\mathbf L}\,m(\mathbf x, t)$, with $\mathbf q(t) = \gamma\int_0^t\mathbf g\,dt'$ (Chapter 4 shows how simulators use it).

    The simulators of Chapters 4 and 5 solve this problem as it stands. The exact methods of Chapter 2 and the formulas of Chapter 3 simplify it in three steps:

    ① **Uniform $T_2$:** relaxation then multiplies the signal by $e^{-T/T_2}$, which cancels in $E$ (Exercise 12). Drop the $T_2$ term.
    ② **Equal densities:** $c_{nm} = 1$; with $\rho = 1$ the start is $m = 1$.
    ③ **One compartment:** impermeable interfaces, so each pore is solved on its own. This is the boxed equation above.

    → Each step is a statement about the tissue, and each can be false: compartments do relax differently (Exercise 12), and membranes do leak. Chapters 4 and 5 are how to find out what a simplification costs.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    With transverse relaxation the equation reads $\partial m/\partial t = D\nabla^2 m - i\gamma\,\mathbf g\cdot\mathbf x\,m - m/T_2$.
    Show that, for a uniform $T_2$, relaxation only multiplies the signal by $e^{-T/T_2}$, so that it cancels in $E = S/S_0$ when $S_0$ is measured at the same echo time.
    """, "12 · Where did T₂ go?")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    ⇒ Substitute $m = e^{-t/T_2}\,\tilde m$. Then $\partial m/\partial t = e^{-t/T_2}\left(\partial\tilde m/\partial t - \tilde m/T_2\right)$.

    ⇒ The $-\tilde m/T_2$ terms on both sides cancel, leaving $\partial\tilde m/\partial t = D\nabla^2\tilde m - i\gamma\,\mathbf g\cdot\mathbf x\,\tilde m$: the equation without relaxation.

    ⇒ So $S(T) = e^{-T/T_2}\tilde S(T)$, and $S_0$ (no gradient, same $T$) carries the same factor. It cancels in the ratio.

    → This fails when $T_2$ differs between compartments: then each compartment is weighted by its own $e^{-T/T_{2,n}}$, which is how echo time changes the apparent volume fractions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same problem in weak form

    The Bloch–Torrey equation makes a demand at every point: both sides must be equal everywhere in $\Omega$. The same problem can be stated as a demand on **averages** instead. That second statement, the weak form, is where most numerical methods start, and it shows how the boundary conditions enter them.

    ⇒ Write the equation as residual $= 0$ (one compartment for now): $\ \partial_tm - \nabla\cdot(D\nabla m) + i\gamma f(t)\,(\mathbf g\cdot\mathbf x)\,m = 0$.

    ⇒ Multiply by an arbitrary **test function** $w(\mathbf x)$ and integrate over $\Omega$. If the result is zero for every $w$, the residual itself must be zero.

    ⇒ Integrate the diffusion term by parts (Green's identity). Half of the derivatives move onto $w$, and a boundary term appears:

    $$\boxed{\ \int_\Omega w\,\partial_tm\,d\mathbf x + \int_\Omega D\,\nabla w\cdot\nabla m\,d\mathbf x + i\gamma f(t)\int_\Omega w\,(\mathbf g\cdot\mathbf x)\,m\,d\mathbf x = \oint_{\partial\Omega}w\,D\,\frac{\partial m}{\partial n}\,dS\qquad\text{for every } w\ }.$$

    This is the **weak form**. It needs only first derivatives of $m$, so $m$ may have kinks, which the equation itself would not allow.

    ⇒ **The boundary term holds the wall conditions.** Look at what $D\,\partial m/\partial n$ is on each kind of wall:

    | Wall | Condition | Boundary term $\oint wD\,\partial m/\partial n$ |
    | --- | --- | --- |
    | impermeable | $D\,\partial m/\partial n = 0$ | $0$: **nothing to add** |
    | relaxing | $D\,\partial m/\partial n = -\varrho\,m$ | $-\varrho\oint w\,m\,dS$ |
    | membrane between $\Omega_n$ and $\Omega_m$ | continuous flux $= \kappa_{nm}(c_{nm}m_m - c_{mn}m_n)$ | a term on $\Gamma_{nm}$ that couples the two sides; for equal densities $-\kappa\oint(m_n - m_m)(w_n - w_m)\,dS$ (Exercise 13) |
    | fixed value $m = m_0$ (Dirichlet) | — | not in the boundary term: must be imposed on $m$ directly |

    → The first three are **natural** conditions: they enter through the boundary term, and every solution of the weak form satisfies them automatically. A fixed value is an **essential** condition, imposed by hand. Diffusion MRI needs only natural ones.

    **Choosing the test functions is choosing a method.** Expand $m$ in a set of functions $\varphi_j$ and use the same functions as test functions (Galerkin's choice, 1915). Each test function gives one equation, so there are as many equations as unknowns:

    | Functions $\varphi_j$ | The weak form becomes | Where |
    | --- | --- | --- |
    | eigenmodes $u_n$ of the pore: smooth, spread over the whole pore | $d\mathbf c/dt = -(D\Lambda + i\gamma g(t)\mathbf B)\,\mathbf c$: diffusion diagonal, the gradient couples every mode | matrix methods, Chapter 2 |
    | hat functions on a mesh: 1 at one node, 0 at all others, linear in between | a sparse system with mass, stiffness and gradient matrices | finite elements, Chapter 4 |
    | indicator functions of grid cells (the equation integrated over each cell) | the change in each cell equals the fluxes through its faces | finite volumes and finite differences, Chapter 4 |

    → The weak form is not a finite-element trick. The matrix method of Chapter 2 is the weak form in the basis of eigenmodes (Exercise 13b).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive the weak form in 1D on $[a, b]$, and show that the boundary term $[w\,D\,\partial_xm]_a^b$ vanishes for reflecting walls whatever $w$ is.

    **(b)** *The matrix method as a weak form.* In a slab with reflecting walls, expand $m = \sum_nc_n(t)\,u_n(x)$ in the orthonormal eigenmodes and take $w = u_k$. Show that the weak form gives $\dot c_k = -D\lambda_kc_k - i\gamma g(t)\sum_nB_{kn}c_n$, with $B_{kn}$ the position matrix of Exercise 5.

    **(c)** For a membrane at $x = 0$ between $[a, 0]$ and $[0, b]$, equal densities, with $m$ and $w$ allowed to jump there: collect the two boundary terms at $x = 0$ and show that they combine into $-\kappa(m_1 - m_2)(w_1 - w_2)$, where 1 and 2 label the two sides.

    **(d)** *Unequal densities.* Redo (c) with the general condition $D\,\partial_{n}m_n = \kappa(c_{nm}m_m - c_{mn}m_n)$, compartment $n$ on the left and $m$ on the right. Show that, moved to the left-hand side, the membrane term is $(w_n, w_m)\,\mathbf Q\,(m_n, m_m)^{\mathsf T}$ with $\mathbf Q = \kappa\begin{pmatrix}c_{mn}&-c_{nm}\\-c_{mn}&c_{nm}\end{pmatrix}$. Check that the columns of $\mathbf Q$ sum to zero and that $(\rho_n, \rho_m)$ is in its null space. What does each property guarantee?
    """, "13 · The weak form")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) $\int_a^bw\,\partial_x(D\partial_xm)\,dx = [wD\partial_xm]_a^b - \int_a^bD\,\partial_xw\,\partial_xm\,dx$.
    - (b) Three integrals: $\int u_ku_n = \delta_{kn}$; $\int u_k'u_n'$ by parts again, using $-u_n'' = \lambda_nu_n$ and $u_n' = 0$ at the walls; and $\int u_k\,x\,u_n = B_{kn}$.
    - (c) On side 1 (left of the membrane) the outward normal at $x = 0$ points in $+x$; on side 2 it points in $-x$. The flux from 1 to 2 is $-D\partial_xm$, the same on both sides, and equals $\kappa(m_1 - m_2)$.
    - (d) Side $n$ contributes $w_n\kappa(c_{nm}m_m - c_{mn}m_n)$ to the right-hand side, side $m$ the same with $n$ and $m$ swapped.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\int_a^bw\,\partial_tm\,dx - \int_a^bw\,\partial_x(D\partial_xm)\,dx + i\gamma g\int_a^bw\,x\,m\,dx = 0$. By parts:

    $$\int_a^bw\,\partial_tm + \int_a^bD\,\partial_xw\,\partial_xm + i\gamma g\int_a^bw\,x\,m = [w\,D\,\partial_xm]_a^b .$$

    ⇒ Reflecting walls: $D\partial_xm = 0$ at $a$ and at $b$, so the right-hand side is zero for every $w$. ✓ Nothing has to be imposed: the condition is natural.

    **(b)** ⇒ First term: $\sum_n\dot c_n\int u_ku_n = \dot c_k$.

    ⇒ Second: $\int Du_k'u_n' = [Du_ku_n']_a^b - D\int u_ku_n'' = 0 + D\lambda_n\int u_ku_n = D\lambda_k\delta_{kn}$. Summed with $c_n$: $D\lambda_kc_k$.

    ⇒ Third: $i\gamma g\sum_nc_n\int u_k\,x\,u_n = i\gamma g\sum_nB_{kn}c_n$. The boundary term is zero (a).

    ⇒ So $\dot c_k = -D\lambda_kc_k - i\gamma g\sum_nB_{kn}c_n$, or $\dot{\mathbf c} = -(D\Lambda + i\gamma g\mathbf B)\mathbf c$: the equation of Chapter 2. ✓

    → In the mode basis the "mass matrix" $\int u_ku_n$ is the identity and the "stiffness matrix" $\int Du_k'u_n'$ is diagonal: the modes are exactly the functions that make both simple. Hat functions make both sparse instead (Chapter 4).

    **(c)** ⇒ Side 1 contributes $+w_1D\partial_xm\big|_{0^-}$, side 2 contributes $-w_2D\partial_xm\big|_{0^+}$. Flux continuity: $-D\partial_xm$ is the same on both sides and equals $\kappa(m_1 - m_2)$.

    ⇒ Sum: $w_1\cdot(-\kappa(m_1 - m_2)) - w_2\cdot(-\kappa(m_1 - m_2)) = -\kappa(m_1 - m_2)(w_1 - w_2)$. ✓

    → $m$ must be allowed to jump at the membrane. This is why finite-element codes put two nodes at the same point of a membrane, one for each side ([Nguyen et al. 2014](https://doi.org/10.1016/j.jcp.2014.01.009); Chapter 4).

    **(d)** ⇒ Moved to the left-hand side: side $n$ gives $w_n\kappa(c_{mn}m_n - c_{nm}m_m)$, side $m$ gives $w_m\kappa(c_{nm}m_m - c_{mn}m_n)$. Collected: $\mathbf Q = \kappa\begin{pmatrix}c_{mn}&-c_{nm}\\-c_{mn}&c_{nm}\end{pmatrix}$. ✓ It is not symmetric unless $\rho_n = \rho_m$.

    ⇒ Columns: $c_{mn} - c_{mn} = 0$ and $-c_{nm} + c_{nm} = 0$. ✓ → The membrane moves magnetization from one side to the other without creating or destroying any.

    ⇒ Null space: $c_{mn}\rho_n - c_{nm}\rho_m = \frac{2\rho_m\rho_n - 2\rho_n\rho_m}{\rho_n + \rho_m} = 0$. ✓ → Without a gradient, density $\rho_n$ on one side and $\rho_m$ on the other stays put: unequal densities alone drive no flux. With $c = 1$ the membrane would wrongly push them towards each other.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The path picture: the average phase over random walks

    Follow single spins. Each spin random-walks along its own path $\mathbf x(t)$ (reflected at walls, as in 1a) and gathers its own phase

    $$\boxed{\ \varphi = \gamma\int_0^T \mathbf g(t)\cdot\mathbf x(t)\,dt\ }.$$

    The voxel contains so many spins that the signal is the **average phase factor** over all possible paths, starting from uniformly spread positions:

    $$\boxed{\ E = \big\langle e^{-i\varphi}\big\rangle_{\text{paths}}\ }.$$

    In the language of probability: the signal is the **characteristic function** of the random phase. Chapter 5 (Monte Carlo) works this way.

    *Sign convention: some texts write $e^{+i\varphi}$. For a refocused echo the two give the same magnitude.*

    **Short pulses: the distribution of displacements.** Make both lobes so short that the spins hardly move during them. The first lobe gives each spin the phase $\mathbf q\cdot\mathbf x(0)$ and the second takes away $\mathbf q\cdot\mathbf x(\Delta)$, so $\varphi = -\mathbf q\cdot\mathbf r$ with $\mathbf r$ the displacement of 1a over the time $\Delta$. Then

    $$\boxed{\ E = \big\langle e^{i\mathbf q\cdot\mathbf r}\big\rangle_{\mathbf x,\mathbf x_0}\ },$$

    the characteristic function of the displacements of 1a, with the Fourier variable $\xi$ set by the gradient: $\xi = q$. Its logarithm has the cumulants of 1a as coefficients, so in this limit $D(t)$ and $K(t)$ are what diffusion and kurtosis imaging estimate. Chapter 3 maps where the limit holds; Chapter 2 shows what changes when the pulses are not short.

    **The two routes agree.** The field and path pictures give exactly the same $E$. This is an example of the **Feynman–Kac relation** between parabolic PDEs and averages over random paths. [Carr & Purcell (1954)](https://doi.org/10.1103/PhysRev.94.630) already computed echo attenuation this way, as an average phase over random walks; the general relation is due to [Kac (1949)](https://doi.org/10.1090/S0002-9947-1949-0027960-X), inspired by [Feynman's (1948)](https://doi.org/10.1103/RevModPhys.20.367) path integrals. You can derive it yourself in a few lines.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    Work in 1D, in a slab with a constant-in-space gradient $g(t)$ along $x$. Define the **phase-weighted density of walkers**

    $$w(x,t) = \Big\langle e^{-i\varphi_t}\;\delta\big(x - x(t)\big)\Big\rangle_{\text{paths}} ,\qquad \varphi_t = \gamma\int_0^t g(t')r_{t'}\,dt' ,$$

    where $x(t)$ is the position of a walker at time $t$. In words: count the walkers that are at $x$ at time $t$, each one weighted by its current phase factor.

    **(a)** Show that $\int w(x,t)\,dx = \langle e^{-i\varphi_t}\rangle_{\text{paths}}$: integrating $w$ gives the path-picture signal.

    **(b)** Write $w(x,t+\delta t)$ in terms of $w(\cdot,t)$ for one short step $\delta t$. In that step a walker moves from $y$ to $x$ with the Gaussian step density $p(x-y)$ of variance $2D\,\delta t$, and gains the phase $\gamma g(t)\,y\,\delta t$.

    **(c)** Expand for small $\delta t$ and show that $w$ obeys the Bloch–Torrey equation.

    **(d)** What are the initial and interface conditions for $w$? Conclude that $\int m\,dx = \langle e^{-i\varphi}\rangle_{\text{paths}}$.
    """, "14 · Derive the Feynman–Kac relation")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) $\int\delta(x - x(t))\,dx = 1$ for every walker.
    - (b) Sum over where the walker was at time $t$: $w(x,t+\delta t) = \int p(x-y)\,e^{-i\gamma g(t) y\,\delta t}\,w(y,t)\,dy$. The walk has no memory, so only the position at $t$ matters.
    - (c) Two small things: the phase factor, $e^{-i\gamma g y\delta t}\approx 1 - i\gamma g y\,\delta t$, and the step. For any smooth $f$, write $y = x - \zeta$, Taylor-expand $f(x-\zeta)$ in $\zeta$, and use $\langle \zeta\rangle_\zeta = 0$ and $\langle \zeta^2\rangle_\zeta = 2D\,\delta t$.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\int w\,dx = \big\langle e^{-i\varphi_t}\int\delta(x - x(t))\,dx\big\rangle_{\text{paths}} = \langle e^{-i\varphi_t}\rangle_{\text{paths}}$. ✓

    **(b)** ⇒ A walker at $x$ at $t+\delta t$ was at some $y$ at time $t$, carried the weight $e^{-i\varphi_t}$ there, and multiplied it by $e^{-i\gamma g(t)y\,\delta t}$ during the step:

    $$w(x,t+\delta t) = \int p(x-y)\; e^{-i\gamma g(t)\,y\,\delta t}\; w(y,t)\,dy .$$

    **(c)** ⇒ **The step.** For a smooth $f$, put $y = x-\zeta$:

    $$\int p(x-y) f(y)\,dy = \int p(\zeta)\Big[f(x) - \zeta f'(x) + \tfrac{\zeta^2}{2}f''(x) - \dots\Big]d\zeta = f(x) + D\,\delta t\, f''(x) + O(\delta t^2),$$

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

    $$\int m(x,T)\,dx = \big\langle e^{-i\varphi}\big\rangle_{\text{paths}} .\qquad\blacksquare$$

    → **What it means.** The magnetization at $x$ *is* the phase-weighted density of spins at $x$. The PDE tracks the density; the random walk tracks the spins.
    → **What it means for simulation.** Step (b), repeated many times, is a recipe: move each walker with a Gaussian step, then add its phase. That is a Monte Carlo simulation (Chapter 5). Written in the eigenbasis of 1a, the same alternation of "diffuse, then add phase" is the multiple propagator method (Chapter 2). Both converge to the Bloch–Torrey solution as $\delta t\to0$, for exactly the reason shown in (c).
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

    | Shortest length | Regime | What the signal does | In this course |
    | --- | --- | --- | --- |
    | $\ell_D$ | free diffusion, or short times | $E \approx e^{-bD}$; walls only add a correction $\propto (S/V)\sqrt{DT}$ | Ch 3: short times and the $(q,t)$ map; the $S/V$ law of 1a |
    | $L$ | motional narrowing (fully restricted) | spins average over the pore; slab: $\ln E \approx -\gamma^2g^2L^4T/(120D)$ ([Robertson 1966](https://doi.org/10.1103/PhysRev.151.273)) | Ch 2: Gaussian phase approximation; Ch 3: motional narrowing |
    | $\ell_g$ | localisation | magnetization survives only in layers of thickness $\sim\ell_g$ at the walls; $\ln E \propto -DT/\ell_g^2 \propto -g^{2/3}T$, not $-g^2$ | Ch 2: the matrix method (the cumulant series fails); Ch 3: localisation |

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

    $$\boxed{\ \ell_g = \left(\frac{D}{\gamma g}\right)^{1/3},\qquad \tau_g = \frac{\ell_g^2}{D}\ }.$$

    → The same length from the table: $\ell_D = \sqrt{DT}$ grows with $T$ and $\ell_q = 1/(\gamma gT)$ shrinks. They are equal at $T = \tau_g$, and there both equal $\ell_g$ (Exercise 15a).

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
        """, "15 · Length scales on real scanners")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
        **(a)** ⇒ $\ell_D^2\,\ell_q = DT\cdot\dfrac{1}{\gamma gT} = \dfrac{D}{\gamma g} = \ell_g^3$. ✓ The gradient length is a geometric mean of the diffusion and phase lengths.

        **(b)** ⇒ $\ell_D = \sqrt{2\times60} = 11.0$ µm.
        ⇒ $g = 80$ mT/m: $\ell_g = \big(2/(2.6752\times10^{-4}\times80)\big)^{1/3} = 4.5$ µm. $\ g = 300$ mT/m: $\ell_g = 2.9$ µm.
        ⇒ **10 µm cell:** $\ell_g < L \approx \ell_D$. The gradient length is the shortest: at strong weighting the signal from such cells enters the localisation regime, and the Gaussian phase picture breaks down (Chapter 2 shows where this starts).
        ⇒ **1 µm axon:** $L \ll \ell_g < \ell_D$: motional narrowing.

        **(c)** ⇒ $\gamma^2g^2L^4T/(120D) = (2.6752\times10^{-4}\times300)^2\times1\times60/(120\times2) = 0.0016$: the signal drops by 0.16%.
        → The size information sits in a tiny attenuation that scales as $L^4$. This is why axon-diameter estimation needs the strongest gradients and very high SNR, and why small axons are hard to tell apart.

        **(d)** ⇒ $\ell_\varrho = 2/10^{-3} = 2000$ µm $\gg 10$ µm: yes. Spins hit the walls many times before relaxing, the loss is uniform at rate $\varrho S/V$, and it simply looks like a shorter $T_2$ for that compartment.
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
    | Interface conditions | impermeable $\partial C/\partial n = 0$; permeable: continuous flux $= \kappa_{nm}(c_{nm}C_m - c_{mn}C_n)$, $\ c_{nm} = 2\rho_n/(\rho_n + \rho_m)$ ($=1$ for equal densities) |
    | Eigenmodes of a pore | $-\nabla^2 u_n = \lambda_n u_n$, $\ \partial u_n/\partial n = 0$; time scales $\tau_n = 1/(D\lambda_n)$ |
    | Propagator | $P(\mathbf x\mid\mathbf x_0,t) = \sum_n u_n(\mathbf x)u_n(\mathbf x_0)e^{-D\lambda_n t}$ |
    | Slab | $u_n = \sqrt{2/L}\cos(n\pi x/L)$, $\ \lambda_n = (n\pi/L)^2$, MSD $\to L^2/6$ |
    | Displacement cumulants | $D(t) = \langle r^2\rangle_{x,x_0}/2t$, $\ K(t) = \langle r^4\rangle_{x,x_0}/\langle r^2\rangle_{x,x_0}^2 - 3$, with $\langle r^k\rangle_{x,x_0} = \int r^k\,\bar P(r,t)\,dr$ and $\bar P(r,t) = \frac1V\int_V P(\mathbf x_0 + r\mid\mathbf x_0,t)\,d\mathbf x_0$ (uniform start); slab: $K>0$ at short times, $K\to-3/5$ at long times |
    | Permeable membrane, absorbing walls | modes of the whole domain; exchange rate $\approx 4\kappa/L$; wall loss $\approx\varrho\,S/V$ |
    | Other geometries | disk and sphere: roots of $J_m'$, $j_l'$; cylindrical surface: $\lambda_n = n^2/R^2$ (periodic, no wall); concentric layers weighted by $R_i/\sum R_j$ |
    | Length scales | $\ell_D = \sqrt{DT}$, $\ell_g = (D/\gamma g)^{1/3}$, $\ell_\varrho = D/\varrho$; the shortest of $L$, $\ell_D$, $\ell_g$ sets the regime |
    | What a gradient does | phase $\varphi = \gamma\int\mathbf g(t)\cdot\mathbf x(t)\,dt$; $\ \mathbf q = \gamma\int\mathbf g\,dt$; static spins refocus at the echo, moving spins keep a phase |
    | Bloch–Torrey | $\partial m/\partial t = D\nabla^2 m - i\gamma\,\mathbf g(t)\cdot\mathbf x\,m$, $\ E = \frac{1}{S_0}\int m\,d\mathbf x$ |
    | Full problem | per compartment $\partial_tm_n = \nabla\cdot(D_n\nabla m_n) - m_n/T_{2,n} - i\gamma f(t)(\mathbf g\cdot\mathbf x)m_n$, $\ m_n(0) = \rho_n$; simplified by uniform $T_2$, equal densities, one compartment |
    | Weak form | $\int w\,\partial_tm + \int D\nabla w\cdot\nabla m + i\gamma f\int w(\mathbf g\cdot\mathbf x)m = \oint wD\,\partial m/\partial n$ for all $w$; walls natural; modes as test functions give Chapter 2, hats give finite elements |
    | Exchange (Kärger) | two Gaussian compartments with hop rates $k_1p_1 = k_2p_2$: $\langle e^{i\xi r}\rangle_{x,x_0} = (1,1)\,e^{-(\xi^2\mathbf D + \mathbf K)t}\,\mathbf p$, $\mathbf K$ = the two slowest modes; $\langle r^2\rangle_{x,x_0} = 2\bar Dt$ always; $K(t) = K_0\,2(x - 1 + e^{-x})/x^2$, $x = t/\tau_{\text{ex}}$ |
    | Path picture (Feynman–Kac) | short pulses: $E = \langle e^{i\mathbf q\cdot\mathbf r}\rangle_{\mathbf x,\mathbf x_0}$; in general $E = \langle e^{-i\varphi}\rangle_{\text{paths}} = \lim_{N\to\infty}\frac1N\sum_{j=1}^N e^{-i\varphi_j}$, $\ \varphi_j = \gamma\int \mathbf g\cdot\mathbf x_j\,dt$ over the random paths $\mathbf x_j(t)$ of $N$ spins |

    ---
    ## Coming next

    **Chapter 2 · Series expansions: two ways to expand the signal**

    Two different series, often confused. **(a)** expands the magnetization in the eigenmodes of 1a: exact once enough modes are kept, at any gradient strength, so it is a numerical method. **(b)** expands $\ln E$ in powers of the gradient: an approximation for weak gradients, which gives closed formulas.

    - **2a · Matrix methods.** The gradient as a matrix $\mathbf B$ in the eigenbasis: the eigenmodes stay the same, the gradient only couples them, $d\mathbf c/dt = -(D\Lambda + i\gamma g(t)\mathbf B)\,\mathbf c$. Grebenkov's MCF: one matrix exponential per piece of constant gradient, exact for piecewise-constant waveforms (Barzykin 1999; Grebenkov 2007). The multiple propagator: diffusion and phase kicks in turn, with Strang splitting and its $\tau^2$ error (Caprihan et al. 1996; Callaghan 1997). The time-ordered exponential (Axelrod & Sen 2001); trapezoids and other waveforms. **How many modes:** smoothness of the mirrored magnetization, Jackson's theorem, and why the couplings fall as $1/n^2$. Beyond simple shapes: modes computed on a mesh (SpinDoctor's matrix formalism), a differentiable version for fitting (ReMiDi), and cylindrical surfaces (Canales-Rodríguez, Tax et al. 2026).
    - **2b · The cumulant expansion.** $\ln E = -\frac12\langle\varphi^2\rangle_{\text{paths}} + \frac1{24}\big(\langle\varphi^4\rangle_{\text{paths}} - 3\langle\varphi^2\rangle_{\text{paths}}^2\big) - \dots$ The second-order term is the **Gaussian phase approximation**: closed formulas from the mode weights and time scales of 1a (Murday & Cotts 1968; Neuman 1974; van Gelderen et al. 1994), the same in frequency as the diffusion spectrum $D(\omega)$ and OGSE (Stepišnik 1993), exact in a harmonic well. Walkers, modes and the GPA side by side: when is the phase Gaussian? The fourth-order term is the **kurtosis** (Jensen et al. 2005). With narrow pulses these are the displacement cumulants $D(t)$ and $K(t)$ of 1a. With finite pulses they become time integrals of multi-time correlation functions, and the measured kurtosis is no longer the displacement kurtosis.
    - **2c · The link.** Grebenkov's MCF moment series computes each cumulant from the modes; the second is $\propto\iint g(t_1)g(t_2)\,[\mathbf B\,e^{-D\Lambda|t_1-t_2|}\mathbf B]_{00}\,dt_1dt_2$. Where the cumulant series stops converging, and why the matrix methods of 2a do not have that limit.
    - *Implementations:* MCFAL, MISST, SpinDoctor, the cylindrical-surface code; GPA formulas in Camino and dmipy.

    **Chapter 3 · Analytical formulas: a map of the PGSE parameter space**

    A PGSE experiment in a pore is fixed by three dimensionless numbers, $D\Delta/L^2$, $D\delta/L^2$ and $\gamma gL^3/D$. Every closed-form result holds in one corner of this box. The chapter computes each corner by comparing the formula with the exact solutions of Chapter 2.

    - **Narrow pulses, the $(q,t)$ map:** the signal as the characteristic function of the displacement, $E(q,\Delta) = \langle e^{iqr}\rangle_{x,x_0}$. Short times: free diffusion, the $b$-matrix (Stejskal & Tanner 1965; Basser et al. 1994), the surface-to-volume ratio from $D(t)$ (Mitra et al. 1992) and the short-time kurtosis. Long times: closed pores, tortuosity and structural disorder (Novikov et al. 2014). Large $q$: diffusion–diffraction and the form factors of slab, cylinder and sphere (Tanner & Stejskal 1968; Callaghan et al. 1991). Where the cumulant series stops.
    - **Long pulses, the $(g,T)$ map** built from the length scales of 1b: free diffusion, motional narrowing (Robertson 1966) and localisation (Stoller, Happer & Dyson 1991; Hürlimann et al. 1995), meeting at a triple point.
    - **Inside the box:** how the edge of the cumulant region moves with $\delta/\Delta$, and a tested rule for it.
    - **Extra axes:** exchange between compartments (Kärger 1985) and relaxing walls (Brownstein & Tarr 1979).
    - *Implementations:* closed-form compartment models, for example the cylinder and sphere models in Camino and dmipy.

    **Chapter 4 · Solving the PDE on a grid or a mesh**

    - **Finite differences:** the domain as voxels; the Laplacian as jumps to neighbours; explicit time steps and their stability limit; periodic boundaries with a phase shift. *Implementation:* MATI.
    - **Finite elements:** the weak form of 1b with hat functions, mass and stiffness matrices, a mesh that follows the cell walls; permeable membranes as double nodes. *Implementation:* SpinDoctor.

    **Chapter 5 · Monte Carlo: simulating the random walk**

    - The algorithm from Exercise 14: Gaussian steps, reflection at walls, phase accumulation, and the average $\langle e^{-i\varphi}\rangle_{\text{paths}}$.
    - Permeable membranes as a crossing probability; choosing the time step; the $1/\sqrt N$ noise floor, compared with an exact eigenmode solution.
    - Walker histories as reusable data: replay packs that answer any later sequence as an inner product ([Fick 2026](https://arxiv.org/abs/2611.01758)).
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
    | 1915 | B. G. Galerkin, *Vestnik Inzhenerov i Tekhnikov* 19:897 | the weighted-residual method with the same functions as unknowns and test functions |
    | 1946 | [F. Bloch, *Phys. Rev.* 70:460](https://doi.org/10.1103/PhysRev.70.460) | nuclear induction; the Bloch equations |
    | 1948 | [R. P. Feynman, *Rev. Mod. Phys.* 20:367](https://doi.org/10.1103/RevModPhys.20.367) | path integrals |
    | 1949 | [M. Kac, *Trans. Am. Math. Soc.* 65:1](https://doi.org/10.1090/S0002-9947-1949-0027960-X) | the Feynman–Kac relation between parabolic PDEs and averages over random paths |
    | 1950 | [E. L. Hahn, *Phys. Rev.* 80:580](https://doi.org/10.1103/PhysRev.80.580) | the spin echo |
    | 1954 | [H. Y. Carr & E. M. Purcell, *Phys. Rev.* 94:630](https://doi.org/10.1103/PhysRev.94.630) | diffusion in spin echoes as a random-walk phase average |
    | 1956 | [H. C. Torrey, *Phys. Rev.* 104:563](https://doi.org/10.1103/PhysRev.104.563) | the Bloch–Torrey equation |
    | 1965 | [E. O. Stejskal & J. E. Tanner, *J. Chem. Phys.* 42:288](https://doi.org/10.1063/1.1695690) | the pulsed-gradient spin echo and its $b$-value |
    | 1966 | [B. Robertson, *Phys. Rev.* 151:273](https://doi.org/10.1103/PhysRev.151.273) | eigenmode expansion for restricted diffusion in a slab; motional narrowing |
    | 1979 | [K. R. Brownstein & C. E. Tarr, *Phys. Rev. A* 19:2446](https://doi.org/10.1103/PhysRevA.19.2446) | eigenmodes with relaxing walls; fast-diffusion and diffusion-limited regimes |
    | 1985 | [J. Kärger, *Adv. Colloid Interface Sci.* 23:129](<https://doi.org/10.1016/0001-8686(85)80018-X>) | two exchanging compartments |
    | 1991 | [S. D. Stoller, W. Happer & F. J. Dyson, *Phys. Rev. A* 44:7459](https://doi.org/10.1103/PhysRevA.44.7459) | the localisation regime and its length scale |
    | 1992 | [P. P. Mitra, P. N. Sen, L. M. Schwartz & P. Le Doussal, *Phys. Rev. Lett.* 68:3555](https://doi.org/10.1103/PhysRevLett.68.3555) | short-time $D(t)$ and the surface-to-volume ratio |
    | 1994 | [T. M. de Swiet & P. N. Sen, *J. Chem. Phys.* 100:5597](https://doi.org/10.1063/1.467127) | bounded diffusion in a constant gradient; regimes set by length scales |
    | 2007 | [D. S. Grebenkov, *Rev. Mod. Phys.* 79:1077](https://doi.org/10.1103/RevModPhys.79.1077) | review: length scales, eigenmode (MCF) description, numerical methods |
    | 2014 | [D. V. Nguyen, J.-R. Li, D. Grebenkov & D. Le Bihan, *J. Comput. Phys.* 263:283](https://doi.org/10.1016/j.jcp.2014.01.009) | the multi-compartment Bloch–Torrey equation in weak form, with membranes as double nodes |
    | 2025 | [E. J. Canales-Rodríguez, C. M. W. Tax et al., *Front. Phys.* 13:1516630](https://doi.org/10.3389/fphy.2025.1516630) | diffusion on cylindrical surfaces; concentric cylinders for myelin |
    | 2026 | [E. J. Canales-Rodríguez, C. M. W. Tax et al., *Phys. Med. Biol.*](https://doi.org/10.1088/1361-6560/aeb2a0) | exact PGSE signal on a cylindrical surface with the spectral matrix formalism ([arXiv 2603.23421](https://arxiv.org/abs/2603.23421)) |
    """)
    return


if __name__ == "__main__":
    app.run()
