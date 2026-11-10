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
    app_title="dMRI simulation · Ch 4 · Grids and meshes",
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
    C_FREE = "#9aa5b1"   # reference
    C_ALT = "#d9622b"    # second method
    C_GREEN = "#2e9e6a"  # third method
    C_PURPLE = "#8a5cd1" # fourth
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
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(D0, GAMMA, expm, np):
    # ---------- exact reference (Chapter 2) ----------
    def slab_B(L, N):
        """Eigenvalues λ_n and B_nm = ∫ u_n (x − L/2) u_m dx of a slab, n, m = 0..N−1."""
        n = np.arange(N)
        m, k = n[:, None], n[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            B = (L / np.pi**2) * ((((-1.0) ** (m - k)) - 1) / (m - k) ** 2 + (((-1.0) ** (m + k)) - 1) / (m + k) ** 2)
            j = np.maximum(m, k)
            B0 = np.sqrt(2) * L * (((-1.0) ** j) - 1) / (j * np.pi) ** 2
        B = np.where((m == 0) | (k == 0), B0, B)
        B = np.where(m == k, 0.0, B)
        return (n * np.pi / L) ** 2, B

    def pgse_mcf(G, L, delta, Delta, N=60, D=D0):
        """Exact PGSE echo in a slab (MCF, Chapter 2)."""
        lam, B = slab_B(L, N)
        Lam = np.diag(lam)
        P1 = expm(-(D * Lam + 1j * GAMMA * G * B) * delta)
        P2 = expm(-D * Lam * (Delta - delta))
        P3 = expm(-(D * Lam - 1j * GAMMA * G * B) * delta)
        return (P3 @ P2 @ P1)[0, 0]

    # ---------- finite differences (finite volumes) in 1D ----------
    def fd_ops(L, n, D=D0, kappa=None, rho=None):
        """Cell-centred grid on [−L/2, L/2]: returns (A, x, h) with dM/dt = −(A + iγg·diag(x)) M.

        Faces carry a conductance c: D/h in the bulk, 1/(h/D + 1/κ) for a membrane at x = 0,
        and nothing (zero flux) at the outer walls. A relaxing wall removes M through 1/(h/2D + 1/ρ)."""
        h = L / n
        x = (np.arange(n) + 0.5) * h - L / 2
        c = np.full(n - 1, D / h)
        if kappa is not None:
            c[n // 2 - 1] = 1.0 / (h / D + 1.0 / kappa)
        A = np.zeros((n, n))
        idx = np.arange(n - 1)
        A[idx, idx] += c / h
        A[idx + 1, idx + 1] += c / h
        A[idx, idx + 1] -= c / h
        A[idx + 1, idx] -= c / h
        if rho is not None:
            cw = 1.0 / (h / (2 * D) + 1.0 / rho)
            A[0, 0] += cw / h
            A[-1, -1] += cw / h
        return A, x, h

    def pgse_pieces(delta, Delta):
        """PGSE as three constant pieces: (sign of g, duration)."""
        return [(+1.0, delta), (0.0, Delta - delta), (-1.0, delta)]

    def fd_signal(G, L, delta, Delta, n=200, dt=0.02, scheme="cn", D=D0):
        """PGSE echo with finite differences; scheme = 'ee' (explicit Euler), 'ie' (implicit Euler) or 'cn'."""
        A, x, h = fd_ops(L, n, D)
        I = np.eye(n)
        M = np.ones(n, complex)
        for s, dur in pgse_pieces(delta, Delta):
            k = int(round(dur / dt))
            Af = A + 1j * GAMMA * s * G * np.diag(x)
            if scheme == "cn":
                S = np.linalg.solve(I + 0.5 * dt * Af, I - 0.5 * dt * Af)
            elif scheme == "ie":
                S = np.linalg.inv(I + dt * Af)
            else:
                S = I - dt * Af
            for _ in range(k):
                M = S @ M
                if not np.isfinite(M[0]) or abs(M[0]) > 1e6:
                    return np.nan
        return M.mean()

    # ---------- finite elements in 1D ----------
    def fem_ops(L, ne, D=D0, kappa=None):
        """Linear elements on [−L/2, L/2]: mass Mm, stiffness K (with D), gradient matrix J, weights w = ∫N_i.

        With kappa, the node at x = 0 is doubled and the membrane adds κ[[1, −1], [−1, 1]]."""
        xn = np.linspace(-L / 2, L / 2, ne + 1)
        nodes = list(xn)
        conn = [(e, e + 1) for e in range(ne)]
        mid = ne // 2
        if kappa is not None:
            nodes.append(xn[mid])
            new = len(nodes) - 1
            conn = [(a, b) if e < mid else (new if a == mid else a, b) for e, (a, b) in enumerate(conn)]
        nodes = np.array(nodes)
        n = len(nodes)
        Mm, K, J = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for a, b in conn:
            xa, xb = nodes[a], nodes[b]
            he = xb - xa
            i = [a, b]
            Mm[np.ix_(i, i)] += he / 6 * np.array([[2, 1], [1, 2]])
            K[np.ix_(i, i)] += D / he * np.array([[1, -1], [-1, 1]])
            J[np.ix_(i, i)] += he / 12 * np.array([[3 * xa + xb, xa + xb], [xa + xb, xa + 3 * xb]])
        if kappa is not None:
            i = [mid, n - 1]
            K[np.ix_(i, i)] += kappa * np.array([[1, -1], [-1, 1]])
        return Mm, K, J, Mm.sum(axis=1), nodes

    def fem_signal(G, L, delta, Delta, ne=200, dt=0.02, D=D0):
        """PGSE echo with linear finite elements and Crank–Nicolson: (Mm + dt/2 A) M⁺ = (Mm − dt/2 A) M."""
        Mm, K, J, w, _ = fem_ops(L, ne, D)
        M = np.ones(len(w), complex)
        for s, dur in pgse_pieces(delta, Delta):
            k = int(round(dur / dt))
            A = K + 1j * GAMMA * s * G * J
            S = np.linalg.solve(Mm + 0.5 * dt * A, Mm - 0.5 * dt * A)
            for _ in range(k):
                M = S @ M
        return (w @ M) / L

    def gen_eig(K, Mm):
        """Generalised symmetric eigenproblem K u = λ Mm u, with Mm-orthonormal eigenvectors."""
        Lc = np.linalg.cholesky(Mm)
        Li = np.linalg.inv(Lc)
        w, V = np.linalg.eigh(Li @ K @ Li.T)
        return w, Li.T @ V

    return fd_ops, fd_signal, fem_ops, fem_signal, gen_eig, pgse_mcf, slab_B


@app.cell(hide_code=True)
def _(D0, GAMMA, np):
    # ---------- periodic media: one cell with a phase-shifted (twisted) boundary ----------
    def periodic_A(q, L, n, D=D0, kappa=None):
        """Operator of dm/dt = −A(q) m for m = M·e^{iq(t)x} on one periodic cell of n cells.

        Each face couples its two cells with conductance c and a phase factor e^{∓iqh}; the face
        between the last and the first cell carries the membrane (permeability κ)."""
        h = L / n
        c = np.full(n, D / h)
        if kappa is not None:
            c[n - 1] = 1.0 / (h / D + 1.0 / kappa)
        A = np.zeros((n, n), complex)
        j = np.arange(n)
        k = (j + 1) % n
        np.add.at(A, (j, j), c / h)
        np.add.at(A, (k, k), c / h)
        np.add.at(A, (j, k), -c / h * np.exp(-1j * q * h))
        np.add.at(A, (k, j), -c / h * np.exp(1j * q * h))
        return A

    def periodic_signal(G, L, delta, Delta, n=40, dt=0.05, D=D0, kappa=None):
        """PGSE echo of an infinite periodic medium, simulated on one cell. Lobes with Crank–Nicolson
        (q changes), the gap exactly (q constant, A Hermitian)."""
        I = np.eye(n)
        m = np.ones(n, complex)
        k1 = int(round(delta / dt))
        for j in range(k1):                                      # first lobe: q rises
            A = periodic_A(GAMMA * G * (j + 0.5) * dt, L, n, D, kappa)
            m = np.linalg.solve(I + 0.5 * dt * A, (I - 0.5 * dt * A) @ m)
        w, V = np.linalg.eigh(periodic_A(GAMMA * G * delta, L, n, D, kappa))
        m = V @ (np.exp(-w * (Delta - delta)) * (V.conj().T @ m))   # gap: exact
        for j in range(k1):                                      # second lobe: q falls
            A = periodic_A(GAMMA * G * (delta - (j + 0.5) * dt), L, n, D, kappa)
            m = np.linalg.solve(I + 0.5 * dt * A, (I - 0.5 * dt * A) @ m)
        return m.mean()

    # ---------- 2D: a disk on a grid (staircase) and on a fitted mesh ----------
    def fd_disk(R, h):
        """Cells of a Cartesian grid whose centres lie in the disk; zero flux to outside cells."""
        m = int(np.ceil(R / h))
        cc = (np.arange(-m, m) + 0.5) * h
        X, Y = np.meshgrid(cc, cc, indexing="ij")
        inside = X**2 + Y**2 < R**2
        idx = -np.ones(inside.shape, int)
        idx[inside] = np.arange(inside.sum())
        n = int(inside.sum())
        A = np.zeros((n, n))
        for di, dj in ((1, 0), (0, 1)):
            I0 = idx[: inside.shape[0] - di, : inside.shape[1] - dj]
            I1 = idx[di:, dj:]
            ok = (I0 >= 0) & (I1 >= 0)
            p, q = I0[ok], I1[ok]
            np.add.at(A, (p, p), 1 / h**2)
            np.add.at(A, (q, q), 1 / h**2)
            np.add.at(A, (p, q), -1 / h**2)
            np.add.at(A, (q, p), -1 / h**2)
        return A, X[inside], Y[inside]

    def disk_mesh(R, nr):
        """Triangle mesh of a disk: ring k has 6k nodes at radius kR/nr; triangles fill the space
        between neighbouring rings. The outer nodes lie exactly on the circle."""
        pts = [(0.0, 0.0)]
        rings = [[0]]
        for k in range(1, nr + 1):
            nk = 6 * k
            th = 2 * np.pi * np.arange(nk) / nk
            ids = list(range(len(pts), len(pts) + nk))
            pts += [(k * R / nr * np.cos(t), k * R / nr * np.sin(t)) for t in th]
            rings.append(ids)
        tris = []
        for k in range(1, nr + 1):
            inner, outer = rings[k - 1], rings[k]
            ni, no = len(inner), len(outer)
            if ni == 1:
                tris += [(inner[0], outer[j], outer[(j + 1) % no]) for j in range(no)]
                continue
            ai = 2 * np.pi * np.arange(ni) / ni
            ao = 2 * np.pi * np.arange(no) / no
            i = j = 0
            while i < ni or j < no:                              # walk round both rings by angle
                nai = ai[i + 1] if i + 1 < ni else 2 * np.pi
                nao = ao[j + 1] if j + 1 < no else 2 * np.pi
                if j < no and (nao <= nai or i >= ni):
                    tris.append((inner[i % ni], outer[j % no], outer[(j + 1) % no]))
                    j += 1
                else:
                    tris.append((inner[i % ni], outer[j % no], inner[(i + 1) % ni]))
                    i += 1
        return np.array(pts), np.array(tris)

    _f = np.array([1.0, 1.0, 2.0, 6.0])
    _T = np.zeros((3, 3, 3))
    for _i in range(3):
        for _j in range(3):
            for _k in range(3):
                _cnt = np.bincount([_i, _j, _k], minlength=3)
                _T[_i, _j, _k] = 2 * np.prod(_f[_cnt]) / 120.0  # ∫N_i N_j N_k dA = 2A·a!b!c!/5!, divided by A

    def fem2d(pts, tris, D=1.0):
        """Linear triangles: mass, stiffness and gradient (x-direction) matrices, assembled."""
        n = len(pts)
        P = pts[tris]
        x, y = P[:, :, 0], P[:, :, 1]
        A = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
        Ke = D * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * A[:, None, None])
        Me = A[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))[None]
        Je = A[:, None, None] * np.einsum("ijk,tk->tij", _T, x)
        rows = np.repeat(tris[:, :, None], 3, axis=2)
        cols = np.repeat(tris[:, None, :], 3, axis=1)
        Mm, K, J = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        np.add.at(Mm, (rows, cols), Me)
        np.add.at(K, (rows, cols), Ke)
        np.add.at(J, (rows, cols), Je)
        return Mm, K, J

    return disk_mesh, fd_disk, fem2d, periodic_signal


@app.cell(hide_code=True)
def _(mo):
    show_answers = mo.ui.switch(label="Teacher view: show all answers", value=False)
    mo.vstack(
        [
            mo.md(
                r"""
    # Chapter 4 · Solving the PDE on a grid or a mesh: finite differences and finite elements

    *© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.*

    *Course: simulating the diffusion MRI signal. Chapters 2 and 3 needed a pore whose eigenmodes or closed forms are known. This chapter solves the Bloch–Torrey equation for any shape, by cutting space into pieces.*

    **What you'll take away**

    1. **The problem every simulator solves:** tissue as compartments, each with its own diffusivity, relaxation time and spin density, joined by permeable interfaces.
    2. **The method of lines:** cut space into pieces and the Bloch–Torrey PDE becomes a large ODE system, with the same structure as the matrix method of Chapter 2.
    3. **Finite differences** as a flux balance between neighbouring cells: walls, membranes and relaxing walls are all rules for the flux through a face.
    4. **Time stepping:** explicit Euler, implicit Euler and Crank–Nicolson, their error orders, and the stability limit $\delta t\le h^2/(2dD)$.
    5. **The matrix form used by GPU codes** such as MATI: jump probabilities between voxels, an exact phase-and-relaxation factor, and the stability limit read as "no voxel gives away more than it holds".
    6. **Periodic media** simulated on a single cell, with a phase-shifted boundary condition.
    7. **Finite elements** from the weak form: why impermeable walls need no code, how a membrane enters, and the mass, stiffness, relaxation, gradient and interface matrices that SpinDoctor solves.
    8. **Curved walls:** a staircase grid against a mesh that follows the wall, and how fast each converges.
    9. How to choose $h$ and $\delta t$, and how to check any solver against Chapters 2 and 3.

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
    | 2 | Series expansions: matrix methods (MCF, multiple propagator, number of modes) and the cumulant expansion, and how they connect | done |
    | 3 | Analytical formulas: the PGSE box, the narrow-pulse $(q,t)$ map, the long-pulse $(g,T)$ map, the inside of the box, relaxing walls and exchange | done |
    | **4** | **Grids and meshes:** the multi-compartment problem, the method of lines, finite differences, time stepping and stability, jump probabilities, periodic cells, finite elements, curved walls | this notebook |
    | 5 | Monte Carlo: simulating the random walk | coming |

    **Units:** lengths in µm, times in ms, gradients in mT/m, $\gamma = 2.6752\times10^{-4}$ rad/(ms·mT/m·µm). $D = 2$ µm²/ms throughout. **Running example:** a slab of width $L = 8$ µm, PGSE with $\delta = 10$ ms, $\Delta = 20$ ms and $G = 300$ mT/m. Its exact echo, from the MCF of Chapter 2, is $E = 0.171071$. Every solver in this chapter is tested against that number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you need from Chapters 1–3

    | Idea | Formula |
    | --- | --- |
    | Bloch–Torrey | $\partial M/\partial t = \nabla\cdot(D\nabla M) - i\gamma\,g(t)\,x\,M$, $\ M(\mathbf r, 0) = 1$, $\ E = \frac1V\int M(\mathbf r, T)\,d\mathbf r$ |
    | Wall and interface conditions | impermeable: $D\,\partial M/\partial n = 0$; membrane: continuous flux $= \kappa\times$ jump in $M$; relaxing wall: $D\,\partial M/\partial n + \rho M = 0$ |
    | Matrix method (Ch 2) | $d\mathbf c/dt = -(D\Lambda + i\gamma g(t)\mathbf B)\,\mathbf c$ in the eigenmodes: $\Lambda$ diagonal, $\mathbf B$ dense |
    | Length scales | $\ell_D = \sqrt{Dt}$, $\ \ell_g = (D/\gamma g)^{1/3}$, phase wavelength $2\pi/q$ |
    | Tortuosity (Ch 3) | membranes with spacing $L$: $1/D_\infty = 1/D + 1/(\kappa L)$ |
    | Exchange mode (Ch 1) | slab $[-a, a]$, membrane at 0: slowest odd mode from $k\tan(ka) = 2\kappa/D$, rate $Dk^2\approx4\kappa/L$ |

    **Refresh first**

    - Taylor: $f(x\pm h) = f(x)\pm hf' + \frac{h^2}2f'' \pm \frac{h^3}6f''' + \frac{h^4}{24}f'''' + \dots$
    - Integration by parts in 1D: $\int_a^b wu'\,dx = [wu]_a^b - \int_a^b w'u\,dx$. In 3D, Green's first identity: $\int_\Omega w\,\nabla\cdot(D\nabla u) = \oint_{\partial\Omega} wD\,\partial_nu\,dS - \int_\Omega D\nabla w\cdot\nabla u$.
    - A linear ODE $\dot y = -\lambda y$ with $\lambda > 0$ decays. A one-step method multiplies $y$ by an amplification factor $G(\lambda\delta t)$ per step; it is stable when $\lvert G\rvert\le1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 4.0 · From tissue to a matrix ODE

    ## The problem: compartments joined by membranes

    Chapters 1–3 solved the Bloch–Torrey equation in one pore at a time: a slab, a cylinder, a sphere. Tissue is a collection of such spaces: the inside of each cell, the space between the cells, perhaps organelles or myelin. Each has its own diffusivity and relaxation, and they are connected through membranes. Before choosing a numerical method, it is worth writing this problem down once in general, because every simulator in this chapter (and the Monte Carlo of Chapter 5) solves exactly this.

    | Symbol | Meaning |
    | --- | --- |
    | $\Omega = \bigcup_n\Omega_n$ | the simulated piece of tissue, split into compartments $\Omega_n$ |
    | $\Gamma_{nm} = \partial\Omega_n\cap\partial\Omega_m$ | the interface (membrane) between compartments $n$ and $m$ |
    | $\Gamma_n = \partial\Omega_n\cap\partial\Omega$ | the part of $\Omega_n$ that lies on the outer boundary (may be empty) |
    | $D_n$, $T_{2,n}$, $\rho_n$ | diffusivity, transverse relaxation time and spin density in $\Omega_n$ |
    | $\kappa_{nm} = \kappa_{mn}$ | permeability of the interface $\Gamma_{nm}$ |
    | $\boldsymbol\nu_n$ | unit normal on $\partial\Omega_n$, pointing out of $\Omega_n$ |
    | $f(t)\,\mathbf g$ | the gradient: a fixed vector $\mathbf g$ times a time profile $f(t)$ ($+1$ and $-1$ during the two PGSE lobes, 0 in between) |

    In each compartment the magnetization $M_n(\mathbf r, t)$ follows the Bloch–Torrey equation, now with relaxation:

    $$\frac{\partial M_n}{\partial t} = \nabla\cdot(D_n\nabla M_n) - \frac{M_n}{T_{2,n}} - i\gamma f(t)\,(\mathbf g\cdot\mathbf r)\,M_n,\qquad \mathbf r\in\Omega_n,\qquad M_n(\mathbf r, 0) = \rho_n .$$

    Three conditions close the problem:

    | Where | Condition | Meaning |
    | --- | --- | --- |
    | outer boundary $\Gamma_n$ | $D_n\,\partial_{\nu}M_n = 0$ | no spins leave the simulated block |
    | interface $\Gamma_{nm}$ | $D_n\,\partial_{\nu_n}M_n = -D_m\,\partial_{\nu_m}M_m$ | what leaves one side enters the other |
    | interface $\Gamma_{nm}$ | $D_n\,\partial_{\nu_n}M_n = \kappa_{nm}\big(c_{nm}M_m - c_{mn}M_n\big)$ | the flux through the membrane is driven by the jump across it |

    The signal is the total magnetization at the echo time, $S = \int_\Omega M(\mathbf r, T_E)\,d\mathbf r$, and $E = S/\int_\Omega\rho$.

    → **The equilibrium factors** $c_{nm} = 2\rho_n/(\rho_n + \rho_m)$ matter only when two compartments hold different spin densities. They make sure that the equilibrium state, $M_n = \rho_n$ everywhere, drives no flux through any membrane (Exercise 6). For equal densities $c_{nm} = 1$, and the condition is the familiar $\kappa\times$ jump of Chapters 1 and 3.

    → **Relaxation** is the easy part. With one $T_2$ everywhere it simply multiplies the signal by $e^{-T_E/T_2}$ (Exercise 4). It matters when compartments relax differently: it then changes their weights in the signal, and with exchange it mixes with diffusion.

    → **The outer boundary is artificial.** A real tissue block continues beyond the simulated one. Reflecting walls are the simplest choice; one periodic cell (4a ⑥) avoids the artificial wall altogether.

    → **The running example** is the smallest case: one compartment (a slab), $\rho = 1$, no relaxation, a gradient along $x$. A membrane in the middle makes it two compartments when we need them.

    ## The method of lines

    Cut space into $n$ pieces and keep time continuous. Each piece carries one unknown, so the magnetization $M(x,t)$ becomes a vector $\mathbf M(t)$ and the Bloch–Torrey PDE becomes a system of $n$ linear ODEs:

    $$\boldsymbol{\mathcal M}\,\frac{d\mathbf M}{dt} = -\big(D\,\mathbf K + i\gamma\,g(t)\,\mathbf X\big)\,\mathbf M .$$

    This is the **method of lines**: every point of space follows its own line in time, coupled to its neighbours. You met this form in Chapter 2, where the pieces were the eigenmodes. What changes from method to method is only the choice of pieces:

    | | Modes (Chapter 2) | Grid (finite differences) | Hat functions (finite elements) |
    | --- | --- | --- | --- |
    | Unknowns | coefficients $c_n$ of the eigenmodes | one value $M_i$ per cell | one value $M_i$ per node |
    | $M(x)$ rebuilt as | $\sum_nc_nu_n(x)$: smooth, global | piecewise constant | piecewise linear |
    | Diffusion matrix $\mathbf K$ | $\Lambda$, diagonal | neighbours only | neighbours only |
    | Gradient matrix $\mathbf X$ | $\mathbf B$, dense | $\mathrm{diag}(x_i)$ | $\mathbf J$, neighbours only |
    | Mass matrix $\boldsymbol{\mathcal M}$ | identity | identity | $\int N_iN_j$, neighbours only |
    | Geometry | needs the modes of the shape | any shape; walls become staircases | any shape; walls lie on the mesh |

    → In the mode basis diffusion is diagonal and the gradient couples every mode. On a grid the gradient is diagonal and diffusion couples neighbours. Same equation, opposite trade-off: modes are compact for simple shapes, grids and meshes work for any shape.

    → After this step, solving the PDE means two separate choices: how to cut **space** (sections 4a–4c) and how to step in **time** (section 4a ④).

    The figure shows the three ways of storing the same magnetization: a slab profile at the end of a short, strong gradient lobe, rebuilt from $n$ unknowns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rep_n = mo.ui.slider(3, 30, step=1, value=8, label="Number of unknowns n", show_value=True)
    rep_n
    return (rep_n,)


@app.cell(hide_code=True)
def _(C_ALT, C_GREEN, C_MAIN, D0, GAMMA, expm, mo, np, plt, rep_n, slab_B):
    _L, _G, _de = 8.0, 1500.0, 2.0
    _lam, _B = slab_B(_L, 60)
    _c = expm(-(D0 * np.diag(_lam) + 1j * GAMMA * _G * _B) * _de)[:, 0]   # state after the first lobe
    _xf = np.linspace(0, _L, 1601)
    _k = np.arange(60)[:, None]
    _U = np.where(_k == 0, 1 / np.sqrt(_L), np.sqrt(2 / _L) * np.cos(_k * np.pi * _xf / _L))
    _Mf = (_c[:, None] * _U).sum(0).real * np.sqrt(_L)   # Re M, normalised so that M(x, 0) = 1
    _n = rep_n.value
    _M_modes = (_c[:_n, None] * _U[:_n]).sum(0).real * np.sqrt(_L)
    _edges = np.linspace(0, _L, _n + 1)
    _cell = np.minimum((_xf / _L * _n).astype(int), _n - 1)
    _avg = np.array([_Mf[_cell == i].mean() for i in range(_n)])
    _M_fd = _avg[_cell]
    _nodes = np.linspace(0, _L, _n)
    _M_fem = np.interp(_xf, _nodes, np.interp(_nodes, _xf, _Mf))

    def _err(f):
        return np.sqrt(np.mean((f - _Mf) ** 2)) / np.sqrt(np.mean(_Mf**2))

    _fig, _axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for _a, _f, _col, _title in (
        (_axs[0], _M_modes, C_MAIN, f"{_n} eigenmodes (Ch 2)"),
        (_axs[1], _M_fd, C_ALT, f"{_n} cells (finite differences)"),
        (_axs[2], _M_fem, C_GREEN, f"{_n} nodes (finite elements)"),
    ):
        _a.plot(_xf, _Mf, color="k", lw=1.0, label="exact")
        _a.plot(_xf, _f, color=_col, lw=2.0, label="representation")
        _a.set(xlabel="x (µm)", title=f"{_title}: error {100 * _err(_f):.1f}%")
    _axs[1].vlines(_edges, -1.2, 1.2, color="0.85", lw=0.6, zorder=0)
    _axs[2].plot(_nodes, np.interp(_nodes, _xf, _Mf), "o", color=C_GREEN, ms=4)
    _axs[0].set(ylabel="Re M(x) after one lobe", ylim=(-1.0, 1.2))
    _axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Slab L = 8 µm after one short, strong lobe (δ = 2 ms, G = 1500 mT/m, so qL = 6.4: one turn of phase across the slab). Error: root-mean-square difference with the exact profile, relative to its size.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Explain in one sentence each why diffusion is diagonal in the mode basis, and why the gradient is diagonal on a grid.

    **(b)** For $n = 1000$ unknowns in 1D, count the nonzero entries of the diffusion and gradient matrices in each basis. Which basis is cheapest per time step?

    **(c)** In the figure, how does each error fall as you double $n$? Explain the order of each from how the representation is built.

    **(d)** The modes win easily in this figure. Why does nobody use them for a segmented neuron?
    """, "1 · Three bases for one equation")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ The modes are the eigenfunctions of the Laplacian, so the Laplacian maps each mode onto itself: $\Lambda$ is diagonal. A grid value only knows its own position, so multiplying by $x$ multiplies each value by its own $x_i$: $\mathrm{diag}(x_i)$.

    **(b)** ⇒ Modes: $\Lambda$ has $10^3$ nonzeros, $\mathbf B$ about $5\times10^5$ (dense, half zero by parity). Grid: the 1D Laplacian has about $3\times10^3$, $\mathrm{diag}(x)$ has $10^3$. Finite elements: $\mathbf K$, $\mathbf J$ and $\boldsymbol{\mathcal M}$ about $3\times10^3$ each.

    → Per step the grid and the mesh cost about $n$ operations, the modes about $n^2$. The modes still win in practice for simple shapes, because they need far fewer unknowns: 20 modes do what 400 grid cells do (Chapter 2).

    **(c)** ⇒ Doubling $n$ divides the finite-difference error by about 2 (first order), the finite-element error by about 4 (second order), and the mode error by much more than 4.

    → A piecewise-constant function differs from a smooth one by about $h\,\lvert M'\rvert$ in each cell: error $\propto h$. Linear interpolation between nodes leaves the curvature, about $h^2\lvert M''\rvert/8$: error $\propto h^2$. The modes converge as fast as the smoothness of the mirrored profile allows (Jackson, Chapter 2): algebraically, with a high power.

    → The finite-difference *solver* is still second order (section 4a): it computes cell values that are accurate to $O(h^2)$ at the cell centres. The first-order error here is only in drawing those values as flat steps.

    **(d)** ⇒ Nobody knows the eigenmodes of a neuron. They have to be computed first, on a mesh, which is a finite-element problem (section 4b ⑦). This is exactly what SpinDoctor's matrix formalism does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 4a · Finite differences

    ## ① The grid, and the Laplacian as a flux balance

    Cut the pore into cells of width $h$, with one value $M_i$ at each cell centre $x_i$. Think of $M_i$ as the amount of magnetization in cell $i$, divided by its volume. Diffusion only moves magnetization between neighbouring cells, through the faces between them.

    ⇒ **Fick's law at a face.** The flux from cell $i$ to cell $i+1$ is $J_{i+1/2} = D\,(M_i - M_{i+1})/h$.

    ⇒ **Balance of one cell.** What enters through one face and leaves through the other changes the content of the cell:

    $$h\,\frac{dM_i}{dt} = J_{i-1/2} - J_{i+1/2} = \frac Dh\big(M_{i-1} - 2M_i + M_{i+1}\big).$$

    ⇒ Divide by $h$: the familiar three-point Laplacian, $D(M_{i-1} - 2M_i + M_{i+1})/h^2$. Taylor-expanding $M_{i\pm1}$ shows that it differs from $D\,M''(x_i)$ by $\frac{Dh^2}{12}M''''$: second order in $h$ (Exercise 2).

    → Writing the scheme as a flux balance (a **finite-volume** view) has two advantages. Magnetization is conserved exactly, because every flux leaves one cell and enters the next. And every kind of wall becomes a rule for the flux through one face.

    ## ② Walls, membranes and relaxing walls as face rules

    Each face gets a **conductance** $c$, so that the flux is $c$ times the difference across it. In matrix form $d\mathbf M/dt = -\mathbf A\mathbf M$, a face between cells $i$ and $j$ adds $c/h$ to $A_{ii}$ and $A_{jj}$, and $-c/h$ to $A_{ij}$ and $A_{ji}$.

    | Face | Flux through the face | Conductance $c$ |
    | --- | --- | --- |
    | bulk, between cells $i$ and $i+1$ | $D(M_i - M_{i+1})/h$ | $D/h$ |
    | impermeable wall | $0$ | $0$ (the same as a "ghost cell" with $M_{\text{ghost}} = M_i$) |
    | membrane with permeability $\kappa$ | $(M_i - M_{i+1})\big/\big(\frac h{2D} + \frac1\kappa + \frac h{2D}\big)$ | $1/(h/D + 1/\kappa)$ |
    | cells with different $D_i$, $D_{i+1}$ | $(M_i - M_{i+1})\big/\big(\frac h{2D_i} + \frac h{2D_{i+1}}\big)$ | the harmonic mean of $D_i$, $D_{i+1}$, divided by $h$ |
    | relaxing wall, relaxivity $\rho$ | $M_i\big/\big(\frac h{2D} + \frac1\rho\big)$ out of the pore | adds $c_w/h$ to $A_{ii}$ only, $c_w = 1/(h/2D + 1/\rho)$ |

    → All the rules come from one idea: **resistances in series**. Between two cell centres the flux passes half a cell, perhaps a membrane, and another half cell. Their resistances $h/2D$, $1/\kappa$, $h/2D$ add. This is the same reasoning as the tortuosity formula of Chapter 3.

    ## ③ The gradient

    The gradient term is local: each cell dephases at its own rate. It adds $i\gamma g(t)x_i$ to $A_{ii}$, a diagonal matrix. Relaxation is just as local: a cell in compartment $n$ gets an extra $1/T_{2,n}$ on the diagonal. Without relaxation, assembled:

    $$\frac{d\mathbf M}{dt} = -\big(\mathbf A + i\gamma g(t)\,\mathrm{diag}(x_i)\big)\,\mathbf M,\qquad E = \frac1n\sum_iM_i\ \text{ at the echo}.$$

    **Recipe**

    ① Choose $h$ (see 4d) and place the cell centres.
    ② Give every face its conductance: bulk, wall, membrane or relaxing wall.
    ③ Assemble $\mathbf A$ and $\mathrm{diag}(x_i)$, start from $\mathbf M = 1$.
    ④ Step in time (next section) through the gradient waveform.
    ⑤ Average $\mathbf M$ over the cells at the echo.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Expand $M(x\pm h)$ in a Taylor series and show that $\frac{M(x-h) - 2M(x) + M(x+h)}{h^2} = M''(x) + \frac{h^2}{12}M''''(x) + O(h^4)$.

    **(b)** Show that putting a ghost cell with $M_0 = M_1$ next to the first cell gives exactly zero flux through the wall, and that the cell then has the stencil $D(M_2 - M_1)/h^2$.

    **(c)** Derive the membrane conductance $c = 1/(h/D + 1/\kappa)$ from three resistances in series. Check the two limits $\kappa\to0$ and $\kappa\to\infty$.

    **(d)** In the relaxing-wall rule, the first cell loses magnetization at rate $c_w/h$. For a fine grid ($h\ll D/\rho$), show that the total loss rate of a slab with two relaxing walls is $2\rho/L$, the fast-diffusion result of Brownstein and Tarr (Chapter 3).
    """, "2 · Faces and fluxes")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) Add the expansions of $M(x+h)$ and $M(x-h)$: the odd powers cancel.
    - (c) Between the two cell centres the flux crosses half a cell of water, the membrane, and half a cell of water. A flux $J$ needs a drop $J\cdot(\text{resistance})$ across each.
    - (d) Start with uniform $M$. Each wall cell loses $c_wM/h$ per unit time, and that cell is a fraction $h/L$ of the slab.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $M(x\pm h) = M \pm hM' + \frac{h^2}2M'' \pm \frac{h^3}6M''' + \frac{h^4}{24}M'''' + O(h^5)$. Adding: $M(x+h) + M(x-h) = 2M + h^2M'' + \frac{h^4}{12}M'''' + O(h^6)$. Subtract $2M$ and divide by $h^2$. ✓

    → The error is $\frac{h^2}{12}M''''$: second order. It is small where $M$ is smooth and large where $M$ varies on the scale of $h$, which is why $h$ must resolve $\ell_g$ and the phase wavelength.

    **(b)** ⇒ The wall flux is $D(M_0 - M_1)/h = 0$. The cell balance becomes $h\dot M_1 = 0 - D(M_1 - M_2)/h$, so $\dot M_1 = D(M_2 - M_1)/h^2$. ✓ It is the same as giving the wall face conductance 0.

    **(c)** ⇒ Drops: $J\frac h{2D}$, $\frac J\kappa$, $J\frac h{2D}$. Total $M_i - M_{i+1} = J\big(\frac hD + \frac1\kappa\big)$, so $J = c\,(M_i - M_{i+1})$ with $c = 1/(h/D + 1/\kappa)$. ✓

    ⇒ $\kappa\to0$: $c\to0$, an impermeable wall. $\kappa\to\infty$: $c\to D/h$, an ordinary bulk face. ✓

    **(d)** ⇒ For $h\ll D/\rho$: $c_w = 1/(h/2D + 1/\rho)\approx\rho$. Each wall cell loses $\rho M/h$ per unit time, and is a fraction $h/L$ of the volume. Two walls: total rate $2\cdot\frac hL\cdot\frac\rho h = \frac{2\rho}L = \rho\frac SV$. ✓

    → On a coarse grid ($h\gtrsim D/\rho$) the half-cell resistance $h/2D$ limits the loss, as it should: the magnetization must first diffuse to the wall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ④ Stepping in time

    The ODE system $d\mathbf M/dt = -\mathbf A_g\mathbf M$, with $\mathbf A_g = \mathbf A + i\gamma g\,\mathrm{diag}(x)$, is stepped with one of three classic one-step methods:

    | Method | One step | Order | Cost per step | Stable for |
    | --- | --- | --- | --- | --- |
    | explicit (forward) Euler | $\mathbf M^+ = (\mathbf I - \delta t\,\mathbf A_g)\,\mathbf M$ | 1 | one sparse product | $\delta t\le h^2/(2dD)$ |
    | implicit (backward) Euler | $(\mathbf I + \delta t\,\mathbf A_g)\,\mathbf M^+ = \mathbf M$ | 1 | one sparse solve | any $\delta t$ |
    | Crank–Nicolson (trapezoid) | $(\mathbf I + \frac{\delta t}2\mathbf A_g)\,\mathbf M^+ = (\mathbf I - \frac{\delta t}2\mathbf A_g)\,\mathbf M$ | 2 | one sparse solve | any $\delta t$ |

    ($d$ is the number of space dimensions.) "Order $p$" means that halving $\delta t$ divides the error at the echo by $2^p$. Crank–Nicolson is the average of the other two ([Crank & Nicolson 1947](https://doi.org/10.1017/S0305004100023197)).

    **Why explicit Euler has a limit.** Look at one Fourier pattern on the grid, $M_j = e^{ikjh}$. The three-point Laplacian maps it onto itself, multiplied by $-\mu$ with $\mu = \frac{4D}{h^2}\sin^2\frac{kh}2$. One explicit step therefore multiplies it by the **amplification factor**

    $$G_{\text{EE}} = 1 - \delta t\,\mu = 1 - 4r\sin^2\tfrac{kh}2,\qquad r = \frac{D\,\delta t}{h^2}.$$

    The fastest pattern on the grid, $kh = \pi$ (neighbouring cells alternate), has $G = 1 - 4r$. For $r > 1/2$ that is below $-1$: this pattern flips sign and grows every step, and the solution blows up. Hence $\delta t\le h^2/(2D)$ in 1D, and $h^2/(2dD)$ in $d$ dimensions. This is the Courant–Friedrichs–Lewy type of limit ([Courant, Friedrichs & Lewy 1928](https://doi.org/10.1007/BF01448839)).

    For the implicit methods: $G_{\text{IE}} = 1/(1 + 4r\sin^2\frac{kh}2)$ and $G_{\text{CN}} = (1 - 2r\sin^2\frac{kh}2)/(1 + 2r\sin^2\frac{kh}2)$. Both stay within $[-1, 1]$ for every $r$. Crank–Nicolson's factor approaches $-1$ for large $r$: the fastest patterns are not damped but flip sign every step. That is harmless for a smooth start like $M = 1$, but it is why Crank–Nicolson can show small wiggles after a sudden change.

    Left: the three factors over all patterns on the grid, for a chosen $r$. Right: the error at the echo of the running example against $\delta t$, for all three (grid of 40 cells, $h = 0.2$ µm, so $h^2/2D = 0.01$ ms).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    stab_r = mo.ui.slider(0.1, 3.0, step=0.05, value=0.6, label="r = Dδt/h²", show_value=True)
    stab_r
    return (stab_r,)


@app.cell(hide_code=True)
def _(fd_signal, np):
    # error against δt for the three schemes on a 40-cell grid (computed once)
    st_n, st_L = 40, 8.0
    st_lim = (st_L / st_n) ** 2 / (2 * 2.0)
    st_ref = fd_signal(300.0, st_L, 10.0, 20.0, n=st_n, dt=0.001, scheme="cn").real
    st_ee_dt = st_lim * np.array([0.1, 0.2, 0.4, 0.8, 1.05])
    st_ee = np.array([fd_signal(300.0, st_L, 10.0, 20.0, n=st_n, dt=d, scheme="ee") for d in st_ee_dt])
    st_dt = np.array([0.0125, 0.025, 0.05, 0.1, 0.2, 0.4])
    st_ie = np.array([fd_signal(300.0, st_L, 10.0, 20.0, n=st_n, dt=d, scheme="ie").real for d in st_dt])
    st_cn = np.array([fd_signal(300.0, st_L, 10.0, 20.0, n=st_n, dt=d, scheme="cn").real for d in st_dt])
    return st_cn, st_dt, st_ee, st_ee_dt, st_ie, st_lim, st_ref


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_GREEN,
    C_MAIN,
    mo,
    np,
    plt,
    st_cn,
    st_dt,
    st_ee,
    st_ee_dt,
    st_ie,
    st_lim,
    st_ref,
    stab_r,
):
    _r = stab_r.value
    _th = np.linspace(0, np.pi, 300)
    _s = np.sin(_th / 2) ** 2
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    _a1.plot(_th / np.pi, 1 - 4 * _r * _s, color=C_ALT, lw=2, label="explicit Euler")
    _a1.plot(_th / np.pi, 1 / (1 + 4 * _r * _s), color=C_GREEN, lw=2, label="implicit Euler")
    _a1.plot(_th / np.pi, (1 - 2 * _r * _s) / (1 + 2 * _r * _s), color=C_MAIN, lw=2, label="Crank–Nicolson")
    _a1.plot(_th / np.pi, np.exp(-4 * _r * _s), color="k", lw=1, ls="--", label="exact: e^{−δt·μ}")
    _a1.axhspan(-1, 1, color="0.92", zorder=0)
    _a1.axhline(-1, color="k", lw=0.8)
    _a1.set(xlabel="kh / π  (1 = neighbouring cells alternate)", ylabel="amplification per step G", ylim=(-3.2, 1.15), title=f"Amplification factors at r = {_r:.2f}")
    _a1.legend(frameon=False, fontsize=8, loc="lower left")

    _ok = np.isfinite(st_ee)
    _a2.loglog(st_ee_dt[_ok], np.abs(st_ee[_ok].real - st_ref), "o-", color=C_ALT, label="explicit Euler")
    for _d in st_ee_dt[~_ok]:
        _a2.plot(_d, 3e-2, "x", color=C_ALT, ms=10, mew=2)
        _a2.text(_d * 1.1, 3e-2, "blows up", color=C_ALT, fontsize=8, va="center")
    _a2.loglog(st_dt, np.abs(st_ie - st_ref), "s-", color=C_GREEN, label="implicit Euler")
    _a2.loglog(st_dt, np.abs(st_cn - st_ref), "^-", color=C_MAIN, label="Crank–Nicolson")
    _c0 = abs(st_cn[0] - st_ref)
    _a2.loglog(st_dt, _c0 * (st_dt / st_dt[0]) ** 2, ":", color=C_MAIN, lw=1)
    _a2.text(st_dt[3], _c0 * (st_dt[3] / st_dt[0]) ** 2 * 3, "∝ δt²", color=C_MAIN, fontsize=8)
    _a2.axvline(st_lim, color="k", lw=0.8, ls="--")
    _a2.text(st_lim * 1.1, 2e-9, "h²/2D", fontsize=8)
    _a2.set(xlabel="time step δt (ms)", ylabel="|error in E| at the echo", ylim=(1e-9, 1e-1), title="Running example on a 40-cell grid")
    _a2.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _verdict = "explicit Euler is stable" if _r <= 0.5 else "explicit Euler is **unstable**: the alternating pattern grows"
    mo.vstack([_fig, mo.md(f"At r = {_r:.2f}: {_verdict}. For this grid, r = {_r:.2f} means δt = {_r * st_lim * 2:.4f} ms.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the right panel says.** Explicit Euler needs 30 000 steps at $\delta t = 0.1\,h^2/2D$ for an error of $2\times10^{-5}$, and the error only halves when $\delta t$ halves. Crank–Nicolson takes steps 20 times *larger* than the explicit limit ($\delta t = 0.2$ ms, 150 steps) and is still 5 times more accurate ($4\times10^{-6}$). Implicit Euler is stable at any step but, being first order, is the least accurate of the three at large steps. Each Crank–Nicolson step costs a sparse solve instead of a product, but in 1D that is cheap.

    → So why do GPU codes such as MATI step explicitly? In 3D a sparse solve is expensive and hard to parallelise, while an explicit step is the same small stencil for every voxel. Millions of voxels updated independently, many times, can beat fewer but sequential solves (section 4d).

    **A trap for explicit Euler: the phase term.** For the gradient alone, one explicit step multiplies $M_i$ by $1 - i\gamma gx_i\,\delta t$, whose modulus is $\sqrt{1 + (\gamma gx_i\delta t)^2} > 1$. Diffusion damps the fast patterns, but the uniform pattern is not damped, and its magnitude creeps up by about $\exp\big(\frac12(\gamma gx)^2\,\delta t\,T\big)$ over the sequence. A common cure is to apply the phase exactly, $M_i\to M_i\,e^{-i\gamma gx_i\delta t}$, and diffusion explicitly: an operator split, as in Chapter 2. That is the update of the next section.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive $G_{\text{IE}}$ and $G_{\text{CN}}$ in the same way as $G_{\text{EE}}$, and show that $\lvert G\rvert\le1$ for every $r$.

    **(b)** In 2D, the five-point Laplacian has $\mu = \frac{4D}{h^2}\big(\sin^2\frac{k_xh}2 + \sin^2\frac{k_yh}2\big)$. Derive the explicit limit $\delta t\le h^2/(4D)$, and guess the 3D one. Evaluate it for a 3D voxel grid with $h = 0.1$ µm and $D = 2$ µm²/ms. How many steps does the 30 ms running example need?

    **(c)** Set $r = 2$ in the left panel. Which patterns does each method get badly wrong? Why does that hardly affect the signal?

    **(d)** Estimate the creep of explicit Euler's phase term for the running example: $x$ up to $L/2 = 4$ µm, $G = 300$ mT/m, $\delta t = h^2/2D$ with $h = 0.1$ µm, $T = 30$ ms.
    """, "3 · Stability and accuracy in time")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) Apply the step to $M_j = e^{ikjh}$ and use that $\mathbf A$ multiplies this pattern by $\mu$. For CN, write $s = 2r\sin^2(kh/2)\ge0$: $G = (1-s)/(1+s)$.
    - (b) The worst pattern alternates in every direction.
    - (c) What is $M$ made of at the start? Which patterns does a smooth $M$ contain?
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Implicit: $(1 + \delta t\mu)G = 1$, so $G_{\text{IE}} = 1/(1 + 4r\sin^2\frac{kh}2)\in(0, 1]$. ✓

    ⇒ CN: $(1 + \frac{\delta t}2\mu)G = 1 - \frac{\delta t}2\mu$. With $s = \frac{\delta t}2\mu = 2r\sin^2\frac{kh}2\ge0$: $G = \frac{1-s}{1+s}\in(-1, 1]$. ✓ It tends to $-1$ for $s\to\infty$.

    **(b)** ⇒ The worst pattern has $k_xh = k_yh = \pi$: $\mu_{\max} = 8D/h^2$. Stability needs $\delta t\,\mu_{\max}\le2$, so $\delta t\le h^2/(4D)$. In 3D: $h^2/(6D)$, i.e. $h^2/(2dD)$.

    ⇒ $h = 0.1$ µm, $D = 2$: $\delta t\le0.01/12 = 8.3\times10^{-4}$ ms $= 0.83$ µs. A 30 ms sequence needs about **36 000 steps**.

    **(c)** ⇒ At $r = 2$ explicit Euler amplifies the fast patterns (blow-up). Implicit Euler damps them, but too little. Crank–Nicolson keeps their size but flips their sign. All three are accurate for slow patterns ($kh\ll1$), where they follow the exact curve.

    → The start $M = 1$ and the smooth dephasing produce only slow patterns on a fine grid. The fast patterns carry almost nothing, so the implicit methods' errors there do not show in $E$. The explicit method is different: rounding errors seed the fast patterns, and instability then amplifies them from $10^{-16}$ to anything.

    **(d)** ⇒ $\gamma Gx = 2.6752\times10^{-4}\times300\times4 = 0.32$ /ms. $\delta t = 0.01/4 = 0.0025$ ms. Creep $\approx\exp(\frac12\times0.32^2\times0.0025\times30) = \exp(0.0039)$: a 0.4% error at the edges of the slab. Small, but systematic, and it does not shrink as fast as the diffusion error; applying the phase exactly removes it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ⑤ The matrix form used by GPU codes: jump probabilities

    Write one explicit Euler step of the flux balance cell by cell, leaving the gradient out for a moment:

    $$M_j^{+} = \big(1 - s_{j\to j-1} - s_{j\to j+1}\big)\,M_j + s_{j-1\to j}\,M_{j-1} + s_{j+1\to j}\,M_{j+1},\qquad s_{j\to k} = \frac{c_{jk}\,\delta t}{h} .$$

    ⇒ Read it as bookkeeping. In one step a fraction $s_{j\to k}$ of the content of cell $j$ moves to cell $k$, and the rest stays. That is why $s$ is called a **jump probability**. In the bulk $s = D\,\delta t/h^2 = r$.

    ⇒ One number per face holds everything about the structure: the diffusivity, a membrane ($c = 1/(h/D + 1/\kappa)$), a wall ($s = 0$), a change of $D$. The structure enters only through these numbers, which are computed once, before the simulation starts. A Monte Carlo simulation instead meets the structure again in every step of every walker (Chapter 5).

    ⇒ The probability to stay must not be negative: $1 - s_{j\to j-1} - s_{j\to j+1}\ge0$, so $r\le\frac12$ in the bulk. This is the stability limit of ④ again, now with a physical reading: **a cell cannot give away more than it holds.**

    **The matrix.** Number the cells $1\dots n$ and collect the jump probabilities in a matrix $\mathbf S$: off the diagonal $S_{kj} = s_{j\to k}$, on the diagonal $S_{jj} = -\sum_ks_{j\to k}$. One diffusion step is $\mathbf M^+ = (\mathbf I + \mathbf S)\,\mathbf M$, and $\mathbf I + \mathbf S = \mathbf I - \delta t\,\mathbf A$ in the notation of ④.

    → Every column of $\mathbf S$ sums to zero: what leaves cell $j$ arrives in another cell. The total magnetization is conserved exactly, step after step.

    **Relaxation and gradient as one factor.** Over one step, relaxation and the gradient act on every cell separately, so they can be applied exactly:

    $$\Phi_j = \exp\!\Big(-i\gamma f(t)\,(\mathbf g\cdot\mathbf r_j)\,\delta t - \frac{\delta t}{T_{2,j}}\Big),\qquad \mathbf M^{n+1} = \boldsymbol\Phi\odot(\mathbf I + \mathbf S)\,\mathbf M^n,$$

    where $\odot$ multiplies entry by entry. This is the **matrix-based finite-difference (MFD)** update of [Xu, Does & Gore (2007)](https://doi.org/10.1088/0031-9155/52/7/N01): diffusion explicitly, everything local exactly. It removes the phase-term trap of ④.

    → In 3D, $\mathbf S$ couples each voxel to its six neighbours. MATI describes a substrate as a 3D matrix of labels (1 for the extracellular space, 2 for the intracellular space), turns the labels into jump probabilities once, and then repeats the same sparse product and entry-wise multiplication at every step: an ideal job for a GPU.

    The table runs this update on the running example. The time step is set through $r$, so $\delta t = rh^2/D$ shrinks with $h^2$.
    """)
    return


@app.cell(hide_code=True)
def _(GAMMA, fd_ops, mo, np, pgse_mcf, pgse_pieces):
    def _mfd_signal(G, L, delta, Delta, n=40, r=0.4, D=2.0, T2=None):
        """PGSE echo with the MFD update M ← Φ ⊙ (I + S) M, with δt = r h²/D."""
        A, x, h = fd_ops(L, n, D)
        dt = r * h**2 / D
        IS = np.eye(n) - dt * A                     # I + S: stay and jump probabilities
        M = np.ones(n, complex)
        for sgn, dur in pgse_pieces(delta, Delta):
            Phi = np.exp(-1j * GAMMA * sgn * G * x * dt - (dt / T2 if T2 else 0.0))
            for _ in range(int(round(dur / dt))):
                M = Phi * (IS @ M)
                if abs(M[0]) > 1e6:
                    return np.nan, dt
        return M.mean(), dt

    _ex = pgse_mcf(300.0, 8.0, 10.0, 20.0).real
    _rows = []
    for _n, _r in ((20, 0.4), (40, 0.4), (80, 0.4), (40, 0.5), (40, 0.55)):
        _E, _dt = _mfd_signal(300.0, 8.0, 10.0, 20.0, n=_n, r=_r)
        _steps = int(round(30.0 / _dt))
        if np.isfinite(_E):
            _rows.append(f"| {_n} | {8.0 / _n:.2f} | {_r:.2f} | {_dt:.4f} | {_steps} | {_E.real:.6f} | {abs(_E.real - _ex):.1e} |")
        else:
            _rows.append(f"| {_n} | {8.0 / _n:.2f} | {_r:.2f} | {_dt:.4f} | {_steps} | blows up | — |")
    _E0, _ = _mfd_signal(300.0, 8.0, 10.0, 20.0, n=40, r=0.4)
    _E2, _ = _mfd_signal(300.0, 8.0, 10.0, 20.0, n=40, r=0.4, T2=50.0)
    mo.md(
        "| cells $n$ | $h$ (µm) | $r$ | $\\delta t$ (ms) | steps | $E$ | error |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(_rows)
        + f"\n\nExact: E = {_ex:.6f}. At fixed r, halving h divides the error by 4 to 6: the time step shrinks with h², so the first-order time error falls as fast as the second-order space error. "
        + f"Above r = 1/2 the alternating pattern grows and the run blows up.\n\n"
        + f"With T₂ = 50 ms in every cell: E = {_E2.real:.6f}, and E(T₂)/E(no relaxation) = {(_E2 / _E0).real:.6f} against e^(−30/50) = {np.exp(-0.6):.6f}."
    )
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Show that every column of $\mathbf S$ sums to zero, and that a diffusion step therefore leaves $\sum_jM_j$ unchanged. What does $\boldsymbol\Phi$ do to the total?

    **(b)** Show that a membrane face has $s = r\big/\big(1 + \frac D{\kappa h}\big)$. Which is larger, $s$ or the bulk $r$? What do $\kappa\to0$ and $\kappa\to\infty$ give?

    **(c)** A tissue has $D_1 = 2$ µm²/ms inside the cells and $D_2 = 3$ µm²/ms outside. Which compartment sets the largest allowed $\delta t$ of an explicit 3D simulation with $h = 0.1$ µm? Does a membrane make the limit stricter?

    **(d)** With one $T_2$ in every cell, show that the MFD signal at the echo is *exactly* $e^{-T_E/T_2}$ times the signal without relaxation (the last line under the table). Why is this no longer true when the two compartments of (c) have different $T_2$?
    """, "4 · Jump probabilities")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) Column $j$ holds what cell $j$ sends out: $s_{j\to k}$ to each neighbour $k$, and $-\sum_ks_{j\to k}$ on the diagonal.
    - (b) $s = c\,\delta t/h$ with $c = 1/(h/D + 1/\kappa)$.
    - (c) The stay probability is smallest where the outgoing jump probabilities are largest.
    - (d) The relaxation part of $\Phi_j$ is the same number for every cell. Does it commute with $\mathbf I + \mathbf S$?
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Column $j$: $\sum_{k\ne j}s_{j\to k} - \sum_ks_{j\to k} = 0$. So $\mathbf 1^{\mathsf T}(\mathbf I + \mathbf S) = \mathbf 1^{\mathsf T}$, and $\sum_jM_j^+ = \sum_jM_j$. ✓

    ⇒ $\Phi_j$ has modulus $e^{-\delta t/T_{2,j}}\le1$ and a phase that differs from cell to cell. It turns each cell's magnetization by its own angle and shrinks it. The total can only lose size: by relaxation, and by the cells pointing in different directions (dephasing).

    **(b)** ⇒ $s = \frac{\delta t}h\cdot\frac1{h/D + 1/\kappa} = \frac{D\delta t}{h^2}\cdot\frac1{1 + D/(\kappa h)} = \frac r{1 + D/(\kappa h)}$. ✓ Always smaller than $r$.

    ⇒ $\kappa\to0$: $s\to0$, a wall. $\kappa\to\infty$: $s\to r$, an ordinary bulk face. ✓

    **(c)** ⇒ The jump probability grows with $D$, so the fastest compartment sets the limit: $\delta t\le h^2/(2dD_{\max}) = 0.01/(6\times3) = 0.56$ µs, set by the extracellular space. A membrane only lowers $s$ on its faces, so it never makes the limit stricter.

    **(d)** ⇒ With one $T_2$: $\boldsymbol\Phi = e^{-\delta t/T_2}\,\boldsymbol\Phi_0$, where $\boldsymbol\Phi_0$ holds only the phases. A scalar commutes with every matrix, so after $N$ steps $\mathbf M = e^{-N\delta t/T_2}\,\mathbf M_0$, with $\mathbf M_0$ the run without relaxation. At the echo $N\delta t = T_E$: exactly $e^{-T_E/T_2}$. ✓

    → With different $T_2$ per compartment, the relaxation factor is a different number in different cells. It no longer commutes with the jumps: a spin's decay depends on how long it has stayed in each compartment. The signal is then not a product of a diffusion part and a relaxation part, and only a simulation (or a model of exchange, as in Chapter 3) can separate them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ⑥ Periodic media: one cell with a phase-shifted boundary

    Tissue is often modelled as a periodic array: the same cell repeated in every direction. Simulating a large block of cells is expensive, and its outer walls are artificial. Can one cell stand for the whole array?

    The diffusion problem is periodic, but the magnetization is not: the gradient makes $M$ wind faster and faster in space. The trick is to take out the winding.

    ⇒ Let $q(t) = \gamma\int_0^tg\,dt'$ and write $M = m\,e^{-iq(t)x}$.

    ⇒ Insert this into the Bloch–Torrey equation. The time derivative of the phase factor produces $-i\gamma gx\,m$, which cancels the gradient term exactly, and the Laplacian picks up the factor: $\partial_x(m\,e^{-iqx}) = e^{-iqx}(\partial_x - iq)\,m$. So

    $$\frac{\partial m}{\partial t} = D\,(\partial_x - iq(t))^2\,m .$$

    ⇒ This equation has no $x$ in it, only derivatives. For a periodic medium and a uniform start ($m = 1$), $m$ stays **periodic**. One cell with ordinary periodic boundaries is now enough.

    ⇒ On the grid, $(\partial_x - iq)^2$ becomes the three-point stencil with phase factors on the neighbours: $\frac D{h^2}\big(e^{-iqh}m_{j+1} - 2m_j + e^{iqh}m_{j-1}\big)$. Each face carries its conductance times a phase factor that changes with $q(t)$.

    ⇒ At the echo $q(T) = 0$, so $E = \frac1V\int M = \frac1V\int m$.

    → **The same trick without a change of variables.** Keep $M$ itself on the grid, with the ordinary stencil and the gradient term of ③. Since $m$ is periodic and $M = m\,e^{-iqx}$, $M$ is periodic up to a phase: $M(x + L) = e^{-iq(t)L}\,M(x)$. So only the face that wraps from the last cell back to the first needs a phase factor, $e^{\mp iq(t)L}$. Across one period the gradient has added the phase $\gamma L\int_0^tg\,dt'$, and the boundary must know it. This is the **revised periodic boundary condition** of [Xu, Does & Gore (2007)](https://doi.org/10.1088/0031-9155/52/7/N01), used in MATI. A plain periodic boundary, $M_{n+1} = M_1$, ignores the phase: it puts a false jump in $M$ at the wrap face, and an error in the signal. Both bookkeepings give the same signal at the echo, where $q = 0$. Finite elements use the same idea under the name pseudo-periodic conditions ([Nguyen et al. 2014](https://doi.org/10.1016/j.jcp.2014.01.009)).

    **Check: tortuosity.** One cell of width $L = 8$ µm with a membrane on its face is an infinite stack of membranes. At long $\Delta$ and small $b$, the signal must decay as $e^{-bD_\infty}$, with $1/D_\infty = 1/D + 1/(\kappa L)$ from Chapter 3. The figure extracts $D_{\text{app}} = -\ln E/b$ from the single-cell simulation (narrow pulses: $\delta = 2$ ms, $G = 10$ mT/m).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    per_kappa = mo.ui.slider(-2, 1, step=0.1, value=-1, label="log₁₀ κ (κ in µm/ms)", show_value=True)
    per_kappa
    return (per_kappa,)


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D0, GAMMA, mo, np, per_kappa, periodic_signal, plt):
    _L, _G, _de = 8.0, 10.0, 2.0
    _kap = 10**per_kappa.value
    _Dinf = 1 / (1 / D0 + 1 / (_kap * _L))
    _Des = np.array([5, 10, 20, 50, 100, 200, 500, 1000.0])
    _Dapp = []
    for _De in _Des:
        _b = (GAMMA * _G * _de) ** 2 * (_De - _de / 3)
        _E = periodic_signal(_G, _L, _de, _De, n=40, dt=0.05, kappa=_kap).real
        _Dapp.append(-np.log(_E) / _b)
    _Dapp = np.array(_Dapp)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.7))
    _a1.semilogx(_Des, _Dapp, "o-", color=C_MAIN, label="one periodic cell (finite differences)")
    _a1.axhline(D0, color=C_FREE, lw=1.2, ls="--", label="D (no membranes)")
    _a1.axhline(_Dinf, color=C_ALT, lw=1.4, ls="--", label=f"D∞ = {_Dinf:.3f} (Ch 3)")
    _a1.set(xlabel="Δ (ms)", ylabel="D_app = −ln E / b  (µm²/ms)", ylim=(0, 2.1), title=f"Stack of membranes, L = {_L:g} µm, κ = {_kap:.3g} µm/ms")
    _a1.legend(frameon=False, fontsize=8)
    _a2.loglog(_Des, np.abs(_Dapp - _Dinf), "o-", color=C_MAIN, label="|D_app − D∞|")
    _a2.loglog(_Des, np.abs(_Dapp[-1] - _Dinf) * _Des[-1] / _Des, ":", color="k", lw=1, label="∝ 1/Δ")
    _a2.set(xlabel="Δ (ms)", ylabel="µm²/ms", title="How D_app approaches D∞")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"At Δ = 1000 ms: D_app = **{_Dapp[-1]:.4f}** against D∞ = {_Dinf:.4f} µm²/ms. κL/D = {_kap * _L / D0:.3g}.")])
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive $\partial_tm = D(\partial_x - iq)^2m$ from the Bloch–Torrey equation with $M = m\,e^{-iq(t)x}$. Then show that the stencil $\frac D{h^2}(e^{-iqh}m_{j+1} - 2m_j + e^{iqh}m_{j-1})$ gives zero for $m = e^{iqx}$, i.e. for a uniform $M$.

    **(b)** Without membranes, what must the single-cell simulation give? Explain why a single cell then reproduces free diffusion exactly, whatever its width.

    **(c)** In the right panel, $D_{\text{app}}$ approaches $D_\infty$ as $1/\Delta$. Chapter 3 said that ordered (periodic) structures approach as $1/t$. Explain the $1/\Delta$ from $\langle X^2\rangle = 2D_\infty t + c$.

    **(d)** Make the membranes very permeable ($\kappa = 10$ µm/ms) and then nearly closed ($\kappa = 0.01$). How long must $\Delta$ be in each case before $D_{\text{app}}$ is within 5% of $D_\infty$? Relate it to the exchange time between cells.
    """, "5 · A periodic cell")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\partial_tM = (\partial_tm - i\dot q\,x\,m)e^{-iqx}$ with $\dot q = \gamma g$. And $\partial_x^2(me^{-iqx}) = e^{-iqx}(\partial_x - iq)^2m$. So the Bloch–Torrey equation reads $(\partial_tm - i\gamma gxm)e^{-iqx} = De^{-iqx}(\partial_x - iq)^2m - i\gamma gx\,me^{-iqx}$: the $x$ terms cancel. ✓

    ⇒ For $m_j = e^{iqx_j}$: $e^{-iqh}e^{iq(x_j+h)} - 2e^{iqx_j} + e^{iqh}e^{iq(x_j-h)} = 0$. ✓ A uniform magnetization does not diffuse, as it should.

    **(b)** ⇒ Without membranes the periodic medium is free space, so $E = e^{-bD}$. The single cell has no walls; with $m$ uniform at the start, the stencil keeps it uniform in space, and $m$ just decays through the phase factors: $m(t) = \exp(-D\int_0^tq^2dt')$, exactly $e^{-bD}$ at the echo. The cell width does not matter. The simulation gives 0.995582 against $e^{-bD} = 0.995582$ for $\delta = 2$, $\Delta = 20$ ms, $G = 20$ mT/m.

    **(c)** ⇒ At long times $\langle X^2\rangle = 2D_\infty t + c$, where $c$ is a constant set by the structure of one cell. Then $D(t) = \langle X^2\rangle/2t = D_\infty + c/(2t)$: a $1/t$ approach. In the figure $(D_{\text{app}} - D_\infty)\Delta\approx2.0$ µm² for $\kappa = 0.1$ µm/ms, from 100 to 1000 ms.

    **(d)** ⇒ $\kappa = 10$: $\kappa L/D = 40$, the membranes hardly matter, $D_\infty = 1.95$, and $D_{\text{app}}$ is within 5% almost at once (it starts close to $D$ anyway). $\kappa = 0.01$: $D_\infty = 0.077$; $D_{\text{app}}$ is still above $D_\infty$ by about 5% only around $\Delta\approx 1000$ ms.

    → The approach takes several exchange times. A spin must cross several membranes before its displacement looks Gaussian. The exit rate of one cell is $k = 2\kappa/L$ (Chapter 3), so $1/k = 400$ ms for $\kappa = 0.01$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 4b · Finite elements

    Finite differences ask the PDE to hold at the cell centres. Finite elements ask something weaker, which turns out to be more useful: that the PDE holds **on average**, against any weight function. The recipe has five steps, and each removes one difficulty.

    ## ① The weak form: weighted residuals

    ⇒ Write the PDE as residual $= 0$: $\ \partial_tM - \nabla\cdot(D\nabla M) + i\gamma g\,x\,M = 0$ in the domain $\Omega$.

    ⇒ Multiply by an arbitrary weight function $w(\mathbf r)$ and integrate over $\Omega$. If the result vanishes for **every** $w$, the residual itself must vanish:

    $$\int_\Omega w\,\partial_tM\,d\mathbf r - \int_\Omega w\,\nabla\cdot(D\nabla M)\,d\mathbf r + i\gamma g\int_\Omega w\,x\,M\,d\mathbf r = 0 .$$

    ⇒ Integrate the middle term by parts (Green's first identity). The second derivatives move half onto $w$, and a boundary term appears:

    $$\int_\Omega w\,\partial_tM + \int_\Omega D\,\nabla w\cdot\nabla M + i\gamma g\int_\Omega w\,x\,M = \oint_{\partial\Omega}w\,D\,\partial_nM\,dS .$$

    This is the **weak form**. It only needs first derivatives of $M$, so $M$ may be built from pieces with kinks.

    ⇒ **The boundary term holds the wall conditions.** Look at what $D\,\partial_nM$ is on each kind of boundary:

    | Boundary | Condition | Boundary term $\oint wD\,\partial_nM$ |
    | --- | --- | --- |
    | impermeable wall | $D\,\partial_nM = 0$ | $0$: **nothing to code** |
    | relaxing wall | $D\,\partial_nM = -\rho M$ | $-\rho\oint w\,M\,dS$: a boundary "mass" term |
    | membrane between $\Omega_1$ and $\Omega_2$ | flux continuous $= \kappa\times$ jump | $-\kappa\oint(M_1 - M_2)(w_1 - w_2)\,dS$, which couples the two sides (equal densities; Exercise 6 for the general case) |
    | fixed value ($M = M_0$) | — | must be imposed on the unknowns directly |

    → The first three are **natural** boundary conditions: they enter the weak form through the boundary integral, and are satisfied automatically by its solution. A fixed value is an **essential** condition, imposed by hand. Diffusion MRI needs only natural ones, which is why reflecting walls cost nothing in finite elements, while finite differences need a rule at every wall face.

    ## ② Galerkin: hat functions as unknowns and as weights

    ⇒ Cut $\Omega$ into elements (intervals in 1D, triangles in 2D, tetrahedra in 3D) with nodes at their corners.

    ⇒ Give each node $j$ a **hat function** $N_j$: 1 at node $j$, 0 at every other node ($N_j(\mathbf r_i) = \delta_{ij}$), linear on each element. Write $M(\mathbf r, t)\approx\sum_jM_j(t)\,N_j(\mathbf r)$: the unknowns are the nodal values.

    ⇒ Use the same hats as weights, $w = N_i$ for each $i$ (**Galerkin**'s choice, 1915). Each weight gives one equation, so there are as many equations as unknowns.

    ⇒ In one element of length $h$ with local coordinate $\xi\in[-1, 1]$, $x = \frac{x_a + x_b}2 + \frac h2\xi$, the two hats are $N_1 = \frac12(1-\xi)$ and $N_2 = \frac12(1+\xi)$. Three nodes per element give quadratic shapes, $\frac12\xi(\xi-1)$, $(1-\xi)(1+\xi)$ and $\frac12\xi(\xi+1)$.

    ## ③ Element matrices and assembly

    ⇒ Insert $M = \sum_jM_jN_j$ and $w = N_i$ into the weak form. Every integral becomes a sum over elements of small matrices:

    $$\boldsymbol{\mathcal M}\frac{d\mathbf M}{dt} = -\big(\mathbf K + i\gamma g(t)\,\mathbf J\big)\mathbf M,\qquad \mathcal M_{ij} = \int N_iN_j,\quad K_{ij} = \int D\,\nabla N_i\cdot\nabla N_j,\quad J_{ij} = \int N_i\,x\,N_j .$$

    For a 1D element $[x_a, x_b]$ of length $h$, all three integrals are exact:

    $$\boldsymbol{\mathcal M}^e = \frac h6\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad \mathbf K^e = \frac Dh\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad \mathbf J^e = \frac h{12}\begin{pmatrix}3x_a + x_b & x_a + x_b\\ x_a + x_b & x_a + 3x_b\end{pmatrix}.$$

    ⇒ **Assembly.** Each element adds its $2\times2$ matrix into the rows and columns of its two nodes. Neighbouring elements share a node, so their contributions overlap there. The result is a sparse matrix that couples each node only to its neighbours.

    → For linear triangles in 2D the same integrals have closed forms: $\mathbf K^e = D(\mathbf b\mathbf b^{\mathsf T} + \mathbf c\mathbf c^{\mathsf T})/(4A)$, with $b_i$, $c_i$ differences of the corner coordinates and $A$ the area; $\mathcal M^e_{ij} = \frac A{12}(1 + \delta_{ij})$; and $\int N_1^aN_2^bN_3^c\,dA = 2A\,\frac{a!\,b!\,c!}{(a+b+c+2)!}$ for $\mathbf J^e$. For curved or higher-order elements the integrals are mapped to a reference element (isoparametric mapping) and done by Gauss quadrature.

    ## ④ The full system: compartments, relaxation and membranes

    With several compartments, each gets its own set of hat functions, living only on the elements of that compartment. A node on an interface $\Gamma_{nm}$ is therefore **doubled**: one copy belongs to $\Omega_n$, the other to $\Omega_m$, so that $M$ can jump across the membrane (⑥ shows it in 1D). Put all nodal values in one vector $\boldsymbol\xi$ and write $\varphi_k$ for the hat function of node $k$. The weak form of the problem of 4.0 then gives

    $$\boldsymbol{\mathcal M}\,\frac{d\boldsymbol\xi}{dt} = -\big(\mathbf S + \mathbf R + \mathbf Q + i\gamma f(t)\,\mathbf J(\mathbf g)\big)\,\boldsymbol\xi,\qquad \xi_k(0) = \rho_n\ \text{ for every node } k \text{ of } \Omega_n .$$

    | Matrix | Entries | Comes from |
    | --- | --- | --- |
    | $\boldsymbol{\mathcal M}$, mass | $\int\varphi_k\varphi_l$ | the time derivative |
    | $\mathbf S$, stiffness | $\int D_n\nabla\varphi_k\cdot\nabla\varphi_l$ | diffusion |
    | $\mathbf R$, relaxation | $\frac1{T_{2,n}}\int_{\Omega_n}\varphi_k\varphi_l$: each compartment's mass matrix divided by its $T_2$ | relaxation |
    | $\mathbf J(\mathbf g)$, gradient | $\int\varphi_k\,(\mathbf g\cdot\mathbf r)\,\varphi_l$ | the gradient |
    | $\mathbf Q$, interface | $\kappa_{nm}\int_{\Gamma_{nm}}\varphi_k\varphi_l$, weighted by $c_{mn}$ or $-c_{nm}$ (Exercise 6) | the membrane term of the weak form |

    → In the 1D notation of ③, $\mathbf K = \mathbf S + \mathbf Q$, $f(t)\,\mathbf J(\mathbf g)$ is $g(t)\,\mathbf J$, and $\mathbf R = 0$.

    → All five matrices are sparse and are assembled once. SpinDoctor ([Li et al. 2019](https://doi.org/10.1016/j.neuroimage.2019.116120)) builds them on a tetrahedral mesh and hands the ODE system to an adaptive solver for stiff problems (MATLAB's `ode23t`, a trapezoidal rule with error control), which picks its own time steps. The signal is $S = \sum_kw_k\xi_k$ with $w_k = \int\varphi_k$.

    ## ⑤ Time stepping and the signal

    Crank–Nicolson with the mass matrix: $\ (\boldsymbol{\mathcal M} + \frac{\delta t}2\mathbf A_g)\,\mathbf M^+ = (\boldsymbol{\mathcal M} - \frac{\delta t}2\mathbf A_g)\,\mathbf M$, with $\mathbf A_g = \mathbf K + i\gamma g\,\mathbf J$. The signal is the integral of $M$: $\ E = \frac1V\sum_iw_iM_i$ with $w_i = \int N_i$, the row sums of $\boldsymbol{\mathcal M}$.

    → Replacing $\boldsymbol{\mathcal M}$ by the diagonal matrix of row sums ("mass lumping") turns linear finite elements on a uniform 1D grid into the finite-difference scheme of 4a. The two methods are close relatives; they part ways in how they handle geometry.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_GREEN, C_MAIN, fem_ops, mo, np, plt):
    _xn = np.array([-4.0, -2.6, -1.5, -0.6, 0.0, 0.9, 2.2, 4.0])          # a non-uniform 1D mesh
    _xf = np.linspace(-4, 4, 800)
    _Mtrue = np.cos(0.7 * _xf) * np.exp(-0.05 * _xf**2)
    _Mn = np.cos(0.7 * _xn) * np.exp(-0.05 * _xn**2)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for _j in range(len(_xn)):
        _e = np.zeros(len(_xn))
        _e[_j] = 1.0
        _Nj = np.interp(_xf, _xn, _e)
        _a1.plot(_xf, _Nj * _Mn[_j], color=C_FREE, lw=0.9)
        if _j == 3:
            _a1.fill_between(_xf, 0, _Nj, color=C_MAIN, alpha=0.15, lw=0)
            _a1.plot(_xf, _Nj, color=C_MAIN, lw=1.5, label="one hat function N₃")
    _a1.plot(_xf, _Mtrue, color="k", lw=1.0, label="M(x)")
    _a1.plot(_xf, np.interp(_xf, _xn, _Mn), color=C_GREEN, lw=2.0, label="Σ Mⱼ Nⱼ (grey: the terms)")
    _a1.plot(_xn, _Mn, "o", color=C_GREEN, ms=5)
    _a1.plot(_xn, np.full(len(_xn), -1.12), "|", color="k", ms=10)
    _a1.set(xlabel="x (µm)", ylim=(-1.2, 1.25), title="Hat functions on a non-uniform mesh")
    _a1.legend(frameon=False, fontsize=8, loc="lower right")

    _Mm, _K, _J, _w, _nodes = fem_ops(8.0, 8, kappa=0.1)
    _a2.imshow(np.where(np.abs(_K) > 1e-12, 1.0, np.nan), cmap="Blues", vmin=0, vmax=1.3)
    _n = _K.shape[0]
    _a2.set_xticks(range(_n))
    _a2.set_yticks(range(_n))
    _a2.set_xticklabels([f"{v:+.0f}" if i < _n - 1 else "0′" for i, v in enumerate(_nodes)], fontsize=7)
    _a2.set_yticklabels([f"{v:+.0f}" if i < _n - 1 else "0′" for i, v in enumerate(_nodes)], fontsize=7)
    _a2.grid(False)
    _a2.set(title="Nonzeros of K: 8 elements, membrane at x = 0", xlabel="node (its x in µm; 0′ = the second node at x = 0)")
    _a2.annotate("membrane couples\n0 and 0′", (_n - 1, _n // 2 - 0.5), xytext=(1.0, _n - 1.6), fontsize=7.5, color=C_ALT, arrowprops=dict(arrowstyle="->", color=C_ALT))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive the weak form in 1D on $[a, b]$, and show that the boundary term $[w\,D\,\partial_xM]_a^b$ vanishes for reflecting walls whatever $w$ is.

    **(b)** For a membrane at $x = 0$ between $[a, 0]$ and $[0, b]$, with $M$ and $w$ allowed to jump there, collect the two boundary terms at $x = 0$ and show that they combine into $-\kappa(M_1 - M_2)(w_1 - w_2)$, where 1 and 2 label the two sides.

    **(c)** Why can the weak form use hat functions, which have kinks, while the strong form $\partial_tM = D\partial_x^2M + \dots$ cannot?

    **(d)** *Unequal spin densities.* Redo (b) with the general interface condition of 4.0, $D_n\partial_{\nu_n}M_n = \kappa(c_{nm}M_m - c_{mn}M_n)$, for compartment $n$ on the left and $m$ on the right. Show that the membrane adds $\kappa\begin{pmatrix}c_{mn}&-c_{nm}\\-c_{mn}&c_{nm}\end{pmatrix}$ to the rows and columns of the two nodes at $x = 0$. Check that its columns sum to zero, and that $(\rho_n, \rho_m)$ is in its null space when $c_{nm} = 2\rho_n/(\rho_n + \rho_m)$. What does each property guarantee?
    """, "6 · The weak form")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (b) On side 1 (left of the membrane) the outward normal at $x = 0$ points in $+x$; on side 2 it points in $-x$. The flux from 1 to 2 is $-D\partial_xM$, the same on both sides, and equals $\kappa(M_1 - M_2)$.
    - (c) What is the second derivative of a hat function at its peak?
    - (d) Side $n$ contributes $w_n\,D_n\partial_{\nu_n}M_n = w_n\kappa(c_{nm}M_m - c_{mn}M_n)$ to the right-hand side, side $m$ the same with $n$ and $m$ swapped. Move both to the left.
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\int_a^bw\,\partial_tM\,dx - \int_a^bw\,\partial_x(D\partial_xM)\,dx + i\gamma g\int_a^bw\,x\,M\,dx = 0$.

    ⇒ By parts: $\int w\,\partial_x(D\partial_xM) = [wD\partial_xM]_a^b - \int D\,\partial_xw\,\partial_xM$. So

    $$\int_a^bw\,\partial_tM + \int_a^bD\,\partial_xw\,\partial_xM + i\gamma g\int_a^bw\,x\,M = [w\,D\,\partial_xM]_a^b .$$

    ⇒ Reflecting walls: $D\partial_xM = 0$ at $a$ and at $b$, so the right side is zero for every $w$. ✓ Nothing has to be imposed: the condition is natural.

    **(b)** ⇒ Side 1 contributes $+w_1D\partial_xM\big|_{0^-}$, side 2 contributes $-w_2D\partial_xM\big|_{0^+}$. Flux continuity: $-D\partial_xM$ is the same on both sides and equals $\kappa(M_1 - M_2)$.

    ⇒ Sum: $w_1\cdot(-\kappa(M_1 - M_2)) - w_2\cdot(-\kappa(M_1 - M_2)) = -\kappa(M_1 - M_2)(w_1 - w_2)$. ✓ Moved to the left-hand side it is $+\kappa(M_1 - M_2)(w_1 - w_2)$, which in 1D adds $\kappa\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ to $\mathbf K$ between the two nodes at $x = 0$ (the extra entries in the right panel).

    → This is why finite-element codes use **double nodes** at membranes ([Nguyen et al. 2014](https://doi.org/10.1016/j.jcp.2014.01.009)): two nodes at the same point, one for each side, so that $M$ can jump.

    **(c)** ⇒ The second derivative of a hat is zero inside each element and a delta function at its nodes: the strong form is not defined for it. The weak form only needs first derivatives, which for a hat are piecewise constant, and their products integrate without trouble.

    **(d)** ⇒ Moved to the left-hand side: row $n$ gets $\kappa(c_{mn}M_n - c_{nm}M_m)$, row $m$ gets $\kappa(c_{nm}M_m - c_{mn}M_n)$. As a matrix acting on $(M_n, M_m)$: $\kappa\begin{pmatrix}c_{mn}&-c_{nm}\\-c_{mn}&c_{nm}\end{pmatrix}$. ✓ It is not symmetric unless $\rho_n = \rho_m$.

    ⇒ Columns: $c_{mn} - c_{mn} = 0$ and $-c_{nm} + c_{nm} = 0$. ✓ → The membrane moves magnetization from one side to the other without creating or destroying any: the total $\int M$ is conserved.

    ⇒ Null space: $c_{mn}\rho_n - c_{nm}\rho_m = \frac{2\rho_m\rho_n - 2\rho_n\rho_m}{\rho_n + \rho_m} = 0$. ✓ → Without a gradient, the equilibrium state with density $\rho_n$ on one side and $\rho_m$ on the other stays put: unequal densities alone drive no flux. With $c = 1$ the membrane would wrongly push the two densities towards each other.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Derive $\boldsymbol{\mathcal M}^e$, $\mathbf K^e$ and $\mathbf J^e$ for a 1D element $[x_a, x_b]$ from $N_1 = (x_b - x)/h$ and $N_2 = (x - x_a)/h$.

    **(b)** Assemble $\mathbf K$ for three equal elements on $[0, 3h]$ (four nodes). Compare its rows with the finite-difference matrix of 4a. Where do they differ, and why?

    **(c)** Show that $w_i = \int N_i$ is the sum of row $i$ of $\boldsymbol{\mathcal M}$, and that $\sum_iw_i = L$. Why does $E = \frac1L\sum_iw_iM_i$ give exactly 1 at $t = 0$?
    """, "7 · Element matrices")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\int_{x_a}^{x_b}N_1^2 = h/3$ and $\int N_1N_2 = h/6$, so $\boldsymbol{\mathcal M}^e = \frac h6\begin{pmatrix}2&1\\1&2\end{pmatrix}$.

    ⇒ $N_1' = -1/h$, $N_2' = 1/h$, constant: $K^e_{ij} = D\,h\,N_i'N_j'$, i.e. $\frac Dh\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$.

    ⇒ $J^e_{11} = \int N_1^2x = \frac h{12}(3x_a + x_b)$, $J^e_{12} = \int N_1N_2x = \frac h{12}(x_a + x_b)$, $J^e_{22} = \frac h{12}(x_a + 3x_b)$ (integrate the quadratics times a linear $x$; a check: $J^e_{11} + J^e_{12} = \int N_1x$, the first moment of $N_1$).

    **(b)** ⇒ $\mathbf K = \frac Dh\begin{pmatrix}1&-1&0&0\\-1&2&-1&0\\0&-1&2&-1\\0&0&-1&1\end{pmatrix}$.

    → The interior rows are $\frac Dh(-1, 2, -1)$, like $h\times$ the finite-difference stencil: the extra $h$ is absorbed by $\boldsymbol{\mathcal M}\approx h\mathbf I$. The end rows have 1 instead of 2: the element simply stops at the wall, which is the zero-flux condition, with no ghost cell. The nodes sit *on* the walls, while finite-difference cells sit half a cell inside.

    **(c)** ⇒ $\sum_jN_j(x) = 1$ everywhere (the hats form a partition of unity), so $\sum_j\mathcal M_{ij} = \int N_i\sum_jN_j = \int N_i = w_i$. And $\sum_iw_i = \int\sum_iN_i = L$.

    ⇒ At $t = 0$, $M_i = 1$ for all nodes, so $E = \frac1L\sum_iw_i = 1$ exactly. ✓
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ⑥ Membranes, and the check against Chapter 1

    Chapter 1 (Exercise 9) found the exchange mode of a slab $[-a, a]$ with a membrane in the middle: its rate $Dk^2$ follows from $k\tan(ka) = 2\kappa/D$. Both discrete operators of this chapter should reproduce it: the finite-difference one through the membrane conductance of 4a ②, the finite-element one through the double node.

    ## ⑦ Modes on the mesh: the matrix method for any shape

    Setting $g = 0$ and looking for $\mathbf M(t) = \mathbf u\,e^{-\lambda t}$ turns the semi-discrete equation into a **generalised eigenproblem**,

    $$\mathbf K\mathbf u = \lambda\,\boldsymbol{\mathcal M}\,\mathbf u ,$$

    whose solutions are the eigenmodes of the meshed shape. With eigenvectors normalised so that $\mathbf U^{\mathsf T}\boldsymbol{\mathcal M}\mathbf U = \mathbf I$, the gradient matrix of Chapter 2 is simply $\mathbf B = \mathbf U^{\mathsf T}\mathbf J\mathbf U$, and the whole of Chapter 2 runs unchanged. This is the matrix formalism of SpinDoctor ([Li, Tran & Nguyen 2020](https://doi.org/10.1002/nbm.4353)), extended to permeable membranes by [Agdestein, Tran & Li (2022)](https://doi.org/10.1002/nbm.4646). ReMiDi ([Khole et al. 2025](https://arxiv.org/abs/2502.01988)) makes it differentiable, so that the shape itself can be fitted to a signal.

    → Chapters 2 and 4 meet here: the mesh computes the modes once, the matrix method uses a few of them for every sequence.
    """)
    return


@app.cell(hide_code=True)
def _(D0, GAMMA, expm, fd_ops, fem_ops, gen_eig, mo, np, pgse_mcf):
    _L, _a = 8.0, 4.0
    _rows = []
    for _kap in (0.01, 0.1, 1.0):
        _lo, _hi = 1e-9, np.pi / (2 * _a) - 1e-12
        for _ in range(200):
            _m = 0.5 * (_lo + _hi)
            if _m * np.tan(_m * _a) > 2 * _kap / D0:
                _hi = _m
            else:
                _lo = _m
        _ex = D0 * _m**2
        _vals = []
        for _n in (20, 80):
            _A, _x, _h = fd_ops(_L, _n, kappa=_kap)
            _vals.append(np.sort(np.linalg.eigvalsh(_A))[1])
            _Mm, _K, _J, _w, _nd = fem_ops(_L, _n, kappa=_kap)
            _vals.append(gen_eig(_K, _Mm)[0][1])
        _rows.append(f"| {_kap:g} | {_ex:.6f} | {4 * _kap / _L:.4f} | {_vals[0]:.6f} | {_vals[1]:.6f} | {_vals[2]:.6f} | {_vals[3]:.6f} |")

    # matrix method on finite-element modes (slab, 100 elements)
    _Mm, _K, _J, _w, _nd = fem_ops(_L, 100)
    _lam, _U = gen_eig(_K, _Mm)
    _ex171 = pgse_mcf(300.0, _L, 10.0, 20.0).real
    _mf = []
    for _N in (5, 10, 20, 40):
        _Lam = np.diag(_lam[:_N])
        _B = _U[:, :_N].T @ _J @ _U[:, :_N]
        _c0 = _U[:, :_N].T @ (_Mm @ np.ones(len(_w)))                    # initial M = 1 in the mode basis
        _c = expm(-(_Lam - 1j * GAMMA * 300.0 * _B) * 10.0) @ expm(-_Lam * 10.0) @ expm(-(_Lam + 1j * GAMMA * 300.0 * _B) * 10.0) @ _c0
        _E = (_w @ (_U[:, :_N] @ _c)) / _L
        _mf.append(f"| {_N} | {_E.real:.6f} | {abs(_E.real - _ex171):.1e} |")
    mo.md(
        "**Exchange-mode rate Dλ (1/ms)** of an 8 µm slab with a membrane in the middle:\n\n"
        "| κ (µm/ms) | exact (Ch 1) | 4κ/L | FD, 20 cells | FEM, 20 elements | FD, 80 cells | FEM, 80 elements |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(_rows)
        + "\n\nFinite differences converge from below, finite elements from above. The simple estimate 4κ/L holds only for κL/D ≪ 1.\n\n"
        + f"**Matrix method on finite-element modes** (100 elements, running example; exact E = {_ex171:.6f}):\n\n"
        "| modes N | E | error |\n| --- | --- | --- |\n" + "\n".join(_mf)
        + "\n\nFrom 10 mesh modes on, the signal matches the exact one as well as the mesh itself allows: the remaining 1.6 × 10⁻⁵ is the error of 100 linear elements, the same as for the direct finite-element run."
    )
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** Use the table. For $\kappa = 1$ µm/ms, how large is the relative error of each method with 20 and with 80 pieces? What order does that suggest?

    **(b)** Why does the finite-element eigenvalue approach from **above**? (Think of the eigenvalue as the minimum of a Rayleigh quotient $\int D\lvert\nabla u\rvert^2\big/\int u^2$ over all allowed $u$.)

    **(c)** Why does $\mathbf U^{\mathsf T}\boldsymbol{\mathcal M}\mathbf U = \mathbf I$ play the role of the orthonormality $\int u_nu_m = \delta_{nm}$ of Chapter 1? And why is $\mathbf B = \mathbf U^{\mathsf T}\mathbf J\mathbf U$ the discrete version of $B_{nm} = \int u_n\,x\,u_m$?
    """, "8 · Membranes and modes on the mesh")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Exact 0.199899. FD: 0.199534 and 0.199876, relative errors $1.8\times10^{-3}$ and $1.2\times10^{-4}$. FEM: 0.200166 and 0.199916, errors $1.3\times10^{-3}$ and $8\times10^{-5}$. Four times more pieces divide the error by about 16: second order in $h$ for both.

    **(b)** ⇒ The exact $\lambda_1$ is the minimum of the Rayleigh quotient over all functions orthogonal to the uniform mode. Galerkin finite elements minimise the same quotient, but only over piecewise-linear functions, a smaller set. A minimum over a smaller set cannot be lower: $\lambda_h\ge\lambda$. (Finite differences have no such variational property, and here they happen to approach from below.)

    **(c)** ⇒ For $u = \sum_iU_{in}N_i$, $\int u_nu_m = \sum_{ij}U_{in}U_{jm}\int N_iN_j = (\mathbf U^{\mathsf T}\boldsymbol{\mathcal M}\mathbf U)_{nm}$. So $\mathbf U^{\mathsf T}\boldsymbol{\mathcal M}\mathbf U = \mathbf I$ says the mesh modes are orthonormal as functions. ✓ In the same way, $\int u_n\,x\,u_m = \sum_{ij}U_{in}U_{jm}\int N_i\,x\,N_j = (\mathbf U^{\mathsf T}\mathbf J\mathbf U)_{nm}$. ✓
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 4c · Curved walls: a staircase or a fitted mesh

    In 1D every wall sits on a face or a node, and finite differences and finite elements are equally good (4d shows it). The difference appears with curved walls. A Cartesian grid can only follow a circle as a **staircase** of cell faces. A triangle mesh puts its boundary nodes **on** the circle, so its boundary is a polygon whose corners lie on the wall.

    The test: the slowest non-uniform mode of a disk of radius $R$ (the cross-section of a cylinder), whose exact eigenvalue is $\lambda_1R^2 = 1.8412^2 = 3.3900$ (Chapter 1). It sets the slowest decay inside an axon, so every restricted-diffusion signal depends on it.
    """)
    return


@app.cell(hide_code=True)
def _(disk_mesh, fd_disk, fem2d, gen_eig, np):
    # λ₁ of a unit disk for several resolutions, both methods (computed once)
    cw_res = np.array([4, 6, 8, 11, 16])
    cw_exact = 1.8411837813**2
    cw_fd, cw_fem, cw_nfd, cw_nfem = [], [], [], []
    for _nh in cw_res:
        _A, _X, _Y = fd_disk(1.0, 1.0 / _nh)
        cw_fd.append(np.sort(np.linalg.eigvalsh(_A))[1])
        cw_nfd.append(len(_X))
        _pts, _tris = disk_mesh(1.0, int(_nh))
        _Mm, _K, _J = fem2d(_pts, _tris)
        cw_fem.append(gen_eig(_K, _Mm)[0][1])
        cw_nfem.append(len(_pts))
    cw_fd, cw_fem = np.array(cw_fd), np.array(cw_fem)
    return cw_exact, cw_fd, cw_fem, cw_nfd, cw_nfem, cw_res


@app.cell(hide_code=True)
def _(mo):
    cw_n = mo.ui.slider(3, 16, step=1, value=6, label="Resolution R/h", show_value=True)
    cw_n
    return (cw_n,)


@app.cell(hide_code=True)
def _(
    C_ALT,
    C_GREEN,
    C_MAIN,
    cw_exact,
    cw_fd,
    cw_fem,
    cw_n,
    cw_nfd,
    cw_nfem,
    cw_res,
    disk_mesh,
    mo,
    np,
    plt,
):
    from matplotlib.patches import Rectangle

    _nh = cw_n.value
    _h = 1.0 / _nh
    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    _th = np.linspace(0, 2 * np.pi, 400)
    _m = int(np.ceil(_nh))
    _cc = (np.arange(-_m, _m) + 0.5) * _h
    for _x in _cc:
        for _y in _cc:
            if _x**2 + _y**2 < 1:
                _a1.add_patch(Rectangle((_x - _h / 2, _y - _h / 2), _h, _h, fc=C_ALT, ec="w", lw=0.6, alpha=0.55))
    _a1.plot(np.cos(_th), np.sin(_th), color="k", lw=1.4)
    _a1.set(title=f"Grid: cells whose centre is inside ({_h:.3f} R)", aspect="equal", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    _a1.axis("off")
    _pts, _tris = disk_mesh(1.0, _nh)
    _a2.triplot(_pts[:, 0], _pts[:, 1], _tris, color=C_GREEN, lw=0.6)
    _a2.plot(np.cos(_th), np.sin(_th), color="k", lw=1.4)
    _a2.set(title=f"Mesh: {len(_pts)} nodes, boundary nodes on the circle", aspect="equal", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    _a2.axis("off")
    _a3.loglog(cw_res, np.abs(cw_fd / cw_exact - 1), "o-", color=C_ALT, label="grid (staircase)")
    _a3.loglog(cw_res, np.abs(cw_fem / cw_exact - 1), "s-", color=C_GREEN, label="mesh (linear triangles)")
    _a3.loglog(cw_res, 0.4 / cw_res, ":", color=C_ALT, lw=1, label="∝ h")
    _a3.loglog(cw_res, 0.4 / cw_res**2, ":", color=C_GREEN, lw=1, label="∝ h²")
    _a3.set(xlabel="R / h", ylabel="relative error of λ₁", title="Slowest mode of a disk", xticks=[4, 6, 8, 11, 16], xticklabels=["4", "6", "8", "11", "16"])
    _a3.minorticks_off()
    _a3.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    _tab = "\n".join(
        f"| {r} | {nf} | {lf:.4f} ({100 * (lf / cw_exact - 1):+.1f}%) | {nm} | {lm:.4f} ({100 * (lm / cw_exact - 1):+.2f}%) |"
        for r, nf, lf, nm, lm in zip(cw_res, cw_nfd, cw_fd, cw_nfem, cw_fem)
    )
    mo.vstack([_fig, mo.md(
        "λ₁R², exact 3.3900:\n\n| R/h | grid cells | grid | mesh nodes | mesh |\n| --- | --- | --- | --- | --- |\n" + _tab
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the result.** With about the same number of unknowns, the mesh is 4 to 16 times more accurate, and the gap grows with resolution. Its error falls as $h^2$, steadily. The grid's error falls only as $h$ on average, and **erratically**: the staircase changes shape in jumps as $h$ shrinks, and a slightly finer grid can even be worse (compare $R/h = 6$ and 8).

    → A grid misplaces the wall by up to half a cell, an $O(h)$ error in the shape, so first-order convergence is the best it can do near curved walls. Finite-difference codes reduce this with finer voxels near the walls or with "cut-cell" corrections. Finite elements avoid it by construction, at the price of generating the mesh.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** The staircase that approximates a circle of radius $R$ is made of horizontal and vertical steps. Show that its perimeter tends to $8R$ as $h\to0$, not to $2\pi R$. What surface-to-volume ratio does a finely voxelised disk therefore have?

    **(b)** Why does the eigenvalue still converge (slowly) when the perimeter does not?

    **(c)** The short-time diffusivity of Chapter 3 depends on $S/V$. What would you check before using a voxel substrate to study short diffusion times?
    """, "9 · The staircase")
    return


@app.cell(hide_code=True)
def _(hint):
    hint(r"""
    - (a) Follow the staircase over one quarter of the circle: how far does it go horizontally in total, and how far vertically?
    - (b) The eigenvalue depends on the volume and on where the boundary is, not on how long it is. How far is the staircase from the circle?
    """)
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ Over a quarter circle the staircase moves $R$ horizontally and $R$ vertically in total, however small the steps: length $2R$ per quarter, $8R$ in all. The ratio to the true perimeter is $8R/2\pi R = 4/\pi = 1.27$.

    ⇒ The area converges to $\pi R^2$, so the voxelised disk has $S/V\to8/(\pi R)$ instead of $2/R$: 27% too large, at every resolution. (In 3D a voxelised sphere has a surface area 1.5 times too large, for the same reason.)

    **(b)** ⇒ The staircase is never further than about $h$ from the circle. The eigenvalue depends on the shape through the region it encloses, which converges to the disk as $h\to0$. The length of a curve is not a continuous function of the curve: a zig-zag can be arbitrarily close to a line and still be longer.

    **(c)** ⇒ At diffusion lengths comparable to the voxel size, spins can feel the steps, and quantities that depend on the wall area, such as the Mitra short-time slope, can inherit the wrong $S/V$. Check against the closed form of Chapter 3 for a shape with known $S/V$ (a cylinder or a sphere), and refine the voxels near the wall until the short-time slope is right. At diffusion lengths much larger than the steps, the walls are effectively smooth again.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 4d · In practice: accuracy, cost and checks

    ## Convergence in 1D

    Both methods solve the running example on the slab, with Crank–Nicolson and $\delta t = 0.02$ ms. The reference is the exact MCF signal of Chapter 2.
    """)
    return


@app.cell(hide_code=True)
def _(C_ALT, C_GREEN, fd_signal, fem_signal, mo, np, pgse_mcf, plt):
    _ex = pgse_mcf(300.0, 8.0, 10.0, 20.0).real
    _ns = np.array([12, 25, 50, 100, 200])
    _efd = np.array([abs(fd_signal(300.0, 8.0, 10.0, 20.0, n=int(k)).real - _ex) for k in _ns])
    _efem = np.array([abs(fem_signal(300.0, 8.0, 10.0, 20.0, ne=int(k)).real - _ex) for k in _ns])
    _fig, _a = plt.subplots(figsize=(6.4, 3.8))
    _a.loglog(8.0 / _ns, _efd, "o-", color=C_ALT, label="finite differences")
    _a.loglog(8.0 / _ns, _efem, "s-", color=C_GREEN, label="finite elements")
    _a.loglog(8.0 / _ns, _efd[2] * (_ns[2] / _ns) ** 2, ":", color="k", lw=1, label="∝ h²")
    _a.set(xlabel="h (µm)", ylabel="|E − E_exact|", title="Running example: error against cell size")
    _a.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"Exact E = {_ex:.6f}. With 200 pieces: finite differences {fd_signal(300.0, 8.0, 10.0, 20.0, n=200).real:.6f}, finite elements {fem_signal(300.0, 8.0, 10.0, 20.0, ne=200).real:.6f}."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both errors fall as $h^2$, four times per halving, and in 1D the two methods are equally accurate. The differences are elsewhere:

    | | Finite differences (e.g. MATI) | Finite elements (e.g. SpinDoctor) |
    | --- | --- | --- |
    | Geometry | voxel grid; curved walls become staircases | triangles or tetrahedra that follow the walls |
    | Input | a segmented image, used directly | a surface mesh, which must be filled with tetrahedra |
    | Walls and membranes | a rule for the flux through every wall face | natural in the weak form; membranes by double nodes |
    | Convergence near curved walls | $O(h)$, erratic | $O(h^2)$ for linear elements |
    | Time stepping | explicit: no solve, but $\delta t\le h^2/(2dD)$ | implicit or adaptive: large steps, one sparse solve each |
    | Cost driver | voxels × steps, trivially parallel (GPU) | meshing, then sparse factorisations; fewer unknowns per accuracy |
    | Extras | periodic cells with a phase-shifted boundary | eigenmodes on the mesh for the matrix method (4b ⑦) |

    → The equation is the same; the difference is how space and time are cut. A grid gives up geometric fidelity and takes tiny time steps, in exchange for an update so simple that a GPU can run millions of voxels at once. A mesh pays for meshing and linear solves, in exchange for walls exactly where they are and large time steps.

    ## Choosing $h$ and $\delta t$

    ① **Space.** $h$ must resolve the finest pattern in $M$: the smallest of the structure you care about, the gradient length $\ell_g = (D/\gamma g)^{1/3}$ and the phase wavelength $2\pi/q$, with at least a few cells per length, better 5–10.
    ② **Time, explicit.** $\delta t\le h^2/(2dD)$ for stability, and small enough to follow the waveform.
    ③ **Time, implicit.** $\delta t$ set by accuracy alone: Crank–Nicolson needs a few steps per change of the gradient, more for oscillating waveforms.
    ④ **Check.** Halve $h$, then $\delta t$, until $E$ changes by less than you can measure.
    ⑤ **Validate.** Before simulating tissue, reproduce a slab, a cylinder or a sphere: the exact MCF of Chapter 2, and the closed forms of Chapter 3 (form factor, motional narrowing, localisation). If two methods disagree on a slab, one of them has a bug.
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
    **(a)** For $G = 300$ mT/m, $\delta = 10$ ms and $D = 2$ µm²/ms, compute $\ell_g$ and the phase wavelength $2\pi/q$ at the end of a lobe. Which one sets $h$?

    **(b)** A 3D voxel simulation of a $20\times20\times20$ µm³ block with $h = 0.2$ µm, for the 30 ms running example. How many voxels and explicit steps? Roughly how many voxel updates, and how long on a GPU that does $10^{10}$ voxel updates per second?

    **(c)** A finite-element mesh of the same block with similar wall accuracy might need $10^5$–$10^6$ nodes and 300 Crank–Nicolson steps, each with a sparse solve. What decides which approach is faster?
    """, "10 · Planning a simulation")
    return


@app.cell(hide_code=True)
def _(answer):
    answer(r"""
    **(a)** ⇒ $\gamma G = 2.6752\times10^{-4}\times300 = 0.0803$ rad/(ms·µm). $\ell_g = (2/0.0803)^{1/3} = 2.9$ µm. $q = \gamma G\delta = 0.80$ µm⁻¹, so $2\pi/q = 7.8$ µm. $\ell_g$ is smaller: $h\lesssim\ell_g/5\approx0.6$ µm, and finer if the structure (membranes, small cells) demands it.

    **(b)** ⇒ $(20/0.2)^3 = 10^6$ voxels. $\delta t\le h^2/(6D) = 0.04/12 = 3.3$ µs, so $30/0.0033\approx9000$ steps. About $9\times10^9$ voxel updates: about one second at $10^{10}$ updates per second. Halving $h$ multiplies this by 32 (8 times more voxels, 4 times more steps).

    **(c)** ⇒ The cost of a sparse solve in 3D grows faster than linearly with the number of nodes and parallelises less well, while the explicit stencil is linear and embarrassingly parallel. A mesh wins when far fewer unknowns are needed (curved walls, local refinement), when large time steps are allowed (long sequences), or when the eigenmodes can be reused for many sequences (4b ⑥). A grid wins for very large segmented volumes on a GPU. Either way, check against Chapters 2 and 3 first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementations

    | Toolbox | Method | Notes |
    | --- | --- | --- |
    | Image-based FD ([Hwang et al. 2003](https://doi.org/10.1002/mrm.10536)) | finite differences on a segmented image | the first voxel-based Bloch–Torrey simulations of tissue |
    | Improved FD ([Xu, Does & Gore 2007](https://doi.org/10.1088/0031-9155/52/7/N01)) | the matrix-based (MFD) update $\boldsymbol\Phi\odot(\mathbf I + \mathbf S)\mathbf M$ with jump probabilities; revised periodic boundary condition | one periodic cell stands for the tissue; the basis of MATI |
    | Adaptive FD ([Li et al. 2014](https://doi.org/10.1088/0031-9155/59/2/441)) | finite volumes with adaptive (stabilised Runge–Kutta) time stepping | larger time steps than explicit Euler |
    | **MATI** ([Xu et al. 2025](https://doi.org/10.1016/j.mri.2025.110428)) | the MFD update on a GPU, with a GUI | substrates as 3D label matrices (extracellular 1, intracellular 2), standard or custom; standard or custom waveforms; fitting of the IMPULSED model by least squares, Bayesian grid search, polynomial regression or dictionary matching |
    | FEM with double nodes ([Nguyen et al. 2014](https://doi.org/10.1016/j.jcp.2014.01.009)) | linear finite elements, pseudo-periodic conditions, membranes by double nodes | the basis of SpinDoctor |
    | **SpinDoctor** ([Li et al. 2019](https://doi.org/10.1016/j.neuroimage.2019.116120)) | finite elements on tetrahedral meshes of cells and neurons; the system $\boldsymbol{\mathcal M}\dot{\boldsymbol\xi} = -(\mathbf S + \mathbf R + \mathbf Q + i\gamma f\mathbf J)\boldsymbol\xi$ solved with `ode23t` | also a matrix formalism on mesh modes ([Li, Tran & Nguyen 2020](https://doi.org/10.1002/nbm.4353)) |
    | ReMiDi ([Khole et al. 2025](https://arxiv.org/abs/2502.01988)) | differentiable matrix formalism on a mesh | fits the shape to a signal |

    Before trusting any of them on a new substrate, reproduce the slab of this chapter, or a cylinder against the closed forms of Chapter 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary of Chapter 4

    | Idea | Formula |
    | --- | --- |
    | The general problem | compartments $\Omega_n$ with $D_n$, $T_{2,n}$, $\rho_n$; outer walls reflecting; on $\Gamma_{nm}$ continuous flux $= \kappa_{nm}(c_{nm}M_m - c_{mn}M_n)$, $\ c_{nm} = 2\rho_n/(\rho_n + \rho_m)$ |
    | Method of lines | $\boldsymbol{\mathcal M}\,d\mathbf M/dt = -(D\mathbf K + i\gamma g(t)\mathbf X)\mathbf M$; modes, grid or hats |
    | FD as flux balance | $h\dot M_i = J_{i-1/2} - J_{i+1/2}$, $\ J = c\,\Delta M$; error $\frac{h^2}{12}M''''$ |
    | Face rules | bulk $D/h$; wall 0; membrane $1/(h/D + 1/\kappa)$; relaxing wall $1/(h/2D + 1/\rho)$: resistances in series |
    | Time stepping | explicit Euler (order 1, $\delta t\le h^2/2dD$); implicit Euler (order 1); Crank–Nicolson (order 2); both implicit methods unconditionally stable |
    | Amplification | $G_{\text{EE}} = 1 - 4r\sin^2\frac{kh}2$, $\ r = D\delta t/h^2$ |
    | MFD update | $\mathbf M^{n+1} = \boldsymbol\Phi\odot(\mathbf I + \mathbf S)\mathbf M^n$; jump probabilities $s_{j\to k} = c_{jk}\delta t/h$, columns of $\mathbf S$ sum to 0; $\Phi_j = e^{-i\gamma f(\mathbf g\cdot\mathbf r_j)\delta t - \delta t/T_{2,j}}$ |
    | Periodic cell | $M = m\,e^{-iq(t)x}$: $\ \partial_tm = D(\partial_x - iq)^2m$, $m$ periodic; neighbour factors $e^{\mp iqh}$, or only the wrap face $e^{\mp iqL}$ |
    | Weak form | $\int w\dot M + \int D\nabla w\cdot\nabla M + i\gamma g\int wxM = \oint wD\partial_nM$; walls natural, membranes $\kappa\oint(M_1 - M_2)(w_1 - w_2)$ |
    | 1D element matrices | $\boldsymbol{\mathcal M}^e = \frac h6(2,1;1,2)$, $\ \mathbf K^e = \frac Dh(1,-1;-1,1)$, $\ \mathbf J^e = \frac h{12}(3x_a + x_b, x_a + x_b; x_a + x_b, x_a + 3x_b)$ |
    | FEM system | $\boldsymbol{\mathcal M}\dot{\boldsymbol\xi} = -(\mathbf S + \mathbf R + \mathbf Q + i\gamma f(t)\mathbf J)\boldsymbol\xi$, $\ \xi(0) = \rho_n$; double nodes on interfaces |
    | Modes on a mesh | $\mathbf K\mathbf u = \lambda\boldsymbol{\mathcal M}\mathbf u$, $\ \mathbf B = \mathbf U^{\mathsf T}\mathbf J\mathbf U$, then Chapter 2 |
    | Curved walls | staircase $O(h)$ and erratic, perimeter $\to4/\pi$ too long; fitted mesh $O(h^2)$ |
    | Choosing $h$, $\delta t$ | $h\lesssim\ell_g/5$, $2\pi/q$/10 and the structure; refine until $E$ stops changing; validate on a slab |

    ---
    ## Coming next

    **Chapter 5 · Monte Carlo: simulating the random walk**

    - No grid and no mesh: walkers take Gaussian steps, reflect at walls, cross membranes with a probability set by $\kappa$, and collect phase; the signal is $\langle e^{-i\varphi}\rangle$ (the Feynman–Kac picture of Chapter 1).
    - The two errors: a bias from the finite step near walls, and noise that falls only as $1/\sqrt N$, checked against the exact signals of Chapters 2 and 3.
    - Walker histories as reusable data: replay packs ([Fick 2026](https://arxiv.org/abs/2611.01758)).
    - *Implementations:* Camino, MC-DC Simulator, Disimpy, RMS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## References

    Where possible, the first paper that introduced each idea.

    | Year | Reference | What it introduced |
    | --- | --- | --- |
    | 1915 | B. G. Galerkin, *Vestnik Inzhenerov i Tekhnikov* 19:897 | the weighted-residual method with the same functions as unknowns and weights |
    | 1928 | [R. Courant, K. Friedrichs & H. Lewy, *Math. Ann.* 100:32](https://doi.org/10.1007/BF01448839) | the stability limit of explicit difference schemes |
    | 1943 | [R. Courant, *Bull. Amer. Math. Soc.* 49:1](https://doi.org/10.1090/S0002-9904-1943-07818-4) | piecewise-linear functions on triangles: the idea of finite elements |
    | 1947 | [J. Crank & P. Nicolson, *Proc. Camb. Phil. Soc.* 43:50](https://doi.org/10.1017/S0305004100023197) | the trapezoidal time step for the heat equation |
    | 2003 | [S. N. Hwang, C.-L. Chin, F. W. Wehrli & D. B. Hackney, *Magn. Reson. Med.* 50:373](https://doi.org/10.1002/mrm.10536) | image-based finite differences for restricted diffusion |
    | 2007 | [J. Xu, M. D. Does & J. C. Gore, *Phys. Med. Biol.* 52:N111](https://doi.org/10.1088/0031-9155/52/7/N01) | improved finite differences; revised periodic boundary condition |
    | 2014 | [D. V. Nguyen, J.-R. Li, D. Grebenkov & D. Le Bihan, *J. Comput. Phys.* 263:283](https://doi.org/10.1016/j.jcp.2014.01.009) | finite elements for the Bloch–Torrey equation; double nodes; pseudo-periodic conditions |
    | 2014 | [J.-R. Li et al., *Phys. Med. Biol.* 59:441](https://doi.org/10.1088/0031-9155/59/2/441) | finite volumes with adaptive time stepping |
    | 2019 | [J.-R. Li et al., *NeuroImage* 202:116120](https://doi.org/10.1016/j.neuroimage.2019.116120) | SpinDoctor |
    | 2020 | [J.-R. Li, T. N. Tran & V.-D. Nguyen, *NMR Biomed.* 33:e4353](https://doi.org/10.1002/nbm.4353) | the matrix formalism on finite-element modes |
    | 2022 | [S. D. Agdestein, T. N. Tran & J.-R. Li, *NMR Biomed.* 35:e4646](https://doi.org/10.1002/nbm.4646) | mesh modes with permeable membranes, for realistic neurons |
    | 2025 | [J. Xu et al., MATI, *Magn. Reson. Imaging* 110428](https://doi.org/10.1016/j.mri.2025.110428) | a GPU finite-difference toolbox with fitting |
    | 2025 | [P. Khole et al., ReMiDi, arXiv 2502.01988](https://arxiv.org/abs/2502.01988) | a differentiable matrix formalism on meshes |
    """)
    return


if __name__ == "__main__":
    app.run()
