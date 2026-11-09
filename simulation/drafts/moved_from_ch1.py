# Sections moved out of Chapter 1 (2026-11-08), to reuse in later chapters:
#   walkers vs modes vs GPA (Exercise "When is the phase Gaussian?")  -> Ch 2b and Ch 5
#   how many modes / Jackson (Exercise "Why the couplings fall as 1/n^2") -> Ch 2a
#   narrow-pulse signal panel, Exercise 5(c), narrow-pulse paragraph   -> Ch 3
# Not runnable on its own: cells depend on the Chapter 1 setup cell (expm, GAMMA, sliders L1, D1).
# Full working version: commit 79d3468, simulation/ch1_foundations.py.

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See it: walkers and modes give the same signal

    A slab of width $L$, PGSE with $\delta = 10$ ms and $\Delta = 20$ ms, $D = 2$ µm²/ms. The **walkers** (Monte Carlo, 40 000 of them, step 0.05 ms) compute $\langle e^{-i\varphi}\rangle$ directly. The **exact** curve solves the Bloch–Torrey equation in the slab's eigenmodes (the matrix method of Chapter 2). The dashed curve keeps only the variance of the phase: $E \approx e^{-\langle\varphi^2\rangle/2}$, the Gaussian phase approximation.
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
        """, "12 · When is the phase Gaussian?")
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

    When you solve the Bloch–Torrey equation in the eigenmodes (Chapter 2), you keep $N$ modes. How large must $N$ be? Approximation theory gives the right way to think about it.

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

    In 2D and 3D the number of modes below a given $\lambda$ grows as $\lambda^{d/2}$ ([Weyl's law](https://doi.org/10.1007/BF01456804), 1912), so the same resolution costs about $(L/\ell)^d$ modes. A linear gradient couples only some of them, which keeps cylinders and spheres cheap (Chapter 2).
    """)
    return


@app.cell(hide_code=True)
def _(exercise):
    exercise(r"""
        **(a)** Compute the cosine coefficients $c_n = \int_0^L x\,u_n(x)\,dx$ of the gradient field on a slab, and show that they fall as $1/n^2$. Relate them to $B_{0n} = \int_0^L u_0\,x\,u_n\,dx$.
        **(b)** Explain the $1/n^2$ with the mirrored function. Would measuring $x$ from the slab's centre change anything?
        **(c)** Use rule 2 above for $L = 8$ µm, $D = 2$ µm²/ms and $G = 300$ mT/m: from which $N$ on is the last mode damped more strongly than it is dephased? Compare with the right-hand panel.
        """, "13 · Why the couplings fall as 1/n²")
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
        → At $G = 300$ mT/m the gradient length is 2.9 µm, about $L/3$, so a handful of modes resolves $m$. In the localisation regime (Chapter 3) $\ell_g$ becomes much smaller than $L$, and the count grows quickly.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    logt6 = mo.ui.slider(-1, 2.5, step=0.05, value=1.7, label="log₁₀ t (t in ms) for the right panel", show_value=True)
    logt6
    return (logt6,)


@app.cell(hide_code=True)
def _(C_ALT, C_FREE, C_MAIN, D1, L1, logt6, mo, np, plt, slab_charfun, slab_cumulants):
    _L, _D = L1.value, D1.value
    _t = np.logspace(-2, 2.7, 300)
    _d, _K = slab_cumulants(_D * _t / _L**2)
    _t6 = 10**logt6.value
    _d6, _K6 = slab_cumulants(np.array([_D * _t6 / _L**2]))
    _k2 = _d6[0] * 2 * _D * _t6
    _k4 = _K6[0] * _k2**2

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a1.semilogx(_t, _d, color=C_MAIN, lw=2, label="D(t) / D")
    _a1.semilogx(_t, _K, color=C_ALT, lw=2, label="K(t)")
    _a1.axhline(0, color=C_FREE, lw=0.8)
    _a1.axhline(-0.6, color=C_ALT, lw=0.8, ls=":", label="−3/5")
    _a1.axvline(_t6, color="k", lw=0.8, ls="--")
    _a1.set(xlabel="t (ms)", ylim=(-0.7, 1.05), title="Diffusivity and kurtosis of the displacement")
    _a1.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.12))

    _q = (np.arange(1, 600) + 0.37) * (3 * 2 * np.pi / _L) / 600
    _E = slab_charfun(_q, _t6, _L, _D)
    _a2.semilogy(_q * _L / (2 * np.pi), np.clip(_E, 1e-12, None), color=C_MAIN, lw=2, label="exact, from the modes")
    _a2.semilogy(_q * _L / (2 * np.pi), np.exp(-_q**2 * _k2 / 2), color=C_FREE, ls="--", label="2nd cumulant (Gaussian)")
    _a2.semilogy(_q * _L / (2 * np.pi), np.exp(-_q**2 * _k2 / 2 + _q**4 * _k4 / 24), color=C_ALT, label="2nd + 4th cumulant")
    _a2.axvline(1, color="k", lw=0.8, ls=":")
    _a2.set(xlabel="qL / 2π", ylabel="⟨exp(iqX)⟩", ylim=(1e-4, 1.3), title=f"Narrow-pulse signal at t = {_t6:.1f} ms")
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"L and D from the sliders above the propagator plot. At t = {_t6:.1f} ms: Dt/L² = **{_D * _t6 / _L**2:.3f}**, D(t)/D = **{_d6[0]:.3f}**, K(t) = **{_K6[0]:+.3f}**. The dotted line in the right panel is qL = 2π."),
    ])
    return



# --- Exercise 5 (c), answer ---
    **(c)** ⇒ $\ln\operatorname{sinc}^2z$ is singular where $\operatorname{sinc}z = 0$, first at $z = \pi$. The series converges only for $qL < 2\pi$.

    → At $qL = 2\pi$ the signal itself goes to zero: the first **diffraction minimum** ([Callaghan et al. 1991](https://doi.org/10.1038/351467a0)). Beyond it no finite number of cumulants describes the signal; you need the whole $E(q)$. In the right panel at $t \gg \tau_1$ the truncated curves already leave the exact one well before the dotted line.

    → The general rule: cumulants work while $q$ times the typical displacement is below about 1, where the typical displacement is the smaller of $\sqrt{Dt}$ and $L$. Chapter 3 draws this as a boundary in the $(q,t)$ plane.


# --- narrow-pulse paragraph (path picture) ---
    **Narrow pulses.** If both gradient pulses are so short ($\delta\to0$) that spins do not move during them, the first pulse gives each spin the phase $\gamma g\delta\,x(0)$ and the second, after refocusing, removes $\gamma g\delta\,x(\Delta)$. So $\varphi = -qX$, with $q = \gamma g\delta$ and the displacement $X = x(\Delta) - x(0)$, and

    $$E(q,\Delta) = \big\langle e^{iqX}\big\rangle ,$$

    the characteristic function of the displacement from 1a: the Fourier transform of $\bar P(X,\Delta)$ ([Stejskal & Tanner 1965](https://doi.org/10.1063/1.1695690)). The cumulants of $\ln E$ in powers of $q$ are then exactly $\kappa_2 = \langle X^2\rangle$, $\kappa_4$, …. With finite pulses this identity breaks: the phase depends on the whole path, not only on its two end points (Chapter 2).
