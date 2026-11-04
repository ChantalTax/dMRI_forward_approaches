# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib"]
# ///
# © 2026 Chantal Tax. All rights reserved.
# Shared for viewing only; not licensed for reuse or redistribution.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="dMRI 02 · From random walks to signal decay")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

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
    C_TISSUE = "#2a6fdb"  # main signal / tissue
    C_WATER = "#9aa5b1"   # free water reference
    C_RESTR = "#d9622b"   # restricted
    GAMMA = 2.675e8       # rad s^-1 T^-1 (1H)
    return C_RESTR, C_TISSUE, C_WATER, GAMMA, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    show_maths = mo.ui.switch(label="Show the maths inline", value=False)
    mo.vstack(
        [
            mo.md(
                r"""
# Module 2 · From random walks to signal decay

*© 2026 Chantal Tax. All rights reserved. Shared for viewing only; not licensed for reuse or redistribution.*

*Backbone path of the diffusion MRI course. Before this: Module 1 (gradient encoding). After this: Module 3 (anisotropy & DTI).*

**What you'll take away**

1. Water molecules random-walk; over a time *t* they spread out by roughly $\sqrt{2Dt}$.
2. A diffusion gradient turns *displacement* into *phase*.
3. Spins dephase, the signal drops, and for free diffusion it drops as $e^{-bD}$.

Every plot below responds to the sliders, so change something, predict what will happen, then look.
Use the switch for derivations: off, they stay folded away; on, they appear in the text.
"""
            ),
            show_maths,
        ]
    )
    return (show_maths,)


@app.cell(hide_code=True)
def _(mo, show_maths):
    def maths(text, title="The maths"):
        """Derivations: inline for physics-minded readers, folded away otherwise."""
        body = mo.md(text)
        if show_maths.value:
            return mo.callout(body, kind="neutral")
        return mo.accordion({f"▸ {title}": body})

    return (maths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## 1 · A random walk

Each water molecule is bumped thousands of times per millisecond, so its path is a random walk.
Set the **diffusivity** $D$ (free water at body temperature ≈ 3 µm²/ms; brain tissue ≈ 0.7–1) and the **diffusion time** $t$.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    D_slider = mo.ui.slider(0.3, 3.0, step=0.1, value=1.0, label="Diffusivity D (µm²/ms)", show_value=True)
    t_slider = mo.ui.slider(1, 100, step=1, value=30, label="Diffusion time t (ms)", show_value=True)
    mo.hstack([D_slider, t_slider], justify="start", gap=2)
    return D_slider, t_slider


@app.cell(hide_code=True)
def _(C_TISSUE, D_slider, mo, np, plt, t_slider):
    _D, _t = D_slider.value, t_slider.value
    _rng = np.random.default_rng(1)
    _nsteps = 200
    _step_sd = np.sqrt(2 * _D * _t / _nsteps)

    # a few 2D trajectories to look at
    _paths = np.cumsum(_rng.normal(0, _step_sd, size=(25, _nsteps, 2)), axis=1)
    _paths = np.concatenate([np.zeros((25, 1, 2)), _paths], axis=1)

    # many 1D walkers for the displacement distribution
    _dx = _rng.normal(0, _step_sd, size=(20000, _nsteps)).sum(axis=1)
    rms_free = np.sqrt(2 * _D * _t)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 4))
    for _p in _paths:
        _a1.plot(_p[:, 0], _p[:, 1], lw=0.7, alpha=0.7)
        _a1.plot(_p[-1, 0], _p[-1, 1], "o", ms=2.5, color="k")
    _a1.add_patch(plt.Circle((0, 0), rms_free, fill=False, ls="--", color=C_TISSUE, lw=1.5))
    _a1.add_patch(plt.Circle((0, 0), 5, fill=False, ls=":", color="k", lw=1))
    _lim = max(4 * rms_free, 12)
    _a1.set(xlim=(-_lim, _lim), ylim=(-_lim, _lim), aspect="equal", xlabel="x (µm)", ylabel="y (µm)",
            title="25 walkers (dotted: a 10 µm cell)")

    _a2.hist(_dx, bins=80, density=True, color=C_TISSUE, alpha=0.35, label="simulated walkers")
    _x = np.linspace(-_lim, _lim, 400)
    _a2.plot(_x, np.exp(-_x**2 / (4 * _D * _t)) / np.sqrt(4 * np.pi * _D * _t), color=C_TISSUE, lw=2,
             label="Gaussian, σ² = 2Dt")
    _a2.set(xlim=(-_lim, _lim), xlabel="displacement along x (µm)", ylabel="probability density",
            title="Where did they end up?")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"RMS displacement along one axis: **{rms_free:.1f} µm**, "
                f"{'larger' if rms_free > 10 else 'smaller'} than a typical 10 µm cell. "
                "Keep that in mind for the branch at the end."
            ),
        ]
    )
    return (rms_free,)


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
Each step is independent with variance $\sigma_s^2$. After $N$ steps the variances add, so $\langle x^2\rangle = N\sigma_s^2$.
With $N = t/\Delta t$ steps and $\sigma_s^2 = 2D\,\Delta t$, this gives

$$\langle x^2 \rangle = 2Dt \qquad(\text{per axis; } 6Dt \text{ in 3D}).$$

The central limit theorem makes the distribution Gaussian whatever the individual step looks like. That is why
free diffusion is fully described by one number, $D$. Keep that idea in mind: once something breaks the Gaussian
shape (Module 4), $D$ alone is no longer enough.
""", title="Why √(2Dt)? Why Gaussian?")
    return


@app.cell(hide_code=True)
def _(mo):
    guess = mo.ui.radio(
        options=["It doubles", "It grows by about 1.4×", "It stays the same", "It halves"],
        label="**Predict first:** you double the diffusion time *t*. What happens to the RMS displacement?",
    )
    guess
    return (guess,)


@app.cell(hide_code=True)
def _(guess, mo):
    if guess.value is None:
        _out = mo.callout(mo.md("Commit to a guess, then test it by doubling *t* with the slider."), kind="info")
    elif guess.value == "It grows by about 1.4×":
        _out = mo.callout(
            mo.md("**Yes.** Displacement grows as √t, so doubling *t* gives √2 ≈ 1.41×. "
                  "This is why we need *long* diffusion times to probe *large* structures."),
            kind="success",
        )
    else:
        _out = mo.callout(
            mo.md("**Not quite.** Try it: set *t* = 20 ms, note the RMS, then set *t* = 40 ms. "
                  "A random walk spreads as √t, not t. Molecules keep reversing direction."),
            kind="warn",
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## 2 · From displacement to phase to signal

A pulsed-gradient spin echo (PGSE) applies a gradient of strength $G$ for duration $\delta$, waits $\Delta$, and then
applies the opposite gradient. A spin that **did not move** comes back in phase. A spin that **moved** a distance $x$ along the gradient keeps a phase of about $\varphi = q\,x$, with $q = \gamma G \delta$.

The signal is the average of all those phase vectors, $S/S_0 = |\langle e^{i\varphi}\rangle|$. More spread in $x$ means more spread in $\varphi$, and therefore less signal.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    G_slider = mo.ui.slider(0, 80, step=1, value=40, label="Gradient G (mT/m)", show_value=True)
    delta_slider = mo.ui.slider(1, 40, step=1, value=20, label="Pulse width δ (ms)", show_value=True)
    Delta_slider = mo.ui.slider(5, 100, step=1, value=40, label="Separation Δ (ms)", show_value=True)
    mo.hstack([G_slider, delta_slider, Delta_slider], justify="start", gap=2, wrap=True)
    return Delta_slider, G_slider, delta_slider


@app.cell(hide_code=True)
def _(Delta_slider, GAMMA, G_slider, delta_slider, mo):
    _G, _d, _Dl = G_slider.value, delta_slider.value, max(Delta_slider.value, delta_slider.value)
    # q in rad/µm, times in ms
    q_now = GAMMA * (_G * 1e-3) * (_d * 1e-3) * 1e-6
    t_diff = _Dl - _d / 3                       # effective diffusion time (ms)
    b_now = q_now**2 * t_diff * 1000            # s/mm²  (1 ms/µm² = 1000 s/mm²)
    _warn = (
        mo.callout(mo.md("Δ must be ≥ δ. Δ = δ is used here."), kind="warn")
        if Delta_slider.value < delta_slider.value
        else mo.md("")
    )
    mo.vstack(
        [
            _warn,
            mo.md(
                f"q = **{q_now:.3f} rad/µm** · effective diffusion time Δ − δ/3 = **{t_diff:.1f} ms** · "
                f"**b = {b_now:,.0f} s/mm²**"
            ),
        ]
    )
    return b_now, q_now, t_diff


@app.cell(hide_code=True)
def _(C_TISSUE, C_WATER, D_slider, b_now, mo, np, plt, q_now, t_diff):
    _D = D_slider.value
    _rng = np.random.default_rng(2)
    _x = _rng.normal(0, np.sqrt(2 * _D * t_diff), 20000)  # Gaussian, as established in §1
    _phi = q_now * _x
    S_now = np.abs(np.mean(np.exp(1j * _phi)))

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 4))

    # left: phase vectors on the unit circle
    _sub = _phi[:300]
    _a1.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.8"))
    for _p in _sub:
        _a1.plot([0, np.cos(_p)], [0, np.sin(_p)], color=C_TISSUE, alpha=0.06, lw=1)
    _m = np.mean(np.exp(1j * _phi))
    _a1.annotate("", xy=(_m.real, _m.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5))
    _a1.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", xticks=[], yticks=[],
            title=f"Phase of 300 spins; black arrow = average (|S| = {S_now:.2f})")
    _a1.grid(False)

    # right: signal vs b
    _b = np.linspace(0, 4000, 300)
    _a2.plot(_b, np.exp(-_b / 1000 * _D), color=C_TISSUE, lw=2, label=f"e$^{{-bD}}$, D = {_D:.1f}")
    _a2.plot(_b, np.exp(-_b / 1000 * 3.0), color=C_WATER, lw=1.5, ls="--", label="free water, D = 3.0")
    _a2.plot([b_now], [S_now], "o", color="k", ms=8, label="this simulation")
    _a2.set(xlabel="b (s/mm²)", ylabel="S / S₀", ylim=(0, 1.05), xlim=(0, 4000), title="Signal decay")
    _a2.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
In the narrow-pulse picture each spin gets phase $\varphi = q x$. For a Gaussian $x$ with variance $2Dt$,
the average of $e^{iqx}$ is the Gaussian's characteristic function:

$$\frac{S}{S_0} = \langle e^{iqx}\rangle = e^{-q^2 \cdot 2Dt/2} = e^{-q^2 t D} \equiv e^{-bD}.$$

For finite pulses the spins also move *during* the gradient, which replaces $\Delta$ by $\Delta - \delta/3$:

$$b = \gamma^2 G^2 \delta^2\left(\Delta - \tfrac{\delta}{3}\right).$$

**Check the units:** with $\gamma = 2.675\times10^8$ rad s⁻¹ T⁻¹, G = 40 mT/m, δ = 20 ms and Δ = 40 ms you get
b ≈ 1500 s/mm², a typical clinical value.
""", title="Deriving S = exp(−bD) and the b-value")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
**Things to try**

- Set **G = 0**. Why is the signal exactly 1, however much the water moves?
- Find two settings with the **same b** but different δ and Δ. Do they give the same signal? *For free diffusion they do. Keep that in mind for the branch.*
- Lower **D** to 0.7 (tissue-like). At b = 1000, how much brighter is tissue than free water?
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
### Branch · What if the water is trapped?

*Optional detour. The backbone continues in Module 3.*

Everything above assumed water can wander freely. In tissue, membranes get in the way.
Here the walkers sit between two reflecting walls a distance $L$ apart, with the gradient across them.
That one change breaks the "single number $D$" picture.
"""),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    L_slider = mo.ui.slider(2, 20, step=1, value=10, label="Wall spacing L (µm)", show_value=True)
    L_slider
    return (L_slider,)


@app.cell(hide_code=True)
def _(D_slider, L_slider, np):
    def _simulate_restricted(D, L, t_max=100.0, dt=0.1, n=4000, seed=3):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, L, n)
        x = x0.copy()
        nsteps = int(t_max / dt)
        sd = np.sqrt(2 * D * dt)
        msd = np.empty(nsteps)
        disp = np.empty((nsteps, n), dtype=np.float32)
        for k in range(nsteps):
            x = x + rng.normal(0, sd, n)
            x = np.abs(x)              # reflect at 0
            x = L - np.abs(L - x)      # reflect at L
            d = x - x0
            disp[k] = d
            msd[k] = np.mean(d**2)
        times = dt * np.arange(1, nsteps + 1)
        return times, msd, disp

    r_times, r_msd, r_disp = _simulate_restricted(D_slider.value, L_slider.value)
    return r_disp, r_msd, r_times


@app.cell(hide_code=True)
def _(
    C_RESTR,
    C_TISSUE,
    D_slider,
    L_slider,
    mo,
    np,
    plt,
    r_disp,
    r_msd,
    r_times,
    t_diff,
):
    _D, _L = D_slider.value, L_slider.value
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 4))

    # left: apparent diffusivity vs time
    _adc = r_msd / (2 * r_times)
    _a1.axhline(_D, color=C_TISSUE, lw=2, label="free: D")
    _a1.plot(r_times, _adc, color=C_RESTR, lw=2, label=f"between walls, L = {_L} µm")
    _a1.axvline(t_diff, color="k", ls=":", lw=1)
    _a1.text(t_diff, _D * 1.02, " your Δ−δ/3", fontsize=8, va="bottom")
    _a1.set(xlabel="diffusion time (ms)", ylabel="apparent D = ⟨x²⟩ / 2t (µm²/ms)",
            ylim=(0, _D * 1.15), xlim=(0, 100), title="Apparent diffusivity depends on time")
    _a1.legend(frameon=False, fontsize=8)

    # right: signal vs b at the current diffusion time
    _k = min(int(round(t_diff / 0.1)) - 1, len(r_times) - 1)
    _dx = r_disp[_k].astype(float)
    _b = np.linspace(0, 10000, 40)
    _q = np.sqrt(_b / 1000 / r_times[_k])
    _S = np.abs(np.exp(1j * np.outer(_q, _dx)).mean(axis=1))
    _bb = np.linspace(0, 10000, 300)
    _adc_now = _adc[_k]
    _a2.semilogy(_bb, np.exp(-_bb / 1000 * _D), color=C_TISSUE, lw=2, label="free: e$^{-bD}$")
    _a2.semilogy(_bb, np.exp(-_bb / 1000 * _adc_now), color=C_RESTR, lw=1.2, ls="--",
                 label=f"Gaussian with same apparent D ({_adc_now:.2f})")
    _a2.semilogy(_b, _S, "o-", color=C_RESTR, ms=3.5, lw=1.2, label="between walls (simulated)")
    _a2.set(xlabel="b (s/mm²)", ylabel="S / S₀ (log scale)", ylim=(1e-3, 1.1), xlim=(0, 10000),
            title=f"Signal at diffusion time {r_times[_k]:.0f} ms")
    _a2.legend(frameon=False, fontsize=8, loc="lower left")
    _fig.tight_layout()

    _plateau = _L**2 / 6
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Free walkers would reach ⟨x²⟩ = 2Dt; trapped walkers level off at L²/6 = **{_plateau:.1f} µm²**. "
                "On a log scale, any Gaussian is a straight line. At high b the trapped signal falls away from the dashed line "
                "with the same apparent D. "
                "**That bend is what kurtosis and multi-compartment models measure** (Module 4)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(maths):
    maths(r"""
Between walls the displacement cannot keep growing. At long times each walker is equally likely to be anywhere in
$[0, L]$, independent of where it started, so

$$\langle x^2 \rangle \to \frac{L^2}{6}, \qquad D_{\text{app}}(t) = \frac{\langle x^2\rangle}{2t} \to \frac{L^2}{12\,t}.$$

The displacement distribution is no longer Gaussian, so $\ln S$ is no longer linear in $b$. Its curvature is the
kurtosis. One caveat: we used the narrow-pulse phase $\varphi = q\,x(\Delta-\delta/3)$. With finite pulses in a
restricted space this is only approximate. A proper treatment (Gaussian phase approximation, or a full phase simulation)
deserves a branch of its own.
""", title="Why the plateau, and why the curve bends")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
**Branch questions**

- At what **L** does restriction stop mattering for your current Δ? Compare L with the free RMS displacement from §1.
- Shorten the diffusion time (lower Δ and δ). Does the apparent D rise towards free D? This is the idea behind **OGSE**: very short effective diffusion times probe very small length scales.
- In the brain, axons are ~1 µm across. With clinical Δ ≈ 40 ms, what should *perpendicular* diffusion look like?

---

**Back to the backbone →** *Module 3: anisotropy & DTI*. Restriction in one direction and not the other produces a diffusion tensor.

**Other branches from here:** Monte Carlo in cylinders · finite-pulse effects · the Gaussian phase approximation · noise floor and Rician bias at high b
""")
    return


if __name__ == "__main__":
    app.run()
