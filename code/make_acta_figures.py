#!/usr/bin/env python3
"""Figures of the paper (matplotlib).  Uses the engine of deep_sieve_improvements.py (same folder).

    python code/make_acta_figures.py [--quick]

Output (pdf and png, in figures/):
    fig_bound                    the bound as a function of theta2, and against the number of Buchstab steps
    fig_fixed_point_and_shifts   fixed point against rectangle bounds along the levels; singular series against counts
    fig_terms                    main term, switched term, room for the weights and the dual bound
The curves are one-sided values on the grid n = 800, rho = 1.002 (n = 400, rho = 1.004 with --quick); the certified
values of the paper are those of data/deep_certificate*.txt, computed on a finer grid."""
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deep_sieve_improvements as D      # noqa: E402

QUICK = "--quick" in sys.argv
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)
NAVY, CRIMSON, EMERALD, AMBER, PURPLE, SLATE = "#1B4F72", "#C0392B", "#117864", "#D68910", "#6C3483", "#566573"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10})
GRID = dict(n=400, rho=1.004, a_min=0.004) if QUICK else dict(n=800, rho=1.002, a_min=0.003)
FINE = GRID if QUICK else dict(n=1600, rho=1.001, a_min=0.002)     # for the panels at the point of Theorem 1.2
T2, T1 = 0.009, 0.398


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print(name, "written", D.clock(), flush=True)


def curve(eng, U, V, mode, lo, hi):
    b = D.scan_theta1(eng, U, V, mode, lo, hi)
    return b["val"] if b else np.nan


# ------------------------------------------------------------------------------------------------ figure 1
def fig_bound():
    small = [0.0008, 0.0012, 0.002] if QUICK else [0.0006, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.0025, 0.003]
    large = [0.005, 0.009, 0.013] if QUICK else [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.014,
                                                 0.016]
    cap = {"rect": [], "fix": []}
    for t2 in small:
        eng = D.Engine(t2, store=False, **GRID)
        U, V = eng.sweep()
        cap["fix"].append(curve(eng, U, V, "capped", 0.42, 0.495))
        del U, V
        eng.store = True
        U0, V0 = eng.rect_UV()
        cap["rect"].append(curve(eng, U0, V0, "capped", 0.42, 0.495))
        print("capped", t2, cap["rect"][-1], cap["fix"][-1], D.clock(), flush=True)
        del eng, U0, V0
    ks = (0, 2, 4, "inf")
    free = {k: [] for k in ks}
    for t2 in large:
        eng = D.Engine(t2, **GRID)
        row = {}
        for k, U, V in D.k_step_tables(eng):
            row[k] = curve(eng, U, V, "free", 0.34, 0.46)
        for k in ks:
            free[k].append(row[k])
        print("free", t2, [round(row[k], 3) for k in ks], D.clock(), flush=True)
        del eng
    # panel (b): theta2 = 0.009, theta1 = 0.398 a node; value and margin at the point of Theorem 1.2
    eng = D.Engine(T2, anchor=T1, **FINE)
    j = eng.j_anchor
    Tsw = D.T_free(T1, T2, s=4.42)[0]
    kk, vals, marg = [], [], []
    for k, U, V in D.k_step_tables(eng, ks=tuple(range(13))):
        b = D.scan_theta1(eng, U, V, "free", 0.34, 0.46)
        m = float(U[eng.n, j]) - Tsw - 0.0935 * eng.J_of(V, j, 0.19)
        if k == "inf":
            vinf, minf = b["val"], m
        else:
            kk.append(k)
            vals.append(b["val"])
            marg.append(m)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    a0 = ax[0]
    for lo, hi, c in ((40, 41, SLATE), (17, 18, AMBER), (15, 16, EMERALD)):
        a0.axhspan(lo, hi, color=c, alpha=0.13, lw=0)
        a0.text(0.0162 if lo == 40 else 0.00062, lo + 0.5, r"$\Omega\leq%d$" % lo, fontsize=6.5, color=c, va="center",
                ha="right" if lo == 40 else "left")
    a0.semilogx(small, cap["rect"], "s:", color=SLATE, ms=3.5, lw=1.1, label="capped levels, rectangle")
    a0.semilogx(small, cap["fix"], "s-", color=PURPLE, ms=3.5, lw=1.3, label="capped levels, fixed point")
    a0.semilogx(large, free[0], "o--", color=AMBER, ms=3.5, lw=1.2, label="free levels, rectangle ($k=0$)")
    a0.semilogx(large, free[2], "^-.", color=CRIMSON, ms=3.5, lw=1.0, label="free levels, $k=2$")
    a0.semilogx(large, free[4], "v-.", color=NAVY, ms=3.5, lw=1.0, label="free levels, $k=4$")
    a0.semilogx(large, free["inf"], "o-", color=EMERALD, ms=3.5, lw=1.6, label="free levels, fixed point")
    a0.set_xlabel(r"$\theta_2$")
    a0.set_ylabel(r"$\min\,(\lambda^{-1}+\theta^{-1})$")
    a0.set_ylim(14, 46)
    a0.set_xlim(0.00055, 0.018)
    a0.legend(fontsize=6.5, loc="center", bbox_to_anchor=(0.5, 0.50))
    a0.set_title(r"(a) the bound as a function of $\theta_2$")
    a1 = ax[1]
    a1.axhspan(15, 16, color=EMERALD, alpha=0.13, lw=0)
    a1.plot(kk, vals, "o-", color=NAVY, ms=3.5, lw=1.4, label=r"$\min\,(\lambda^{-1}+\theta^{-1})$")
    a1.axhline(vinf, color=NAVY, ls=":", lw=0.9)
    a1.axhline(16, color=EMERALD, lw=0.9)
    a1.set_xticks(range(0, 13, 2))
    a1.set_xlabel("number $k$ of Buchstab steps")
    a1.set_ylabel(r"$\min\,(\lambda^{-1}+\theta^{-1})$", color=NAVY)
    a1.set_ylim(15.4, 18.2)
    a2 = a1.twinx()
    a2.spines["right"].set_visible(True)
    a2.plot(kk, marg, "D-", color=CRIMSON, ms=3, lw=1.2, label=r"$u-T-\lambda J$")
    a2.axhline(0, color=CRIMSON, ls="--", lw=0.8)
    a2.axhline(minf, color=CRIMSON, ls=":", lw=0.9)
    a2.set_ylabel(r"$u-T-\lambda J$ at the point of Theorem 1.2", color=CRIMSON, fontsize=8.5)
    a2.set_ylim(-0.09, 0.03)
    a1.set_title(r"(b) $\theta_2=0.009$: Buchstab steps")
    save(fig, "fig_bound")


# ------------------------------------------------------------------------------------------------ figure 2
def fig_fixed_point_and_shifts():
    eng = D.Engine(T2, anchor=T1, **FINE)
    j = eng.j_anchor
    U0, V0 = eng.rect_UV()
    U0, V0 = U0[:, j].astype(float), V0[:, j].astype(float)
    U, V = eng.sweep()
    U, V = U[:, j].astype(float), V[:, j].astype(float)
    X = eng.X
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    a0 = ax[0]
    k = X >= 0.03
    a0.plot(X[k], V0[k], "--", color=AMBER, lw=1.2, label=r"$V^{(0)}$ (rectangle)")
    a0.plot(X[k], V[k], "-", color=CRIMSON, lw=1.5, label=r"$V^{\sharp}$ (fixed point)")
    a0.plot(X[k], U0[k], "--", color=SLATE, lw=1.2, label=r"$U^{(0)}$ (rectangle)")
    a0.plot(X[k], U[k], "-", color=NAVY, lw=1.5, label=r"$U^{\sharp}$ (fixed point)")
    a0.fill_between(X[k], U0[k], U[k], color=NAVY, alpha=0.15, lw=0)
    a0.fill_between(X[k], V[k], np.minimum(V0[k], 8), color=CRIMSON, alpha=0.12, lw=0)
    a0.set_ylim(0, 6)
    a0.set_xlim(0.03, 0.5)
    a0.set_xlabel(r"level $X$")
    a0.set_title(r"(a) bounds at $(\theta_1,\theta_2)=(0.398,0.009)$")
    a0.legend(fontsize=7.5, loc="upper right")
    a0.annotate(r"$+%.1f\%%$" % (100 * (U[-1] / U0[-1] - 1)), xy=(0.5, U[-1]), xytext=(0.40, 1.9), fontsize=8, color=NAVY,
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.6))
    # (b) singular series against counts
    shifts = [-6, -4, 2, 4, 6, 8, 12, 14, 20, 30, 42, 2310]
    x = 5 * 10 ** 6 if QUICK else 6 * 10 ** 7
    N = D.shift_counts(x, shifts)
    C = {a: D.C_shift(a, 10 ** 6) for a in shifts}
    a1 = ax[1]
    pos = np.arange(len(shifts))
    pred = np.array([C[a] / C[2] for a in shifts])
    obs = np.array([N[a] / N[2] for a in shifts])
    err = obs * np.sqrt(1.0 / np.array([N[a] for a in shifts]) + 1.0 / N[2])
    a1.bar(pos, pred, width=0.62, color=NAVY, alpha=0.28, edgecolor=NAVY, lw=0.8, label=r"$C(a)/C(2)$ (Euler product)")
    a1.errorbar(pos, obs, yerr=2 * err, fmt="o", color=CRIMSON, ms=3.5, lw=1.0, capsize=2,
                label=r"$N_a/N_2$, $p\leq 6\cdot10^7$ ($\pm2\sigma$)")
    k4 = shifts.index(4)
    a1.plot([pos[k4]], [pred[k4] * math.sqrt(math.log(x / 2) / math.log(x / 18))], marker="_", color=AMBER, ms=11, mew=1.6,
            ls="none", label="second-order correction")
    a1.set_xticks(pos)
    a1.set_xticklabels([str(a) for a in shifts], fontsize=7, rotation=45)
    a1.set_xlabel(r"shift $a$")
    a1.set_ylim(0, 2.25)
    a1.legend(fontsize=7, loc="upper left")
    a1.set_title(r"(b) singular series against counts")
    save(fig, "fig_fixed_point_and_shifts")


# ------------------------------------------------------------------------------------------------ figure 3
def dual_bound(t2):
    """dual bound for R = 14 at theta2 from data/deep_level3.txt (None if absent)"""
    try:
        txt = open(os.path.join(DATA, "deep_level3.txt"), encoding="utf8").read()
    except OSError:
        return None
    blk = txt.split("theta2 %.4f:" % t2)
    if len(blk) < 2:
        return None
    m = re.search(r"J_psi >= ([0-9.]+)", blk[1])
    return float(m.group(1)) if m else None


def fig_terms():
    eng = D.Engine(T2, **GRID)
    U0, V0 = eng.rect_UV()
    U, V = eng.sweep()
    j = np.nonzero((eng.a >= 0.30) & (eng.a <= 0.492))[0]
    t1 = eng.a[j]
    Tf = np.array([D.T_free(float(t), T2, s=4.42)[0] for t in t1])
    Tc = np.array([D.T_capped(float(t), T2) for t in t1])
    u, u0 = U[eng.n, j].astype(float), U0[eng.n, j].astype(float)
    Jd = dual_bound(T2)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    a0 = ax[0]
    a0.plot(t1, u, "-", color=NAVY, lw=1.6, label=r"$u^{\sharp}$ (fixed point)")
    a0.plot(t1, u0, "--", color=NAVY, lw=1.0, label=r"$u_0$ (rectangle)")
    a0.plot(t1, Tf, "-", color=CRIMSON, lw=1.6, label=r"$T$ (free levels)")
    a0.plot(t1, Tc, ":", color=CRIMSON, lw=1.4, label=r"$T$ (capped levels)")
    room = u - Tf
    a0.plot(t1, np.where(room > 0, room, np.nan), "-", color=EMERALD, lw=1.4, label=r"room $u^{\sharp}-T$")
    a0.fill_between(t1, 0, np.maximum(room, 0), color=EMERALD, alpha=0.12, lw=0)
    if Jd:
        a0.axhline(Jd, color=PURPLE, ls="--", lw=1.0)
        a0.text(0.302, Jd + 0.03, r"dual bound for $J_\psi$, $R=14$", fontsize=7.5, color=PURPLE)
    a0.set_xlabel(r"$\theta_1$")
    a0.set_ylim(0, 1.6)
    a0.set_xlim(0.30, 0.495)
    a0.set_title(r"(a) main and switched terms, $\theta_2=0.009$")
    a0.legend(fontsize=7, loc="upper right")
    a1 = ax[1]
    d = 1 - 2 * t1
    k = d > 0.012
    a1.loglog(d[k], u0[k], "--", color=NAVY, lw=1.0, label=r"$u_0$")
    a1.loglog(d[k], u[k], "-", color=NAVY, lw=1.6, label=r"$u^{\sharp}$")
    a1.loglog(d[k], Tf[k], "-", color=CRIMSON, lw=1.6, label=r"$T$ (free)")
    a1.loglog(d[k], Tc[k], ":", color=CRIMSON, lw=1.4, label=r"$T$ (capped)")
    dd = np.array([0.02, 0.3])
    a1.loglog(dd, D.C1 * np.sqrt(2 * dd), color=SLATE, lw=0.8, alpha=0.7)
    a1.loglog(dd, D.KAPPA0 * 2 * (8 / 3) * dd ** 1.5, color=SLATE, lw=0.8, alpha=0.7)
    a1.text(0.021, 0.50, r"$C_1\sqrt{2\delta}$", fontsize=8, color=SLATE)
    a1.text(0.021, 0.085, r"$\frac{16}{3}e^{\gamma}C_1\delta^{3/2}$", fontsize=8, color=SLATE)
    a1.set_xlabel(r"$\delta=1-2\theta_1$")
    a1.set_ylim(0.01, 3)
    a1.legend(fontsize=7, loc="lower right")
    a1.set_title(r"(b) the same against $\delta$")
    save(fig, "fig_terms")


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name, fn in (("bound", fig_bound), ("shifts", fig_fixed_point_and_shifts), ("terms", fig_terms)):
        if not which or name in which:
            fn()
