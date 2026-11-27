#!/usr/bin/env python3
"""Vector-sieve level optimisation and Buchstab iteration for primes p = m^2 + n^2 + 1 with p + a almost prime.

Standalone (numpy; mpmath for the interval certificate of stage `level1`; scipy only for the exploratory `level3`).

    python code/deep_sieve_improvements.py level1     # free switching levels: explicit witness, interval arithmetic
    python code/deep_sieve_improvements.py level2     # Buchstab iteration on the inert primes: k steps and fixed point
    python code/deep_sieve_improvements.py level3     # exploration: general weights (linear programme), what-if tables
    python code/deep_sieve_improvements.py level4     # general even shifts a: local factors, singular series, counts
    python code/deep_sieve_improvements.py all

Outputs: data/deep_level<k>.txt (and the same text on stdout).

Conventions.  theta1: exponent of the sifting range of the primes q = 3 (4) dividing p - 1 (half-dimensional sieve,
functions F1, f1);  theta2: exponent of the sifting range of all primes dividing p + a (linear sieve, F2, f2);
level of distribution x^{1/2}.  All statements are in units of x (log x)^{-5/2}.

One-sided ("certified") numbers.  Every quantity that enters a certified inequality is computed as a bound in the
safe direction:
  * the sieve functions are enclosed on a grid, [lo, hi], using only the delay equations and the monotonicity of
    F (decreasing) and f (increasing);
  * suprema/infima over splittings of the level are replaced by the value at explicit admissible splittings;
  * integrals are bounded cell by cell, using monotonicity in the level and in the sifting range of *valid sieve
    bounds* (a bound valid at level X and range a is valid at every level >= X; an upper bound stays valid for a
    larger range, a lower bound for a smaller range).  No regularity of the computed tables is assumed.
Floating point: double precision, tables stored in single precision after outward rounding, and every row is padded
outwards by 1e-9.  Stage level1 does not rely on this: it is carried out in interval arithmetic (mpmath.iv).
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)
T0 = time.time()
_fh = None


def open_log(name):
    global _fh
    if _fh:
        _fh.close()
    _fh = open(os.path.join(DATA, name), "w", encoding="utf8")


def say(s=""):
    print(s, flush=True)
    if _fh:
        _fh.write(s + "\n")
        _fh.flush()


def clock():
    return "[%6.0f s]" % (time.time() - T0)


GAMMA = 0.57721566490153286
EG = math.exp(GAMMA)
C1 = 2.0 * math.sqrt(EG / math.pi)      # sqrt(s) F1(s) = C1 on (0, 2]
C2 = 2.0 * EG                           # s F2(s) = C2 on (0, 3]
PAD = 1e-9

# ----------------------------------------------------------------------------------------------------------------
# 1. sieve functions: enclosures on a grid
# ----------------------------------------------------------------------------------------------------------------
NT = 40000            # nodes per unit of s
SMAX = 14             # beyond SMAX: F in [1, F_hi(SMAX)], f in [f_lo(SMAX), 1]


class SieveTables:
    """lo/hi enclosures of F1, f1 (kappa = 1/2, beta = 1) and F2, f2 (kappa = 1, beta = 2) at the nodes s = i/NT.

    half-dimensional:  (sqrt(s) F1)' = f1(s-1) / (2 sqrt s)  (s > 2),   (sqrt(s) f1)' = F1(s-1) / (2 sqrt s)  (s > 3),
                       F1 = C1 / sqrt s on (0, 2],  f1 = C1 log(sqrt s + sqrt(s-1)) / sqrt s on [1, 3];
    linear:            (s F2)' = f2(s-1)  (s > 3),   (s f2)' = F2(s-1)  (s > 4),
                       F2 = C2 / s on (0, 3],  f2 = C2 log(s-1) / s on [2, 4].
    On a cell [s_{i-1}, s_i] the delayed function is enclosed by its values at the two delayed nodes (monotonicity),
    and int dt/(2 sqrt t) = sqrt(s_i) - sqrt(s_{i-1}) exactly."""

    def __init__(self, nt=NT, smax=SMAX):
        self.nt, self.smax = nt, smax
        M = nt * smax
        s = np.arange(M + 1) / nt
        s[0] = 1e-300
        rs = np.sqrt(s)
        up, dn = 1 + 1e-13, 1 - 1e-13
        # ---- half-dimensional
        P_lo = np.full(M + 1, C1 * dn)
        P_hi = np.full(M + 1, C1 * up)              # sqrt(s) F1
        Q_lo = np.zeros(M + 1)
        Q_hi = np.zeros(M + 1)                      # sqrt(s) f1
        k = slice(nt, 3 * nt + 1)
        q = C1 * np.log(rs[k] + np.sqrt(np.maximum(s[k] - 1, 0.0)))
        Q_lo[k], Q_hi[k] = q * dn, q * up
        Q_lo[nt] = 0.0
        F_lo, F_hi = P_lo / rs, P_hi / rs
        f_lo, f_hi = Q_lo / rs, Q_hi / rs
        drs = np.diff(rs)                           # drs[i-1] = sqrt(s_i) - sqrt(s_{i-1})
        for b in range(2, smax):
            idx = np.arange(b * nt + 1, (b + 1) * nt + 1)
            P_hi[idx] = (P_hi[b * nt] + np.cumsum(drs[idx - 1] * f_hi[idx - nt])) * up
            P_lo[idx] = (P_lo[b * nt] + np.cumsum(drs[idx - 1] * f_lo[idx - 1 - nt])) * dn
            F_hi[idx], F_lo[idx] = P_hi[idx] / rs[idx] * up, P_lo[idx] / rs[idx] * dn
            if b >= 3:
                Q_hi[idx] = (Q_hi[b * nt] + np.cumsum(drs[idx - 1] * F_hi[idx - 1 - nt])) * up
                Q_lo[idx] = (Q_lo[b * nt] + np.cumsum(drs[idx - 1] * F_lo[idx - nt])) * dn
                f_hi[idx], f_lo[idx] = Q_hi[idx] / rs[idx] * up, Q_lo[idx] / rs[idx] * dn
        self.F1_lo, self.F1_hi, self.f1_lo, self.f1_hi = F_lo, F_hi, f_lo, f_hi
        # ---- linear
        U_lo = np.full(M + 1, C2 * dn)
        U_hi = np.full(M + 1, C2 * up)              # s F2
        W_lo = np.zeros(M + 1)
        W_hi = np.zeros(M + 1)                      # s f2
        k = slice(2 * nt, 4 * nt + 1)
        w = C2 * np.log(np.maximum(s[k] - 1, 1.0))
        W_lo[k], W_hi[k] = w * dn, w * up
        W_lo[2 * nt] = 0.0
        F_lo, F_hi = U_lo / s, U_hi / s
        f_lo, f_hi = W_lo / s, W_hi / s
        d = 1.0 / nt
        for b in range(3, smax):
            idx = np.arange(b * nt + 1, (b + 1) * nt + 1)
            U_hi[idx] = (U_hi[b * nt] + np.cumsum(d * f_hi[idx - nt])) * up
            U_lo[idx] = (U_lo[b * nt] + np.cumsum(d * f_lo[idx - 1 - nt])) * dn
            F_hi[idx], F_lo[idx] = U_hi[idx] / s[idx] * up, U_lo[idx] / s[idx] * dn
            if b >= 4:
                W_hi[idx] = (W_hi[b * nt] + np.cumsum(d * F_hi[idx - 1 - nt])) * up
                W_lo[idx] = (W_lo[b * nt] + np.cumsum(d * F_lo[idx - nt])) * dn
                f_hi[idx], f_lo[idx] = W_hi[idx] / s[idx] * up, W_lo[idx] / s[idx] * dn
        self.F2_lo, self.F2_hi, self.f2_lo, self.f2_hi = F_lo, F_hi, f_lo, f_hi
        self.M = M

    # floor / ceil node of s (s is an array); nodes beyond SMAX are clipped, see the callers
    def _fl(self, s):
        return np.clip(np.floor(s * self.nt * (1 - 1e-14)).astype(np.int64), 0, self.M)

    def _ce(self, s):
        return np.clip(np.ceil(s * self.nt * (1 + 1e-14)).astype(np.int64), 0, self.M)

    # upper bounds of F (decreasing): value at the floor node; lower bounds: ceil node, and 1 beyond SMAX
    def F1_up(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 2.0, C1 / np.sqrt(np.maximum(s, 1e-300)) * (1 + 1e-13), self.F1_hi[self._fl(s)])

    def F1_dn(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 2.0, C1 / np.sqrt(np.maximum(s, 1e-300)) * (1 - 1e-13),
                        np.where(s >= self.smax, 1.0, self.F1_lo[self._ce(s)]))

    def f1_dn(self, s):
        s = np.asarray(s, float)
        t = np.clip(s, 1.0, 3.0)
        cl = C1 * np.log(np.sqrt(t) + np.sqrt(t - 1.0)) / np.sqrt(t) * (1 - 1e-13)
        return np.where(s < 1.0, 0.0, np.where(s <= 3.0, cl, self.f1_lo[self._fl(s)]))

    def f1_up(self, s):
        s = np.asarray(s, float)
        t = np.clip(s, 1.0, 3.0)
        cl = C1 * np.log(np.sqrt(t) + np.sqrt(t - 1.0)) / np.sqrt(t) * (1 + 1e-13)
        return np.where(s < 1.0, 0.0, np.where(s <= 3.0, cl, np.where(s >= self.smax, 1.0, self.f1_hi[self._ce(s)])))

    def F2_up(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 3.0, C2 / np.maximum(s, 1e-300) * (1 + 1e-13), self.F2_hi[self._fl(s)])

    def F2_dn(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 3.0, C2 / np.maximum(s, 1e-300) * (1 - 1e-13),
                        np.where(s >= self.smax, 1.0, self.F2_lo[self._ce(s)]))

    def f2_dn(self, s):
        s = np.asarray(s, float)
        t = np.clip(s, 2.0, 4.0)
        cl = C2 * np.log(t - 1.0) / t * (1 - 1e-13)
        return np.where(s < 2.0, 0.0, np.where(s <= 4.0, cl, self.f2_lo[self._fl(s)]))

    def f2_up(self, s):
        s = np.asarray(s, float)
        t = np.clip(s, 2.0, 4.0)
        cl = C2 * np.log(t - 1.0) / t * (1 + 1e-13)
        return np.where(s < 2.0, 0.0, np.where(s <= 4.0, cl, np.where(s >= self.smax, 1.0, self.f2_hi[self._ce(s)])))


_ST = None


def tables():
    global _ST
    if _ST is None:
        _ST = SieveTables()
    return _ST


# ----------------------------------------------------------------------------------------------------------------
# 2. constants
# ----------------------------------------------------------------------------------------------------------------
def prime_list(n):
    sv = np.ones(n + 1, dtype=bool)
    sv[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = False
    return np.nonzero(sv)[0]


_CONST = {}


def constants(pmax=10 ** 7):
    """A = C kappa e^{-3 gamma/2}/4 and K = 2 c_j c_2^2 c_3^* for the shift a = 2 (primes up to pmax; every factor is
    1 + O(p^-2) with |log factor| <= 5/p^2 for p > 100, so the tail changes each product by at most exp(+-5/pmax))."""
    if pmax in _CONST:
        return _CONST[pmax]
    p = prime_list(pmax).astype(np.float64)
    p1, p3 = p[p % 4 == 1], p[p % 4 == 3]
    lg = lambda v: float(np.exp(np.sum(np.log(v))))
    C = 2.25 * lg(1 - (3 * p3[p3 > 3] - 1) / (p3[p3 > 3] - 1) ** 3) * lg(1 - 1 / (p1 - 1) ** 2)
    K3 = lg(1 - 1 / p3 ** 2)
    kappa = math.sqrt(math.pi / 2 * K3)
    pp = p[p > 3]
    cj = lg(1 - 1 / (pp - 2) ** 2)
    c2 = lg(1 - 1 / (p[p > 2] - 1) ** 2)
    kp = (1 - 1 / (p1 - 1) ** 2) ** 2 * (1 - 1 / (p1 - 2) ** 2)
    c3s = math.sqrt(K3 / 2) * lg((1 - 1 / p1) * (1 + p1 ** 2 / ((p1 - 1) ** 3 * kp)))
    A = C * kappa * math.exp(-1.5 * GAMMA) / 4
    K = 2 * cj * c2 ** 2 * c3s
    tail = math.exp(12.0 / pmax)
    out = dict(C=C, kappa=kappa, cj=cj, c2=c2, c3s=c3s, A=A, K=K, A_lo=A / tail, A_hi=A * tail, K_hi=K * tail,
               K_lo=K / tail)
    _CONST[pmax] = out
    return out


def C_theta_hi(t1, n=20000):
    """upper bound of C(theta1) = int_0^{1-2 theta1} log((1-b-theta1)/theta1) db/(b^{1/2}(1-b)); with b = u^2 the
    integrand is 2 log((1-u^2-theta1)/theta1)/(1-u^2): a decreasing factor times an increasing factor."""
    u = np.linspace(0.0, math.sqrt(1 - 2 * t1), n + 1)
    lg = np.log(np.maximum((1 - u * u - t1) / t1, 1.0))
    w = 2.0 / (1 - u * u)
    return float(np.sum(lg[:-1] * w[1:] * np.diff(u))) * (1 + 1e-12)


def C_theta(t1, n=4000):
    u = np.linspace(0.0, math.sqrt(1 - 2 * t1), n + 1)
    y = 2 * np.log(np.maximum((1 - u * u - t1) / t1, 1.0)) / (1 - u * u)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(u)))


def W_free(t2, s=None):
    """free switching levels: (e^-gamma F2(s)/2) / (theta2 (1/2 - s theta2)), upper bound; s = None: minimise over a grid"""
    T = tables()
    if s is None:
        ss = np.arange(1.0, min(0.49 / t2, SMAX - 0.01), 0.01)
    else:
        ss = np.array([float(s)])
    w = (math.exp(-GAMMA) * T.F2_up(ss) / 2) / (t2 * (0.5 - ss * t2)) * (1 + 1e-12)
    k = int(np.argmin(w))
    return float(w[k]), float(ss[k])


def W_capped(t1, t2):
    return 1.0 / (t2 * min(t1, 0.5 - t2))


# ----------------------------------------------------------------------------------------------------------------
# 3. rectangle bounds (the vector sieve of [NX, Proposition 3.10]) and the Buchstab recursion on the inert primes
# ----------------------------------------------------------------------------------------------------------------
S2_LOW = np.concatenate([np.arange(2.0, 8.0, 0.125), np.geomspace(8.0, 40.0, 28)])     # splittings for the lower bound
S2_UPP = np.concatenate([np.arange(1.0, 8.0, 0.125), np.geomspace(8.0, 40.0, 28)])     # ... and for the upper bound


class Engine:
    """Tables on the grid  X_i = i h (levels, h = 1/(2n)),  a_j = a_0 rho^j (exponents of the inert sifting range).

    Gauge: u = a^{-1/2} Phi, v = a^{-1/2} G, so that  S(W_d; x^a, x^e) / (X g(d) V_2(x^e) c (log x)^{-1/2})
    lies between u - o(1) and v + o(1), where V_1(x^a) = c a^{-1/2} (log x)^{-1/2}.
        U[i, j]  is a lower bound valid for every level >= X_i and every range <= a_j,
        V[i, j]  is an upper bound valid for every level >= X_i and every range >= a_j.
    Buchstab on the cell [a_l, a_{l+1}):  sum_{q in cell} S(W_{dq}; q, .) is at most  (1/2) log(a_{l+1}/a_l) V[i', l]  and
    at least (1/2) log(a_{l+1}/a_l) U[i', l+1],  where X_{i'} <= X_i - a_{l+1} (the level left after dividing by q)."""

    def __init__(self, e, n=800, rho=1.002, a_min=0.004, anchor=0.4, verbose=False, store=True, kmain=0.5, kaux=1):
        self.e, self.n, self.h = e, n, 0.5 / n
        self.store = store          # keep the rectangle tables (needed for the k-step iteration)
        self.kmain = kmain          # dimension of the main sieve: 1/2 (inert primes of p - 1) or 1 (switched term)
        j_lo = int(math.floor(math.log(a_min / anchor) / math.log(rho)))
        j_hi = int(math.floor(math.log(0.4999 / anchor) / math.log(rho)))
        self.a = anchor * rho ** np.arange(j_lo, j_hi + 1, dtype=np.float64)
        self.j_anchor = -j_lo
        self.m = len(self.a)
        self.ra = self.a ** -kmain                                           # V_main(x^a) is proportional to a^-kappa
        self.lw = kmain * np.log(self.a[1:] / self.a[:-1]) * (1 + 1e-13)     # kappa log(a_{l+1}/a_l), l = 0..m-2
        self.lw_lo = kmain * np.log(self.a[1:] / self.a[:-1]) * (1 - 1e-13)  # rounded down, for the upper bounds (4.3)
        self.X = np.arange(n + 1) * self.h
        T = tables()
        self.T = T
        self.kaux = kaux            # dimension of the auxiliary sieve: 1 (one shift) or 2 (the product of two shifts)
        if kaux == 1:
            self.s2L, self.s2U = S2_LOW, S2_UPP
            self.f2L, self.F2L = T.f2_dn(self.s2L), T.F2_up(self.s2L)
            self.F2U = T.F2_up(self.s2U)
        else:
            K = k2tables()
            tail = np.geomspace(12.0, 60.0, 24)
            self.s2L = np.concatenate([np.arange(K.beta + 1e-9, 12.0, 0.1), tail])
            self.s2U = np.concatenate([np.arange(K.beta - 1 + 1e-9, 12.0, 0.1), tail])
            self.f2L, self.F2L, self.F2U = K.f_dn(self.s2L), K.F_up(self.s2L), K.F_up(self.s2U)
        self.psi_m = None
        self.verbose = verbose

    # ---- rectangle obstacles at level X_i:  psi^- = a^{-1/2} Phi_0 (lower),  psi^+ = a^{-1/2} G_0 (upper)
    def rect_row(self, i):
        T, e, a = self.T, self.e, self.a
        X = self.X[i]
        m = self.m
        half = self.kmain == 0.5
        Fm_up, Fm_dn, fm_dn = (T.F1_up, T.F1_dn, T.f1_dn) if half else (T.F2_up, T.F2_dn, T.f2_dn)
        beta = 1.0 if half else 2.0                                       # sifting limit of the main sieve
        lo = np.zeros(m)
        ok2 = self.s2L <= (X - beta * a.min()) / e
        if ok2.any():
            s2 = self.s2L[ok2]
            s1 = (X - e * s2[None, :]) / a[:, None]                      # main variable, must be >= beta
            F1u, F1d, f1d = Fm_up(s1), Fm_dn(s1), fm_dn(s1)
            val = self.f2L[ok2][None, :] * F1d - self.F2L[ok2][None, :] * (F1u - f1d)
            val = np.where(s1 >= beta, val, 0.0)
            lo = np.maximum(0.0, val.max(axis=1))
        up = np.full(m, np.inf)
        ok2 = self.s2U < X / e
        if ok2.any():
            s2 = self.s2U[ok2]
            s1 = (X - e * s2[None, :]) / a[:, None]                      # > 0 (half-dimensional), >= 1 (linear)
            val = self.F2U[ok2][None, :] * Fm_up(np.maximum(s1, 1e-12))
            up = np.where(s1 >= beta - 1.0, np.where(s1 > 0, val, np.inf), np.inf).min(axis=1)
        return lo * self.ra * (1 - 1e-12), up * self.ra * (1 + 1e-12)

    def rect_tables(self):
        if not self.store:
            return None, None
        if self.psi_m is None:
            n, m = self.n, self.m
            self.psi_m = np.zeros((n + 1, m), np.float32)
            self.psi_p = np.full((n + 1, m), np.inf, np.float32)
            for i in range(1, n + 1):
                lo, up = self.rect_row(i)
                self.psi_m[i] = np.nextafter(lo.astype(np.float32), np.float32(-np.inf)).clip(min=0)
                self.psi_p[i] = np.nextafter(up.astype(np.float32), np.float32(np.inf))
        return self.psi_m, self.psi_p

    def _levels(self, i):
        """index i' of the largest grid level <= X_i - a_{l+1}, for l = 0..m-2 (negative: no level left)"""
        return np.floor((self.X[i] - self.a[1:]) / self.h - 1e-9).astype(np.int64)

    def sweep(self, Uprev=None, Vprev=None):
        """One pass over the levels.  With Uprev = Vprev = None the delayed values are read from the tables being
        built (rows below i are complete): this is the fixed point of the recursion on the grid.  Otherwise the
        delayed values are read from (Uprev, Vprev): one more Buchstab step."""
        n, m = self.n, self.m
        psi_m, psi_p = self.rect_tables()
        U = np.zeros((n + 1, m), np.float32)
        V = np.full((n + 1, m), np.inf, np.float32)
        Us, Vs = (U, V) if Uprev is None else (Uprev, Vprev)
        col = np.arange(m - 1)
        for i in range(1, n + 1):
            if psi_m is None:
                pm, pp = self.rect_row(i)
            else:
                pm, pp = psi_m[i].astype(np.float64), psi_p[i].astype(np.float64)
            if self.verbose and i % 400 == 0:
                print("      level %d/%d %s" % (i, n, clock()), flush=True)
            lev = self._levels(i)
            okl = lev >= 1
            levc = np.where(okl, lev, 0)
            cV = np.where(okl, Vs[levc, col].astype(np.float64), np.inf)          # upper bounds of the cell sums
            cU = np.where(okl, Us[levc, col + 1].astype(np.float64), 0.0)         # lower bounds of the cell sums
            bad = ~np.isfinite(cV)
            lbad = int(np.argmax(bad)) if bad.any() else m - 1                    # first cell that cannot be crossed
            Kg = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, self.lw * cV))])
            Kp = np.concatenate([[0.0], np.cumsum(self.lw_lo * cU)])
            x = pm + Kg
            row_u = np.maximum.accumulate(x) - Kg - PAD
            row_u[lbad + 1:] = pm[lbad + 1:]
            row_u = np.maximum(row_u, pm)
            ppf = np.where(np.isfinite(pp), pp, 1e300)
            row_v = np.minimum.accumulate(ppf + Kp) - Kp + PAD
            row_v = np.where(row_v > 1e200, np.inf, row_v)
            # monotone closures: lower bounds extend to smaller ranges and higher levels, upper bounds to larger
            # ranges and higher levels
            row_u = np.maximum.accumulate(row_u[::-1])[::-1]
            row_v = np.minimum.accumulate(row_v)
            u32 = np.nextafter(row_u.astype(np.float32), np.float32(-np.inf)).clip(min=0)
            v32 = np.nextafter(row_v.astype(np.float32), np.float32(np.inf))
            U[i] = np.maximum(u32, U[i - 1])
            V[i] = np.minimum(v32, V[i - 1])
        return U, V

    def rect_UV(self):
        """the rectangle bounds themselves, with the monotone closures (step k = 0)"""
        psi_m, psi_p = self.rect_tables()
        U = np.maximum.accumulate(psi_m[:, ::-1], axis=1)[:, ::-1]
        V = np.minimum.accumulate(psi_p, axis=1)
        U = np.maximum.accumulate(U, axis=0)
        V = np.minimum.accumulate(V, axis=0)
        return U, V

    # ---- the criterion at theta1 = a_j:  main = A u / theta2,  Richert = (A/theta2) int (1/alpha - 1/theta) v d alpha
    def richert_tables(self, V, j):
        """cells alpha in [1/2 - X_{i+1}, 1/2 - X_i], on which v <= V[i, j]; returns the cell boundaries and the
        cumulative sums S1(i0) = sum_{i >= i0} V_i log(alpha_i/alpha_{i+1}), S2(i0) = sum_{i >= i0} V_i (alpha_i - alpha_{i+1}),
        the sums being restricted to alpha >= theta2."""
        n, e = self.n, self.e
        al_hi = 0.5 - self.X[:n]                # upper end of cell i, i = 0..n-1
        al_lo = np.maximum(0.5 - self.X[1:n + 1], e)
        v = V[:n, j].astype(np.float64)
        ok = al_hi > e
        t1 = np.where(ok, v * np.log(np.where(ok, al_hi / al_lo, 1.0)), 0.0)
        t2 = np.where(ok, v * (al_hi - al_lo), 0.0)
        t1 = np.where(np.isfinite(t1), t1, np.inf)
        S1 = np.cumsum(t1[::-1])[::-1]
        S2 = np.cumsum(np.where(np.isfinite(t2), t2, 0.0)[::-1])[::-1]
        return al_hi, S1, S2

    def J_of(self, V, j, theta):
        """upper bound of J = int_{theta2}^{theta} (1/alpha - 1/theta) v(1/2 - alpha) d alpha at theta1 = a_j, for a
        theta that is a cell boundary 1/2 - X_i"""
        al_hi, S1, S2 = self.richert_tables(V, j)
        i0 = int(round((0.5 - theta) / self.h))
        assert abs(al_hi[i0] - theta) < 1e-12, "theta must be of the form 1/2 - i h"
        return float((S1[i0] - S2[i0] / theta) * (1 + 1e-12))

    def best(self, U, V, j, Tsw, th_min=0.05, th_max=0.45):
        """minimise 1/lambda + 1/theta subject to u - T - lambda J >= 0, theta on the cell boundaries (units of Sect. 4)"""
        u = float(U[self.n, j])
        if u <= Tsw:
            return None
        al_hi, S1, S2 = self.richert_tables(V, j)
        k = (al_hi >= th_min) & (al_hi <= th_max) & np.isfinite(S1)
        if not k.any():
            return None
        th = al_hi[k]
        J = self.kaux * (S1[k] - S2[k] / th) * (1 + 1e-12)          # upper bound of kaux int (1/alpha - 1/theta) v
        lam = (u - Tsw) / J
        val = 1 / lam + self.kaux / th
        q = int(np.argmin(val))
        return dict(val=float(val[q]), theta=float(th[q]), lam=float(lam[q]), u=u, T=Tsw, J=float(J[q]),
                    theta1=float(self.a[j]), theta2=self.e)


def scan_theta1(eng, U, V, mode, t1_lo=0.30, t1_hi=0.49, stride=1):
    """best point over the nodes theta1 = a_j; mode 'free' (Proposition 3.1) or 'capped' (levels of [NX])"""
    best = None
    wfree = None
    for j in range(0, eng.m, stride):
        t1 = float(eng.a[j])
        if t1 < t1_lo or t1 > t1_hi or t1 + 2 * eng.e > 0.5:
            continue
        if mode == "free":
            if wfree is None:
                wfree = T_free(0.4, eng.e)[1]
            Tsw = T_free(t1, eng.e, s=wfree)[0]
        else:
            Tsw = T_capped(t1, eng.e)
        b = eng.best(U, V, j, Tsw)
        if b and (best is None or b["val"] < best["val"]):
            best = dict(b, j=j)
    return best


# ----------------------------------------------------------------------------------------------------------------
# 4. the criterion in normalised units
# ----------------------------------------------------------------------------------------------------------------
# With A = C kappa e^{-3 gamma/2}/4 (main term) and K = 2 c_j c_2^2 c_3^* (switched term) one has the identity
#       K / A = 2 e^{3 gamma/2} / sqrt(pi) = e^gamma C1                       (KAPPA0 below),
# all Euler products cancel.  Dividing the criterion of [NX, Section 5] by A / theta2 gives
#       M = u - T - lambda J > 0,      u = Phi / sqrt(theta1),
#       T = (C1/2) C(theta1) F2(s) / (1/2 - s theta2)            (free levels, any 1 <= s < 1/(2 theta2)),
#       T = e^gamma C1 C(theta1) / min(theta1, 1/2 - theta2)       (levels capped as in [NX]),
#       J = int_{theta2}^{theta} (1/alpha - 1/theta) v(1/2 - alpha) d alpha,   v = G / sqrt(theta1).
KAPPA0 = EG * C1


def T_free(t1, t2, s=None, ncell=4000):
    T = tables()
    ss = np.arange(1.0, min(0.49 / t2, SMAX - 0.01), 0.01) if s is None else np.array([float(s)])
    w = T.F2_up(ss) / (0.5 - ss * t2)
    k = int(np.argmin(w))
    return 0.5 * C1 * C_theta_hi(t1, ncell) * float(w[k]) * (1 + 1e-12), float(ss[k])


def T_capped(t1, t2, ncell=4000):
    return KAPPA0 * C_theta_hi(t1, ncell) / min(t1, 0.5 - t2) * (1 + 1e-12)


def rect_point(t1, t2, X=0.5, ds=0.005):
    """one-sided rectangle bounds at a single point: u_0 = Phi_0/sqrt(theta1) (lower) and v_0 = G_0/sqrt(theta1) (upper)
    at level X; splittings on a grid of step ds"""
    T = tables()
    X = np.atleast_1d(np.asarray(X, float))
    s2 = np.arange(2.0, min((X.max() - t1) / t2, SMAX * 3.0) + 1e-12, ds)
    lo = np.zeros(len(X))
    if len(s2):
        s1 = (X[:, None] - t2 * s2[None, :]) / t1
        val = T.f2_dn(s2)[None, :] * T.F1_dn(s1) - T.F2_up(s2)[None, :] * (T.F1_up(s1) - T.f1_dn(s1))
        lo = np.maximum(0.0, np.where(s1 >= 1.0, val, 0.0).max(axis=1))
    s2 = np.concatenate([np.arange(1.0, 10.0, ds), np.geomspace(10.0, 60.0, 200)])
    s1 = (X[:, None] - t2 * s2[None, :]) / t1
    up = np.where(s1 > 0, T.F2_up(s2)[None, :] * T.F1_up(np.maximum(s1, 1e-12)), np.inf).min(axis=1)
    return lo / math.sqrt(t1) * (1 - 1e-12), up / math.sqrt(t1) * (1 + 1e-12)


def rect_criterion(t1, t2, mode, ncell=1500):
    """rectangle bounds (no Buchstab step): returns (u, T, alpha-nodes, cumulative Richert sums) for the point"""
    u = float(rect_point(t1, t2)[0][0])
    Tsw = T_free(t1, t2)[0] if mode == "free" else T_capped(t1, t2)
    al = np.geomspace(t2, 0.46, ncell + 1)
    v = rect_point(t1, t2, X=0.5 - al[1:])[1]          # level decreases with alpha: v at the upper end of each cell
    return u, Tsw, al, v


def rect_best(t1, t2, mode):
    u, Tsw, al, v = rect_criterion(t1, t2, mode)
    if u <= Tsw:
        return None
    S1 = np.concatenate([[0.0], np.cumsum(v * np.log(al[1:] / al[:-1]))])
    S2 = np.concatenate([[0.0], np.cumsum(v * np.diff(al))])
    th = al[1:]
    J = (S1[1:] - S2[1:] / th) * (1 + 1e-12)
    ok = (th > 0.03) & np.isfinite(J)
    lam = (u - Tsw) / J[ok]
    val = 1 / lam + 1 / th[ok]
    q = int(np.argmin(val))
    return dict(val=float(val[q]), theta=float(th[ok][q]), lam=float(lam[q]), u=u, T=Tsw, J=float(J[ok][q]), theta1=t1,
                theta2=t2)


def fmtn(b):
    if b is None:
        return "no admissible point"
    return ("1/lam+1/theta = %.4f => r = %d | theta1 %.5f theta2 %.5f theta %.4f lambda %.5f | u %.5f T %.5f J %.5f"
            % (b["val"], math.ceil(b["val"] - 1e-12) - 1, b["theta1"], b["theta2"], b["theta"], b["lam"], b["u"], b["T"], b["J"]))


# ----------------------------------------------------------------------------------------------------------------
# 5. LEVEL 1: free switching levels, explicit witnesses in interval arithmetic
# ----------------------------------------------------------------------------------------------------------------
def iv_setup():
    from mpmath import iv, mp
    iv.dps = 30
    mp.dps = 30
    return iv


def iv_monotone_sum(iv, g, a, b, n):
    """enclosure of int_a^b g for an increasing g >= 0: h sum g(x_i) <= int <= h sum g(x_{i+1})"""
    h = (iv.mpf(b) - iv.mpf(a)) / n
    tot = iv.mpf(0)
    for i in range(n):
        tot += g(iv.mpf(a) + i * h)
    lo = (tot * h).a
    hi = ((tot - g(iv.mpf(a)) + g(iv.mpf(b))) * h).b
    return iv.mpf([lo, hi])


def iv_F2(iv, s, n):
    """F2(s) = 2 e^gamma (1 + int_2^{s-1} log(u-1)/u du) / s  for 3 <= s <= 5 (log(u-1)/u increases on [2, 4])"""
    c2 = 2 * iv.exp(iv.euler)
    if s == 3:
        return c2 / 3
    J = iv_monotone_sum(iv, lambda u: iv.log(u - 1) / u, 2, iv.mpf(s) - 1, n)
    return c2 * (1 + J) / iv.mpf(s)


def iv_f2(iv, s, n):
    """f2(s) = 2 e^gamma (log(s-1) + int_2^{s-2} (log(v-1)/v) log((s-1)/(v+1)) dv) / s  for 4 <= s <= 5; on a cell the
    integrand lies between g(v_i) log((s-1)/(v_{i+1}+1)) and g(v_{i+1}) log((s-1)/(v_i+1)), g(v) = log(v-1)/v"""
    c2 = 2 * iv.exp(iv.euler)
    s = iv.mpf(s)
    if s == 4:
        return c2 * iv.log(3) / 4
    a, b = iv.mpf(2), s - 2
    h = (b - a) / n
    lo = iv.mpf(0)
    hi = iv.mpf(0)
    g0 = iv.mpf(0)
    L0 = iv.log((s - 1) / 3)
    for i in range(n):
        v1 = a + (i + 1) * h
        g1 = iv.log(v1 - 1) / v1
        L1 = iv.log((s - 1) / (v1 + 1))
        lo += g0 * L1
        hi += g1 * L0
        g0, L0 = g1, L1
    J = iv.mpf([(lo * h).a, (hi * h).b])
    return c2 * (iv.log(s - 1) + J) / s


def iv_Ctheta(iv, t1, n):
    """C(theta1) = int_0^{sqrt(1-2 theta1)} 2 log((1-u^2-theta1)/theta1) du / (1-u^2): decreasing times increasing"""
    t1 = iv.mpf(t1)
    U = iv.sqrt(1 - 2 * t1)
    h = U / n
    lo = iv.mpf(0)
    hi = iv.mpf(0)
    l0 = iv.log((1 - t1) / t1)
    w0 = iv.mpf(2)
    for i in range(n):
        u1 = (i + 1) * h
        if i == n - 1:
            l1 = iv.mpf(0)
        else:
            l1 = iv.log((1 - u1 * u1 - t1) / t1)
        w1 = 2 / (1 - u1 * u1)
        lo += l1 * w0
        hi += l0 * w1
        l0, w0 = l1, w1
    return iv.mpf([(lo * h).a, (hi * h).b])


def witness_weighted(iv, t1, t2, th, lam, s, n=4000):
    """Normalised by C1:  M/C1 = (1/sqrt c)[f2(s) - F2(s)(1 - l0)] - F2(s) C(theta1)/(2c) - lam F2(s) Psi,
       c = 1/2 - s theta2, s0 = c/theta1 in [1, 2], l0 = log(sqrt s0 + sqrt(s0 - 1)),
       Psi = int_{theta2}^{theta} (1/alpha - 1/theta) d alpha / sqrt(c - alpha)   (closed form)."""
    t1, t2, th, lam, s = [iv.mpf(str(v)) for v in (t1, t2, th, lam, s)]
    c = iv.mpf(1) / 2 - s * t2
    s0 = c / t1
    assert s0.a >= 1 and s0.b <= 2 and (c - th).a > 0
    l0 = iv.log(iv.sqrt(s0) + iv.sqrt(s0 - 1))
    F2s, f2s = iv_F2(iv, s, n), iv_f2(iv, s, n)
    Cth = iv_Ctheta(iv, t1, n)
    rc, ta, tb = iv.sqrt(c), iv.sqrt(c - t2), iv.sqrt(c - th)
    Psi = iv.log((rc + ta) * (rc - tb) / ((rc - ta) * (rc + tb))) / rc - 2 * (ta - tb) / th
    main = (f2s - F2s * (1 - l0)) / rc
    sw = F2s * Cth / (2 * c)
    rich = F2s * Psi
    M = main - sw - lam * rich
    return dict(F2=F2s, f2=f2s, C=Cth, Psi=Psi, main=main, sw=sw, rich=rich, M=M, s0=s0, l0=l0, c=c)


def witness_unweighted(iv, t1, t2, sf, sw_, n=20000):
    """G/C1 = (1/sqrt c_f)[f2(s_f) - F2(s_f)(1 - l0)] - F2(s_W) C(theta1)/(2 (1/2 - s_W theta2))"""
    t1, t2, sf, sw_ = [iv.mpf(str(v)) for v in (t1, t2, sf, sw_)]
    c = iv.mpf(1) / 2 - sf * t2
    s0 = c / t1
    assert s0.a >= 1 and s0.b <= 2
    l0 = iv.log(iv.sqrt(s0) + iv.sqrt(s0 - 1))
    F2f, f2f, F2w = iv_F2(iv, sf, n), iv_f2(iv, sf, n), iv_F2(iv, sw_, n)
    Cth = iv_Ctheta(iv, t1, 10 * n)
    main = (f2f - F2f * (1 - l0)) / iv.sqrt(c)
    sw = F2w * Cth / (2 * (iv.mpf(1) / 2 - sw_ * t2))
    return dict(F2f=F2f, f2f=f2f, F2w=F2w, C=Cth, main=main, sw=sw, G=main - sw)


def ivs(x, d=8):
    return "[%.*f, %.*f]" % (d, float(x.a), d, float(x.b))


def level1(quick=False):
    open_log("deep_level1.txt")
    say("LEVEL 1.  Free switching levels (rectangle bounds of the vector sieve)   %s" % clock())
    say("=" * 110)
    c = constants(10 ** 6 if quick else 10 ** 7)
    say("constants (shift a = 2):  C = %.6f  kappa = %.6f  A = C kappa e^{-3g/2}/4 = %.6f" % (c["C"], c["kappa"], c["A"]))
    say("                          c_j = %.6f  c_2 = %.6f  c_3^* = %.6f  K = 2 c_j c_2^2 c_3^* = %.6f" % (c["cj"], c["c2"], c["c3s"], c["K"]))
    say("identity K/A = 2 e^{3 gamma/2}/sqrt(pi):  K/A = %.9f,  2 e^{3g/2}/sqrt(pi) = %.9f" % (c["K"] / c["A"], KAPPA0))
    say("  (so the criterion does not involve any Euler product; units below: A x (log x)^{-5/2} / theta2)")
    T = tables()
    say("sieve tables: step 1/%d, enclosure widths at s = 5: F1 %.1e f1 %.1e F2 %.1e f2 %.1e" % (
        NT, T.F1_hi[5 * NT] - T.F1_lo[5 * NT], T.f1_hi[5 * NT] - T.f1_lo[5 * NT], T.F2_hi[5 * NT] - T.F2_lo[5 * NT],
        T.f2_hi[5 * NT] - T.f2_lo[5 * NT]))
    say("")
    say("(a) one-sided optimisation, rectangle bounds; theta1 in steps of 0.0005")
    grids = {"capped": [0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.002], "free": [0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008]}
    if quick:
        grids = {"capped": [0.0012], "free": [0.006]}
    for mode in ("capped", "free"):
        glob = None
        for t2 in grids[mode]:
            best = None
            lo, hi = (0.44, 0.49) if mode == "capped" else (0.37, 0.42)
            for t1 in np.arange(lo, hi + 1e-9, 0.0025):
                if t1 + 2 * t2 > 0.5:
                    break
                b = rect_best(float(t1), t2, mode)
                if b and (best is None or b["val"] < best["val"]):
                    best = b
            if best:
                for t1 in np.arange(best["theta1"] - 0.0025, best["theta1"] + 0.0025 + 1e-9, 0.0005):
                    if t1 + 2 * t2 > 0.5:
                        break
                    b = rect_best(float(t1), t2, mode)
                    if b and b["val"] < best["val"]:
                        best = b
            say("    %-6s theta2 %.4f: %s  %s" % (mode, t2, fmtn(best), clock()))
            if best and (glob is None or best["val"] < glob["val"]):
                glob = best
        say("  => %s levels: %s" % (mode, fmtn(glob)))
        if mode == "free" and glob:
            say("     (optimal s in T at theta2 = %.4f: %.2f)" % (glob["theta2"], T_free(glob["theta1"], glob["theta2"])[1]))
    say("")
    say("(b) unweighted criterion u - T > 0 (rectangle bounds): largest theta2")
    for mode in ("capped", "free"):
        def gmax(t2):
            best = -1e9
            for t1 in np.arange(0.33, 0.495, 0.001):
                if t1 + 2 * t2 > 0.5:
                    break
                u = float(rect_point(float(t1), t2)[0][0])
                Tsw = T_free(float(t1), t2)[0] if mode == "free" else T_capped(float(t1), t2)
                best = max(best, u - Tsw)
            return best
        lo, hi = 1e-3, 0.03
        for _ in range(12 if quick else 22):
            mid = 0.5 * (lo + hi)
            if gmax(mid) > 0:
                lo = mid
            else:
                hi = mid
        say("    %-6s: theta2* = %.6f, 1/theta2* = %.3f  %s" % (mode, lo, 1 / lo, clock()))
    say("")
    say("(c) explicit witness for r = 17, interval arithmetic (mpmath.iv, 30 digits)")
    iv = iv_setup()
    P = dict(t1="0.4", t2="0.006", th="0.19", lam="0.08", s="5")
    w = witness_weighted(iv, n=2000 if quick else 20000, **P)
    say("    (theta1, theta2, theta, lambda) = (%s, %s, %s, %s), all splittings with s2 = %s" % (P["t1"], P["t2"], P["th"], P["lam"], P["s"]))
    say("    c = 1/2 - 5 theta2 = %s, s0 = c/theta1 = %s, l0 = log(sqrt s0 + sqrt(s0-1)) = %s" % (ivs(w["c"], 4), ivs(w["s0"], 4), ivs(w["l0"])))
    say("    F2(5) in %s   f2(5) in %s" % (ivs(w["F2"]), ivs(w["f2"])))
    say("    C(theta1) in %s   Psi in %s" % (ivs(w["C"]), ivs(w["Psi"])))
    say("    main  (1/sqrt c)[f2 - F2 (1 - l0)]  in %s" % ivs(w["main"]))
    say("    switched  F2 C(theta1)/(2c)         in %s" % ivs(w["sw"]))
    say("    Richert   F2 Psi                    in %s" % ivs(w["rich"]))
    say("    M/C1 = main - switched - lambda Richert in %s   =>  %s" % (ivs(w["M"]), "POSITIVE" if w["M"].a > 0 else "NOT CERTIFIED"))
    v = 1 / float(P["lam"]) + 1 / float(P["th"])
    say("    1/lambda + 1/theta = %.5f < 18  =>  omega(p+2) <= 17   %s" % (v, clock()))
    say("")
    say("(d) explicit witness for the unweighted bound, interval arithmetic")
    for (t1, t2, sf, sw_) in (("0.3694", "0.02041", "4.1", "4"), ("0.368", "0.0202", "4.1", "4")):
        w = witness_unweighted(iv, t1, t2, sf, sw_, n=2000 if quick else 20000)
        say("    theta1 = %s, theta2 = %s (1/theta2 = %.4f), s_f = %s, s_W = %s:" % (t1, t2, 1 / float(t2), sf, sw_))
        say("      F2(s_f) %s f2(s_f) %s F2(s_W) %s C %s" % (ivs(w["F2f"]), ivs(w["f2f"]), ivs(w["F2w"]), ivs(w["C"])))
        say("      main %s  switched %s  G/C1 in [%.3e, %.3e]  =>  %s, omega(p+2) <= %d   %s" % (
            ivs(w["main"]), ivs(w["sw"]), float(w["G"].a), float(w["G"].b), "POSITIVE" if w["G"].a > 0 else "NOT CERTIFIED",
            math.floor(1 / float(t2)), clock()))
    say("done %s" % clock())


# ----------------------------------------------------------------------------------------------------------------
# 6. LEVEL 2: Buchstab's identity on the inert primes; k steps and the fixed point on the grid
# ----------------------------------------------------------------------------------------------------------------
KSTEPS = (0, 1, 2, 3, 4, 6, 8, 12)


def k_step_tables(eng, ks=KSTEPS):
    """yields (k, U, V) for the k-step iteration, k in ks, and finally ('inf', U, V) for the fixed point"""
    U, V = eng.rect_UV()
    k = 0
    for target in ks:
        while k < target:
            U, V = eng.sweep(U, V)
            k += 1
        yield k, U, V
    del U, V
    yield "inf", *eng.sweep()


def level2(quick=False):
    open_log("deep_level2.txt")
    say("LEVEL 2.  Buchstab's identity on the primes q = 3 (4) of p - 1, coupled with the linear sieve on p + a   %s" % clock())
    say("=" * 110)
    say("units: A x (log x)^{-5/2} / theta2;  criterion  M = u - T - lambda J > 0;  all values one-sided (see the header)")
    g = dict(n=400, rho=1.004, a_min=0.004) if quick else dict(n=1600, rho=1.001, a_min=0.002)
    say("")
    say("(a) free switching levels: min of 1/lambda + 1/theta after k Buchstab steps; grid n = %(n)d, rho = %(rho)s, a_min = %(a_min)s" % g)
    th2s = [0.006, 0.009] if quick else [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011]
    glob = {}
    for t2 in th2s:
        eng = Engine(t2, **g)
        row = []
        for k, U, V in k_step_tables(eng):
            b = scan_theta1(eng, U, V, "free", 0.36, 0.44)
            row.append((k, b))
            if b and (k not in glob or b["val"] < glob[k]["val"]):
                glob[k] = b
        say("    theta2 %.4f (m = %d): " % (t2, eng.m) + "  ".join("k=%s: %.3f" % (k, b["val"]) if b else "k=%s: -" % k for k, b in row) + "  " + clock())
        del eng
    say("  best over theta2:")
    for k in list(KSTEPS) + ["inf"]:
        say("    k = %-3s  %s" % (k, fmtn(glob.get(k))))
    say("")
    say("(b) levels capped as in [NX] (the setting of the Erratum), rectangle bounds and fixed point; same grid")
    for t2 in ([0.0012] if quick else [0.0010, 0.0012, 0.0016, 0.0020, 0.0025]):
        eng = Engine(t2, store=False, **g)
        U0, V0 = None, None
        U, V = eng.sweep()
        say("    theta2 %.4f: fixed point %s  %s" % (t2, fmtn(scan_theta1(eng, U, V, "capped", 0.44, 0.49)), clock()))
        del eng, U, V
    say("")
    best = glob["inf"]
    t2 = best["theta2"]
    t1 = round(best["theta1"], 3)
    say("(c) refinement of the grid at theta2 = %.4f, fixed point, theta1 = %.3f (a node of every grid)" % (t2, t1))
    grids = [(400, 1.004, 0.004), (800, 1.002, 0.004)] if quick else [(800, 1.002, 0.004), (1600, 1.001, 0.002), (3200, 1.0005, 0.001)]
    last = None
    for (n, rho, amin) in grids:
        eng = Engine(t2, n=n, rho=rho, a_min=amin, anchor=t1, store=False, verbose=True)
        U, V = eng.sweep()
        j = eng.j_anchor
        assert abs(eng.a[j] - t1) < 1e-12
        Tsw, s_opt = T_free(t1, t2, ncell=20000)
        b = eng.best(U, V, j, Tsw)
        say("    n = %4d, rho = %s, a_min = %s (m = %d): %s  %s" % (n, rho, amin, eng.m, fmtn(b), clock()))
        last = (eng, U, V, j, Tsw, s_opt, b)
    eng, U, V, j, Tsw, s_opt, b = last
    say("")
    say("(d) witness on the finest grid")
    theta = round(b["theta"] * 200) / 200             # a multiple of 0.005, hence a cell boundary 1/2 - i h
    J = eng.J_of(V, j, theta)
    u = float(U[eng.n, j])
    lam_max = (u - Tsw) / J
    lam = math.floor(lam_max * 0.997 * 1e4) / 1e4
    M = u - Tsw - lam * J
    say("    (theta1, theta2, theta, lambda) = (%.3f, %.4f, %.3f, %.4f),  s = %.2f in T" % (t1, t2, theta, lam, s_opt))
    say("    u >= %.6f   T <= %.6f   J <= %.6f   (lambda_max = %.5f)" % (u, Tsw, J, lam_max))
    say("    M = u - T - lambda J >= %.6f  => %s" % (M, "POSITIVE" if M > 0 else "NOT CERTIFIED"))
    say("    1/lambda + 1/theta = %.5f  =>  omega(p + a) <= %d" % (1 / lam + 1 / theta, math.ceil(1 / lam + 1 / theta - 1e-12) - 1))
    say("    rectangle bounds at the same point: u_0 >= %.6f, and J_0 <= %.6f" % (
        float(rect_point(t1, t2)[0][0]), rect_J(t1, t2, theta)))
    say("done %s" % clock())


def rect_J(t1, t2, theta, ncell=3000):
    al = np.geomspace(t2, theta, ncell + 1)
    v = rect_point(t1, t2, X=0.5 - al[1:])[1]
    return float(np.sum(v * (np.log(al[1:] / al[:-1]) - np.diff(al) / theta)) * (1 + 1e-12))


# ----------------------------------------------------------------------------------------------------------------
# 6b. CERTIFY: the fixed point of Theorem 1.2 with exact integer arithmetic in the recursion
# ----------------------------------------------------------------------------------------------------------------
# What is rigorous here, and how.
#   * Transcendental functions are evaluated only in interval arithmetic (mpmath.iv): the constants C1, C2, the
#     closed forms sqrt(s) f1(s) = C1 log(sqrt s + sqrt(s-1)) on [1, 3] and s f2(s) = C2 log(s-1) on [2, 4] at the
#     nodes s = i/NT, the integral C(theta1), F2(s) in T, and the logarithms in J.
#   * Everything else that is done in floating point consists of IEEE-754 additions, subtractions, multiplications,
#     divisions and square roots (correctly rounded, relative error <= 2^-53 each).  Every quantity is moved outwards
#     by a relative 1e-10 after each block of operations; the longest chain is a sum of NT = 40000 non-negative
#     terms (accumulated relative error below 5e-12).
#   * The Buchstab recursion (4.2)-(4.3) is carried out in exact integer arithmetic: table entries are integers in
#     units of 1e-9, the weights mu_l are integers in units of 1e-10 (rounded up in (4.2), down in (4.3)), each
#     product is rounded in the safe direction by an exact integer division, and sums, maxima and minima of integers
#     are exact.  mu_l = (1/2) log(1 + t) is bounded by t - t^2/2 + t^3/3 - t^4/4 <= log(1+t) <= t - t^2/2 + t^3/3.
#   * Upper bounds larger than VCAP are replaced by "no bound"; this is valid and keeps the products inside 64 bits.
SC = 10 ** 9
MU_SC = 10 ** 10
VCAP = 300
IINF = np.int64(2 ** 62)
RM = 1e-10


class CertTables:
    """as SieveTables, with the closed forms taken from interval arithmetic at the nodes (cached in data/)"""

    def __init__(self, nt=NT, smax=SMAX):
        self.nt, self.smax = nt, smax
        cache = os.path.join(DATA, "sieve_nodes_iv_%d.npz" % nt)
        if os.path.exists(cache):
            z = np.load(cache)
            Qn_lo, Qn_hi, Wn_lo, Wn_hi, cst = z["Q_lo"], z["Q_hi"], z["W_lo"], z["W_hi"], z["cst"]
        else:
            iv = iv_setup()
            down = lambda v: float(np.nextafter(float(v.a), -np.inf))
            up = lambda v: float(np.nextafter(float(v.b), np.inf))
            c1 = 2 * iv.sqrt(iv.exp(iv.euler) / iv.pi)
            c2 = 2 * iv.exp(iv.euler)
            Qn_lo, Qn_hi = np.zeros(2 * nt + 1), np.zeros(2 * nt + 1)
            Wn_lo, Wn_hi = np.zeros(2 * nt + 1), np.zeros(2 * nt + 1)
            for i in range(2 * nt + 1):
                r = iv.mpf(i) / nt
                q = c1 * iv.log(iv.sqrt(1 + r) + iv.sqrt(r))           # sqrt(s) f1(s), s = 1 + i/nt
                Qn_lo[i], Qn_hi[i] = max(0.0, down(q)), up(q)
                if i > 0:
                    w = c2 * iv.log(1 + r)                             # s f2(s), s = 2 + i/nt
                    Wn_lo[i], Wn_hi[i] = max(0.0, down(w)), up(w)
            cst = np.array([down(c1), up(c1), down(c2), up(c2)])
            np.savez(cache, Q_lo=Qn_lo, Q_hi=Qn_hi, W_lo=Wn_lo, W_hi=Wn_hi, cst=cst)
        self.C1_lo, self.C1_hi, self.C2_lo, self.C2_hi = [float(v) for v in cst]
        M = nt * smax
        self.M = M
        s = np.arange(M + 1) / nt
        s[0] = 1e-300
        rs = np.sqrt(s)
        up, dn = 1 + RM, 1 - RM
        drs = (1.0 / nt) / (rs[1:] + rs[:-1])                           # sqrt(s_i) - sqrt(s_{i-1}), computed stably
        P_lo, P_hi = np.full(M + 1, self.C1_lo), np.full(M + 1, self.C1_hi)
        Q_lo, Q_hi = np.zeros(M + 1), np.zeros(M + 1)
        Q_lo[nt:3 * nt + 1], Q_hi[nt:3 * nt + 1] = Qn_lo, Qn_hi
        F_lo, F_hi = P_lo / rs * dn, P_hi / rs * up
        f_lo, f_hi = Q_lo / rs * dn, Q_hi / rs * up
        for b in range(2, smax):
            idx = np.arange(b * nt + 1, (b + 1) * nt + 1)
            P_hi[idx] = (P_hi[b * nt] + np.cumsum(drs[idx - 1] * f_hi[idx - nt])) * up
            P_lo[idx] = (P_lo[b * nt] + np.cumsum(drs[idx - 1] * f_lo[idx - 1 - nt])) * dn
            F_hi[idx], F_lo[idx] = P_hi[idx] / rs[idx] * up, P_lo[idx] / rs[idx] * dn
            if b >= 3:
                Q_hi[idx] = (Q_hi[b * nt] + np.cumsum(drs[idx - 1] * F_hi[idx - 1 - nt])) * up
                Q_lo[idx] = (Q_lo[b * nt] + np.cumsum(drs[idx - 1] * F_lo[idx - nt])) * dn
                f_hi[idx], f_lo[idx] = Q_hi[idx] / rs[idx] * up, Q_lo[idx] / rs[idx] * dn
        self.F1_lo, self.F1_hi, self.f1_lo = F_lo, F_hi, f_lo
        U_lo, U_hi = np.full(M + 1, self.C2_lo), np.full(M + 1, self.C2_hi)
        W_lo, W_hi = np.zeros(M + 1), np.zeros(M + 1)
        W_lo[2 * nt:4 * nt + 1], W_hi[2 * nt:4 * nt + 1] = Wn_lo, Wn_hi
        F_lo, F_hi = U_lo / s * dn, U_hi / s * up
        f_lo, f_hi = W_lo / s * dn, W_hi / s * up
        d = 1.0 / nt
        for b in range(3, smax):
            idx = np.arange(b * nt + 1, (b + 1) * nt + 1)
            U_hi[idx] = (U_hi[b * nt] + np.cumsum(d * f_hi[idx - nt])) * up
            U_lo[idx] = (U_lo[b * nt] + np.cumsum(d * f_lo[idx - 1 - nt])) * dn
            F_hi[idx], F_lo[idx] = U_hi[idx] / s[idx] * up, U_lo[idx] / s[idx] * dn
            if b >= 4:
                W_hi[idx] = (W_hi[b * nt] + np.cumsum(d * F_hi[idx - 1 - nt])) * up
                W_lo[idx] = (W_lo[b * nt] + np.cumsum(d * F_lo[idx - nt])) * dn
                f_hi[idx], f_lo[idx] = W_hi[idx] / s[idx] * up, W_lo[idx] / s[idx] * dn
        self.F2_lo, self.F2_hi, self.f2_lo = F_lo, F_hi, f_lo

    def _fl(self, s):
        return np.clip(np.floor(s * self.nt * (1 - 1e-12)).astype(np.int64), 0, self.M)

    def _ce(self, s):
        return np.clip(np.ceil(s * self.nt * (1 + 1e-12)).astype(np.int64), 0, self.M)

    def F1_up(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 2.0, self.C1_hi / np.sqrt(np.maximum(s, 1e-300)) * (1 + RM), self.F1_hi[self._fl(s)])

    def F1_dn(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 2.0, self.C1_lo / np.sqrt(np.maximum(s, 1e-300)) * (1 - RM),
                        np.where(s >= self.smax, 1.0, self.F1_lo[self._ce(s)]))

    def f1_dn(self, s):                 # f1 increasing: value at the floor node, also inside the closed-form range
        s = np.asarray(s, float)
        return np.where(s < 1.0, 0.0, self.f1_lo[self._fl(s)])

    def F2_up(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 3.0, self.C2_hi / np.maximum(s, 1e-300) * (1 + RM), self.F2_hi[self._fl(s)])

    def F2_dn(self, s):
        s = np.asarray(s, float)
        return np.where(s <= 3.0, self.C2_lo / np.maximum(s, 1e-300) * (1 - RM),
                        np.where(s >= self.smax, 1.0, self.F2_lo[self._ce(s)]))

    def f2_dn(self, s):
        s = np.asarray(s, float)
        return np.where(s < 2.0, 0.0, self.f2_lo[self._fl(s)])


class CertK2:
    """Dimension 2, certified.  beta is enclosed by a sign change of q(s) = s^3 - 6 s^2 + 9 s - 8/3 in interval
    arithmetic.  With G = F/A, g = f/A (the solution for A = 1): G = 1/s^2 on (0, beta + 1], s^2 g(s) =
    2 [log(t-1) - 1/(t-1)]_beta^s on [beta, beta + 2] (interval arithmetic at the nodes), then the delay equations
    as in CertTables.  Since f increases to 1 and F decreases to 1,  1/G(S) <= A <= 1/g(S)  for every S: this
    encloses A by the tables themselves."""

    def __init__(self, nt=NT, smax=SMAX + 2):
        self.nt, self.smax = nt, smax
        iv = iv_setup()
        b0 = K2Tables.BETA
        q = lambda s: s ** 3 - 6 * s ** 2 + 9 * s - iv.mpf(8) / 3
        lo, hi = iv.mpf(b0 - 1) - iv.mpf("1e-12"), iv.mpf(b0 - 1) + iv.mpf("1e-12")
        # q' = 3 (s - 1)(s - 3) > 0 for s > 3, so a sign change at some s > 3 locates the largest zero
        assert q(lo).b < 0 < q(hi).a and lo.a > 3
        self.b_lo, self.b_hi = float(lo.a) + 1 - 1e-13, float(hi.b) + 1 + 1e-13
        self.beta = b0
        M = nt * smax
        self.M = M
        s = np.arange(M + 1) / nt
        up, dn = 1 + RM, 1 - RM
        i0 = int(math.ceil(self.b_hi * nt))                 # first node >= beta
        i2 = int(math.floor((self.b_lo + 2) * nt))          # last node <= beta + 2
        cache = os.path.join(DATA, "sieve_nodes_iv_k2_%d.npz" % nt)
        if os.path.exists(cache):
            z = np.load(cache)
            Qn_lo, Qn_hi = z["Q_lo"], z["Q_hi"]
        else:
            B = iv.mpf([self.b_lo, self.b_hi])
            cB = iv.log(B - 1) - 1 / (B - 1)
            Qn_lo, Qn_hi = np.zeros(i2 - i0 + 1), np.zeros(i2 - i0 + 1)
            for k, i in enumerate(range(i0, i2 + 1)):
                x = iv.mpf(i) / nt
                v = 2 * (iv.log(x - 1) - 1 / (x - 1) - cB)
                Qn_lo[k] = max(0.0, float(np.nextafter(float(v.a), -np.inf)))
                Qn_hi[k] = float(np.nextafter(float(v.b), np.inf))
            np.savez(cache, Q_lo=Qn_lo, Q_hi=Qn_hi)
        P_lo, P_hi = np.full(M + 1, 1.0), np.full(M + 1, 1.0)          # s^2 G
        Q_lo, Q_hi = np.zeros(M + 1), np.zeros(M + 1)                  # s^2 g
        Q_lo[i0:i2 + 1], Q_hi[i0:i2 + 1] = Qn_lo, Qn_hi
        s2 = s * s
        s2[0] = 1.0
        G_lo, G_hi = P_lo / s2 * dn, P_hi / s2 * up
        g_lo, g_hi = Q_lo / s2 * dn, Q_hi / s2 * up
        ds2 = (1.0 / nt) * (s[1:] + s[:-1]) * up
        ds2l = (1.0 / nt) * (s[1:] + s[:-1]) * dn
        iP = int(math.floor((self.b_lo + 1) * nt))          # nodes <= iP have s^2 G = 1 exactly
        start = iP + 1
        while start <= M:
            idx = np.arange(start, min(start + nt, M + 1))
            P_hi[idx] = (P_hi[start - 1] + np.cumsum(ds2[idx - 1] * g_hi[idx - nt])) * up
            P_lo[idx] = (P_lo[start - 1] + np.cumsum(ds2l[idx - 1] * g_lo[idx - 1 - nt])) * dn
            G_hi[idx], G_lo[idx] = P_hi[idx] / s2[idx] * up, P_lo[idx] / s2[idx] * dn
            jdx = idx + (i2 - iP)
            jdx = jdx[jdx <= M]
            if len(jdx):
                Q_hi[jdx] = (Q_hi[jdx[0] - 1] + np.cumsum(ds2[jdx - 1] * G_hi[jdx - 1 - nt])) * up
                Q_lo[jdx] = (Q_lo[jdx[0] - 1] + np.cumsum(ds2l[jdx - 1] * G_lo[jdx - nt])) * dn
                g_hi[jdx], g_lo[jdx] = Q_hi[jdx] / s2[jdx] * up, Q_lo[jdx] / s2[jdx] * dn
            start += nt
        self.A_lo = 1.0 / G_hi[M] * dn
        self.A_hi = 1.0 / g_lo[M] * up
        self.F_hi = G_hi * self.A_hi * up
        self.f_lo = g_lo * self.A_lo * dn

    def F_up(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.floor(s * self.nt * (1 - 1e-12)).astype(np.int64), 0, self.M)
        return np.where(s <= self.b_lo + 1, self.A_hi / np.maximum(s, 1e-300) ** 2 * (1 + RM), self.F_hi[i])

    def f_dn(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.floor(s * self.nt * (1 - 1e-12)).astype(np.int64), 0, self.M)
        return np.where(s <= self.b_hi, 0.0, self.f_lo[i])


def certify(t1=0.398, t2=0.009, theta=0.19, lam=0.0935, s_sw="4.42", n=1600, rho=1.001, a_min=0.002, log=True, kaux=1):
    """Certificate of u - T - lambda J > 0 at one point, fixed point of (4.2)-(4.3) on the grid, integer arithmetic."""
    if log:
        open_log("deep_certificate.txt" if kaux == 1 else "deep_certificate_pairs.txt")
    say("CERTIFICATE for Theorem %s: fixed point on the grid n = %d, rho = %s, b_0 = %s   %s" % (
        "1.2" if kaux == 1 else "1.4 (two shifts, dimension 2)", n, rho, a_min, clock()))
    say("=" * 110)
    global _ST
    saved, _ST = _ST, CertTables()
    T = _ST
    say("sieve tables: closed forms at the nodes from interval arithmetic; C1 in [%.15f, %.15f]  %s" % (T.C1_lo, T.C1_hi, clock()))
    eng = Engine(t2, n=n, rho=rho, a_min=a_min, anchor=t1, store=False, kaux=kaux)
    _ST = saved
    eng.T = T
    if kaux == 1:
        eng.f2L, eng.F2L, eng.F2U = T.f2_dn(eng.s2L), T.F2_up(eng.s2L), T.F2_up(eng.s2U)
    else:
        K = CertK2()
        say("dimension 2: beta in [%.13f, %.13f], A in [%.7f, %.7f] (from 1/G(S) <= A <= 1/g(S), S = %d)  %s" % (
            K.b_lo, K.b_hi, K.A_lo, K.A_hi, K.smax, clock()))
        assert eng.s2L.min() > K.b_hi and eng.s2U.min() > K.b_hi - 1
        eng.f2L, eng.F2L, eng.F2U = K.f_dn(eng.s2L), K.F_up(eng.s2L), K.F_up(eng.s2U)
    a, m, h = eng.a, eng.m, eng.h
    j1 = eng.j_anchor
    assert a[j1] == t1
    ra = 1.0 / np.sqrt(a)
    eng.ra = np.ones(m)                                  # rect_row then returns Phi_0, G_0 without the gauge
    t = (a[1:] - a[:-1]) / a[:-1]
    t_hi, t_lo = t * (1 + 1e-12), t * (1 - 1e-12)
    mu_hi = 0.5 * (t_hi - t_lo ** 2 / 2 + t_hi ** 3 / 3) * (1 + 1e-12)
    mu_lo = 0.5 * (t_lo - t_hi ** 2 / 2 + t_lo ** 3 / 3 - t_hi ** 4 / 4) * (1 - 1e-12)
    MU_hi = np.ceil(mu_hi * MU_SC).astype(np.int64) + 1
    MU_lo = np.maximum(np.floor(mu_lo * MU_SC).astype(np.int64) - 1, 0)
    DIV = MU_SC
    assert int(MU_hi.max()) * VCAP * SC < 2 ** 63, 'products must fit into 64 bits'
    U = np.zeros((n + 1, m), np.int64)
    V = np.full((n + 1, m), IINF, np.int64)
    col = np.arange(m - 1)
    cap = VCAP * SC
    for i in range(1, n + 1):
        lo, up = eng.rect_row(i)
        pm = np.maximum(np.floor((lo * ra * (1 - RM) - 1e-12) * SC).astype(np.int64) - 1, 0)
        upg = up * ra * (1 + RM)
        pp = np.where(np.isfinite(upg) & (upg < VCAP), np.ceil(np.where(np.isfinite(upg), upg, 0.0) * SC).astype(np.int64) + 1, IINF)
        lev = np.floor((eng.X[i] - a[1:]) / h - 1e-9).astype(np.int64)
        okl = lev >= 1
        levc = np.where(okl, lev, 0)
        cV = np.where(okl, V[levc, col], IINF)
        cU = np.where(okl, U[levc, col + 1], 0)
        bad = cV >= cap
        lbad = int(np.argmax(bad)) if bad.any() else m - 1
        termV = np.where(bad, 0, -((-(MU_hi * np.where(bad, 0, cV))) // DIV))          # ceil of the exact product
        termU = (MU_lo * cU) // DIV                                                   # floor of the exact product
        Kg = np.concatenate([[0], np.cumsum(termV)])
        Kp = np.concatenate([[0], np.cumsum(termU)])
        row_u = np.maximum.accumulate(pm + Kg) - Kg
        row_u[lbad + 1:] = pm[lbad + 1:]
        row_u = np.maximum(row_u, pm)
        ppf = np.where(pp >= cap, IINF, pp)
        row_v = np.minimum.accumulate(ppf + Kp) - Kp
        row_v = np.where(row_v >= cap, IINF, row_v)
        row_u = np.maximum.accumulate(row_u[::-1])[::-1]
        row_v = np.minimum.accumulate(row_v)
        U[i] = np.maximum(row_u, U[i - 1])
        V[i] = np.minimum(row_v, V[i - 1])
        if i % 400 == 0:
            say("    level %d/%d  %s" % (i, n, clock()))
    # ---- the three terms, in interval arithmetic
    iv = iv_setup()
    u = iv.mpf(int(U[n, j1])) / SC
    i0 = int(round((0.5 - theta) / h))
    th = iv.mpf(1) / 2 - iv.mpf(i0) / (2 * n)
    assert abs(float(th.mid) - theta) < 1e-12
    e = iv.mpf(str(t2))
    J = iv.mpf(0)
    for i in range(i0, n):
        hi = iv.mpf(1) / 2 - iv.mpf(i) / (2 * n)
        lo_ = iv.mpf(1) / 2 - iv.mpf(i + 1) / (2 * n)
        if hi.b <= e.a:
            break
        if lo_.a < e.b:
            lo_ = e
        assert V[i, j1] < cap
        J += (iv.mpf(int(V[i, j1])) / SC) * (iv.log(hi / lo_) - (hi - lo_) / th)
    c1 = 2 * iv.sqrt(iv.exp(iv.euler) / iv.pi)
    sw = iv.mpf(s_sw)
    if kaux == 1:
        Fsw = iv_F2(iv, sw, 20000)
    else:
        assert float(sw.a) >= K.b_hi - 1
        Fsw = iv.mpf([1.0, float(K.F_up(np.array([float(sw.b)]))[0])])      # only the upper end is used
    Tsw = c1 / 2 * Fsw * iv_Ctheta(iv, str(t1), 20000) / (iv.mpf(1) / 2 - sw * e)
    J = kaux * J
    lam_iv = iv.mpf(str(lam))
    Mv = u - Tsw - lam_iv * J
    val = 1 / lam_iv + kaux / th
    say("point: (theta1, theta2, theta, lambda) = (%s, %s, %s, %s), s = %s in T; %d ranges" % (t1, t2, theta, lam, s_sw, m))
    say("    u  >= %.9f   (integer table entry %d / 1e9)" % (float(u.a), int(U[n, j1])))
    say("    T  <= %.9f   (interval arithmetic)" % float(Tsw.b))
    say("    %sJ <= %.9f   (integer table entries, logarithms in interval arithmetic)" % ("2" if kaux == 2 else " ", float(J.b)))
    say("    u - T - %slambda J >= %.6f   =>  %s" % ("2 " if kaux == 2 else "", float(Mv.a), "POSITIVE" if Mv.a > 0 else "NOT CERTIFIED"))
    say("    1/lambda + %d/theta <= %.6f  =>  omega <= %d   %s" % (kaux, float(val.b), math.ceil(float(val.b)) - 1, clock()))
    return float(Mv.a)


# ----------------------------------------------------------------------------------------------------------------
# 7. LEVEL 3: exploration beyond r = 15 (what the method can and cannot give)
# ----------------------------------------------------------------------------------------------------------------
def switched_fixed_point(t2, n, rho, a_min):
    """Upper bound for the pair of linear sieves of the switched term (primality of 2 m p1 p2 + 1, sifting range x^b;
    (2 m p1 p2 + 1 + a, P(x^theta2)) = 1) with Buchstab's identity on the first sieve.  Returns
    (w_rect, w_fix) in units of 2 e^gamma: the rectangle value min_s F2(s)/(1/2 - s theta2) and the fixed point
    min_b G(1/2; theta2, b)/b / (2 e^gamma);  T = (C1/2) C(theta1) w."""
    T = tables()
    ss = np.arange(1.0, min(0.49 / t2, SMAX - 0.01), 0.01)
    w_rect = float((T.F2_up(ss) / (0.5 - ss * t2)).min())
    eng = Engine(t2, n=n, rho=rho, a_min=a_min, store=False, kmain=1.0)
    U, V = eng.sweep()
    w_fix = float(V[eng.n].min()) / C2 * (1 + 1e-12)
    return w_rect, min(w_rect, w_fix)


def lp_weights(eng, V, j, R1, amax=0.40, cell=0.002):
    """General weights w(n) = 1 - sum_{q | n} psi(log q/log x), psi >= 0 a step function on cells of width `cell` in
    [theta2, amax), psi = 0 beyond.  Constraint: every R1 exponents >= theta2 with sum <= 1 have sum psi >= 1 (so that
    w(n) <= 0 when omega(n) >= R1); for a step function this is a knapsack problem on the left endpoints.
    Cost: upper bound of int psi(alpha) v(1/2 - alpha) d alpha/alpha.  Linear programme with cutting planes (HiGHS)."""
    from scipy.optimize import linprog
    h, e = eng.h, eng.e
    stride = max(1, int(round(cell / h)))
    d = h * stride
    nc = int(math.floor((amax - e) / d))
    lo = e + d * np.arange(nc)
    hi = lo + d
    lev = np.floor((0.5 - hi) / h - 1e-9).astype(int)
    w = V[lev, j].astype(float) * np.log(hi / lo)
    B = int(math.floor((1 - R1 * e) / d + 1e-9))

    def separate(psi):
        cost = np.append(psi, 0.0)                       # index nc: a prime with exponent >= amax, weight 0
        INF = 1e18
        dp = np.full((R1 + 1, B + 1), INF)
        dp[0, 0] = 0.0
        arg = np.zeros((R1 + 1, B + 1), np.int32)
        for t in range(1, R1 + 1):
            best = np.full(B + 1, INF)
            ba = np.zeros(B + 1, np.int32)
            for c in range(min(nc, B) + 1):
                cand = np.full(B + 1, INF)
                cand[c:] = dp[t - 1, :B + 1 - c] + cost[c]
                msk = cand < best
                best[msk] = cand[msk]
                ba[msk] = c
            dp[t], arg[t] = best, ba
        b = int(np.argmin(dp[R1]))
        val = float(dp[R1, b])
        tup = []
        for t in range(R1, 0, -1):
            c = int(arg[t, b])
            tup.append(c)
            b -= c
        return val, tup

    rows, psi = [], np.zeros(nc)
    for it in range(600):
        val, tup = separate(psi)
        if val >= 1 - 1e-7:
            break
        r = np.zeros(nc)
        for c in tup:
            if c < nc:
                r[c] += 1
        rows.append(r)
        res = linprog(w, A_ub=-np.array(rows), b_ub=-np.ones(len(rows)), bounds=[(0, None)] * nc, method="highs")
        psi = res.x
    val, tup = separate(psi)
    psi = psi / min(val, 1.0)
    # Richert weights restricted to the same cells (value at the left endpoint), for a like-for-like comparison
    bestR = None
    for th in lo[lo > 1.0 / R1 + 1e-9]:
        lam = 1.0 / (R1 - 1.0 / th)
        cR = float(w @ (lam * np.maximum(0.0, 1 - lo / th)))
        if bestR is None or cR < bestR[0]:
            bestR = (cR, th, lam)
    return dict(cost=float(w @ psi), psi=psi, lo=lo, iters=it, minsum=val, richert=bestR)


def lp_dual(eng, V, j, R1, cell=0.001):
    """Lower bound, valid for EVERY measurable psi >= 0 with (6.1), of J_psi = int psi(alpha) v(1/2 - alpha) d alpha/alpha,
    where v is the step function of the table V[., j].

    Cells c = [lo_c, hi_c) of width d cover [theta2, 1/2 - theta2); beyond, v is infinite, so psi = 0 a.e. there.
    If the right end-points of R1 cells sum to at most 1, every choice of points in these cells is admissible in (6.1);
    averaging (6.1) over the product of the cells gives  sum_m avg_{c_m}(psi) >= 1  (a cell beyond the range has
    average 0).  Moreover J_psi >= sum_c wl_c avg_c(psi) with wl_c = d min_{cell} v/alpha.  So for weights y_r >= 0 on
    such configurations with  sum_r y_r #{m: c_m^{(r)} = c} <= wl_c  for all c:   J_psi >= sum_r y_r.
    The y_r are the dual variables of the cutting-plane LP; the inequalities are re-checked at the end."""
    from scipy.optimize import linprog
    h, e = eng.h, eng.e
    stride = max(1, int(round(cell / h)))
    d = h * stride
    nc = int(math.ceil((0.5 - e - e) / d))
    lo = e + d * np.arange(nc)
    hi = lo + d
    levs = np.arange(eng.n)
    al_hi = 0.5 - eng.X[:eng.n]                         # level cell i: alpha in (1/2 - X_{i+1}, 1/2 - X_i]
    dens = V[:eng.n, j].astype(float) / al_hi           # lower bound of v/alpha on level cell i
    wl = np.empty(nc)
    for c in range(nc):
        k = (al_hi > lo[c] - 1e-12) & (al_hi - h < hi[c] + 1e-12)
        wl[c] = d * dens[k].min() * (1 - 1e-9)
    wl = np.where(np.isfinite(wl), wl, 1e6)
    B = int(math.floor((1 - R1 * (e + d)) / d + 1e-9))

    def separate(psi):
        cost = np.append(psi, 0.0)
        INF = 1e18
        dp = np.full((R1 + 1, B + 1), INF)
        dp[0, 0] = 0.0
        arg = np.zeros((R1 + 1, B + 1), np.int32)
        for t in range(1, R1 + 1):
            best = np.full(B + 1, INF)
            ba = np.zeros(B + 1, np.int32)
            for c in range(min(nc, B) + 1):
                cand = np.full(B + 1, INF)
                cand[c:] = dp[t - 1, :B + 1 - c] + cost[c]
                msk = cand < best
                best[msk] = cand[msk]
                ba[msk] = c
            dp[t], arg[t] = best, ba
        b = int(np.argmin(dp[R1]))
        val = float(dp[R1, b])
        tup = []
        for t in range(R1, 0, -1):
            c = int(arg[t, b])
            tup.append(c)
            b -= c
        return val, tup

    rows, psi, res = [], np.zeros(nc), None
    for it in range(1500):
        val, tup = separate(psi)
        if val >= 1 - 1e-7:
            break
        assert sum(e + (c + 1) * d for c in tup) <= 1 + 1e-9
        r = np.zeros(nc)
        for c in tup:
            if c < nc:
                r[c] += 1
        rows.append(r)
        res = linprog(wl, A_ub=-np.array(rows), b_ub=-np.ones(len(rows)), bounds=[(0, None)] * nc, method="highs")
        psi = res.x
    Amat = np.array(rows)
    y = np.maximum(-res.ineqlin.marginals, 0.0)
    load = Amat.T @ y
    scale = min(1.0, float(np.min(np.where(load > 0, wl / np.maximum(load, 1e-300), np.inf))))
    y = y * scale * (1 - 1e-9)
    assert np.all(Amat.T @ y <= wl)
    return dict(J_dual=float(y.sum()), n_config=int((y > 0).sum()), iters=it, primal=float(wl @ psi), cells=nc)


def level3(quick=False):
    open_log("deep_level3.txt")
    say("LEVEL 3.  Exploration: can the method reach r = 14?   %s" % clock())
    say("=" * 110)
    g = dict(n=400, rho=1.004, a_min=0.004) if quick else dict(n=1600, rho=1.001, a_min=0.002)
    th2s = [0.009] if quick else [0.008, 0.009, 0.010, 0.011, 0.012]
    say("grid n = %(n)d, rho = %(rho)s, a_min = %(a_min)s; fixed point on the inert primes throughout; one-sided values" % g)
    say("")
    say("(a) Buchstab's identity on the first linear sieve of the switched term: T = (C1/2) C(theta1) w(theta2)")
    FACT = (1.0, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70, 0.50)
    res, keep = {}, None
    for t2 in th2s:
        w_rect, w_fix = switched_fixed_point(t2, **g)
        eng = Engine(t2, store=False, **g)
        U, V = eng.sweep()
        row = {}
        for tau in sorted(set(FACT) | {w_fix / w_rect}, reverse=True):
            best = None
            for j in range(eng.m):
                t1 = float(eng.a[j])
                if t1 < 0.34 or t1 > 0.44:
                    continue
                b = eng.best(U, V, j, tau * 0.5 * C1 * C_theta_hi(t1, 4000) * w_rect * (1 + 1e-12))
                if b and (best is None or b["val"] < best["val"]):
                    best = dict(b, j=j)
            row[tau] = best
        res[t2] = (w_rect, w_fix, row)
        say("    theta2 %.4f: w_rect = %.5f, w_fix = %.5f (ratio %.4f; the value 2 would be the ratio %.4f)" % (
            t2, w_rect, w_fix, w_fix / w_rect, 2 / w_rect))
        say("        with T (rectangle): %s" % fmtn(row[1.0]))
        say("        with T (fixed pt.): %s  %s" % (fmtn(row[w_fix / w_rect]), clock()))
        # dual certificate against omega <= 14: at the nodes theta1 with the largest budget/cost for Richert's weights
        cand = []
        for jj in range(0, eng.m, max(1, eng.m // 600)):
            t1 = float(eng.a[jj])
            if t1 < 0.36 or t1 > 0.43:
                continue
            bud = float(U[eng.n, jj]) - 0.5 * C1 * C_theta_hi(t1, 4000) * w_fix
            al_hi, S1, S2 = eng.richert_tables(V, jj)
            kk = (al_hi > 0.08) & (al_hi < 0.45) & np.isfinite(S1)
            th = al_hi[kk]
            costR = float(np.min(np.where(15 - 1 / th > 0, (S1[kk] - S2[kk] / th) / (15 - 1 / th), np.inf)))
            cand.append((bud / costR, jj, bud))
        cand.sort(reverse=True)
        pick = sorted(set([cand[0][1]] + [c[1] for c in cand if abs(c[1] - cand[0][1]) in (4 * max(1, eng.m // 600),)]))
        for jj in pick[:1 if quick else 3]:
            bud = [c[2] for c in cand if c[1] == jj][0]
            Dl = lp_dual(eng, V, jj, 15, cell=0.004 if quick else 0.001)
            say("        dual bound, omega <= 14, theta1 %.4f: budget u - T = %.5f, J_psi >= %.5f for every psi (%d configurations, "
                "%d cells; LP value %.5f)  => %s  %s" % (eng.a[jj], bud, Dl["J_dual"], Dl["n_config"], Dl["cells"], Dl["primal"],
                                                        "EXCLUDED" if Dl["J_dual"] > bud else "not excluded", clock()))
        if keep is None or row[w_fix / w_rect]["val"] < keep[3]["val"]:
            keep = (eng, U, V, row[w_fix / w_rect], w_fix / w_rect)
    say("")
    say("(b) what if the switched term were smaller: min over theta2 of 1/lambda + 1/theta with T replaced by tau T")
    for tau in FACT:
        vals = [res[t2][2][tau] for t2 in th2s if res[t2][2][tau]]
        b = min(vals, key=lambda z: z["val"])
        say("    tau = %.2f: %s" % (tau, fmtn(b)))
    say("")
    eng, U, V, b, tau = keep
    j = b["j"]
    say("(c) general weights 1 - sum_{q | n} psi(log q / log x) by linear programming, at theta1 = %.4f, theta2 = %.4f" % (b["theta1"], b["theta2"]))
    say("    budget u - T = %.5f (T with the fixed point of (a))" % (b["u"] - b["T"]))
    for R1 in (16, 15, 14):
        L = lp_weights(eng, V, j, R1)
        cR, thR, lamR = L["richert"]
        say("    omega <= %d: cost of the optimal step weights %.5f (budget/cost %.4f); Richert weights on the same cells %.5f "
            "(theta = %.3f, budget/cost %.4f); %d cuts, min sum psi = %.6f  %s" % (
                R1 - 1, L["cost"], (b["u"] - b["T"]) / L["cost"], cR, thR, (b["u"] - b["T"]) / cR, L["iters"], L["minsum"], clock()))
    say("done %s" % clock())


# ----------------------------------------------------------------------------------------------------------------
# 8. LEVEL 4: general even shifts a
# ----------------------------------------------------------------------------------------------------------------
def local_main(l, a):
    """density of the residues p mod l that survive in the main term: p a unit, p + a != 0, and p != 1 if l = 3 (4)"""
    from fractions import Fraction
    ok = [p for p in range(1, l) if (p + a) % l != 0 and not (l % 4 == 3 and p == 1)]
    return Fraction(len(ok), l - 1)


def local_switched(l, a):
    """the same for the switched term, built from its own ingredients: n = 2 m p1 p2, sieves on n + 1 (density g1) and
    on n + 1 + a (density g2), joint density g12; for l = 1 (4) the sum over the power of l dividing m."""
    from fractions import Fraction
    units = range(1, l)
    g1 = Fraction(sum(1 for n in units if (n + 1) % l == 0), l - 1)
    g2 = Fraction(sum(1 for n in units if (n + 1 + a) % l == 0), l - 1)
    g12 = Fraction(sum(1 for n in units if (n + 1) % l == 0 and (n + 1 + a) % l == 0), l - 1)
    E = 1 - g1 - g2 + g12                                  # l does not divide m
    if l % 4 == 3:
        return E
    Em = Fraction(0 if (1 + a) % l == 0 else 1)           # l | m: n = 0 mod l, n + 1 = 1, n + 1 + a = 1 + a
    return E + Em * Fraction(1, l - 1)                    # sum_{k >= 1} l^{-k} = 1/(l-1)


def C_shift(a, pmax=10 ** 6):
    """C(a) = 2 prod_{l > 2} M_l(a) (1 - 1/l)^{-nu_l}, nu_l = 2 for l = 3 (4), 1 for l = 1 (4); for a = 1 (3) the factor
    at l = 3 is replaced by (1/9)(1 - 1/3)^{-2} = 1/4 (the classes p = 10, 19 mod 27)."""
    p = prime_list(pmax)
    p = p[p > 2].astype(np.float64)
    rho = np.where(p % 4 == 3, 3.0, 2.0)
    nu = np.where(p % 4 == 3, 2.0, 1.0)
    special = [l for l in set(prime_factors(abs(a)) + prime_factors(abs(a + 1))) if l > 2]
    logs = np.log(1 - (rho - 1) / (p - 1), where=(p > 3), out=np.zeros_like(p)) - nu * np.log(1 - 1 / p) * (p > 3)
    tot = float(np.sum(logs))
    for l in special:
        if l == 3:
            continue
        r = 3.0 if l % 4 == 3 else 2.0                      # generic number of excluded classes; the true one:
        tot += math.log(float(local_main(l, a))) - math.log(1 - (r - 1) / (l - 1))
    f3 = 0.25 if a % 3 == 1 else 0.5 / (1 - 1 / 3) ** 2
    return 2 * f3 * math.exp(tot)


def prime_factors(n):
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def shift_counts(x, shifts):
    """N_a = #{p <= x: p = 3 (8), p - 1 free of prime factors = 3 (4), p + a prime} (for a = 1 (3): p = 10, 19 (27) and
    (p - 1)/9 free of such factors); returns {a: N_a}"""
    sv = np.ones(x + 3000, dtype=bool)
    sv[:2] = False
    for i in range(2, int((x + 3000) ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = False
    inert_free = np.ones(x + 1, dtype=bool)
    for q in np.nonzero(sv[:x // 2 + 1])[0]:
        if q % 4 == 3:
            inert_free[q::q] = False
    P = np.nonzero(sv[:x + 1])[0]
    P = P[(P % 8 == 3) & (P > 3000)]
    std = P[inert_free[P - 1]]
    P27 = P[(P % 27 == 10) | (P % 27 == 19)]
    mod = P27[inert_free[(P27 - 1) // 9]]
    return {a: int(np.sum(sv[(mod if a % 3 == 1 else std) + a])) for a in shifts}


def level4(quick=False):
    open_log("deep_level4.txt")
    say("LEVEL 4.  General even shifts a: p = m^2 + n^2 + 1, p + a almost prime   %s" % clock())
    say("=" * 110)
    shifts = [2, -2, 4, -4, 6, -6, 8, 10, 12, 14, 20, 30, 42, 100, 2310, -12]
    say("(a) local factors: main term M_l(a) and switched term S_l(a), exact rational arithmetic, primes 3 <= l < 200")
    pr = [int(q) for q in prime_list(200) if q > 2]
    bad = 0
    for a in shifts:
        diff = [l for l in pr if local_main(l, a) != local_switched(l, a)]
        zero = [l for l in pr if local_main(l, a) == 0]
        bad += len(diff)
        spec = sorted(set(l for l in prime_factors(abs(a)) + prime_factors(abs(a + 1)) if l > 2))
        say("    a = %5d: M_l = S_l for all l: %s;  primes dividing a(a+1): %s;  M_l = 0 at l in %s" % (a, not diff, spec, zero))
    say("    => the local factors of the two terms agree for every prime and every shift: %s" % (bad == 0))
    say("       (M_3 = 0 exactly when a = 1 mod 3: then 3 | (p - 1)(p + a); one takes p = 10, 19 mod 27, so that 9 || p - 1)")
    say("")
    say("(b) singular series C(a) = 2 prod_{l>2} M_l(a)(1 - 1/l)^{-nu_l}  (A(a) = C(a) kappa e^{-3g/2}/4, K(a) = 2 e^{3g/2} A(a)/sqrt(pi))")
    Cs = {a: C_shift(a, 10 ** 6) for a in shifts}
    for a in shifts:
        say("    a = %5d: C(a) = %.6f   C(a)/C(2) = %.6f%s" % (a, Cs[a], Cs[a] / Cs[2], "   (classes mod 27)" if a % 3 == 1 else ""))
    say("    check: C(2) = %.6f against C = %.6f of [NX]" % (Cs[2], constants(10 ** 6)["C"]))
    say("")
    x = 5 * 10 ** 6 if quick else 6 * 10 ** 7
    say("(c) counts up to x = %.0e of primes p = 3 (8) with p - 1 free of prime factors = 3 (4) and p + a prime" % x)
    say("    (for a = 1 (3): p = 10, 19 (27) and (p - 1)/9 free of prime factors = 3 (4)); prediction N_a/N_2 -> C(a)/C(2)")
    sv = np.ones(x + 3000, dtype=bool)
    sv[:2] = False
    for i in range(2, int((x + 3000) ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = False
    inert_free = np.ones(x + 1, dtype=bool)
    for q in np.nonzero(sv[:x // 2 + 1])[0]:
        if q % 4 == 3:
            inert_free[q::q] = False
    P = np.nonzero(sv[:x + 1])[0]
    P = P[(P % 8 == 3) & (P > 3000)]
    std = P[inert_free[P - 1]]
    P27 = P[(P % 27 == 10) | (P % 27 == 19)]
    mod = P27[inert_free[(P27 - 1) // 9]]
    N2 = int(np.sum(sv[std + 2]))
    say("    #{p <= x: p = 3 (8), p - 1 free of inert primes} = %d;  N_2 = %d   %s" % (len(std), N2, clock()))
    for a in shifts:
        base = mod if a % 3 == 1 else std
        Na = int(np.sum(sv[base + a]))
        extra = ""
        if a % 3 == 1:
            extra = "  [standard classes: %d]" % int(np.sum(sv[std + a]))
        pred = Cs[a] / Cs[2]
        if a % 3 == 1:          # the two-squares condition is on (p-1)/18 instead of (p-1)/2: second-order factor
            pred2 = pred * math.sqrt(math.log(x / 2) / math.log(x / 18))
            extra = "  [with the factor (log(x/2)/log(x/18))^(1/2): %.4f]" % pred2 + extra
        say("    a = %5d: N_a = %6d   N_a/N_2 = %.4f   predicted %.4f   (statistical error about %.3f)%s" % (
            a, Na, Na / N2, pred, (Na / N2) * math.sqrt(1 / max(Na, 1) + 1 / N2), extra))
    say("done %s" % clock())


# ----------------------------------------------------------------------------------------------------------------
# 9. PAIRS: two shifts, p - 1 sifted by the primes = 3 (4) and (p + a1)(p + a2) by all primes (dimension 2)
# ----------------------------------------------------------------------------------------------------------------
class K2Tables:
    """beta-sieve of dimension 2: beta - 1 is the largest zero of q(s) = s^3 - 6 s^2 + 9 s - 8/3 (the polynomial
    solution of (s q(s))' = 2 q(s) + 2 q(s+1)), F(s) = A/s^2 on (0, beta + 1], f = 0 on (0, beta],
    (s^2 f)' = 2 s F(s-1) (s > beta), (s^2 F)' = 2 s f(s-1); so s^2 f(s) = 2A [log(t-1) - 1/(t-1)]_beta^s on
    [beta, beta + 2].  A = 43.4968874... is fixed by F, f -> 1 (the tables below give |F - 1|, |f - 1| < 1e-6 at
    s = 14 for A in [A_LO, A_HI]; cf. [FI, Chapter 11]).  Enclosures on the grid s = i/nt as in SieveTables."""
    BETA = 4.83398659670404
    A_LO, A_HI = 43.4968874, 43.4968875

    def __init__(self, nt=NT, smax=SMAX + 2):
        self.nt, self.smax, self.beta = nt, smax, self.BETA
        b = self.BETA
        assert abs((b - 1) ** 3 - 6 * (b - 1) ** 2 + 9 * (b - 1) - 8 / 3) < 1e-12
        self.b_lo, self.b_hi = b - 1e-12, b + 1e-12
        M = nt * smax
        self.M = M
        s = np.arange(M + 1) / nt
        s[0] = 1e-300
        up, dn = 1 + 1e-12, 1 - 1e-12
        cf = lambda x: np.log(np.maximum(x - 1, 1e-300)) - 1 / np.maximum(x - 1, 1e-300)
        P_lo, P_hi = np.full(M + 1, self.A_LO), np.full(M + 1, self.A_HI)          # s^2 F
        Q_lo, Q_hi = np.zeros(M + 1), np.zeros(M + 1)                             # s^2 f
        k = (s > b + 1e-9) & (s <= b + 2)
        Q_lo[k] = 2 * self.A_LO * (cf(s[k]) - cf(self.b_hi)) * dn - 1e-10
        Q_hi[k] = 2 * self.A_HI * (cf(s[k]) - cf(self.b_lo)) * up + 1e-10
        Q_lo = np.maximum(Q_lo, 0.0)
        s[0] = 1.0
        F_lo, F_hi = P_lo / s ** 2 * dn, P_hi / s ** 2 * up
        f_lo, f_hi = Q_lo / s ** 2 * dn, Q_hi / s ** 2 * up
        s[0] = 0.0
        ds2 = (s[1:] - s[:-1]) * (s[1:] + s[:-1])
        iP = int(math.floor((b + 1) * nt))            # last node <= beta + 1
        iQ = int(math.floor((b + 2) * nt))            # last node <= beta + 2
        start = iP + 1
        while start <= M:
            idx = np.arange(start, min(start + nt, M + 1))
            P_hi[idx] = (P_hi[start - 1] + np.cumsum(ds2[idx - 1] * f_hi[idx - nt])) * up
            P_lo[idx] = (P_lo[start - 1] + np.cumsum(ds2[idx - 1] * f_lo[idx - 1 - nt])) * dn
            F_hi[idx], F_lo[idx] = P_hi[idx] / s[idx] ** 2 * up, P_lo[idx] / s[idx] ** 2 * dn
            jdx = idx + (iQ - iP)
            jdx = jdx[jdx <= M]
            if len(jdx):
                Q_hi[jdx] = (Q_hi[jdx[0] - 1] + np.cumsum(ds2[jdx - 1] * F_hi[jdx - 1 - nt])) * up
                Q_lo[jdx] = (Q_lo[jdx[0] - 1] + np.cumsum(ds2[jdx - 1] * F_lo[jdx - nt])) * dn
                f_hi[jdx], f_lo[jdx] = Q_hi[jdx] / s[jdx] ** 2 * up, Q_lo[jdx] / s[jdx] ** 2 * dn
            start += nt
        self.F_lo, self.F_hi, self.f_lo, self.f_hi = F_lo, F_hi, f_lo, f_hi

    def F_up(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.floor(s * self.nt * (1 - 1e-14)).astype(np.int64), 0, self.M)
        return np.where(s <= self.BETA + 1, self.A_HI / np.maximum(s, 1e-300) ** 2 * (1 + 1e-12), self.F_hi[i])

    def f_dn(self, s):
        s = np.asarray(s, float)
        i = np.clip(np.floor(s * self.nt * (1 - 1e-14)).astype(np.int64), 0, self.M)
        return np.where(s <= self.b_hi, 0.0, self.f_lo[i])


_K2 = None


def k2tables():
    global _K2
    if _K2 is None:
        _K2 = K2Tables()
    return _K2


def T_pair(t1, t2, ncell=4000):
    """switched term for two shifts: (C1/2) C(theta1) min_{s >= beta - 1} F^{(2)}(s)/(1/2 - s theta2)"""
    K = k2tables()
    ss = np.arange(K.beta - 1 + 1e-9, min(0.49 / t2, K.smax - 0.01), 0.01)
    w = K.F_up(ss) / (0.5 - ss * t2)
    k = int(np.argmin(w))
    return 0.5 * C1 * C_theta_hi(t1, ncell) * float(w[k]) * (1 + 1e-12), float(ss[k])


def local_pair(l, shifts):
    """(M_l, S_l) for a tuple of shifts, exact: main term (p a unit, p + a_i != 0, p != 1 if l = 3 (4)) and
    switched term (n = 2 m p1 p2: units, plus the residue n = 0 with mass 1/(l-1) if l = 1 (4))"""
    from fractions import Fraction
    Mv = Fraction(sum(1 for p in range(1, l) if all((p + a) % l for a in shifts) and not (l % 4 == 3 and p == 1)), l - 1)
    E = Fraction(sum(1 for n in range(1, l) if (n + 1) % l and all((n + 1 + a) % l for a in shifts)), l - 1)
    if l % 4 == 1 and all((1 + a) % l for a in shifts):
        E += Fraction(1, l - 1)
    return Mv, E


def pairs(quick=False):
    open_log("deep_pairs.txt")
    say("PAIRS.  p = m^2 + n^2 + 1 with (p + a1)(p + a2) almost prime: semi-linear sieve x sieve of dimension 2   %s" % clock())
    say("=" * 110)
    K = k2tables()
    nt = K.nt
    say("dimension 2: beta = %.12f, A in [%.7f, %.7f]" % (K.beta, K.A_LO, K.A_HI))
    for x in (5.0, 5.834, 6.0, 7.0, 8.0, 10.0, 14.0):
        i = int(round(x * nt))
        say("    s = %6.3f: F in [%.7f, %.7f]   f in [%.7f, %.7f]" % (x, K.F_lo[i], K.F_hi[i], K.f_lo[i], K.f_hi[i]))
    say("")
    say("(a) local factors M_l = S_l for pairs of shifts (exact arithmetic, 3 <= l < 200); admissible if M_l > 0 for all l")
    pr = [int(q) for q in prime_list(200) if q > 2]
    for sh in ((2, 6), (2, 8), (6, 8), (2, 12), (6, 12), (-4, 2), (2, 4), (2, 30)):
        loc = [local_pair(l, sh) for l in pr]
        say("    (a1, a2) = %-9s: M_l = S_l for all l: %s;  admissible: %s" % (
            sh, all(a == b for a, b in loc), all(a > 0 for a, _ in loc)))
    say("")
    g = dict(n=400, rho=1.004, a_min=0.004) if quick else dict(n=1600, rho=1.001, a_min=0.002)
    say("(b) criterion u - T - 2 lambda J > 0, omega((p+a1)(p+a2)) < 1/lambda + 2/theta; one-sided values, grid n = %(n)d, "
        "rho = %(rho)s, b_0 = %(a_min)s" % g)
    glob = None
    for t2 in ([0.004] if quick else [0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]):
        eng = Engine(t2, kaux=2, **g)
        out = {}
        for name in ("rect", "fix"):
            U, V = eng.rect_UV() if name == "rect" else eng.sweep()
            best = None
            for j in range(eng.m):
                t1 = float(eng.a[j])
                if t1 < 0.33 or t1 > 0.47:
                    continue
                b = eng.best(U, V, j, T_pair(t1, t2)[0], th_min=0.05, th_max=0.5 - (K.beta - 1) * t2 - 1e-9)
                if b and (best is None or b["val"] < best["val"]):
                    best = dict(b, j=j)
            out[name] = best
        say("    theta2 %.4f: rectangle %s" % (t2, fmtn(out["rect"]).replace("1/lam+1/theta", "1/lam+2/theta")))
        say("                   fixed pt. %s  %s" % (fmtn(out["fix"]).replace("1/lam+1/theta", "1/lam+2/theta"), clock()))
        if out["fix"] and (glob is None or out["fix"]["val"] < glob[0]["val"]):
            glob = (out["fix"], eng, U, V)
        else:
            del eng, U, V
    b, eng, U, V = glob
    j = b["j"]
    theta = math.floor(b["theta"] * 200) / 200
    Tsw, s_opt = T_pair(b["theta1"], b["theta2"], ncell=20000)
    J = 2 * eng.J_of(V, j, theta)
    u = float(U[eng.n, j])
    lam = math.floor((u - Tsw) / J * 0.997 * 1e4) / 1e4
    say("")
    say("(c) witness: (theta1, theta2, theta, lambda) = (%.5f, %.4f, %.3f, %.4f), s = %.2f in T" % (b["theta1"], b["theta2"], theta, lam, s_opt))
    say("    u >= %.6f   T <= %.6f   2J <= %.6f   u - T - 2 lambda J >= %.6f" % (u, Tsw, J, u - Tsw - lam * J))
    v = 1 / lam + 2 / theta
    say("    1/lambda + 2/theta = %.4f  =>  omega((p+a1)(p+a2)) <= %d" % (v, math.ceil(v - 1e-12) - 1))
    say("done %s" % clock())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    QUICK = "--quick" in sys.argv
    stage = args[0] if args else "all"
    if stage in ("level1", "all"):
        level1(QUICK)
    if stage in ("level2", "all"):
        level2(QUICK)
    if stage in ("certify", "all"):
        if QUICK:
            certify(theta=0.19, lam=0.09, n=400, rho=1.004, a_min=0.004)
        else:
            certify()
            certify(t1=0.4, t2=0.0035, theta=0.2, lam=0.036, s_sw="8.17", kaux=2)
    if stage in ("level3", "all"):
        level3(QUICK)
    if stage in ("level4", "all"):
        level4(QUICK)
    if stage in ("pairs", "all"):
        pairs(QUICK)
