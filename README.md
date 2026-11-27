# Vector-sieve level optimization, Buchstab's identity, and almost primes p + a for primes p = m^2 + n^2 + 1

Ruqing Chen (GUT Geoservice Inc., Montreal). Manuscript: `acta_arith_almost_primes.pdf` (source: `.tex`).

The paper continues K. Nath and L. Xie, *Almost primes and primes that are sums of two squares plus 1*,
Acta Arith. 223 (2026), 51-82, and their Erratum. Here `Omega(n)` is the number of prime factors of `n` with
multiplicity, and each statement holds for `>> x (log x)^(-5/2)` primes `p <= x` of the form `m^2 + n^2 + 1`
(`>> x (log x)^(-7/2)` in Theorem 1.4).

| | statement | method |
|---|---|---|
| Theorem 1.1 | `Omega(p+2) <= 17` (`<= 48` without weights) | free levels in the switching step; explicit witness in interval arithmetic |
| Theorem 1.2 | `Omega(p+2) <= 15` | Buchstab's identity on the primes `q = 3 (mod 4)` inside the vector sieve; certificate in exact integer and interval arithmetic |
| Theorem 1.3 | `Omega(p+a) <= 15` for every even `a != 0` | the Euler products of the main and switched terms agree prime by prime; their ratio is `2 e^(3 gamma/2) / sqrt(pi)` |
| Theorem 1.4 | `Omega((p+a1)(p+a2)) <= 37`, `a1 != a2` even, non-zero, neither `= 1 (mod 3)` | sieve of dimension 2 on the product; same certificate |

## Files

| path | content |
|---|---|
| `acta_arith_almost_primes.tex`, `.pdf` | the paper (15 pages) |
| `code/deep_sieve_improvements.py` | all computations, in stages (see below) |
| `code/make_acta_figures.py` | the three figures |
| `data/deep_level1.txt` | Section 3: optimisation with rectangle bounds, interval witnesses for 17 and 48 |
| `data/deep_level2.txt` | Section 4: Buchstab steps `k = 0, ..., 12` and fixed point, grid refinement (Tables 1, 2) |
| `data/deep_certificate.txt`, `data/deep_certificate_pairs.txt` | the certified inequalities of Theorems 1.2 and 1.4 (Table 5) |
| `data/deep_level4.txt` | Section 5: local factors, singular series, prime counts up to `6e7` |
| `data/deep_pairs.txt` | Section 5.1: two shifts |
| `data/deep_level3.txt` | Section 6: switched term, dual bounds, linear programmes (Tables 3, 4) |
| `figures/` | `fig_bound.pdf`, `fig_fixed_point_and_shifts.pdf`, `fig_terms.pdf` |

## Reproduction

Python 3 with `numpy`, `mpmath` (interval arithmetic), `scipy` (linear programmes, stage `level3` only) and
`matplotlib` (figures).

```
python code/deep_sieve_improvements.py level1     # about 7 min
python code/deep_sieve_improvements.py level2     # about 30 min, 0.5 GB of memory
python code/deep_sieve_improvements.py certify    # about 3 min: Theorems 1.2 and 1.4
python code/deep_sieve_improvements.py level3     # about 20 min
python code/deep_sieve_improvements.py level4     # under 1 min
python code/deep_sieve_improvements.py pairs      # about 11 min
python code/make_acta_figures.py                  # about 20 min
```

`all` runs every stage; `--quick` uses coarse grids (a few minutes in total, values less sharp). Each stage
writes `data/deep_<stage>.txt`. The stage `certify` caches the interval-arithmetic node values of the sieve
functions in `data/sieve_nodes_iv_*.npz` on its first run.

## How the numbers are certified

* Section 3 (Theorem 1.1): interval arithmetic throughout.
* Theorems 1.2 and 1.4: transcendental functions are evaluated in interval arithmetic only; the remaining
  floating-point work uses the four operations and square roots with outward rounding; the Buchstab recursion
  itself runs in exact integer arithmetic (table entries in units of `1e-9`).
* The exploratory tables (Tables 1-4) are one-sided values in double precision with outward rounding.
