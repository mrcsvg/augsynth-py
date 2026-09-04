# Dependency compatibility

How augsynth-py's dependency floors are chosen, and how to install it into an
environment that is still pinned to the numpy 1.x ABI.

The short version: **augsynth-py supports numpy 1.26+ and numpy 2.x**, and
declares low floors with no upper caps so that a downstream pin can win the
resolution. If a co-installed library forces numpy 1.x, install the `numpy1`
extra.

---

## 1. Supported ranges

| Dependency | Floor      | Upper bound | Notes                                                        |
|------------|------------|-------------|--------------------------------------------------------------|
| Python     | 3.11       | —           | `requires-python = ">=3.11"`                                  |
| `numpy`    | 1.26       | none        | `src/` uses no numpy 2-only API; both stub generations type-check |
| `scipy`    | 1.12       | none        | Not imported by `src/` today; kept as a declared numerics dependency |
| `cvxpy`    | 1.4.1      | none        | 1.4.1 is the oldest release shipping the CLARABEL solver used by `Synth` |
| `polars`   | 1.0        | none        | Only stable 1.x API surface (`DataFrame`, `Series`, `exclude`) |
| `joblib`   | 1.3        | none        | `power.py` uses only `Parallel`/`delayed`, stable since well before 1.3 |

No upper caps are declared in `[project.dependencies]`. Capping in a library
propagates to every consumer; the `numpy1` extra exists so that the narrowing
is opt-in.

## 2. The numpy 1.x / numpy 2.x fault line

The only dependency that forces the choice is `cvxpy`, because it is the one
component of the stack with compiled extensions:

| cvxpy   | Requires        | numpy 1.x usable |
|---------|-----------------|------------------|
| 1.4.x   | `numpy>=1.15`   | yes              |
| 1.5.x   | `numpy>=1.15`   | yes              |
| 1.6.x   | `numpy>=1.20`   | yes              |
| 1.7.x   | `numpy>=1.21.6` | yes              |
| **1.8.0** | **`numpy>=2.0.0`** | **no**        |
| 1.9.x   | `numpy>=2.0.0`  | no               |

So `cvxpy<1.8` is exactly the condition for staying on numpy 1.x, and that is
what the `numpy1` extra encodes:

```toml
numpy1 = ["numpy>=1.26,<2", "cvxpy>=1.4.1,<1.8"]
```

```bash
pip install "augsynth-py[numpy1]"
```

This also explains a resolution that ends up on numpy 2 unintentionally: with
an open upper bound, pip takes the newest cvxpy, and any cvxpy ≥ 1.8 drags
`numpy>=2.0.0` in with it. Constraining cvxpy constrains numpy for free.

## 3. Co-installing with an existing pinned environment

`pip` reconciles only the requirements named in a single command — it does not
re-check the requirements of packages that are *already installed* (that is the
origin of the familiar `pip's dependency resolver does not currently take into
account all the packages that are installed` message). Adding augsynth-py to a
populated environment therefore needs the surrounding pins restated.

A worked example, for an environment holding `econml<2`-style numpy pins plus a
library requiring `cvxpy>=1.4.1,<1.5.0`:

```bash
# constraints.txt
numpy>=1.26.4,<2
cvxpy>=1.4.1,<1.5

pip install -c constraints.txt augsynth-py
```

Both constraints are satisfiable against augsynth-py's floors, and the unit
suite passes on that exact set (see §4).

A related note on `numba`: augsynth-py does not depend on it, directly or
transitively, so a `numba` conflict in such an environment comes from
elsewhere. It is worth resolving in the same pass, because `numba` carries its
own numpy ceiling — a `numba<0.62` pin (e.g. from `tslearn`) and a `numpy<2`
pin are mutually compatible, but only at `numba` 0.61.x or below.

## 4. What is verified

`tests/unit/` passes, and `mypy --strict src/` is clean, on:

- the declared floors — numpy 1.26.0, scipy 1.12.0, cvxpy 1.4.1, polars 1.0.0,
  joblib 1.3.0, Python 3.11;
- the constrained-but-current set — numpy 1.26.4, cvxpy 1.4.4, latest scipy /
  polars / joblib;
- the unconstrained current set (numpy 2.x, cvxpy 1.9.x), which is what the
  main `unit-tests` matrix installs.

The floors are guarded by the `unit-tests-min-deps` job in
`.github/workflows/ci.yml`. When a floor moves in `pyproject.toml`, move the
pins in that job with it.

The R parity suite (`tests/validation_against_r/`) runs in its own workflow
with an unconstrained resolution. Its only extra Python dependency is `rpy2`.

## 5. Adding a dependency, later

Two rules keep this property from eroding:

1. Prefer a dependency that supports both numpy generations. If it supports only
   numpy 2, it belongs behind an extra, not in `[project.dependencies]`.
2. Set the floor at the oldest release whose API the code actually uses, then
   let `unit-tests-min-deps` prove it. A floor raised for no reason is a
   constraint exported to every consumer.
