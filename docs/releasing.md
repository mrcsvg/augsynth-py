# Releasing augsynth-py

Runbook for publishing a release to PyPI. The publish itself is automated by
[`.github/workflows/release.yml`](../.github/workflows/release.yml) via PyPI
Trusted Publishing; this document covers the one-time setup and the
per-release steps.

## One-time setup (before the first release)

1. Create a PyPI account (with 2FA) at <https://pypi.org> if you don't have
   one.
2. Register the **pending publisher** for the project — because the project
   does not exist on PyPI yet, this is done under
   *Account settings → Publishing → Add a new pending publisher* with:
   - PyPI project name: `augsynth-py`
   - Owner: `mrcsvg`
   - Repository: `augsynth-py`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repo, create the `pypi` environment
   (*Settings → Environments → New environment*). Optionally add yourself as
   a required reviewer — that turns every publish into a one-click manual
   approval.

No API tokens are created or stored anywhere; the workflow authenticates via
OIDC (trusted publishing).

## Per-release steps

1. Make sure `main` is green (CI **and** the validation-against-R workflow).
2. Update `src/augsynth_py/_version.py` to the release version (the release
   workflow refuses to publish when the tag and `_version.py` disagree).
3. Move the `[Unreleased]` items in `CHANGELOG.md` under a new
   `[X.Y.Z] - YYYY-MM-DD` heading and update the link references at the
   bottom.
4. Land those changes on `main` via the normal PR flow.
5. Tag and push:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "augsynth-py X.Y.Z"
   git push origin vX.Y.Z
   ```

6. The `Release to PyPI` workflow runs: lint + unit-test gate, build,
   `twine check`, then publish (pausing for approval if the `pypi`
   environment has required reviewers).
7. Verify: `pip install augsynth-py==X.Y.Z` in a clean venv and run the
   README quickstart.
8. Create a GitHub Release for the tag (paste the changelog section) —
   *Releases → Draft a new release → choose the tag*.

## Sanity checks that are automated

- Tag ↔ `_version.py` agreement (release workflow, build job).
- `twine check --strict` on both sdist and wheel.
- sdist contents are restricted via `[tool.hatch.build.targets.sdist]`
  `only-include` in `pyproject.toml` — in particular `notebooks/` and
  `papers/` are excluded (paper PDFs are not tracked in git at all; see
  `papers/SOURCES.md`). If you add top-level directories that belong in the
  sdist, extend that list.

## Versioning policy

Semantic versioning with the 0.x caveat: minor bumps (0.3 → 0.4) may break
the API; patch bumps must not. The first API-stable release will be 1.0.0,
not before the GeoLift-style orchestration layer (power analysis, market
selection) has shipped and survived real use.
