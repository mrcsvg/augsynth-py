# Notebooks

Exploratory notebooks live here. They are **not shipped with the package**
and are excluded from CI.

Two rules:

1. **Notebooks are for exploration, not for tests.** If a notebook produces
   a result worth keeping, the code moves into `src/` and the verification
   moves into `tests/`. The notebook stays as documentation of the thought
   process, not as the source of truth.

2. **Do not commit large output cells.** Use a `.gitignore` rule or run
   `nbstripout` before pushing. Big notebooks with embedded plots wreck the
   diff history.

The `.gitignore` at the repo root excludes `*.ipynb` by default. Add
specific notebooks you want tracked with a force-add: `git add -f
notebooks/<name>.ipynb`.
