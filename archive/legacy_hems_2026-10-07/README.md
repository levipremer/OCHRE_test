# Legacy HEMS archive

Saved and verified 2026-10-07. Active simulator files were not moved or replaced.

- `committed_snapshot/`: 55 source/configuration/documentation/model files from
  commit `e286f438250d8649487d9de569ee725e930ca5a9` (2026-10-01,
  "Publish single-home HEMS simulations and scenario MPC"). Repository-relative
  paths are preserved, including the shared five-home Python modules/configs.
- `committed_legacy.zip`: the same Git export, retained as a recovery bundle.
- `pre_unification_ev/`: the earlier EV MPC controller captured before this
  session consolidated it into `controllers/ev.py`. It retains event-triggered
  solves and the former mutually exclusive cycling selection.
- `manifest.json`: source commit, per-file SHA-256 hashes, export hash, and
  verification method. Extracted files match the ZIP byte-for-byte. Git's
  Windows export converted some text line endings to CRLF; those files match
  Git after LF normalization. Binary model files match without normalization.

## Completeness and recovery limits

This is **not a complete pre-unification October 6 working-tree snapshot**.
Uncommitted intermediate versions not captured separately cannot be recovered
from the October 1 commit. The recovered EV controller is a separate partial
snapshot, not a claim that the committed snapshot contains the EV/V2H revision.
No legacy sources or current user-owned configuration edits were deleted.

External weather, mobility/water datasets, generated outputs, virtual environments,
and native OCHRE sources are not duplicated here. Archived configuration paths
may reference historical filenames. Treat this as a source/reference archive,
not an independently runnable installation. Restore selected files only into a
separate checkout/directory after reviewing their versions and dependencies;
do not overwrite the active unified simulator wholesale.
