# Recovery audit

The umbrella repository was restored after accidental deletion and then re-audited before new work was added.

## Recovered state

- 11 original consolidated project directories were present after restore.
- The recovered `projects/learning-to-branch-milp/` snapshot was compared against the surviving preserved copy.
- **30/30 source files matched by Git blob SHA and file mode.**
- No original recovered project directory was overwritten during the post-recovery enhancement.

## Added after recovery

Two new projects were added because they cover distinct solver decisions not represented as standalone modules in the restored repository:

1. `learning-to-search-bnb-nodes`
2. `learning-to-select-primal-heuristics`

The literature and benchmark protocol were also centralized under `docs/`.

## Canonical-source rule

The restored 11 project snapshots are treated as the canonical recovered code. Experimental reconstructions made during the recovery window are not used to replace them.
