# Handoff notes — natural-spread Junction discovery (straight-through branch)

Branch: `feature/rebuild-multi-junction-from-single`
File: `pygame_simulator/Single_junction_sph_dfs_Multi_Hop.py`
Last commit: `47c936e Add straight-through branch discovery to natural-spread Junction detection`

## What this is

`Single_junction_sph_dfs_Multi_Hop.py` already implements a natural-spread,
no-forward-sensor K-hop Junction/branch discovery pipeline (see the module
docstring and `SpaceRecognitionState`). Verified end-to-end via headless runs:
root Leader/cap forms -> natural lateral spread detected -> cohort
clustering/validation -> Junction confirmed -> Max-Min d-cluster branch
Leader election -> rollout-cost-ordered sequential DFS -> local
clearance/density/pressure-based dead-end detection (no coordinates) ->
Marker PENDING->COMPLETED->BLOCKING consensus -> next branch -> all
branches COMPLETED/BLOCKING -> KHOP_COMPLETE. No crashes observed.

## The bug that was found and partially fixed this session

The physical map is a genuine 3-way junction (straight/UP + LEFT + RIGHT),
but the discovery pipeline only ever confirmed 2 cohorts (the two turning
branches) and silently dropped the straight-through one.

Root cause: a branch is normally "witnessed" by a NORMAL robot that slips
past the front Leader into new territory and accumulates travel/dwell
there (`detect_persistent_non_corridor_motion`,
`detect_khop_directional_clusters`). A straight-through exit can never be
witnessed this way — nothing is permitted ahead of the front Leader in its
own heading, so nobody can "slip past" it to prove that direction.

### Fix applied (in `detect_khop_directional_clusters`, ~line 5210)

1. Broadened the observation role filter from `role == "NORMAL"` to
   `role in KHOP_DYNAMIC_ROLES` (`{"NORMAL", "KHOP_LEADER", "KHOP_SHEPHERD"}`)
   — captured cap members are the same fluid population, just a different
   capture-tree role.
2. Added `detect_khop_straight_continuation_cluster` (new function, right
   after `detect_khop_directional_clusters`): evidences the straight lane
   from the root group's own measured state instead of a witness —
   `max(forward-travel-since-formation, root.forward_clearance)` for travel,
   `root.connectivity_ratio` / base-connectivity for connectivity, corridor
   width from `root.estimated_width`, and an instantaneous per-member
   heading-variance for direction stability. This candidate is merged into
   the same `clusters` list and judged by the same
   `validate_directional_cohorts` thresholds as a turning cohort.

### Confirmed by headless testing

- `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python Single_junction_sph_dfs_Multi_Hop.py --headless-steps=N`
  (see the module docstring / top of file for all `--headless-*` /
  `--initial-leader-id=` CLI flags).
- The straight-through candidate **is now detected** every frame during
  `CONTROLLED_SPREAD` (shows up in the `clusters=[...]` headless summary
  with heading ≈ `(0.0, -1.0)`, size ≈ root's captured member count).
- It is **not yet validated** in the runs tried so far —
  `valid-cohorts=2`, only `T_left`/`T_right` end up in the branch queue.

## Remaining problem: `direction_variance` for the straight cluster

`NATURAL_SPREAD_COHORT_MAX_VARIANCE = 0.13`. Two ways of computing the
straight cluster's `direction_variance` were tried, both borderline:

- **Instantaneous** (`1 - |sum of per-member khop_velocity_ema headings| / count`,
  same formula the turning clusters use): very noisy frame to frame
  (observed swinging between ~0.01 and ~0.95 within the same
  ~0.5s window) — the cap's individual members jitter locally to hold
  their slot even while the group's net motion is stable, so this signal
  rarely stays under 0.13 for the full `NATURAL_SPREAD_COHORT_MIN_DWELL`
  (0.48s) needed to validate. **This is what's currently committed.**
- **EMA** (`1 - root.heading_stability_ema`, already maintained by
  `update_khop_group_statistics` every `KHOP_CAPTURE_REFRESH_TIME`): much
  smoother, converges monotonically (observed 0.455 -> 0.135 over the
  same window) but in the tested run never quite crossed under 0.130
  before the two turning cohorts finished their own validation and the
  split fired with just 2.

Because the split triggers as soon as `len(valid clusters) >= 2`
(`NATURAL_SPREAD_MIN_COHORTS`), the straight cluster is racing the turning
clusters' own dwell timers and is currently losing by a small margin.

### Things to try next (not yet attempted)

1. Switch `direction_variance` to the EMA form and either loosen
   `NATURAL_SPREAD_COHORT_MAX_VARIANCE` slightly for this case or seed
   `root.heading_stability_ema` from its value at the moment
   `begin_khop_controlled_spread` fires (it was presumably already
   stable during ordinary `ADVANCING`, before the slowdown/jitter
   transient this EMA is currently recovering from).
2. Don't let the split fire the instant 2 valid clusters exist while a
   3rd detected-but-unresolved candidate is still accumulating dwell —
   give still-forming candidates a short grace window (bounded by
   `NATURAL_SPREAD_CANDIDATE_TIMEOUT = 5.0s`) before finalizing with
   fewer than all currently-tracked clusters. This is the more general
   fix and would also help future non-3-way topologies.
3. Re-derive the diagnostic prints (removed before commit — see git
   history / this session's transcript for the exact `print(...)` block
   that was in `detect_khop_straight_continuation_cluster`) if more
   frame-by-frame data is needed again.

## Repo housekeeping

- `single_junction_sph_dfs_Multi_Hop.py.bak-*` (4 files, lowercase `s`,
  under `pygame_simulator/`) are local backup snapshots from earlier
  work sessions. They are **not committed** (left untracked on purpose —
  they're large near-duplicates of this file, not source). Ask before
  committing or deleting them.
