# Handoff notes — natural-spread Junction discovery (straight-through branch)

Branch: `feature/rebuild-multi-junction-from-single`
File: `pygame_simulator/Single_junction_sph_dfs_Multi_Hop.py`

## Status: RESOLVED

The straight-through (UP) branch is now reliably discovered alongside the
two turning branches (LEFT/RIGHT). Verified via headless runs
(`--headless-steps=3000`): all three branches (`T_up`, `T_left`, `T_right`)
are confirmed, rollout-ordered, explored one at a time, each reaches a
correctly local-evidence-only dead-end confirmation, each Marker goes
`PENDING -> COMPLETED -> BLOCKING`, and the run ends at `KHOP_COMPLETE`
with all groups `GATEKEEPING`. No crashes.

## What this is

`Single_junction_sph_dfs_Multi_Hop.py` implements a natural-spread,
no-forward-sensor K-hop Junction/branch discovery pipeline (see the module
docstring and `SpaceRecognitionState`). Root Leader/cap forms -> natural
lateral spread detected -> cohort clustering/validation -> Junction
confirmed -> Max-Min d-cluster branch Leader election -> rollout-cost-
ordered sequential DFS -> local clearance/density/pressure-based dead-end
detection (no coordinates) -> Marker consensus -> next branch -> all
branches COMPLETED/BLOCKING -> KHOP_COMPLETE.

## The bug that was found and fixed

The physical map is a genuine 3-way junction (straight/UP + LEFT + RIGHT),
but the discovery pipeline originally only ever confirmed 2 cohorts (the
turning branches) and silently dropped the straight-through one.

Root cause: a branch is normally "witnessed" by a NORMAL robot that slips
past the front Leader into new territory and accumulates travel/dwell
there (`detect_persistent_non_corridor_motion`,
`detect_khop_directional_clusters`). A straight-through exit can never be
witnessed this way — nothing is permitted ahead of the front Leader in its
own heading, so nobody can "slip past" it to prove that direction.

### Fix (all in `Single_junction_sph_dfs_Multi_Hop.py`, ~line 5210-5420
and the `CONTROLLED_SPREAD` block of `update_khop_capture_state`)

1. `detect_khop_directional_clusters`: broadened the observation role
   filter from `NORMAL`-only to `KHOP_DYNAMIC_ROLES`
   (`{"NORMAL", "KHOP_LEADER", "KHOP_SHEPHERD"}`) — captured cap members
   are the same fluid population, just a different capture-tree role. No
   longer requires >=2 turning clusters up front before returning
   anything (down to 1, or 0 if only the straight candidate applies).
2. `detect_khop_straight_continuation_cluster` (new): evidences the
   straight lane from the root group's own measured state instead of a
   witness observation:
   - `mean_travel`: `max(forward-travel-since-formation, root.forward_clearance)`
     — the same local clearance probe already used (inverted) for
     dead-end confirmation stands in for "no one has slipped past to
     prove this territory," since CONTROLLED_SPREAD deliberately holds
     the root's own displacement still while cohorts are validated.
   - `direction_variance = 0.0` — **this is the key correction**. That
     field is meant to express uncertainty in an *inferred* heading (a
     turning cohort's true direction is unknown and estimated from
     scattered witness velocities). The straight lane has no such
     uncertainty: its heading isn't inferred, it *is* `root.heading` by
     construction. Per-member cap velocity noise (tried both raw
     instantaneous and `root.heading_stability_ema`-smoothed) was
     answering a different question — the cap's own in-place
     slot-holding jitter — and has a steady-state floor around 0.13-0.3
     that never reliably clears `NATURAL_SPREAD_COHORT_MAX_VARIANCE`
     (0.13) regardless of smoothing.
   - `connectivity_ratio` / base-connectivity from `root.connectivity_ratio`
     and root members' `connected_to_base`; `lateral_width` from
     `root.estimated_width`.
   This candidate is merged into the same `clusters` list and judged by
   the identical `validate_directional_cohorts` thresholds as a turning
   cohort.
3. `update_khop_capture_state` (`CONTROLLED_SPREAD` stage): added a
   bounded grace period (`NATURAL_SPREAD_STRAGGLER_GRACE_TIME = 1.5s`,
   tracked in `KHopCaptureState.straggler_wait_elapsed`) so that once
   `>= NATURAL_SPREAD_MIN_COHORTS` have validated, the split does not
   finalize immediately if another detected-but-not-yet-valid candidate
   is still present — it waits up to the grace window (still bounded
   well under the overall `NATURAL_SPREAD_CANDIDATE_TIMEOUT = 5.0s`) for
   the straggler to either validate or drop out. This turned out not to
   be strictly necessary once `direction_variance = 0.0` let the
   straight candidate validate on the same fast timescale as the turning
   ones (`candidate=0.56` in the successful run, same as the pre-fix
   2-cohort baseline), but it's kept as a general safety margin for
   topologies where a genuine straggler is slower.

## How to test

```
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python Single_junction_sph_dfs_Multi_Hop.py --headless-steps=7000
```
(see the module docstring / top of file for all `--headless-*` /
`--initial-leader-id=` CLI flags). Look for `valid-cohorts=3`, a
`[K-hop] swarm gathered back at Base: N/760` line with N at or above
~95%, and a final `[Headless]` summary with `stage=COMPLETE`,
`command=KHOP_COMPLETE`, and all three groups (`T_up`, `T_left`,
`T_right`) `RELEASED` (not `GATEKEEPING` — see below). 7000 steps is
needed, not 3000: the walk back to Base after the last branch closes
takes real time.

## Second round of fixes: Shepherd physical formation + real return-to-base

Found after watching the actual GUI run (not just headless logs).
Status: **RESOLVED**, verified via a 7000-step headless run reaching
`stage=COMPLETE` with 736/760 (96.8%) actually gathered at Base and all
three groups `RELEASED`.

1. **Shepherd cap shape.** `khop_cap_slot_offsets` built a tapered
   three-row "crescent" centered on the Leader (its own comment called
   it that), leaving both edges of the corridor uncovered — a real
   robot line can't pass that off as a gate, fluid slips around the
   tapered ends. Rewritten so every row spans the full sensed corridor
   width evenly (spreads fewer robots wider rather than clustering at
   center), with rows stacked straight behind one another instead of
   tapering inward with depth.
2. **Un-activated branches had no blocking force.** Only completed
   `GATEKEEPING` groups got the physical Gatekeeper repulsion field;
   a `FORMING`/`WAITING` (not-yet-its-turn) branch had a cap shape but
   nothing stopping a foreign stream from drifting into it.
   `compute_khop_gatekeeping_force` now applies the same repulsion to
   any robot that is not a `FORMING`/`WAITING` group's own cap member.
3. **Mission completion left the swarm wherever it stopped** (the
   "awkward triangle" at the end) — `khop_state.stage` jumped straight
   from all-branches-`BLOCKING` to `COMPLETE`/`DONE`, with every
   Gatekeeper frozen at its branch mouth forever and everything else
   just damped in place. Added a `RETURNING_TO_BASE` stage:
   `release_khop_survivors_for_return` stands every remaining
   Gatekeeper/Marker/stray Leader/Shepherd/Relay down to `NORMAL`,
   `compute_khop_route_force` drives everyone home via
   `direction_toward_base_path`, and completion now requires
   `KHOP_RETURN_BASE_READY_RATIO` (95%) of the swarm actually inside
   the Base rectangle for `KHOP_RETURN_BASE_DWELL` (0.5s) before
   `COMPLETE` fires.
   - Also required: `RETURNING_TO_BASE` needed the same **packed
     equilibrium spacing + reduced pressure scale** the legacy
     `RETURN_TO_BASE` phase used
     (`RETURN_PACKED_EQUILIBRIUM_SCALE` / `RETURN_PACKING_PRESSURE_SCALE`,
     applied in `adaptive_equilibrium_radius` and `compute_pressures`).
     The Base rectangle physically cannot hold ~760 robots at ordinary
     flow spacing — without compaction the ready ratio was unreachable
     and a 6000-frame run just sat in `RETURNING_TO_BASE` forever with
     Gatekeepers cosmetically stuck reporting `GATEKEEPING`. Fixed by
     also setting `group.state = RELEASED` in
     `release_khop_survivors_for_return`.

## Repo housekeeping

- `single_junction_sph_dfs_Multi_Hop.py.bak-*` (4 files, lowercase `s`,
  under `pygame_simulator/`) are local backup snapshots from earlier
  work sessions. They are **not committed** (left untracked on purpose —
  they're large near-duplicates of this file, not source). Ask before
  committing or deleting them.
