# Handoff notes — natural-spread Junction discovery (straight-through branch)

Branch: `feature/rebuild-multi-junction-from-single`
File: `pygame_simulator/Single_junction_sph_dfs_Multi_Hop.py`

## START HERE — session handoff (continuing in a fresh session)

This file is being read by a **new session** picking up from the last
pushed commit on this branch. Everything below "START HERE" is the
accumulated history from the prior session, kept as reference — read
this section first for where things actually stand.

### What's committed and safe
Every code change described in this file (rounds 1-3 below) is already
committed and pushed to `origin/feature/rebuild-multi-junction-from-single`.
Latest commit: `b98b661 Fix physical Shepherd/Gatekeeper behavior from
GUI feedback`. Nothing is sitting uncommitted.

### What's NOT yet confirmed — do this first

**UPDATE: the 14000-step run finished and found a real, confirmed bug.**
Leader-duplication stays fixed (three distinct Leaders, density sane for
`T_up`/`T_left`: 0.71-1.1x) and T_up/T_left both reach `Dead-end
confirmed` -> `Marker BLOCKING` correctly. But **`T_right` (the last
branch in queue) never reaches dead-end and appears genuinely stuck**:
at frame 8000 it read `clearance=2.8, rho=0.61, P=0.0, contact=0.426,
dead_end_dwell=0.0`; at frame 14000 (6000 frames / ~100s later) it read
**the exact same numbers, unchanged**. `clearance=2.8` means the front
is already sitting at the dead-end wall; `contact=0.426` alone would
already satisfy `compression_observed` (>= `KHOP_DEAD_END_MIN_CONTACT_
COMPRESSION` = 0.28). The blocker is `density_observed`:
`KHOP_DEAD_END_DENSITY_RATIO` was raised to 0.70 this round (see round
3, item 4), and T_right's local density appears to have hit a genuine
steady state at 0.61 -- below threshold -- and is not climbing further
no matter how long it waits.

Likely cause (not yet root-caused): T_right is the **third and last**
branch activated -- by the time it's `EXPLORING`, the other two branches
already absorbed a large share of the swarm into their own Gatekeeping
lines, and/or `KHOP_UNASSIGNED_FOLLOW_FORCE` still isn't reliably
funneling enough of the remaining ~277+ un-arrived robots
(`split-N=483` at that point, i.e. under two-thirds of 760 had even
joined a stream) into a queue behind T_right's cap fast enough to build
real density there. This may be branch-order-dependent (last branch
starves) rather than a generic threshold problem -- worth checking
whether T_right specifically has less corridor length or a smaller
share of the swarm reaching it, or whether `KHOP_DEAD_END_DENSITY_RATIO`
0.70 was simply too aggressive and should come down (e.g. to something
between the old 0.25 and new 0.70 -- try ~0.45-0.55) while keeping the
tightened pressure/contact-compression floors, which already proved
sufficient on their own (T_up/T_left both actually triggered via
contact compression, pressure was near-zero for both: 0.035 and 0.005).

**First step in this session**: decide whether to (a) loosen
`KHOP_DEAD_END_DENSITY_RATIO`, (b) investigate why the last-queued
branch specifically starves of density, or (c) drop density from the
AND and let pressure-OR-contact alone gate it (both already proved to
correlate with genuine crowding in the T_up/T_left data) -- then re-run
to confirm all three branches complete and the run reaches `stage=
COMPLETE`:
```
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python Single_junction_sph_dfs_Multi_Hop.py --headless-steps=14000
```

**Also not yet done**: GUI visual re-confirmation of the 8 issues
reported in round 3 below. All of that round's evidence so far is
headless logs/numbers only -- nobody has looked at the actual rendered
window since those fixes went in.

### What to do after verification passes (already decided, don't re-ask)
Apply normalization ideas from a swarm-distribution-as-environment-sensor
paper the user shared, **scoped to normalization only** (both explicitly
confirmed by the user in the prior session -- do not re-litigate):
- Replace absolute robot-count thresholds (`NATURAL_SPREAD_ANOMALY_ROBOTS`
  = 7, `KHOP_SPLIT_CLUSTER_MIN_SIZE`, `KHOP_SPLIT_MIN_GROUP_SIZE`, etc.)
  with ratios normalized against the locally-observable robot count
  (e.g. `khop_reachable_robot_count(leader, NATURAL_SPREAD_LOCAL_HOPS)`),
  so behavior stays consistent as cap/witness-pool sizes change (this is
  exactly the class of bug that caused the Leader-duplication issue: an
  absolute threshold's meaning silently shifted when the root cap grew
  from 29 to ~38 members this round).
- **Explicitly out of scope for now** (user deferred this, don't build
  it unprompted): a unified weighted composite score
  `S_J(t) = w1*Ŵ + w2*σ_perp² + w3*V_perp + w4*R_branch` replacing the
  current AND-gated multi-condition validation in
  `validate_directional_cohorts`/`detect_persistent_non_corridor_motion`.
  The existing local/multi-feature/time-accumulated design already
  satisfies most of the paper's principles; only the normalization gap
  was judged worth fixing immediately, given how much was already
  in flight and unverified. Formal robustness testing (varying robot
  count, corridor/branch width, Junction geometry, noise -- paper
  section 7) was also identified as a gap but not scheduled.
- If asked to continue past normalization, re-read the full paper-summary
  message in the conversation this file cannot include verbatim -- ask
  the user to re-paste it if starting genuinely fresh with no chat
  history, since it's substantial (a full section-by-section mapping of
  a distributed-sensing paper's principles onto this codebase).

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

## Third round of fixes: physical Shepherd/Gatekeeper behavior (GUI-reported)

Status: **CODE COMPLETE, PARTIALLY VERIFIED**. Found via live GUI screenshots
(not just headless logs) that the physical formation didn't match the
intended design (reference sketches showed straight full-width caps
sitting right at each Branch mouth). Eight distinct issues were reported;
all now have code fixes applied. A 14000-step extended headless run to
confirm full 3-branch completion end-to-end is running as of this note
(started, not yet finished) -- an 8000-step run already confirmed the
critical bug below is fixed and density stays sane, but hadn't finished
all 3 branches within that window. **GUI visual confirmation of the
original 8 complaints has not happened yet** -- only headless log/number
evidence so far.

### Issues reported and fixes applied

1. **Cap formation too slow -> leaks before the line is ready.** Individual
   per-member repulsion (`compute_khop_gatekeeping_force`) was only ever as
   good as wherever physical bodies currently were mid-formation. Replaced
   with `compute_khop_barrier_segment_force`: a straight wall-segment
   repulsion computed from the group's *target* line position
   (`khop_barrier_line_point`), active immediately from group creation
   regardless of how far individual members have physically converged.
2. **Cap should be one straight line spanning the corridor width, fixed in
   shape, not thick, densely packed (no gaps).**
   - `KHOP_CAPTURE_THICKNESS_ROWS`: 3 -> 1 (no more multi-row requirement).
   - New shared `khop_cap_line_spacing()` (`ROBOT_RADIUS * 2.30`, near the
     physical minimum) used by both `khop_required_shepherd_count` (how
     many robots) and `khop_cap_slot_offsets` (where they sit) -- these
     were previously inconsistent (count assumed tight spacing, placement
     used a much wider one), which is exactly what silently reintroduced
     extra rows even with thickness=1.
   - `KHOP_CAP_EQUILIBRIUM_DISTANCE` (SPH pair-repulsion equilibrium
     between physical cap bodies) now equals the same tight spacing, so
     the physical repulsion doesn't fight the slot-attraction force's
     tighter target.
   - `KHOP_CAP_FORM_TOLERANCE` raised to 1.5x equilibrium distance (was
     0.85x) so formation-complete dwell is still reachable at the new
     tight spacing.
   - A `WAITING` (fully formed, not yet its turn) group's roster/slots are
     no longer re-selected via BFS every refresh cycle -- this is what was
     making an already-formed gate visibly keep changing.
   - Color legend: green NORMAL robot = `khop_stream_id` matches the
     currently active Branch; blue (density-colored) = not yet assigned to
     that stream.
3. **General swarm has no force toward the active Branch, just scatters.**
   `KHOP_UNASSIGNED_FOLLOW_FORCE` (see prior section) was too weak
   (`KHOP_STREAM_FORCE` = 2.5x `MOTION_SPEED_MULTIPLIER`) to meaningfully
   out-compete raw SPH pressure diffusion once ~700 robots are involved.
   Raised to `10.0 * MOTION_SPEED_MULTIPLIER` (4x stronger), still well
   under `KHOP_SHEPHERD_FORCE`/`KHOP_LEADER_FORCE` so the cap stays ahead.
4. **Pressure Push (backtrack) seems to start on a timer, not real
   crowding.** `KHOP_DEAD_END_DENSITY_RATIO` (0.25), `KHOP_DEAD_END_
   MIN_PRESSURE_RATIO` (0.006), `KHOP_DEAD_END_MIN_CONTACT_COMPRESSION`
   (0.075) were all so low that ordinary ambient density right after the
   front merely arrived already cleared them -- dead-end confirmation was
   effectively gated by clearance + dwell alone. Raised to 0.70 / 0.05 /
   0.28 respectively so it requires a real queue. Confirmed in the 8000-
   step run: T_up/T_left dead-ends now fire at density=0.71-0.74,
   contact=0.41-0.44, clearly past ambient.
5/6/7/8. **Gatekeepers/Markers sit near the open Junction center instead
   of each Branch's actual (narrower) mouth** -- root cause of the
   "awkward cross shape in the middle," "Shepherd doesn't actually block
   anything," and "why did everyone suddenly compress toward the center"
   complaints. `KHOP_MARKER_OFFSET` (was `COMM_SAFE_DISTANCE * 0.55` ~=
   15px) and `KHOP_BRANCH_STAGING_OFFSET` (was `* 1.65` ~= 46px) left both
   well inside the ~84px-wide open Junction cross-section for every
   Branch, regardless of which one, so they visually clustered together
   near the middle instead of spreading to each Branch's actual confined
   corridor. Raised to `* 2.20` (~62px) and `* 3.40` (~95px) so they clear
   the Junction opening and land inside the real Branch corridor.

### Critical bug found and fixed during verification of the above

**Two Branch groups ended up sharing the same Leader robot**, causing
severe overcompression (density ratio observed at 5.5x reference,
pressure ratio 4.05x -- a real jam, both groups' cap/barrier lines
computed from literally the same physical robot). Root cause: the earlier
straight-through-branch fix (second round, above) had broadened
`detect_khop_directional_clusters`'s observation role filter to include
`KHOP_DYNAMIC_ROLES` (not just `NORMAL`), so a robot still captured in
root's own cap could *also* get swept into a turning k-means cluster's
`robot_ids`. With this round's larger cap size (29 -> ~35-38, from the
tighter spacing needing more robots), that overlap became likely enough
to actually happen. Two fixes, both applied:
- `detect_khop_directional_clusters` now explicitly excludes any robot
  still in `root.member_robot_ids` from the turning-cluster observation
  loop -- it's the straight-continuation candidate's evidence exclusively.
- `split_khop_from_clusters` now tracks `claimed_robot_ids` across
  clusters and subtracts them before each cluster's own Max-Min run and
  stream assignment, so overlapping `cluster.robot_ids` (from any future
  source, not just this one) can never again let two groups agree on the
  same Leader.

Verified fixed in the 8000-step re-run: three groups, three distinct
Leaders (740/743/723), density ratios back in the 0.6-1.1 range.

## Repo housekeeping

- `single_junction_sph_dfs_Multi_Hop.py.bak-*` (4 files, lowercase `s`,
  under `pygame_simulator/`) are local backup snapshots from earlier
  work sessions. They are **not committed** (left untracked on purpose —
  they're large near-duplicates of this file, not source). Ask before
  committing or deleting them.
