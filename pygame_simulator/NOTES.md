# Handoff notes — natural-spread Junction discovery (straight-through branch)

Branch: `feature/rebuild-multi-junction-from-single`
File: `pygame_simulator/Single_junction_sph_dfs_Multi_Hop.py`

## START HERE — session handoff (continuing in a fresh session)

This file is being read by a **new session** picking up from the last
pushed commit on this branch. Everything below "START HERE" is the
accumulated history from prior sessions, kept as reference — read this
section first for where things actually stand.

### What's committed and safe
Every code change described in this file (rounds 1-6, the T_right fix,
and the normalization pass below) is committed **locally**. **As of this
note, round 6 (root Leader re-election) has been handed to the user as a
raw file (not a patch -- see the session-infra note below for why) to
apply and push themselves**, since this session's own git-push
credential was 403 all session (see the session-infra note further
down). Check `git log origin/claude/pygame-simulator-review-aj16vc` vs
local `HEAD` to see the actual gap -- if `origin` is still at `67bd414`,
round 6 hasn't landed yet. The net *executable* diff from `67bd414` is
just the re-election extension (~35 real lines across
`update_khop_front_leader` and `update_khop_capture_state`'s ADVANCING
block) plus some comment-only additions carried over from round 5's
dead-end documentation. Nothing is sitting *uncommitted* either way -- if
`git status` shows a dirty working tree, that's this session's
in-progress edits, not something to discard.

**If starting a fresh session and `git status` looks confusing**: this
session's local branch and `origin`'s branch diverged in commit
granularity partway through (some rounds were squashed differently when
pushed locally vs. how they were committed in-session) -- if `git log
--graph --all` shows two divergent lines from a shared ancestor, that's
expected, not corruption. What matters is whether the *file content* at
each tip matches, not the commit-by-commit history -- diff the actual
`.py` file between local `HEAD` and `origin/claude/pygame-simulator-
review-aj16vc` to check, not just commit hashes.

For prior sessions' pushes: the commit graph on GitHub may not exactly
match individual commit messages described below for rounds 1-4 -- some
fixes were handed to the user as raw file content (not `git am` patches,
which kept failing with "corrupt patch") and squashed into fewer commits
when they committed locally. The code content is byte-identical to what's
described here regardless of exact commit boundaries.

### Round 4 addendum: root cap was also blocking the doorway
Found right after the three fixes below, from the user noticing the root
cap sits at the Junction *entrance boundary* (confirmed via coordinate
probe: `leader_pos=(398.3, 390.6)` vs `junction_rect` bottom edge at
`y=392` -- within 1.4px of the doorway, not the Junction's geometric
center at `y=350`) and asking why it doesn't obstruct the trailing swarm
still flowing up from Base. It turned out it did: `khop_cap_slot_offsets`
is a generic function shared by every `KHopShepherdGroup`, so the root
group got the exact same full-corridor-width barrier-line shape round 3
built specifically for branch gates -- even though the root group never
receives the actual blocking force (`compute_khop_barrier_segment_force`
only applies to `khop_branch_groups()`, i.e. groups *with* a
`parent_group_id`, and root's is always `None`). A ~30-member dense line
sitting right across the doorway still created real physical crowding
for the trailing body to push through even without an artificial block.
Added `KHOP_ROOT_CAP_HALF_WIDTH` (half the branch-gate width) and made
`khop_cap_slot_offsets` use it for the root group specifically, packing
the same member count into a narrower, deeper column that leaves both
flanks of the corridor open. Verified via a 14000-step run: still
`stage=COMPLETE`, all groups `RELEASED`, `valid-cohorts=3`, 722/760
(95.0%) at Base -- and branch formation now completes markedly faster
(within a few hundred frames instead of a few thousand), consistent with
the trailing swarm no longer queuing behind root's cap to reach the
Junction.

### Round 5: root cap width extremes + a gather-gate, both tried and reverted

User asked for two more things on top of round 4's narrow root-cap
column: (1) even less footprint (zero lateral width, true single-file)
since the narrow column still theoretically deflects the trailing body,
and (2) a hard requirement that the general swarm actually gather at the
Junction before the first branch starts exploring, not just that every
branch cap finish forming. Both were implemented, and **both
independently regressed and were reverted** -- this is the single most
important thing to know before touching either area again.

**Zero-width root cap (reverted, see `khop_cap_slot_offsets`'s comment
for the in-code version of this).** `KHOP_ROOT_CAP_HALF_WIDTH = 0.0` made
every row's per-member lateral targets literally coincide (the shared
row-math floors at 2 members per row regardless of width, and at
half_width=0 both land at the same point). Two regressions from this,
found via a 14000-step run that stalled at <500 frames after 5+ minutes
(normally completes well under an hour):
- Perfectly-overlapping target pairs made SPH neighbor/pressure
  computation pathologically expensive once those members were released
  back into the general crowd.
- Even before release, the shape itself is a single-file chain ~38
  members deep, which took 6500+ frames just for root cap formation to
  converge, and `CONTROLLED_SPREAD` never once reached
  `JUNCTION_CONFIRMED` in the full 14000-frame run -- root's Leader just
  kept advancing straight up the Up corridor past the Junction instead
  of the natural-spread anomaly ever validating.

Fix: reverted to the round-4 narrow-but-nonzero width
(`ROBOT_RADIUS * 2.30 * 1.5`), which was already proven working. Zero
width is **not** a safe direction to explore further without redesigning
the shape (e.g. a wide-but-thin ellipse, or capping single-file depth and
falling back to 2-wide beyond some length) rather than just shrinking the
existing row-based shape to its limit.

**Gather-before-explore gate (reverted).** Added a check (`khop_state.
junction_gather_dwell`, `KHOP_JUNCTION_GATHER_READY_RATIO`/`_DWELL`,
`khop_junction_gather_ratio()`) delaying only the *first* branch
activation until 60% of the whole swarm was physically inside the
Junction/branch regions (`get_robot_region`), on top of the existing
"every cap finished FORMING" requirement. This is **not a discovery
decision** -- confirmed still correct, that stays strictly
local-evidence-only -- it only gates when the already-confirmed branches
start being explored, using the same fixed-geometry-steering category
already accepted for `direction_toward_branch_path`/`_base_path`.

Despite that reasoning, it caused a real, **fully deterministic**
regression: `T_left` got stuck in `RETURNING` for the rest of a
14000-frame run, density climbing to 4.5x reference (normal peak is
~1.0-1.3x), never completing its Marker handshake back. This was not a
random fluke -- this codebase has no RNG seed, so identical code always
produces an identical run (same Leader IDs, same rollout costs, frame for
frame), which made root-causing this tractable:
1. Checked out the last known-good pushed commit (`ffb23bd`) in a
   separate `git worktree` and ran it fresh: completed cleanly
   (`stage=COMPLETE`, `T_left` dead-end at `density=1.09`), with
   byte-identical Leader IDs/rollout costs to the broken run -- so this
   wasn't pre-existing, something in the two new changes caused it.
2. Disabled just the gather-gate (kept the root-cap-width revert above)
   and reran: reproduced the clean baseline exactly. This isolated the
   gather-gate specifically, not the root-cap change, as the cause.

Root cause understood only at the level of "delaying the first
activation shifts population/timing state enough to cascade into a later
branch's return navigation failing" -- not traced further into exactly
*why* `T_left`'s Marker handshake specifically breaks, given the time
already spent. The gather-gate code was fully removed (not just
disabled) with a comment at the call site in `update_khop_capture_state`
recording what was tried and why, so a future session doesn't just
flip it back on. If asked to implement this again, a safer approach
would cap the maximum delay tightly, or gate branch-cap *formation*
itself rather than *activation* (so the delay can't outlast anything
else in the state machine) -- don't just re-enable this version.

**Verified after both reverts**: fresh 14000-step run reproduces the
`ffb23bd` baseline byte-for-byte -- `stage=COMPLETE`, all three groups
`RELEASED`, `valid-cohorts=3`, 722/760 (95.0%) at Base. The net
*executable* diff from `ffb23bd` is nothing; only comments recording
these two dead ends were added.

### Round 6: root Leader re-election during ADVANCING — RESOLVED, verified

User's framing this round: the root cap shouldn't be a fixed formation
that marches to the Junction with a Leader locked in from the very start
-- if the swarm holds a rigid lead formation while advancing, that
suppresses the free lateral spread the whole natural-spread Junction
detection design depends on. Branch groups already re-elect their front
Leader during `EXPLORING` (`update_khop_front_leader`, distance/dwell
hysteresis via `KHOP_LEADER_REELECT_DISTANCE`/`_DWELL`); root never had
this at all -- whoever `initialize_khop_capture`'s FRONT_LOCAL selection
picked once stayed authoritative for the rest of the march to the
Junction, even if a different robot genuinely pulled ahead.

**Fix**: extended `update_khop_front_leader`'s early-return guard to
also allow root through. Root doesn't carry its own `EXPLORING` state
the way branches do (`group.state` does become `EXPLORING` once cap
formation finishes, but that's shared with `CONTROLLED_SPREAD`, where
the Leader must stay fixed as the `split_origin`/`heading` reference for
cohort-displacement measurement) -- gated on `khop_state.stage ==
"ADVANCING"` instead, reusing the same hysteresis constants already
proven for branches, not new ones. Added the matching call site in the
`ADVANCING` block of `update_khop_capture_state`, same ordering as the
branch refresh loop (re-election before the capture tree rebuilds each
cycle, so a swap takes effect the same cycle instead of lagging one
behind).

This only applies to root's advance from Base to the Junction. Each
branch's own Shepherd gate formation (rigid corridor-width row, guarding
the branch entrance once a branch starts being explored) is untouched --
that's supposed to stay fixed, since it functions as a physical barrier.

**A process failure worth recording**: a follow-up request the same
round -- remove root's physical slot-alignment force entirely
(`compute_khop_cap_force`/`update_khop_cap_formation`), so Leader
tracking stays but nothing pulls members to a fixed slot position -- was
implemented and *committed in the same commit* as the (already-verified)
re-election change, with the commit message itself flagging the
slot-force part as "not yet independently verified, verification in
progress." That combined-state verification run then found a real
regression: `T_right` stuck in `RETURNING` for the rest of a 14000-frame
run, density `4.35x` reference, never completing its Marker handshake --
the exact same failure signature as round 5's gather-gate regression,
just on a different branch. A `git checkout -- <file>` was run to
revert, but that only discards *uncommitted* changes; the slot-force
removal was already baked into the prior commit, so the regression-
causing code silently remained on disk through a full session
compaction, and was only caught by re-diffing against the last verified
commit before finally pushing. **Lesson: never bundle a verified change
and an unverified one into the same commit, even if the unverified part
is added right at the end of a long session** -- commit the verified
part first, then verify the risky addition in isolation before it
touches history at all.

The slot-force-removal idea itself is not concluded to be wrong -- it
just hasn't been isolated and verified on its own yet. If revisited, test
it alone (re-election already merged, nothing else changed), not bundled
with anything else, using the same worktree-diff isolation method round
5 used.

**Verified**: fresh 14000-step run (re-election only, slot-force removal
excluded) reaches `stage=COMPLETE`, all three groups `RELEASED`,
722/760 at Base -- the final `[Headless]` summary line is byte-identical
to a separately captured known-good re-election-only run, confirming the
fix reproduces exactly that state with no regression.

### Latest round: T_right dead-end starvation — RESOLVED, verified

A completed 14000-step run (from the session before this one) found a
real, confirmed bug: `T_up`/`T_left` both reached `Dead-end confirmed`
correctly, but **`T_right` (the last-queued branch) never did** --
local density plateaued at a genuine steady state of `rho=0.61`,
permanently below `KHOP_DEAD_END_DENSITY_RATIO` (0.70), unchanged for
6000+ frames, while `clearance=2.8` (front already at the wall) and
`contact=0.426` (already above the 0.28 floor) both already showed real
crowding.

**Fix** (`update_khop_dead_end_detection`): `density_observed` was a
separate hard AND alongside `compression_observed` (pressure OR
contact). Density-ratio is area-normalized, so a branch whose corridor
happens to be wider reads a lower ratio for the same physical jam --
requiring it in addition to an already-crowding-proven contact/pressure
signal reintroduced exactly the single-point-of-failure gate the round-3
multi-signal design was meant to avoid, just shifted from "too easy" to
"impossible for some branches." Folded density into the same three-way
OR as pressure/contact instead: `clearance` and `forward_speed` stay
hard gates, but any one of density/pressure/contact now independently
proves crowding.

**Verified** via a fresh 14000-step run with this fix (plus the
normalization change below) applied together:
```
[Headless] frames=14000, stage=COMPLETE
groups=[T_left RELEASED, T_up RELEASED, T_right RELEASED]
valid-cohorts=3, visited=[T_up, T_left, T_right]
swarm gathered back at Base: 724/760 (95.3%)
```
`T_right` itself confirmed dead-end at `density=0.63, pressure=0.003,
contact=0.427` -- density still below the old 0.70 threshold, contact
alone now correctly gates it. No regression in `T_up`/`T_left` (both
still confirm as before, `T_up` even fired at `density=0.65` this run,
below 0.70, again via contact alone -- consistent with the fix working
as intended, not just papering over one run's numbers).

### Latest round: normalization — DONE, verified in the same run above
Applied the normalization ideas from a swarm-distribution-as-environment-
sensor paper the user shared, **scoped to normalization only** as
explicitly decided in a prior session:
- Replaced `NATURAL_SPREAD_ANOMALY_ROBOTS` (fixed at 7) and
  `KHOP_SPLIT_CLUSTER_MIN_SIZE` (fixed at 18, previously aliased to
  `KHOP_SPLIT_MIN_GROUP_SIZE`) with `natural_spread_anomaly_threshold()`
  and `khop_split_cluster_min_size()`, each a ratio of
  `khop_state.last_split_group_size` (the locally observable population
  at `NATURAL_SPREAD_LOCAL_HOPS`) with a floor. Ratios (0.0145, 0.0373)
  were calibrated to reproduce the previously tuned absolute values at
  the population actually measured at anomaly detection (~483 robots).
- `KHOP_SPLIT_MIN_GROUP_SIZE` was deliberately **left absolute** and
  decoupled from the new `khop_split_cluster_min_size()` -- it's a
  physical/geometric floor (corridor-width-driven cap/gate viability,
  connected-tree re-election, retry pool size), not a detection-evidence
  threshold, so normalizing it against swarm population would be
  conflating two different questions.
- **Still explicitly out of scope** (user deferred this, don't build it
  unprompted): a unified weighted composite score
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

### Round 4: GUI-driven physical formation fixes — RESOLVED, verified

Done this session, using `--headless-snapshot=<path>` (see
`save_khop_headless_snapshot`, module docstring for the flag -- renders
one physical frame via the SDL dummy driver to a PNG without needing a
real display) plus the user's own live GUI run on their local machine.
This is the first time any of round 3's physical-formation claims were
actually looked at as a rendered image rather than just headless log
numbers, and it surfaced three real, previously-undiscovered bugs:

1. **Skewed branch heading -> diagonal, undersized barrier lines.**
   `split_khop_from_clusters` set a turning branch's `group.heading` once
   at creation from `cluster.heading` -- a k-means mean of scattered
   pre-filter witness observations, individually allowed to differ from
   the true branch axis by up to `NATURAL_SPREAD_COHORT_MAX_ANGLE_STD`
   (25 degrees) -- and never corrected it afterward, despite it being the
   forward axis for corridor-width probes, cap slot/barrier-line
   orientation, and the dead-end forward-clearance probe. Fixed by
   recomputing it from the elected branch Leader's real displacement from
   the split origin once assigned (a much cleaner post-Max-Min-election
   signal), **excluding** the straight-through cluster
   (`direction_variance == 0.0`) since its heading is exact by
   construction and a first attempt that didn't exclude it visibly
   regressed `T_up`'s cap convergence (caught via a background regression
   run before committing, not after).
2. **General swarm not funneling into the active branch.**
   `compute_khop_unassigned_follow_force` pushed every `NORMAL` robot with
   a single raw constant-direction force along the active branch's
   heading -- fine for a robot with a clear line to that branch's mouth,
   but a robot already inside a *different* branch corridor just got
   pushed into that corridor's own side wall, going nowhere. Added
   `direction_toward_branch_path` (mirrors the existing
   `direction_toward_base_path` already used for `RETURNING_TO_BASE`):
   route through the current corridor's entrance and the Junction center
   first if not already in the active branch's own region. Confirmed via
   snapshot: the previously-unrelated branch corridor (`T_up`, while
   `T_right` was active) went from visibly full of stream-assigned
   (green) robots to essentially none.
3. **Shepherd width converging too slowly, and never fixed.** Measured
   via a temporary probe: `khop_estimate_corridor_width` blended toward
   each new two-sided wall reading at a slow 20%-per-refresh-cycle rate,
   taking ~30 refresh cycles to even approach its own (already
   conservative, ~90% of the true 84px corridor) asymptote, and kept
   re-measuring for as long as a group stayed `FORMING`/`EXPLORING`
   rather than ever locking -- a physical barrier that keeps resizing
   itself doesn't read as "one fixed line." Added
   `KHopShepherdGroup.width_locked`: take the first valid two-sided
   reading directly (no blending) and lock it for the group's lifetime.
   Confirmed via probe: locks immediately at the same ~75.6px value the
   old code used to take dozens of cycles to approach.

**Verified together** via a fresh 14000-step run with all three fixes
(plus the T_right fix and normalization) applied: `stage=COMPLETE`, all
three groups `RELEASED`, `valid-cohorts=3`, all three branches visited,
723/760 (95.1%) gathered at Base. Snapshots at intermediate frames
visually confirm straight full-width barrier lines and no green
(stream-assigned) robots left in inactive-branch corridors.

**Known remaining gap, not yet addressed:** even the locked ~75.6px
width leaves roughly a 4px margin at each end between the barrier line
and the physical wall (robot diameter is ~2.24px) -- theoretically wide
enough for one robot to squeeze past at an endpoint, though no actual
mass leakage has been observed in any completed run's Base-return
percentage (consistently ~95%+). This comes from a deliberately
conservative safety margin in `khop_local_clearance` (used for general
collision-avoidance probing too, not just this estimate) plus probe-step
quantization. Narrowing it specifically for the corridor-width estimate
(without weakening general collision-safety probing elsewhere) was
raised but not implemented this session -- worth a follow-up if leakage
is ever actually observed rather than just theoretically possible.

### Session-infra note: cloud-session git push was broken for one session
Not a code issue, don't re-investigate if seen again without cause: for
this entire session, this environment's git proxy issued a **read-only**
credential (`git push` → `Permission ... denied`; MCP `push_files`/
`create_branch` → `403 Resource not accessible by integration`) despite
the account itself having working push access (a session immediately
prior had pushed fine, e.g. the `9f74b2b8` commit below), and it never
cleared for the rest of the session even after retrying many times over
several hours. Confirmed via the "Claude Code on the web" docs
(code.claude.com/docs/en/claude-code-on-the-web) that repo write access
comes from GitHub App authorization or `/web-setup` token sync at the
*account* level, not anything fixable per-session from inside the
sandbox. Worked around by handing the final file content directly to the
user (a `git am` patch bundle failed with "corrupt patch" -- likely
mangled by a Windows copy/paste/download round-trip -- so raw file
downloads that got `copy`'d over the local working tree were used
instead) for them to commit and push from their local machine, which
worked immediately once done that way. `mcp__github__*` write tools came
back later in the same session (after a server reconnect) and were used
directly to fix a stale `NOTES.md` push (see next paragraph), so the
credential issue may have been specific to the git-push proxy path
rather than every write path -- worth noting if it recurs. If a fresh
session hits the same symptom, don't spend time on GitHub App
installation UI -- retry a few times, and if it doesn't clear, hand the
user the raw changed files (not a patch, to avoid the corruption risk)
to commit locally, or try the `mcp__github__` write tools directly since
they may work independently of the git-push proxy.

One more thing worth flagging for a future session: the local `copy`
command the user ran to overwrite `NOTES.md` from a downloaded file
picked up a stale/unrelated older copy instead of the one actually sent
that session (confirmed after the fact: the pushed `NOTES.md` matched
the very first version from before this session even started, not
anything written during it) -- the `.py` source file copied correctly in
the same round, so this seems specific to `NOTES.md`, possibly a stale
file already sitting in the user's Downloads folder under the same name
from an unrelated earlier occasion. Worth double-checking a freshly
`copy`'d `NOTES.md` actually changed (e.g. `git diff` before committing)
rather than assuming a successful `copy`/`commit`/`push` sequence means
the content is right.

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
