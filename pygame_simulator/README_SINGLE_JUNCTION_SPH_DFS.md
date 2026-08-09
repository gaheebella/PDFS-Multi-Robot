# Single-Junction SPH Physical DFS

This document records the current reproducible state of
`single_junction_sph_dfs_environment.py`.

## What the simulator implements

- A pressure-driven SPH swarm released from a Base into a three-branch Junction.
- Emmons-inspired Junction inference from sequential local angular distribution,
  lateral expansion, and persistent branch-crossing cohorts.
- NORMAL-to-NORMAL branch voting. The Anchor stores and relays consensus; it
  does not choose a branch on behalf of the swarm.
- Eguchi-inspired command/observed-velocity tracking error for indirect contact
  evidence and obstacle-response samples.
- Base-rooted line-of-sight communication with reactive Breadcrumb Relays.
- Persistent selected-branch `FRONTIER_SHEPHERD` robots that move with the
  NORMAL front, detect a dead end through direct forward contact, then remain
  the return Shepherd line during pressure-driven backtracking.
- Physical guards at unselected branch mouths. Logical `CLOSED` states are
  messages, not simulator geofences.
- Adaptive, leader-seeded K-hop thick mouth walls for high-pressure swarms.

## Current exploration sequence

1. All branch mouths remain physically open during initial SPH diffusion.
2. A Junction is confirmed only after lateral expansion and persistent robots
   physically cross each candidate mouth.
3. A terminal robot with no farther outward same-branch neighbor becomes the
   branch guard Leader.
4. A minimum K-hop group first forms a full-width line at each observed branch
   frontier.
5. NORMAL peer consensus selects the next unvisited branch.
6. The selected mouth recruits enough connected robots to form a dense
   full-width `FRONTIER_SHEPHERD` line. Slot spacing is kept below the local
   `SAFE_RADIUS`; the current corridor produces 17 moving Shepherds.
7. Every unselected branch recruits additional connected NORMAL robots up to
   four hops and forms a two-to-four-layer physical wall on the branch side of
   its mouth.
8. Selected-branch flow remains paused until every thick mouth wall is in its
   assigned position and stable for the configured dwell time.
9. At the selected dead end, multiple frontier Shepherds must directly contact
   the boundary across a sufficient lateral span while forward speed remains
   low. The same dense Shepherd IDs then form the return piston.
10. Pressure-driven backtracking returns the swarm to the Junction. Thick walls
    at still-unvisited branches retain the same robot IDs, anchors, column
    count, and layer count; they are not collapsed and rebuilt between visits.
11. After every branch is visited, temporary roles are released and the swarm
    returns to Base.

## Physical gatekeeper behavior

The simulator no longer blocks an unselected branch in `is_walkable()` and no
longer applies a map-aware virtual valve or final one-way hard gate. Instead:

- Unselected branch guards remain real `JUNCTION_GUARD` robots.
- Each guard communicates its branch-facing orientation.
- Nearby NORMAL robots use only relative guard position and that communicated
  direction to move back toward the Junction.
- Overlapping guard influence disks and the physical depth of the layered wall
  resist SPH pressure without an invisible barrier.

The HUD separates the logical command from its physical implementation:

- `Gate commands (no geofence)`
- `Physical mouth guards`
- `Thick K-hop walls`

## Thick K-hop mouth walls

The original guard count covered only corridor width, producing one row of
approximately 5 robots in a 160-robot run or 9 robots in a 680-robot run. That
row was too thin to resist sustained junction pressure.

The current policy is `ADAPTIVE_KHOP_LAYERED_MOUTH_WALL_V1`:

- Columns are calculated from observed effective branch width.
- Layer count is selected from local arriving mass/density and swarm scale,
  clamped to 2--4 layers.
- Existing frontier guards are the seed group.
- Additional NORMAL robots are recruited through the communication graph, up
  to `JUNCTION_GUARD_MAX_HOPS = 4`.
- Slots form full-width axial rows entirely on the branch side of the mouth.
- Once a valid unvisited wall is formed, it persists across branch switches and
  is never recomputed to a smaller width.
- When that branch is selected, the moving cross-section is replenished to the
  width/Safe-radius target instead of falling back to five robots.
- Exploration begins only after `wall_ready` remains true for the formation
  dwell time.

Observed validation configurations:

| Swarm size | Moving selected line | Thick unselected wall | Result |
|---:|---:|---:|---|
| 160 | 17 robots | 5 columns x 3 layers = 15 per unselected branch | UP and RIGHT exploration, contact-confirmed dead-end inference, and backtracking completed with the same 17 IDs |
| 680 | width/Safe-radius target (17 in the current corridor) | observed 8--9 columns x 3 layers = 24--27 per unselected branch | Thick walls form before selected flow; a 72-second post-change smoke run completed without errors but did not reach a full branch switch at the lower FPS |

## Dead-end inference

A known terminal coordinate is not used as the state-transition trigger. The
frontier confirms a dead end from:

- direct forward bumper/proximity contact memory on frontier Shepherds;
- sufficient ratio of contacting Shepherds;
- contact spanning enough of the corridor width;
- low observed forward speed;
- persistence for the configured dwell time.

The renderer and collision mask still know the test fixture geometry. The
controller also retains known branch axes/regions for several projections and
role filters, so this is not yet a fully localization-free system.

## Remaining research limitations

- The environment contains one known cross-shaped test fixture.
- Branch labels/directions (`UP`, `LEFT`, `RIGHT`) and several region tests are
  still global-map abstractions.
- Anchor election still uses a predefined Junction region and parking slots.
- Shepherd and guard target formations still use branch-relative slots.
- The selected dead-end return boundary retains a simulator curtain mechanism.
- Completed-branch Pebble/Marker robots are not implemented yet.
- Recursive multi-junction DFS tree construction and repair are not implemented.

Recommended next changes should be isolated and validated one at a time:

1. Leave a small physical Pebble/Marker group at completed branch mouths.
2. Let an elected Anchor hold its current pose instead of moving to a map slot.
3. Preserve a moving Shepherd Tree's relative K-hop thickness without flattening
   every robot onto one cross-section.
4. Replace branch axes and region membership with cohort-derived local frames
   and communicated edge signatures.

## Requirements

- Python 3.12 recommended
- `pygame-ce==2.5.7`
- Windows, Linux, or macOS with an SDL-compatible display

No external map, image, or data asset is required for the single-file simulator.

## Continue on another local machine

Until Draft PR #1 is merged into `main`, use the published working branch:

```powershell
git clone https://github.com/gaheebella/PDFS-Multi-Robot.git
cd PDFS-Multi-Robot
git switch claude/pygame-simulator-review-aj16vc
powershell -ExecutionPolicy Bypass -File .\pygame_simulator\run_single_junction_sph_dfs.ps1
```

Detailed system state, problem/root-cause/fix history, validation evidence, and
next-work checklist are recorded in
[`SYSTEM_PROBLEM_SOLUTION_LOG.md`](SYSTEM_PROBLEM_SOLUTION_LOG.md).

## Quick start on Windows

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\pygame_simulator\run_single_junction_sph_dfs.ps1
```

To choose a different swarm size:

```powershell
.\pygame_simulator\run_single_junction_sph_dfs.ps1 -RobotCount 160
```

The script creates `pygame_simulator/.venv-sph-dfs`, installs the pinned
dependency, sets `SPH_DFS_ROBOT_COUNT`, and starts the simulator.

## Manual setup

```powershell
cd pygame_simulator
python -m venv .venv-sph-dfs
.\.venv-sph-dfs\Scripts\Activate.ps1
python -m pip install -r requirements-sph-dfs.txt
$env:SPH_DFS_ROBOT_COUNT = "680"
python .\single_junction_sph_dfs_environment.py
```

Linux/macOS:

```bash
cd pygame_simulator
python3 -m venv .venv-sph-dfs
source .venv-sph-dfs/bin/activate
python -m pip install -r requirements-sph-dfs.txt
SPH_DFS_ROBOT_COUNT=680 python single_junction_sph_dfs_environment.py
```

## Controls

| Key | Action |
|---|---|
| `Space` | Pause/resume |
| `R` | Reset |
| `D` | Toggle density coloring |
| `V` | Toggle diagnostic regions |
| `C` | Toggle communication links |
| `Esc` | Exit |

The simulator writes `pygame_simulator/sph_dfs_experiment_summary.csv` on a
completed run or user exit. This generated file is intentionally ignored by Git.

## Headless smoke test

On an SDL-capable CI or local shell:

```powershell
$env:SDL_VIDEODRIVER = "dummy"
$env:SPH_DFS_ROBOT_COUNT = "160"
python .\pygame_simulator\single_junction_sph_dfs_environment.py
```

For automated runs, post a Pygame `QUIT` event from a timer because the normal
program loop is interactive.
