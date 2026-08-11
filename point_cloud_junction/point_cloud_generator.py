"""
point_cloud_generator.py

Anchor 기준 2D LiDAR / Point Cloud 생성기.

핵심 원칙
---------
- 시뮬레이터 내부에서는 벽 좌표와 Anchor world pose를 사용해 ray casting을 한다.
- detector로 넘기는 데이터에는 Global map, Anchor global 좌표,
  Junction 좌표, Branch ID/개수/방향을 포함하지 않는다.
- detector 입력은 Anchor 기준 local angle/range 뿐이다.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LidarScan:
    angle_deg: np.ndarray
    range_m: np.ndarray
    hit: np.ndarray
    local_x: np.ndarray
    local_y: np.ndarray

    def detector_input(self):
        """Junction detector에 전달할 최소 입력."""
        return self.angle_deg.copy(), self.range_m.copy()


def cross2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_distance(ray_origin, ray_direction, seg_start, seg_end, eps=1e-10):
    """
    Ray: O + tD, t >= 0
    Segment: P + uS, 0 <= u <= 1

    교차하면 ray origin에서 교차점까지 거리 t를 반환.
    교차하지 않으면 None.
    """
    o = np.asarray(ray_origin, dtype=float)
    d = np.asarray(ray_direction, dtype=float)
    p = np.asarray(seg_start, dtype=float)
    q = np.asarray(seg_end, dtype=float)

    s = q - p
    denom = cross2d(d, s)

    if abs(denom) < eps:
        return None

    po = p - o
    t = cross2d(po, s) / denom
    u = cross2d(po, d) / denom

    if t >= 0.0 and -eps <= u <= 1.0 + eps:
        return float(t)

    return None


def simulate_lidar_scan(
    wall_segments,
    anchor_xy,
    *,
    anchor_yaw_deg=0.0,
    angle_min_deg=-180.0,
    angle_max_deg=180.0,
    angle_step_deg=1.0,
    max_range_m=6.0,
    noise_std_m=0.0,
    dropout_probability=0.0,
    seed=None,
):
    """
    Anchor에서 ray casting하여 local 2D LiDAR scan 생성.

    Parameters
    ----------
    wall_segments : ndarray, shape (N, 2, 2)
        시뮬레이터 내부 ground-truth 벽 선분.
    anchor_xy : (x, y)
        시뮬레이터 내부 Anchor world 위치.
    anchor_yaw_deg : float
        Anchor world heading.
    angle_* : float
        Anchor 기준 상대 센서 각도 설정.
    max_range_m : float
        센서 최대 거리.
    noise_std_m : float
        hit range에 적용할 Gaussian noise 표준편차.
    dropout_probability : float
        실제 hit를 no-return으로 만들 확률.

    Returns
    -------
    LidarScan
        Anchor local frame 기준 scan.
    """
    walls = np.asarray(wall_segments, dtype=float)
    anchor = np.asarray(anchor_xy, dtype=float)

    if walls.ndim != 3 or walls.shape[1:] != (2, 2):
        raise ValueError("wall_segments must have shape (N, 2, 2)")
    if anchor.shape != (2,):
        raise ValueError("anchor_xy must be length 2")
    if angle_step_deg <= 0:
        raise ValueError("angle_step_deg must be > 0")
    if max_range_m <= 0:
        raise ValueError("max_range_m must be > 0")

    angles_local = np.arange(
        angle_min_deg,
        angle_max_deg,
        angle_step_deg,
        dtype=float,
    )

    ranges = np.full(len(angles_local), max_range_m, dtype=float)
    hits = np.zeros(len(angles_local), dtype=bool)

    yaw_rad = np.deg2rad(anchor_yaw_deg)

    for i, local_angle_deg in enumerate(angles_local):
        world_angle = yaw_rad + np.deg2rad(local_angle_deg)
        direction = np.array([np.cos(world_angle), np.sin(world_angle)])

        nearest = max_range_m
        found = False

        for wall in walls:
            distance = ray_segment_distance(
                anchor,
                direction,
                wall[0],
                wall[1],
            )

            if distance is not None and distance <= nearest:
                nearest = distance
                found = True

        if found:
            ranges[i] = nearest
            hits[i] = True

    rng = np.random.default_rng(seed)

    if noise_std_m > 0 and hits.any():
        ranges[hits] += rng.normal(0.0, noise_std_m, int(hits.sum()))
        ranges[hits] = np.clip(ranges[hits], 0.0, max_range_m)

    if dropout_probability > 0 and hits.any():
        dropout = (rng.random(len(ranges)) < dropout_probability) & hits
        hits[dropout] = False
        ranges[dropout] = max_range_m

    # Point Cloud는 Anchor local frame에서 계산
    theta = np.deg2rad(angles_local)
    local_x = ranges * np.cos(theta)
    local_y = ranges * np.sin(theta)

    return LidarScan(
        angle_deg=angles_local,
        range_m=ranges,
        hit=hits,
        local_x=local_x,
        local_y=local_y,
    )


def make_cross_corridor_walls(half_width_m=1.0, extent_m=10.0):
    """
    테스트용 +자 통로의 벽 선분 생성.

    이 wall 정보는 오직 sensor simulation 내부 ground truth다.
    detector에는 전달하지 않는다.
    """
    w = float(half_width_m)
    L = float(extent_m)

    return np.array(
        [
            # top
            [[-w, L], [w, L]],
            [[w, L], [w, w]],
            [[-w, w], [-w, L]],

            # right
            [[w, w], [L, w]],
            [[L, w], [L, -w]],
            [[L, -w], [w, -w]],

            # bottom
            [[w, -w], [w, -L]],
            [[w, -L], [-w, -L]],
            [[-w, -L], [-w, -w]],

            # left
            [[-w, -w], [-L, -w]],
            [[-L, -w], [-L, w]],
            [[-L, w], [-w, w]],
        ],
        dtype=float,
    )


def save_local_scan_csv(scan, path):
    """
    local scan만 CSV로 저장.
    Global wall/Anchor/Junction/Branch 정보는 저장하지 않는다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "range_m", "hit", "local_x_m", "local_y_m"])

        for a, r, h, x, y in zip(
            scan.angle_deg,
            scan.range_m,
            scan.hit,
            scan.local_x,
            scan.local_y,
        ):
            writer.writerow([
                f"{a:.6f}",
                f"{r:.6f}",
                int(h),
                f"{x:.6f}",
                f"{y:.6f}",
            ])


def plot_demo(walls, anchor_xy, scan, save_path=None, show=True):
    """
    왼쪽: sensor simulator의 ground truth 확인용.
    오른쪽: Anchor가 실제 detector에 넘길 수 있는 local point cloud.
    """
    anchor = np.asarray(anchor_xy, dtype=float)

    fig = plt.figure(figsize=(12, 5.5))

    ax1 = fig.add_subplot(1, 2, 1)
    for wall in walls:
        ax1.plot(
            [wall[0, 0], wall[1, 0]],
            [wall[0, 1], wall[1, 1]],
            linewidth=2,
        )
    ax1.scatter(anchor[0], anchor[1], marker="x", s=80, label="Anchor")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Simulator ground truth")
    ax1.set_xlabel("world x [m]")
    ax1.set_ylabel("world y [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(
        scan.local_x[scan.hit],
        scan.local_y[scan.hit],
        s=10,
        label="LiDAR returns",
    )

    no_hit = ~scan.hit
    if no_hit.any():
        ax2.scatter(
            scan.local_x[no_hit],
            scan.local_y[no_hit],
            s=6,
            alpha=0.25,
            label="No return / max range",
        )

    ax2.scatter([0], [0], marker="x", s=80, label="Anchor local origin")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Anchor-local point cloud")
    ax2.set_xlabel("local x [m]")
    ax2.set_ylabel("local y [m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle-step", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=6.0)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--anchor-x", type=float, default=0.25)
    parser.add_argument("--anchor-y", type=float, default=-0.15)
    parser.add_argument("--anchor-yaw", type=float, default=17.0)
    parser.add_argument("--csv", type=str, default="local_scan.csv")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    walls = make_cross_corridor_walls(
        half_width_m=1.0,
        extent_m=10.0,
    )

    anchor_xy = (args.anchor_x, args.anchor_y)

    scan = simulate_lidar_scan(
        walls,
        anchor_xy,
        anchor_yaw_deg=args.anchor_yaw,
        angle_step_deg=args.angle_step,
        max_range_m=args.max_range,
        noise_std_m=args.noise,
        dropout_probability=args.dropout,
        seed=7,
    )

    save_local_scan_csv(scan, args.csv)

    print("=== Local LiDAR / Point Cloud generated ===")
    print(f"samples     : {len(scan.angle_deg)}")
    print(f"hits        : {int(scan.hit.sum())}")
    print(f"no returns  : {int((~scan.hit).sum())}")
    print(f"csv         : {Path(args.csv).resolve()}")
    print()
    print("Detector receives only:")
    print("    angles_deg, ranges_m = scan.detector_input()")
    print()
    print("Detector does NOT receive:")
    print("    walls")
    print("    anchor global x/y/yaw")
    print("    junction coordinates")
    print("    branch labels/count/directions")

    plot_demo(
        walls,
        anchor_xy,
        scan,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()