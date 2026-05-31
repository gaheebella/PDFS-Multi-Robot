#!/usr/bin/env python3
"""
follower_node.py  –  Phase 4 (multi-follower, relay role-handoff)

실행 파라미터 (ros2 param):
  robot_id        : 'follower1' | 'follower2'   (기본 follower1)
  initial_role    : 'EXPLORER'  | 'IDLE'        (기본 EXPLORER)

토픽 규칙:
  follower1  →  /follower_odom, /follower_cmd_vel, /follower_scan
  follower2  →  /follower2_odom, /follower2_cmd_vel, /follower2_scan

역할 전환 신호:
  발행: /relay_event  (std_msgs/String)  "RELAY_PLACED x=.. y=.."
  구독: /relay_event  →  IDLE 로봇이 수신 후 EXPLORER 전환

상태머신:
  EXPLORER        : leader breadcrumb 추종
  RELAY_CANDIDATE : relay 배치 임박
  STATIONARY_RELAY: 정지, /relay_event 발행
  IDLE            : 대기, /relay_event 수신 시 EXPLORER 전환
"""
import math
import collections
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class FollowerController(Node):
    def __init__(self):
        super().__init__('follower_controller')

        # ── ROS 파라미터 ─────────────────────────────────────────────────────
        self.declare_parameter('robot_id',     'follower1')
        self.declare_parameter('initial_role', 'EXPLORER')
        self.robot_id    = self.get_parameter('robot_id').value
        initial_role     = self.get_parameter('initial_role').value

        # 토픽 prefix 결정
        if self.robot_id == 'follower2':
            odom_topic = '/follower2_odom'
            cmd_topic  = '/follower2_cmd_vel'
            scan_topic = '/follower2_scan'
        else:
            odom_topic = '/follower_odom'
            cmd_topic  = '/follower_cmd_vel'
            scan_topic = '/follower_scan'

        # ── 상태머신 ─────────────────────────────────────────────────────────
        # EXPLORER | RELAY_CANDIDATE | STATIONARY_RELAY | IDLE
        self.robot_state = initial_role   # 'EXPLORER' or 'IDLE'

        # ── Relay Placement 파라미터 ─────────────────────────────────────────
        self.comm_range      = 2.0
        self.relay_threshold = 0.85          # 2.0 * 0.85 = 1.7m
        self.last_relay_pos  = None
        self.relay_positions = []
        # ── junction 우선 배치 가드 ──
        self.min_relay_gap = 0.6        # 직전 relay에서 이만큼 떨어져야 junction 배치 허용
        self.junction_confirm_n = 3     # 연속 N tick junction 감지 시 확정 (오탐 방지)
        self._junction_count = 0        # junction 연속 감지 카운터
        self.explorer_start_pos = None

        # ── Leader 우선권 양보 상태머신 ──────────────────────────────────────
        self._yield_state       = 'follow'
        self._prev_leader_dist  = None
        self._prev_leader_pos   = None
        self.approach_detect_dist = 0.50
        self.approach_clear_dist  = 0.65
        self.backoff_speed        = -0.10

        # ── odom 안정화 ──────────────────────────────────────────────────────
        self._init_ticks = 0
        self._init_done  = False

        # ── 제어 파라미터 ─────────────────────────────────────────────────────
        self.follow_distance   = 0.01
        self.k_linear          = 0.8
        self.k_angular         = 2.5
        self.max_linear        = 0.35
        self.max_angular       = 1.5
        self.search_angular_speed = 1.0

        # ── 히스테리시스 ─────────────────────────────────────────────────────
        self.ROTATE_ENTER = 0.4
        self.ROTATE_EXIT  = 0.15
        self._rotating    = False

        # ── Breadcrumb ────────────────────────────────────────────────────────
        self.use_breadcrumb       = False
        self.path_record_min_dist = 0.03
        self.max_path_length      = 3000
        self.waypoint_reach_dist  = 0.08
        self.lookahead_steps      = 4
        self.leader_path          = collections.deque(maxlen=self.max_path_length)
        self.waypoint_idx         = 0

        # ── 장애물 회피 ──────────────────────────────────────────────────────
        self.obstacle_stop_dist  = 0.35
        self.obstacle_clear_dist = 0.42
        self.obstacle_turn_speed = 1.0
        self._avoiding           = False

        # ── 포즈 캐시 ────────────────────────────────────────────────────────
        self.leader_pose   = None
        self.follower_pose = None
        self.scan_msg      = None

        # ── 구독 / 발행 ──────────────────────────────────────────────────────
        self.leader_sub = self.create_subscription(
            Odometry, '/leader_odom', self.leader_callback, 10)
        self.follower_sub = self.create_subscription(
            Odometry, odom_topic, self.follower_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(
            Twist, cmd_topic, 10)

        # relay_event: 발행(STATIONARY_RELAY) + 구독(IDLE)
        self.relay_event_pub = self.create_publisher(
            String, '/relay_event', 10)
        self.relay_event_sub = self.create_subscription(
            String, '/relay_event', self.relay_event_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f'[{self.robot_id}] init: state={self.robot_state}, '
            f'odom={odom_topic}, cmd={cmd_topic}'
        )

    # ------------------------------------------------------------------ #
    #  콜백
    # ------------------------------------------------------------------ #

    def leader_callback(self, msg):
        self.leader_pose = msg.pose.pose
        lx = msg.pose.pose.position.x
        ly = msg.pose.pose.position.y

        if self.leader_path:
            last_x, last_y = self.leader_path[-1]
            if math.hypot(lx - last_x, ly - last_y) <= self.path_record_min_dist:
                return
        self.leader_path.append((lx, ly))

    def follower_callback(self, msg):
        self.follower_pose = msg.pose.pose

    def scan_callback(self, msg):
        self.scan_msg = msg

    def relay_event_callback(self, msg):
        """
        /relay_event 수신 처리
        IDLE 로봇만 반응 → EXPLORER 전환

        핵심: follower2의 relay 기준점(last_relay_pos)을
              자기 시작 위치가 아니라 "follower1이 배치한 relay 위치"로 설정한다.
              그래야 다음 relay 판단 거리가 통신망 기준으로 올바르게 누적된다.
        """
        if self.robot_state != 'IDLE':
            return

        # payload 예: "RELAY_PLACED id=follower1 x=1.727 y=-0.000"
        tokens = msg.data.split()
        x = None
        y = None
        for token in tokens:
            if token.startswith('x='):
                try:
                    x = float(token.split('=')[1])
                except ValueError:
                    pass
            elif token.startswith('y='):
                try:
                    y = float(token.split('=')[1])
                except ValueError:
                    pass

        if x is not None and y is not None:
            self.last_relay_pos = (x, y)

        # follower2가 EXPLORER가 되는 순간의 자기 위치를 새 출발점으로 저장
        if self.follower_pose is not None:
            self.explorer_start_pos = (
                self.follower_pose.position.x,
                self.follower_pose.position.y
            )
        else:
            self.explorer_start_pos = None

        self.robot_state    = 'EXPLORER'
        self.use_breadcrumb = False
        self.waypoint_idx   = 0
        self._junction_count = 0   # 깨어날 때 junction 카운터 리셋
        # 자기 위치를 relay base로 덮어쓰지 않도록 odom 안정화 단계를 건너뜀
        self._init_done  = True
        self._init_ticks = 10

        self.get_logger().warn(
            f'[{self.robot_id}] relay_event received → IDLE → EXPLORER, '
            f'explorer_start={self.explorer_start_pos}, last_relay={self.last_relay_pos} '
            f'(payload: {msg.data})'
        )

    # ------------------------------------------------------------------ #
    #  유틸리티
    # ------------------------------------------------------------------ #

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def get_front_min(self):
        if self.scan_msg is None:
            return 10.0
        ranges = self.scan_msg.ranges
        n = len(ranges)
        front = list(ranges[0:30]) + list(ranges[n - 30:])
        return min((r for r in front if math.isfinite(r)), default=10.0)

    def get_rear_min(self):
        if self.scan_msg is None:
            return 10.0
        ranges = self.scan_msg.ranges
        n = len(ranges)
        rear = list(ranges[n // 2 - 30: n // 2 + 30])
        return min((r for r in rear if math.isfinite(r)), default=10.0)

    def get_side_space(self):
        if self.scan_msg is None:
            return ('L', 10.0)

    def detect_junction(self):
        """
        LaserScan 기반 topology 판정 (감지만, 배치 안 함).
        반환: 'CORRIDOR' | 'CORNER' | 'T_JUNCTION' | 'CROSS' | 'DEAD_END' | 'UNKNOWN'
        """
        if self.scan_msg is None:
            return 'UNKNOWN', 0.0, 0.0, 0.0
        ranges = self.scan_msg.ranges
        n = len(ranges)
        open_th_side  = 1.5   # 좌우: 1.5m 이상 트여야 개방 (직선 벽 0.3과 구분)
        open_th_front = 0.9   # 앞: 0.9m 이하면 막힘(전방에 벽/분기 임박)

        def sector_min(center, half=20):
            vals = []
            for k in range(center - half, center + half + 1):
                r = ranges[k % n]
                if math.isfinite(r):
                    vals.append(r)
            return min(vals) if vals else 10.0

        # 로봇 기준: front=0deg, left=90deg, right=270deg
        front = sector_min(0)
        left  = sector_min(n // 4)
        right = sector_min((3 * n) // 4)

        # 이 맵은 직선에서도 front가 0.8 안팎으로 짧게 나온다.
        # 따라서 front는 보조 신호로만 쓰고, 좌우 개방 여부로 junction을 판정한다.
        l_open = left  >= open_th_side
        r_open = right >= open_th_side
        f_open = front >= 1.2        # front는 확실히 트일 때(1.2+)만 개방으로

        if l_open and r_open:
            # 좌우 모두 열림 → 사거리/T자
            kind = 'CROSS' if f_open else 'T_JUNCTION'
        elif l_open or r_open:
            # 한쪽만 열림 → 코너/분기
            kind = 'CORNER'
        else:
            # 좌우 모두 막힘 → 직선 복도 (front가 막혀 보여도 직선으로 본다)
            # 단, front까지 아주 가까우면(벽에 붙음) DEAD_END
            kind = 'DEAD_END' if front < 0.4 else 'CORRIDOR'
        return kind, front, left, right

    def select_waypoint(self, fx, fy):
        path = list(self.leader_path)
        n = len(path)
        if n == 0:
            return None

        self.waypoint_idx = min(self.waypoint_idx, n - 1)

        while self.waypoint_idx < n - 1:
            wx, wy = path[self.waypoint_idx]
            if math.hypot(fx - wx, fy - wy) < self.waypoint_reach_dist:
                self.waypoint_idx += 1
            else:
                break

        lag = n - 1 - self.waypoint_idx
        if lag > 10:
            self.waypoint_idx = n - 1 - self.lookahead_steps

        target_idx = min(self.waypoint_idx + self.lookahead_steps, n - 1)
        return path[target_idx]

    # ------------------------------------------------------------------ #
    #  장애물 회피
    # ------------------------------------------------------------------ #

    def handle_obstacle(self, cmd, e_theta):
        front_min = self.get_front_min()

        # ── relay 근처에서는 회피 금지 ────────────────────────────────────
        # follower2처럼 '남의 relay'를 추월해야 하는 경우에만 적용.
        # follower1은 자기 시작점이 last_relay라 이 가드를 쓰면 안 된다.
        if (self.robot_id == 'follower2'
                and self.last_relay_pos is not None
                and self.follower_pose is not None):
            fx = self.follower_pose.position.x
            fy = self.follower_pose.position.y
            rlx, rly = self.last_relay_pos
            dist_to_relay = math.hypot(fx - rlx, fy - rly)
            if dist_to_relay < 0.7:
                # relay 코앞: 회피 끄고 약간 측면으로 비켜 직진
                self._avoiding = False
                # 좁은 복도라 옆으로 비킬 공간이 없다 → 진행 방향(e_theta) 유지하며 천천히 직진
                cmd.linear.x  = 0.12
                cmd.angular.z = max(min(self.k_angular * e_theta, 0.5), -0.5)
                self.get_logger().info(
                    f'[{self.robot_id}] PASS_RELAY: dist_to_relay={dist_to_relay:.2f}, '
                    f'front={front_min:.2f}, e_theta={e_theta:.2f} (회피 무시, 통과)',
                    throttle_duration_sec=0.5
                )
                return True, cmd

        if not self._avoiding:
            if front_min < self.obstacle_stop_dist:
                self._avoiding = True
        else:
            if front_min >= self.obstacle_clear_dist:
                self._avoiding = False

        if self._avoiding:
            cmd.linear.x  = 0.0
            cmd.angular.z = (
                self.obstacle_turn_speed if e_theta >= 0
                else -self.obstacle_turn_speed
            )
            self.get_logger().info(
                f'[{self.robot_id}] AVOIDING: front={front_min:.2f}, '
                f'turn={"L" if e_theta >= 0 else "R"}',
                throttle_duration_sec=0.5
            )
            return True, cmd

        return False, cmd

    # ------------------------------------------------------------------ #
    #  relay 배치 처리  (EXPLORER → RELAY_CANDIDATE → STATIONARY_RELAY)
    # ------------------------------------------------------------------ #

    def _check_junction_placement(self, fx, fy, cmd):
        """
        junction 우선 배치 (topology-aware). 통신 범위와 무관하게 항상 검사.
        반환: (placed, cmd)  placed=True면 호출자는 publish 후 return
        """
        if self.last_relay_pos is None:
            return False, cmd
        if self.robot_state not in ('EXPLORER', 'RELAY_CANDIDATE'):
            return False, cmd

        sx, sy = self.last_relay_pos
        dist_from_relay = math.hypot(fx - sx, fy - sy)

        jkind, jf, jl, jr = self.detect_junction()
        is_junction = jkind in ('T_JUNCTION', 'CROSS', 'CORNER')
        if is_junction and dist_from_relay >= self.min_relay_gap:
            self._junction_count += 1
        else:
            self._junction_count = 0

        if self._junction_count >= self.junction_confirm_n:
            self.robot_state = 'STATIONARY_RELAY'
            self.relay_positions.append((fx, fy))
            self.last_relay_pos = (fx, fy)
            self.explorer_start_pos = None
            self._junction_count = 0

            event_msg = String()
            event_msg.data = f'RELAY_PLACED id={self.robot_id} x={fx:.3f} y={fy:.3f}'
            self.relay_event_pub.publish(event_msg)
            self.get_logger().warn(
                f'[{self.robot_id}] JUNCTION_RELAY_PLACED ({jkind}): '
                f'x={fx:.3f}, y={fy:.3f}, dist_from_relay={dist_from_relay:.2f}m '
                f'(junction 우선 배치) → relay_event published'
            )
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            return True, cmd
        return False, cmd

    def _check_relay_placement(self, fx, fy, cmd):
        """
        반환: (should_stop, cmd)
          should_stop=True  → 호출자는 publish 후 return
          should_stop=False → 이동 로직 계속
        """
        if self.last_relay_pos is None:
            return False, cmd

        sx, sy = self.last_relay_pos
        dist_from_relay = math.hypot(fx - sx, fy - sy)
        relay_trigger_dist = self.comm_range * self.relay_threshold   # 1.70m


        # EXPLORER → RELAY_CANDIDATE  (90% 도달)
        if self.robot_state == 'EXPLORER':
            if dist_from_relay >= relay_trigger_dist * 0.90:
                self.robot_state = 'RELAY_CANDIDATE'
                self.get_logger().warn(
                    f'[{self.robot_id}] RELAY_CANDIDATE: '
                    f'dist={dist_from_relay:.2f}m (trigger={relay_trigger_dist:.2f}m)'
                )

        # RELAY_CANDIDATE → STATIONARY_RELAY  (100% 도달)
        if self.robot_state == 'RELAY_CANDIDATE':
            self.get_logger().info(
                f'[{self.robot_id}] RELAY_CANDIDATE: '
                f'dist={dist_from_relay:.2f}m / {relay_trigger_dist:.2f}m',
                throttle_duration_sec=0.5
            )
            # RELAY_CANDIDATE 동안에는 추종/회피와 무관하게 전진을 강제한다.
            # (leader가 멈췄거나 앞에 정지한 relay가 있어도 1.70m를 넘겨 relay를 완료)
            if dist_from_relay < relay_trigger_dist:
                fyaw = self.quaternion_to_yaw(self.follower_pose.orientation)
                # last_relay -> 현재위치 방향(=전진 방향)으로 직진
                head_x = fx - sx
                head_y = fy - sy
                target_yaw = math.atan2(head_y, head_x)
                e = self.normalize_angle(target_yaw - fyaw)
                cmd.angular.z = max(min(self.k_angular * e, self.max_angular), -self.max_angular)
                cmd.linear.x = self.max_linear if abs(e) < 0.5 else 0.1
                return True, cmd
            if dist_from_relay >= relay_trigger_dist:
                self.robot_state = 'STATIONARY_RELAY'
                self.relay_positions.append((fx, fy))
                self.last_relay_pos = (fx, fy)
                self.explorer_start_pos = None

                # ── Phase 4 핵심: relay_event 발행 → IDLE 로봇 깨우기 ──────
                event_msg = String()
                event_msg.data = (
                    f'RELAY_PLACED id={self.robot_id} '
                    f'x={fx:.3f} y={fy:.3f}'
                )
                self.relay_event_pub.publish(event_msg)

                self.get_logger().warn(
                    f'[{self.robot_id}] RELAY_PLACED: '
                    f'x={fx:.3f}, y={fy:.3f}, '
                    f'dist_from_base={dist_from_relay:.2f}m  '
                    f'→ relay_event published'
                )
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                return True, cmd

        # 이미 STATIONARY_RELAY
        if self.robot_state == 'STATIONARY_RELAY':
            cmd.linear.x  = 0.0
            cmd.angular.z = 0.0
            return True, cmd

        return False, cmd

    # ------------------------------------------------------------------ #
    #  메인 제어 루프
    # ------------------------------------------------------------------ #

    def control_loop(self):
        cmd = Twist()

        # ── IDLE: 완전 대기 ──────────────────────────────────────────────────
        if self.robot_state == 'IDLE':
            self.cmd_pub.publish(cmd)
            return
        
        # ── STATIONARY_RELAY: 중계기 상태에서는 완전 정지 ─────────────────────
        if self.robot_state == 'STATIONARY_RELAY':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        if self.leader_pose is None or self.follower_pose is None:
            self.cmd_pub.publish(cmd)
            return

        # ── odom 안정화 대기 ─────────────────────────────────────────────────
        if not self._init_done:
            self._init_ticks += 1
            if self._init_ticks >= 10:
                self._init_done = True
                # last_relay_pos가 아직 없을 때만 자기 위치로 초기화.
                # (follower2가 relay_event로 이미 기준점을 받았다면 덮어쓰지 않음)
                if self.last_relay_pos is None:
                    self.last_relay_pos = (
                        self.follower_pose.position.x,
                        self.follower_pose.position.y
                    )
                if self.explorer_start_pos is None:
                    self.explorer_start_pos = (
                        self.follower_pose.position.x,
                        self.follower_pose.position.y
                    )
                self.get_logger().info(
                    f'[{self.robot_id}] odom stabilized. '
                    f'relay base=({self.last_relay_pos[0]:.2f}, '
                    f'{self.last_relay_pos[1]:.2f})'
                )
            else:
                self.cmd_pub.publish(cmd)
                return

        fx   = self.follower_pose.position.x
        fy   = self.follower_pose.position.y
        fyaw = self.quaternion_to_yaw(self.follower_pose.orientation)

        lx = self.leader_pose.position.x
        ly = self.leader_pose.position.y
        leader_dist = math.hypot(lx - fx, ly - fy)

        # ── [관찰용] junction 감지 로그 (배치 안 함, 판단만 출력) ──────────
        jkind, jf, jl, jr = self.detect_junction()
        if jkind not in ('CORRIDOR', 'UNKNOWN'):
            self.get_logger().warn(
                f'[{self.robot_id}] JUNCTION_DETECT={jkind} '
                f'front={jf:.2f} left={jl:.2f} right={jr:.2f}',
                throttle_duration_sec=0.5
            )
        else:
            self.get_logger().info(
                f'[{self.robot_id}] topo={jkind} '
                f'front={jf:.2f} left={jl:.2f} right={jr:.2f}',
                throttle_duration_sec=2.0
            )

        # ── Relay placement 판단 ─────────────────────────────────────────────
        # 통신망 안(leader_dist <= comm_range)일 때만 relay 판단.
        # OUT_OF_RANGE(따라잡기 중)에는 거리가 누적돼도 relay를 놓지 않는다.
        # 통신 범위 안에서만 배치 검사 (따라잡기 중에는 배치 금지).
        # junction 우선 → 거리 기준 순서.
        if leader_dist <= self.comm_range:
            placed, cmd = self._check_junction_placement(fx, fy, cmd)
            if placed:
                self.cmd_pub.publish(cmd)
                return
            should_stop, cmd = self._check_relay_placement(fx, fy, cmd)
            if should_stop:
                self.cmd_pub.publish(cmd)
                return

        # ── 통신 범위 밖 복귀 ────────────────────────────────────────────────
        if leader_dist > self.comm_range:
            self._rotating    = False
            self._avoiding    = False
            self._yield_state = 'follow'
            self._prev_leader_dist = None
            self._prev_leader_pos  = None
            self.use_breadcrumb    = False
            self.waypoint_idx      = 0

            theta_target = math.atan2(ly - fy, lx - fx)
            e_theta      = self.normalize_angle(theta_target - fyaw)

            cmd.angular.z = max(min(self.k_angular * e_theta,
                                    self.max_angular), -self.max_angular)
            cmd.linear.x  = self.max_linear if abs(e_theta) < 0.3 else 0.0

            front_min = self.get_front_min()
            if front_min < self.obstacle_stop_dist:
                cmd.linear.x = 0.0

            self.get_logger().info(
                f'[{self.robot_id}] OUT_OF_RANGE: d={leader_dist:.2f}, '
                f'e_theta={e_theta:.2f}',
                throttle_duration_sec=1.0
            )
            self.cmd_pub.publish(cmd)
            return

        # ── Leader 우선권 양보 ───────────────────────────────────────────────
        if self._prev_leader_pos is not None:
            leader_vel_x = lx - self._prev_leader_pos[0]
            leader_vel_y = ly - self._prev_leader_pos[1]
            to_f_x = fx - lx
            to_f_y = fy - ly
            dist_lf = math.hypot(to_f_x, to_f_y)
            if dist_lf > 1e-6:
                to_f_x /= dist_lf
                to_f_y /= dist_lf
            dot = leader_vel_x * to_f_x + leader_vel_y * to_f_y
            leader_coming = (dot > 0.001 and leader_dist < self.approach_detect_dist)
        else:
            leader_coming = False
            dot = 0.0
        self._prev_leader_pos = (lx, ly)

        if self._yield_state == 'follow':
            if leader_coming:
                self._yield_state = 'yielding'
                self._avoiding    = False
                self.get_logger().info(
                    f'[{self.robot_id}] YIELD_START: d={leader_dist:.2f}'
                )
        elif self._yield_state == 'yielding':
            if leader_dist > self.approach_clear_dist:
                self._yield_state = 'follow'
                self.get_logger().info(
                    f'[{self.robot_id}] YIELD_END: d={leader_dist:.2f}'
                )

        if self._yield_state == 'yielding':
            self._avoiding = False
            rear_min = self.get_rear_min()
            if rear_min > 0.25:
                cmd.linear.x  = self.backoff_speed
                cmd.angular.z = 0.0
            else:
                side, _ = self.get_side_space()
                cmd.linear.x  = 0.0
                cmd.angular.z = self.obstacle_turn_speed if side == 'L' else -self.obstacle_turn_speed
            self.cmd_pub.publish(cmd)
            return

        # ── 초기 직접 추종 ───────────────────────────────────────────────────
        if not self.use_breadcrumb:
            dx       = lx - fx
            dy       = ly - fy
            distance = leader_dist

            if len(self.leader_path) >= 10:
                path_list   = list(self.leader_path)
                closest_idx = min(
                    range(len(path_list)),
                    key=lambda i: math.hypot(fx - path_list[i][0], fy - path_list[i][1])
                )
                self.waypoint_idx   = closest_idx
                self.use_breadcrumb = True
                self.get_logger().info(
                    f'[{self.robot_id}] → breadcrumb mode, wp_idx={self.waypoint_idx}'
                )

            if distance < 0.05:
                self.cmd_pub.publish(cmd)
                return

            theta_target  = math.atan2(dy, dx)
            e_theta       = self.normalize_angle(theta_target - fyaw)
            cmd.angular.z = max(min(self.k_angular * e_theta,
                                    self.max_angular), -self.max_angular)

            if distance > self.follow_distance:
                cmd.linear.x = min(self.k_linear * distance, self.max_linear)
            elif distance > 0.01:
                cmd.linear.x = min(self.k_linear * distance * 0.5, self.max_linear)
            else:
                cmd.linear.x = 0.0

            if self._yield_state != 'yielding':
                avoided, cmd = self.handle_obstacle(cmd, e_theta)
                if avoided:
                    self.cmd_pub.publish(cmd)
                    return

            self.get_logger().info(
                f'[{self.robot_id}] [DIRECT] state={self.robot_state}, '
                f'd={distance:.2f}, v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
                throttle_duration_sec=1.0
            )
            self.cmd_pub.publish(cmd)
            return

        # ── Breadcrumb waypoint 추종 ─────────────────────────────────────────
        waypoint = self.select_waypoint(fx, fy)
        if waypoint is None:
            self.cmd_pub.publish(cmd)
            return

        wx, wy   = waypoint
        dx       = wx - fx
        dy       = wy - fy
        distance = math.hypot(dx, dy)

        theta_target = math.atan2(dy, dx)
        e_theta      = self.normalize_angle(theta_target - fyaw)

        if self._yield_state != 'yielding':
            avoided, cmd = self.handle_obstacle(cmd, e_theta)
            if avoided:
                self.cmd_pub.publish(cmd)
                return

        if self._rotating:
            if abs(e_theta) < self.ROTATE_EXIT:
                self._rotating = False
        else:
            if abs(e_theta) > self.ROTATE_ENTER:
                self._rotating = True

        cmd.angular.z = max(min(self.k_angular * e_theta,
                                self.max_angular), -self.max_angular)

        if distance <= self.waypoint_reach_dist or self._rotating:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = min(self.k_linear * distance, self.max_linear)

        self.get_logger().info(
            f'[{self.robot_id}] [BREAD] state={self.robot_state}, '
            f'wp=({wx:.2f},{wy:.2f}), d={distance:.2f}, '
            f'leader_d={leader_dist:.2f}, '
            f'v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
            throttle_duration_sec=1.0
        )
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = FollowerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()