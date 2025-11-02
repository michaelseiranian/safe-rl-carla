# safe-rl-carla\team_code_safe_rl\safe_rl_agent.py:
import os
import numpy as np
import carla
import math
from typing import Optional
from leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent
from safe_rl.algorithms.lagu import LagUAgent, LagUHyper
from safe_rl.env.carla_cmdp_env import MAX_ACTORS, ACTOR_FEATURES


def get_entry_point():
    return "SafeRLAgent"


class SafeRLAgent(AutonomousAgent):
    """
    Lag-U Safe-RL agent for CARLA Leaderboard (vector-only).
    * Observation  : speed (scalar, normalized) + guidance vector (cte, heading[, tl_red, tl_dist])
    * Action       : [steer, throttle, brake] ∈ [-1,1]^3
    * Safety cost  : lane invasion + collision (impulse-weighted)
    """

    # ───────────────────────── constructor ────────────────────────── #
    # NOTE: leaderboard integration: use vec_dim=2 (cte, heading), actors omitted (zeros)
    def setup(self, _path_to_conf_file=None):
        self.training = True  # will be toggled below

        cfg = LagUHyper()     # hyper-parameters dataclass
        # Vector-only learner: image tensor is unused; pass a dummy shape.
        self.obs_shape = (0,)
        self.act_dim   = 2   # [steer, accel]
        self.algorithm = LagUAgent(
            obs_shape=self.obs_shape,
            act_dim=self.act_dim,
            max_action=1.0,
            cfg=cfg,
        )

        # ─── load pretrained weights when running *evaluation* ───
        ckpt_path = os.getenv("SAFE_RL_CKPT_PATH", None)
        fine_tune = os.getenv("SAFE_RL_FINE_TUNE", "0") == "1"
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[SafeRLAgent] loading weights from {ckpt_path}")
            self.algorithm.load_models(ckpt_path)
            self.training = fine_tune      # freeze unless explicit opt‑in
        else:
            print("[SafeRLAgent] no SAFE_RL_CKPT_PATH set – "
                  "running in *training* mode")


        # per-episode trackers
        self.last_speed      = None   # scalar
        self.last_vec        = None   # guidance vector (length adapts to policy: 2 or 4)
        self.last_action_vec = None
        self.last_info       = None
        self.last_actr_flat  = None   # zero actor-vector from previous step

        self.step_counter        = 0
        self.train_every_n_steps = 4

        # This will be populated by the leaderboard framework
        self._map = None

        # Declared for static checkers; the evaluator will set this at runtime.
        self._vehicle: Optional[carla.Vehicle] = None
        # Cache for route calculation to improve performance
        self._prev_wp_idx = 0

    # ───────────────────────── sensors list ───────────────────────── #
    def sensors(self):
        """
        Define the minimal sensor suite (vector-only).
        GNSS and IMU are used to compute the guidance vector.
        """
        return [
            # Safety sensors
            {"type": "sensor.other.lane_invasion", "id": "lane_invasion"},
            {"type": "sensor.other.collision",     "id": "collision", "sensor_tick": 0.05},
            # State sensors for guidance vector
            {"type": "sensor.speedometer", "id": "speed"},
            {"type": "sensor.other.imu",   "id": "imu"},
            {"type": "sensor.other.gnss",  "id": "gnss"},
        ]

    # ──────────────────────── main control loop ───────────────────── #
    def run_step(self, input_data, timestamp):
        # --- build current state ------------------------------------------------
        speed, vec = self._process_obs(input_data)

        lane_ev = input_data["lane_invasion"][1]   # carla.LaneInvasionEvent
        coll_ev = input_data["collision"][1]       # carla.CollisionEvent or None

        cur_info = {
            "speed_kph": float(input_data["speed"][1]["speed"]),
            "lane_marks": getattr(lane_ev, "crossed_lane_markings", []),
            "collision":  coll_ev,
        }

        # No actor list available here → feed zeros (size: MAX_ACTORS*ACTOR_FEATURES)
        actors_flat = np.zeros(MAX_ACTORS * ACTOR_FEATURES, dtype=np.float32)

        # Determine the vector length expected by the loaded policy:
        # in_dim = 1 (speed) + vec_dim + MAX_ACTORS*ACTOR_FEATURES
        try:
            in_dim = int(self.algorithm.actor.head[0].in_features)
            vec_dim_required = in_dim - 1 - (MAX_ACTORS * ACTOR_FEATURES)
        except Exception:
            vec_dim_required = 2  # safe fallback
        vec_dim_required = max(2, int(vec_dim_required))

        # Zero-pad (cte, he) to expected length (2 or 4). Extra dims (e.g., TL flag/dist) = 0.
        vec_padded = np.zeros(vec_dim_required, dtype=np.float32)
        vec_padded[:min(2, vec_dim_required)] = vec[:min(2, vec_dim_required)]

        # --- bootstrap episode --------------------------------------------------
        if self.last_speed is None:
            # First step of the episode
            self.last_speed        = speed
            self.last_vec        = np.zeros(vec_dim_required, dtype=np.float32)  # initial vector zeros
            self.last_action_vec = np.zeros(2, dtype=np.float32)
            self.last_info       = {
                "speed_kph": 0.0,
                "lane_marks": [],
                "collision":  None,
            }
            self.last_actr_flat  = actors_flat
            return carla.VehicleControl(brake=1.0)   # first tick: hand-brake

        # --- store transition from previous tick -------------------------------
        reward = self._compute_reward(self.last_info, cur_info)
        cost   = self._compute_cost  (self.last_info, cur_info)
        done   = False # Leaderboard handles episode end; we never set True mid‑route.

        # previous (already padded) vector + speed
        last_spd = self.last_speed
        last_vec = self.last_vec

        # Vector-only transition (actors omitted → zeros)
        self.algorithm.store(
            last_spd,            # s_spd
            last_vec,            # s_vec
            self.last_actr_flat, # s_actr (zeros)
            self.last_action_vec,# a
            reward,              # r
            cost,                # c
            speed,               # s2_spd
            vec_padded,          # s2_vec
            actors_flat,         # s2_actr (zeros)
            done                 # d
        )

        # --- periodic training step --------------------------------------------
        self.step_counter += 1
        if self.training and self.step_counter % self.train_every_n_steps == 0:
            self.algorithm.train(log=False)

        # --- act ----------------------------------------------------------------
        action_vec = self.algorithm.act(speed, vec_padded, actors_flat, explore=self.training)
        control    = self._vector_to_control(action_vec)

        # --- update trackers for the next step ----------------------------------
        self.last_speed      = speed
        self.last_vec        = vec_padded
        self.last_action_vec = action_vec
        self.last_info       = cur_info
        self.last_actr_flat  = actors_flat

        return control

    # ─────────────────────── termination hook ────────────────────── #
    def destroy(self):
        """Save model weights at the end of the run."""
        if self.training:
            fname = f"./lagu_checkpoint_steps_{self.step_counter}.pth"
            print(f"[SafeRLAgent] saving checkpoint → {fname}")
            self.algorithm.save_models(fname)

    # ─────────────────────── helper functions ────────────────────── #
    def _get_route_errors(self, ego_tf):
        """
        Calculates the cross-track and heading errors of the agent with respect to the global plan.
        """
        # The global plan is a list of (carla.Location, RoadOption) tuples.
        route = self._global_plan_world_coord
        if not route:
            return 0.0, 0.0

        ego_loc = ego_tf.location

        # Find the closest waypoint in a search window around the last known index
        # This is an optimization to avoid searching the entire route every step.
        min_dist = float('inf')
        closest_wp_idx = self._prev_wp_idx

        search_start = max(0, self._prev_wp_idx - 20)
        search_end = min(len(route), self._prev_wp_idx + 20)

        for i in range(search_start, search_end):
            dist = ego_loc.distance(route[i][0])
            if dist < min_dist:
                min_dist = dist
                closest_wp_idx = i

        self._prev_wp_idx = closest_wp_idx

        # Get the transform of the target waypoint on the route
        map_wp = self._map.get_waypoint(route[closest_wp_idx][0])
        wp_tf = map_wp.transform

        # Vector from waypoint to ego
        vec_to_ego = ego_loc - wp_tf.location

        # Cross-track error (CTE): project the vector onto the waypoint's right vector
        wp_right_vec = wp_tf.get_right_vector()
        cte = vec_to_ego.x * wp_right_vec.x + vec_to_ego.y * wp_right_vec.y

        # Heading error (HE): angle between ego's and waypoint's forward vectors
        ego_fwd_vec = ego_tf.get_forward_vector()
        wp_fwd_vec = wp_tf.get_forward_vector()
        dot_product = ego_fwd_vec.x * wp_fwd_vec.x + ego_fwd_vec.y * wp_fwd_vec.y
        he_rad = math.acos(np.clip(dot_product, -1.0, 1.0))

        # Get the sign of the angle from the cross product's Z component
        cross_product_z = ego_fwd_vec.x * wp_fwd_vec.y - ego_fwd_vec.y * wp_fwd_vec.x
        if cross_product_z < 0:
            he_rad *= -1.0

        return float(cte), float(he_rad)

    def _process_obs(self, input_data):
        # Speed (CARLA speedometer reports m/s)
        speed_ms  = float(input_data["speed"][1]["speed"])
        speed_kph = speed_ms * 3.6
        speed_norm = np.float32(np.clip(speed_kph / 90.0, 0.0, 2.0))

        # Guidance Vector
        gps_data = input_data['gnss'][1]
        imu_data = input_data['imu'][1]
        ego_tf = carla.Transform(
            carla.Location(x=gps_data[0], y=gps_data[1], z=gps_data[2]),
            carla.Rotation(yaw=math.degrees(imu_data[2]))
        )

        cte, he = self._get_route_errors(ego_tf)

        # Normalize errors to be roughly in [-1, 1] for the network
        cte_norm = np.clip(cte / 4.0, -1.0, 1.0)  # Clip at 4 meters
        he_norm = np.clip(he / (math.pi / 2), -1.0, 1.0) # Clip at 90 degrees
        # Minimal traffic-light features (matches env): (red flag, approx. distance ahead)
        tl_red = 0.0
        tl_dist = 1.0
        try:
            ego_fwd = ego_tf.get_forward_vector()
            # Guard: evaluator sets self._vehicle at runtime
            veh = self._vehicle
            if veh is not None:
                tl = veh.get_traffic_light()
                if tl is not None and isinstance(tl, carla.TrafficLight):
                    if tl.get_state() == carla.TrafficLightState.Red:
                        tl_red = 1.0
                    d = tl.get_transform().location - ego_tf.location
                    ahead = max(d.x*ego_fwd.x + d.y*ego_fwd.y, 0.0)
                    tl_dist = float(np.clip(ahead / 50.0, 0.0, 1.0))
        except Exception:
            pass

        vec_norm = np.array([cte_norm, he_norm, tl_red, tl_dist], dtype=np.float32)

        return speed_norm, vec_norm

    @staticmethod
    def _vector_to_control(v):
        """
        2-D action: v=[steer, accel] in [-1,1]^2.
        Map to CARLA control with mutually-exclusive accel:
          throttle=relu(accel), brake=relu(-accel).
        """
        s = float(np.clip(v[0], -1.0, 1.0))
        steer = 0.0 if abs(s) < 0.02 else s
        accel = float(np.clip(v[1], -1.0, 1.0))
        throttle_raw = max(accel, 0.0)
        brake_raw    = max(-accel, 0.0)
        throttle = 0.0 if throttle_raw < 0.02 else throttle_raw
        brake    = 0.0 if brake_raw    < 0.15 else brake_raw
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    @staticmethod
    def _compute_reward(last, cur):
        # Simple progress proxy, can be replaced with route completion later
        return cur["speed_kph"] * 0.1

    @staticmethod
    def _compute_cost(last, cur):
        cost = 0.0

        # lane invasion – charge per event (sensor already emits only the marks crossed this tick)
        cost += 0.5 * len(cur["lane_marks"])

        # collision – penalise first detection within episode
        if cur["collision"] is not None and last["collision"] is None:
            impulse = cur["collision"].normal_impulse.length()
            speed_ms = cur["speed_kph"] / 3.6
            cost += 5.0 + 0.05 * impulse + 0.2 * speed_ms

        return cost