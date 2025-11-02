# safe-rl-carla\safe_rl\train_lagu.py:
import shutil

import torch
from safe_rl.env.carla_cmdp_env import CarlaCMDPEnv
from pathlib import Path
from safe_rl.algorithms.lagu import LagUAgent, LagUHyper
import argparse
import uuid, wandb
import os
import csv
import time
import math
import faulthandler, signal, sys


faulthandler.enable(file=sys.stderr, all_threads=True)
# Trigger a full dump by sending SIGUSR1 to the python PID
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

# Use the same run‑id across stages if it already exists
_RUN_ID = os.environ.get("WANDB_RUN_ID")
if _RUN_ID is None:
    _RUN_ID = str(uuid.uuid4())
    os.environ["WANDB_RUN_ID"] = _RUN_ID     # export for child stages


def train_one_env(env, agent, total_steps, args):
    """
    Run the main training loop for a given environment and a set number of steps.
    This function contains the body of the original main loop.
    """

    NO_PROGRESS_TIMEOUT = 30          # seconds without a single store
    last_store_t = time.monotonic()

    obs, _ = env.reset()
    speed, vec, actors = obs["speed"][0], obs["vector"], obs["actors"].reshape(-1)
    episode_ret, episode_cost = 0.0, 0.0
    ep_steps = 0
    ep_speed_accum = 0.0
    prev_completion = 0.0

    os.makedirs(args.save_dir, exist_ok=True)

    slow_step_streak = 0
    stagnation_streak = 0
    for t in range(total_steps):
        t_rel_stage = max(0, agent.global_step - int(getattr(agent, "_stage_step0", 0)))
        # Linearly decay actor token dropout across this stage if requested
        if getattr(args, "actor_dropout_decay_steps", 0):
            s0 = getattr(agent, "_stage_step0", 0)
            t_rel = max(0, agent.global_step - s0)
            frac = min(1.0, t_rel / float(args.actor_dropout_decay_steps))
            p0 = float(getattr(args, "actor_dropout_start", agent.cfg.actor_dropout_p))
            p1 = float(getattr(args, "actor_dropout_end",   max(0.0, p0)))
            agent.cfg.actor_dropout_p = p0 + (p1 - p0) * frac

        t_step_start = time.monotonic()
        act = agent.act(speed, vec, actors, explore=True)
        # --- one‑off sanity check ------------------------------------------
        if agent.global_step == 0:      # print only on the very first step
            print("[debug] sample action:", act, flush=True)
        # env reward is always 0.0 (by design); capture but ignore
        next_obs, rew_env, terminated, truncated, info = env.step(act)
        step_dt = time.monotonic() - t_step_start
        # --- latency watchdog: if two slow (>8s) steps in a row, refresh sensors ---
        if step_dt > 8.0:
            slow_step_streak += 1
            if slow_step_streak >= 2:
                print("[train] step latency high twice; refreshing sensors", flush=True)
                try:
                    env._purge_sensors()
                    env._spawn_sensors()
                except Exception as e:
                    print("[train] sensor refresh failed:", e, flush=True)
                slow_step_streak = 0
        else:
            slow_step_streak = 0

        cost = float(info['cost'])

        # ----- reward shaping -----
        completion = float(info.get("route_completion", prev_completion))
        # progress is reported as a percent (0..100). Convert to fraction-of-route delta.
        delta_prog = max(0.0, (completion - prev_completion) / 100.0)
        speed_norm = float(next_obs["speed"][0])  # ~0..1.5
        vec_now    = next_obs["vector"]
        # Read first two entries as (cte, he); tolerate extra dims for TL features
        cte_norm = float(vec_now[0]) if len(vec_now) > 0 else 0.0
        he_norm  = float(vec_now[1]) if len(vec_now) > 1 else 0.0
        # Optional traffic-light features (if vec_dim >= 4)
        tl_red   = float(vec_now[2]) if len(vec_now) > 2 else 0.0
        tl_dist  = float(vec_now[3]) if len(vec_now) > 3 else 1.0  # 1.0 ~ far
        if args.reward_mode == "speed":
            # strong drive to go straight:
            rew  = speed_norm * 5.0                 # encourage movement
            rew += delta_prog * args.prog_scale
            rew -= abs(cte_norm) * 2.0              # punish lateral error
            rew -= abs(he_norm)  * 1.0              # punish heading error

        elif args.reward_mode == "blend":
            # Progress-dominant shaping with explicit speed target
            # 10% route progress in one step → +10 reward
            rew  = 100.0 * delta_prog
            # Encourage ~45 km/h (speed_norm≈0.5) with a smooth bump
            target_speed = 0.5
            speed_reward = math.exp(-4.0 * (speed_norm - target_speed) ** 2)
            rew += 5.0 * speed_reward
            # Lane keeping penalties
            rew -= 2.0 * abs(cte_norm)
            rew -= 1.0 * abs(he_norm)
            # Anti-parking penalty when essentially stopped
            if speed_norm < 0.02:
                rew -= 2.0

        else:  # progress-only
            # progress + target-speed shaping so "full throttle" isn't always optimal
            steer_mag = abs(act[0])
            # Optional warmup inside a progress stage: use 'blend' for the first N stage-steps
            if getattr(args, "progress_warmup_steps", 0) and t_rel_stage < int(args.progress_warmup_steps):
                rew = delta_prog * args.prog_scale + speed_norm * getattr(args, "speed_scale", 0.1)
            else:
                rew  = delta_prog * args.prog_scale
            # Target around ~0.45 normalized (≈40 kph), fairly wide
            target, sigma = 0.45, 0.20
            speed_shape = math.exp(-((speed_norm - target) ** 2) / (2.0 * sigma * sigma))
            rew += 1.0 * speed_shape  # ↑ slightly stronger to avoid "park is fine"
            rew -= steer_mag  * args.steer_penalty
            # keep the car centred in lane
            rew -= (abs(cte_norm) + 0.5*abs(he_norm)) * 0.5
            # penalty for standing still
            rew -= 0.15 * float(speed_norm < 0.05)
        # (Optional, small TL shaping — keeps learning from cost primarily)
        # If red and close (tl_dist small), penalize speed a bit to encourage braking.
        # Comment out if you want *pure* cost-driven TL behavior.
        near_red = tl_red * max(0.0, 1.0 - tl_dist)  # grows when red & closer
        if near_red > 0.0:
            rew -= 0.2 * near_red * speed_norm


        episode_cost += cost
        ep_steps += 1
        ep_speed_accum += speed_norm

        agent.global_step += 1
        if args.log_interval > 0 and (agent.global_step % args.log_interval == 0):
            try:
                wandb.log({
                    "global_step": agent.global_step,
                    "trainer/step": agent.global_step,
                    "env/reward": rew,
                    "env/cost": cost,
                    "env/route_completion": completion,
                    "env/episode_return": episode_ret,
                    "env/episode_cost": episode_cost,
                    "env/speed_norm": speed_norm,
                    "safety/lambda": agent.lmbd.item(),
                }, step=agent.global_step)
            except Exception as e:
                print("[LagU] wandb.log failed:", e, flush=True)


        next_speed, next_vec = next_obs["speed"][0], next_obs["vector"]
        next_actors          = next_obs["actors"].reshape(-1)

        # Skip storing invalid transitions if the env hard-reset mid-step
        if info.get("hard_reset", False):
            print("[train] Detected env hard_reset → skipping store and resetting episode trackers", flush=True)
            # Reset episodic trackers and continue from the fresh observation returned by env.step()
            speed, vec, actors = next_speed, next_vec, next_actors
            prev_completion = 0.0
            episode_ret, episode_cost = 0.0, 0.0
            ep_steps = 0
            ep_speed_accum = 0.0
            last_store_t = time.monotonic()
            continue

        done = bool(terminated or truncated)

        # Only add a *small* success bonus when the episode truly terminates
        # (i.e., reached goal), not when truncated by timeout/stagnation.
        if done and terminated:
            # Success bonus ~ full progress worth
            rew += args.prog_scale
            # Optional: time-efficiency bonus (finish sooner → more)
            steps_per_timeout = max(1, int(env._timeout_s / 0.05))   # 0.05 s tick
            time_frac = min(1.0, ep_steps / steps_per_timeout)
            rew += 0.5 * args.prog_scale * (1.0 - time_frac)

        # ----- scale rewards up before storing (costs unchanged) -----
        REWARD_SCALE = 10.0
        rew = rew * REWARD_SCALE

        episode_ret += rew

        agent.store(speed, vec, actors, act, rew, cost, next_speed, next_vec, next_actors, done)
        last_store_t = time.monotonic()          # we *did* get a transition
        agent.train(log=True)

        speed, vec, actors = next_speed, next_vec, next_actors

        # update baseline for next step
        prev_completion = completion

        if done:
            print(f"Episode finished: R={episode_ret:.1f} | C={episode_cost:.2f}")

            # ---- Per-episode per-km metrics (for reports) ----
            try:
                km = max(1e-6, float(info.get("km_travelled_est", 0.0)))
                wandb.log({
                    "episode/route_completion": completion,
                    "episode/cost": episode_cost,
                    "episode/cost_per_km": episode_cost / km,
                    "episode/collisions_per_km": float(info.get("collisions_total", 0)) / km,
                    "episode/lane_marks_per_km": float(info.get("lane_marks_total", 0)) / km,
                    "episode/red_ticks_per_km": float(info.get("red_ticks_total", 0)) / km,
                    "episode/timeout": float("timeout_s" in info),
                    "episode/stagnation": float(info.get("stagnation", False)),
                }, step=agent.global_step)
            except Exception as e:
                print("[train] wandb.log (episode metrics) failed:", e, flush=True)

            # ── NEW: episodic λ update (preferred) ────────────────────────────
            try:
                if getattr(agent.cfg, "episodic_lambda", False):
                    avg_gap_per_step = (episode_cost - agent.cfg.cost_limit_init * max(1, ep_steps)) / max(1, ep_steps)
                    # Exponential moving average of cost gap
                    with torch.no_grad():
                        agent.ema_cost_gap.mul_(0.9).add_(0.1 * torch.tensor([avg_gap_per_step], device=agent.cfg.device))
                        mean_speed_norm = ep_speed_accum / max(1, ep_steps)
                        if mean_speed_norm > agent.cfg.lambda_update_min_speed:
                            alpha = agent.cfg.lambda_alpha0 / math.sqrt(max(1, agent.pi_step))
                            agent.lmbd.add_(alpha * agent.ema_cost_gap)
                            agent.lmbd.clamp_(0.0, agent.cfg.lambda_clip)
            except Exception as e:
                print("[train] episodic lambda update failed:", e, flush=True)

            # Reset episodic accumulators
            ep_steps = 0
            ep_speed_accum = 0.0

            # ---- NEW: if repeated stagnation is ending episodes, purge stale replay ----
            try:
                if bool(info.get("stagnation", False)):
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0
                if stagnation_streak >= 3:
                    print("[train] Detected 3 consecutive stagnation terminations → clearing replay and soft-reset λ", flush=True)
                    agent.replay.clear()
                    with torch.no_grad():
                        agent.lmbd.mul_(0.25)
                    stagnation_streak = 0
            except Exception:
                pass

            obs,_ = env.reset()
            speed, vec  = obs["speed"][0], obs["vector"]
            actors       = obs["actors"].reshape(-1)
            episode_ret, episode_cost = 0.0, 0.0
            prev_completion = 0.0  # reset for new episode

        # ───────── Replay-buffer guard ─────────
        if time.monotonic() - last_store_t > NO_PROGRESS_TIMEOUT:
            print(f"[train] No transitions for >{NO_PROGRESS_TIMEOUT}s → resetting env.", flush=True)
            obs, _ = env.reset()
            speed, vec  = obs["speed"][0], obs["vector"]
            actors      = obs["actors"].reshape(-1)
            last_store_t = time.monotonic()  # restart watchdog

        CKPT_INTERVAL = 50_000
        if (agent.global_step + 1) % CKPT_INTERVAL == 0:
            tag = (getattr(agent.cfg, "baseline", "lagu") or "lagu").lower()
            ckpt_path = os.path.join(args.save_dir, f"{tag}_ckpt_{agent.global_step + 1}.pth")
            agent.save_models(ckpt_path)
            print(f"[train] Saved checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000,
                        help="Traffic Manager port")
    parser.add_argument("--manifest",
                        help="CSV with columns: routes_xml,scenarios_json,steps")
    parser.add_argument("--routes", help="Single routes.xml – ignored if --manifest")
    parser.add_argument("--scenarios")
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=int(2e6), help="training env steps")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reward-mode", choices=["speed", "blend", "progress"], default="speed",
                        help="speed: speed-only shaping; blend: progress+speed; progress: progress-only.")
    parser.add_argument("--baseline", choices=["lagu","lag","td3"], default="lagu",
                        help="lagu: uncertainty+adaptive budget; lag: fixed-budget Lagrangian; td3: reward-only.")
    # Progress reward ≃ route‑percent → keep it around 0‑1
    parser.add_argument("--prog-scale", type=float, default=10.0,
                        help="scaling for progress reward (fraction of route).")
    parser.add_argument("--speed-scale", type=float, default=0.1, help="scaling for speed_norm in blend mode.")  # used in blend
    parser.add_argument("--replay-size", type=int, default=200000,
                        help="capacity of replay buffer (transitions)")
    # No down‑scaling → critics see true magnitudes
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Multiplicative scale for rewards & costs in replay (must match ReplayBuffer.scale).")
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=1000,
                        help="Env steps between WandB scalar logs.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (0 → non‑deterministic)")
    # NEW – shaping terms for pure‑progress mode
    parser.add_argument("--speed-bonus",  type=float, default=0.2,
                        help="extra reward = speed_norm * speed_bonus")
    parser.add_argument("--steer-penalty", type=float, default=0.3,
                        help="penalise |steer| with this coefficient")

    args = parser.parse_args()

    if wandb.run is None:
        try:
            # Determine a more robust name for the run
            run_name = "lagu_curriculum" if args.manifest else f"lagu_{os.path.basename(args.routes)}_{args.reward_mode}"
            # Respect env; default to 'allow' so a new run is created if it doesn't exist yet
            resume_mode = os.environ.get("WANDB_RESUME", "allow")
            wandb.init(
                project="safe-rl-carla",
                id=_RUN_ID if _RUN_ID else None,
                resume=resume_mode if _RUN_ID else None,
                config=vars(args),
                name=run_name
            )
            # Tell WandB to use our global_step for the x-axis everywhere
            wandb.define_metric("global_step")
            for prefix in ("env/*","loss/*","actor/*","safety/*","uncertainty/*"):
                wandb.define_metric(prefix, step_metric="global_step")
            try:
                wandb.config.update({"baseline": str(getattr(args, "baseline", "lagu"))}, allow_val_change=True)
            except Exception:
                pass
        except Exception as e:
            print("[LagU] wandb.init failed:", e, flush=True)

    # This helper function creates an environment instance
    def make_env(rxml,
                 sjson,
                 *,
                 route_idx: int = 0,
                 disable_cost: bool = False,
                 enable_pedestrian: bool = False,
                 enable_traffic: bool = False,
                 n_traffic: int = 8,
                 tm_port: int = None,
                 seed: int = None,
                 red_moving_thresh_ms: float = 1.4,
                 timeout_speed_ref: float = 4.0,
                 success_completion_pct: float = 99.0):

        # Use full cost weights so costs exceed the tighter budget
        cw = dict(collisions=1.0, off_lane=0.5, red=0.2)
        if disable_cost:  # warm-up stage → 10 × softer, not zero
            cw = {k: v * 0.1 for k, v in cw.items()}
        # Traffic Manager port: follow the UE4 launch (default: RPC+600)
        tm_port = tm_port if tm_port is not None else (args.tm_port or (args.port + 600))
        return CarlaCMDPEnv(host=args.host,
                            port=args.port,
                            routes_file=rxml,
                            scenarios_file=sjson,
                            route_index=route_idx,
                            cost_weights=cw,
                            enable_pedestrian=enable_pedestrian,
                            enable_traffic=enable_traffic,
                            n_traffic=int(n_traffic),
                            traffic_manager_port=tm_port,
                            seed=seed,
                            red_moving_thresh_ms=float(red_moving_thresh_ms),
                            timeout_extra_s=10.0,
                            timeout_speed_ref=float(timeout_speed_ref),
                            success_completion_pct=float(success_completion_pct))

    # We create the agent ONCE, so it persists across all training stages
    cfg = LagUHyper()
    cfg.seed = args.seed
    cfg.baseline = args.baseline  # make baseline visible to the agent

    # Baseline selection (CLI default can be overridden per-stage via manifest)
    try:
        cfg.baseline = args.baseline
    except Exception:
        cfg.baseline = "lagu"

    # ── Establish curriculum or single stage *before* creating agent ──
    if args.manifest:
        with open(args.manifest, encoding="utf-8-sig") as f:
            reader = csv.DictReader(
                (ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")),
                skipinitialspace=True
            )

            def _clean_val(val):
                if val is None:
                    return ""
                return val.strip() if isinstance(val, str) else str(val).strip()

            curriculum = []
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    if not k:
                        continue
                    key = k.strip() if isinstance(k, str) else str(k).strip()
                    cleaned[key] = _clean_val(v)
                # ignore completely empty rows
                if any(v != "" for v in cleaned.values()):
                    curriculum.append(cleaned)
        required = {"routes_xml", "scenarios_json", "steps"}
        for i, row in enumerate(curriculum, 1):
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"[manifest] Row {i} is missing columns {missing}. "
                    f"Keys present: {list(row.keys())}"
                )
        def _to_int(x, default=0):
            try: return int(str(x).strip())
            except Exception: return default
        normalized = []
        for row in curriculum:
            # skip rows that are header dups or empty
            if all((k == (str(v).strip() if v is not None else "")) for k, v in row.items()):
                continue
            row["route_index"] = _to_int(row.get("route_index", 0), 0)
            row["steps"]       = _to_int(row.get("steps", 0), 0)
            normalized.append(row)
        curriculum = normalized
    else:
        curriculum = [dict(routes_xml=args.routes,
                           scenarios_json=args.scenarios,
                           steps=args.steps)]

    # Create a short-lived env for FIRST stage to infer shapes (vec_dim now = 4)
    if not curriculum:
        raise RuntimeError(
            "Curriculum parsed empty. Check your manifest CSV and job script: "
            "ensure the header row is not being passed as the only row."
        )
    first = curriculum[0]
    first_label = (first.get("label", "") or "").strip()
    first_enable_traf = (first_label.startswith("traffic")
                         or str(first.get("enable_traffic", "0")).strip().lower() in ("1","true","yes"))
    # Robust parse: allow blanks and float/scientific strings, default → 8
    try:
        first_n_traf = int(float(str(first.get("n_traffic", 8)).strip() or "8"))
    except Exception:
        first_n_traf = 8
    first_red_thresh = 3.0 if first_enable_traf else 1.4
    shape_env = make_env(first["routes_xml"], first["scenarios_json"],
                         route_idx=first.get("route_index", 0),
                         disable_cost=first_label.startswith("bootstrap") or first_label.startswith("ped_cross"),
                         enable_pedestrian=first_label.startswith("ped_cross"),
                         enable_traffic=first_enable_traf,
                         n_traffic=first_n_traf,
                         tm_port=(args.tm_port or (args.port + 600)),
                         red_moving_thresh_ms=first_red_thresh,
                         seed=args.seed)
    vec_dim = shape_env.observation_space["vector"].shape[0]
    act_dim = shape_env.action_space.shape[0]
    shape_env.close()

    # Build agent once with correct dims
    agent = LagUAgent(obs_shape=(0,),  # placeholder, vector-only
                      act_dim=act_dim,
                      max_action=1.0,
                      cfg=cfg,
                      vec_dim=vec_dim,
                      replay_size=args.replay_size,
                      replay_scale=args.reward_scale)

    # ───────────────── resume (optional) ──────────────────
    if args.resume:
        # Look for both periodic ckpts and final ckpts for this baseline
        patterns = [f"{args.baseline}_ckpt_*.pth", f"{args.baseline}_final_*.pth"]
        cands = []
        for pat in patterns:
            cands.extend(Path(args.save_dir).glob(pat))
        def _step_num(p):
            try:
                return int(p.stem.split("_")[-1])
            except Exception:
                return -1
        if cands:
            latest = max(cands, key=_step_num)
            print(f"[train] Resuming from checkpoint {latest}")
            agent.load_models(str(latest))
        else:
            print(f"[train] --resume specified but no checkpoint found for baseline '{args.baseline}'. Starting fresh.")

    # ── tell WandB to track weights, biases & grads of the *actor* network ──
    try:
        wandb.watch(agent.actor, log="all", log_freq=1000)
    except Exception as e:
        print("[train_lagu] wandb.watch failed:", e, flush=True)

    # -------- training curriculum --------------


    # -------- single agent across all routes ----
    # Initialize for change-detection without linter warnings
    prev_stage = None
    prev_stage_args = None
    env = None
    for stage in curriculum:
        if env is not None:
            # The close method needs to be implemented in your CarlaCMDPEnv
            # to properly destroy actors and disconnect
            try:
                env.close()
            except AttributeError:
                print("[train] Warning: env.close() method not found. Implement it to clean up CARLA actors.")
                env._destroy_episode() # Fallback to destroying actors
                time.sleep(2)


        label = (stage.get("label", "") or "").strip()
        # define 'warmup' again (was removed during refactor)
        warmup = label.startswith("bootstrap")
        ped_stage = label.startswith("ped_cross")
        disable_cost = warmup or ped_stage
        # Only spawn the episodic pedestrian on ped_cross stages
        enable_ped  = ped_stage

        enable_traf = (
            label.startswith("traffic")
            or str(stage.get("enable_traffic", "0")).strip().lower() in ("1", "true", "yes")
        )
        # Robust parse: allow blanks and float/scientific strings, default → 8
        try:
            n_traf = int(float(str(stage.get("n_traffic", 8)).strip() or "8"))
        except Exception:
            n_traf = 8

        # Compute red-light moving threshold *before* using it
        red_thresh = float(
            stage.get("red_moving_thresh_ms", 3.0 if enable_traf else 1.4)
            or (3.0 if enable_traf else 1.4)
        )
        if red_thresh <= 0:
            red_thresh = 1.4

        env = make_env(
            stage["routes_xml"], stage["scenarios_json"],
            route_idx=stage.get("route_index", 0),
            disable_cost=disable_cost,
            enable_pedestrian=enable_ped,
            enable_traffic=enable_traf,
            n_traffic=n_traf,
            tm_port=(args.tm_port or (args.port + 600)),
            seed=args.seed,
            red_moving_thresh_ms=red_thresh,
            # Read per-stage override directly from the stage dict; fall back to 4.0s ref speed.
            # This avoids referencing an uninitialized local `stage_args`.
            timeout_speed_ref=float(stage.get("timeout_speed_ref", 4.0)),
            # New: allow early-success curriculum (e.g., 10%, 25%, 50%)
            success_completion_pct=float(stage.get("success_completion_pct", 99.0)
                                         or 99.0)
        )

        stagn_limit = int(stage.get("stagnation", 100))
        env.stagnation_steps = stagn_limit

        # stage["steps"] was coerced to int earlier; fall back to CLI if it's 0
        run_steps = stage["steps"] if stage["steps"] > 0 else int(args.steps)

        # ── 1.4 Per-stage overrides from CSV (optional columns) ──
        # Reward keys: reward_mode, speed_bonus, steer_penalty, speed_scale, prog_scale
        # Safety/robustness keys: cost_limit_init, lambda_alpha0, lambda_alpha0_after,
        #                         lambda_alpha0_ramp_after, delta_clip_mult,
        #                         safe_guard_abs_sqrtvar, T_thresh
        # Exploration/regularization: actor_dropout_start, actor_dropout_end, actor_dropout_decay_steps
        from types import SimpleNamespace
        stage_args = SimpleNamespace(**vars(args))
        override_keys = (
            "reward_mode", "speed_bonus", "steer_penalty", "speed_scale", "prog_scale",
            "cost_limit_init", "lambda_alpha0", "lambda_alpha0_after", "lambda_alpha0_ramp_after",
            "delta_clip_mult", "safe_guard_abs_sqrtvar", "T_thresh",
            "actor_dropout_start", "actor_dropout_end", "actor_dropout_decay_steps",
            "baseline", "timeout_speed_ref", "success_completion_pct",
            # NEW: expose bonus & env red-light threshold and optional progress warmup
            "bonus_coef", "red_moving_thresh_ms", "progress_warmup_steps"
        )
        for k in override_keys:
            if k in stage and str(stage[k]).strip() != "":
                v = stage[k]
                if k in ("speed_bonus","steer_penalty","speed_scale","prog_scale",
                         "cost_limit_init","lambda_alpha0","lambda_alpha0_after",
                         "delta_clip_mult","safe_guard_abs_sqrtvar","T_thresh",
                         "actor_dropout_start","actor_dropout_end",
                         "bonus_coef","red_moving_thresh_ms","progress_warmup_steps"):
                    try:
                        v = float(v)
                    except Exception:
                        pass
                if k in ("lambda_alpha0_ramp_after","actor_dropout_decay_steps"):
                    try:
                        v = int(v)
                    except Exception:
                        pass
                stage_args.__dict__[k] = v

        # ---- reset learner state between tasks ----
        # Treat ped_cross like a warm-start too (new dynamics due to walker)
        if warmup or ped_stage:
            agent.replay.clear()      # drop off‑policy clutter
            agent.lmbd.zero_()        # fresh dual variable
        # ── Stage-relative exploration + actor freeze (defaults; adjusted below) ──
        agent._eps_start = 0.35             # initial ε right after stage switch
        agent._eps_decay_steps = 5000       # decay horizon within the stage (env steps)
        # Defer setting _stage_actor_freeze_updates/_stage_random_steps until after we
        # detect regime changes (route/traffic/reward); see below.
        agent._stage_step0 = agent.global_step

        # Apply stage-specific safety/robustness overrides to cfg
        if hasattr(stage_args, "cost_limit_init"):      agent.cfg.cost_limit_init = stage_args.cost_limit_init
        if hasattr(stage_args, "lambda_alpha0"):        agent.cfg.lambda_alpha0 = stage_args.lambda_alpha0
        if hasattr(stage_args, "lambda_alpha0_after"):  agent.cfg.lambda_alpha0_after = stage_args.lambda_alpha0_after
        if hasattr(stage_args, "lambda_alpha0_ramp_after"): agent.cfg.lambda_alpha0_ramp_after = stage_args.lambda_alpha0_ramp_after
        if hasattr(stage_args, "delta_clip_mult"):      agent.cfg.delta_clip_mult = stage_args.delta_clip_mult
        if hasattr(stage_args, "safe_guard_abs_sqrtvar"): agent.cfg.safe_guard_abs_sqrtvar = stage_args.safe_guard_abs_sqrtvar
        if hasattr(stage_args, "T_thresh"):             agent.cfg.T_thresh = stage_args.T_thresh
        if hasattr(stage_args, "bonus_coef"):           agent.cfg.bonus_coef = stage_args.bonus_coef
        if hasattr(stage_args, "baseline") and stage_args.baseline:
            agent.cfg.baseline = str(stage_args.baseline).lower()
        # Stage-local actor-dropout schedule
        agent._stage_step0 = agent.global_step
        if hasattr(stage_args, "actor_dropout_start"):
            agent.cfg.actor_dropout_p = stage_args.actor_dropout_start

        # ── Detect regime changes and do light resets (λ, replay) ──
        changed_route  = (prev_stage is not None) and (stage["routes_xml"] != prev_stage["routes_xml"])
        changed_traf   = (prev_stage is not None) and (
            bool(int(stage.get("enable_traffic", 0) or 0)) != bool(int(prev_stage.get("enable_traffic", 0) or 0))
        )
        changed_reward = (prev_stage_args is not None) and (stage_args.reward_mode != prev_stage_args.reward_mode)
        prev_was_ped   = (prev_stage is not None) and str(prev_stage.get("label","")).strip().startswith("ped_cross")
        changed_ped    = (ped_stage != prev_was_ped)

        if changed_route or changed_traf or changed_reward or changed_ped:
            # Fast adaptation on new regime
            agent.replay.clear()
            with torch.no_grad():
                agent.lmbd.mul_(0.25)  # soft reset; use .zero_() if things still stall
            # --- NEW: temporarily boost actor LR for fast adaptation on the new regime
            agent._lr_boost_steps = 10000
            try:
                for g in agent.pi_opt.param_groups:
                    g["lr"] = 2e-4
            except Exception:
                pass


        # Finalize per-stage gates **after** change detection:
        # Shorter pure-random window; slightly longer if brand-new geometry/traffic.
        agent._stage_random_steps = 800 if (changed_route or changed_traf or changed_reward or changed_ped) else 300
        # Shorter actor freeze so π can adapt within ~few hundred updates.
        agent._stage_actor_freeze_updates = 200 if (changed_route or changed_traf or changed_ped) else 120

        # ── Gentle defaults for first "traffic-lite" exposures ── (enable_traf already computed)
        if enable_traf:
            # If not overridden in CSV, apply safer defaults
            if "lambda_alpha0" not in stage:      agent.cfg.lambda_alpha0 = min(agent.cfg.lambda_alpha0, 1e-4)
            if "delta_clip_mult" not in stage:    agent.cfg.delta_clip_mult = min(agent.cfg.delta_clip_mult, 0.5)
            if "bonus_coef" not in stage:         agent.cfg.bonus_coef = min(getattr(agent.cfg, "bonus_coef", 0.02), 0.01)
            if "actor_dropout_start" not in stage:
                agent.cfg.actor_dropout_p = 0.15
            if "actor_dropout_end" not in stage:
                # set an end value if we’re decaying this stage
                try:
                    stage_args.actor_dropout_end = 0.10
                except Exception:
                    pass

        print(f"\n[train] >>> Stage '{stage.get('label', 'Default')}' for "
              f"{run_steps} steps on {os.path.basename(stage['routes_xml'])} "
              f"stagnation={stagn_limit} <<<\n")

        # Explicitly print resolved reward mode for sanity
        try: print(f"[train] reward_mode={stage_args.reward_mode}")
        except Exception: pass

        # Call the training loop for the current stage
        train_one_env(env, agent, run_steps, stage_args)
        # Keep previous for next-stage change detection
        prev_stage = stage
        prev_stage_args = stage_args
    # --------------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------------
    tag = (getattr(agent.cfg, "baseline", "lagu") or "lagu").lower()
    final_ckpt = os.path.join(args.save_dir, f"{tag}_final_{agent.global_step}.pth")
    agent.save_models(final_ckpt)

    print(f"[train] Curriculum finished – weights saved to {final_ckpt}")
    return


if __name__ == "__main__":
    main()