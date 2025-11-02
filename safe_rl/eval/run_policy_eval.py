# safe-rl-carla\safe_rl\eval\run_policy_eval.py:
import argparse, csv, os, pathlib, time, importlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from safe_rl.env.carla_cmdp_env import CarlaCMDPEnv, MAX_ACTORS, ACTOR_FEATURES

def make_env(
        host: str, port: int, routes_xml: str, scenarios_json: str,
        route_index: int, tm_port: int, red_moving_thresh_ms: float,
        enable_traffic: bool, n_traffic: int, seed: int,
        cost_weights: Optional[Dict[str, float]] = None
) -> CarlaCMDPEnv:
    scen = scenarios_json
    if scen and scen.lower().endswith("none.json"):
        scen = None

    ped = False
    if scenarios_json:
        base = os.path.basename(scenarios_json).lower()
        ped = base.startswith("ped")

    env = CarlaCMDPEnv(
        host=host,
        port=port,
        routes_file=routes_xml,
        scenarios_file=scen,
        route_index=route_index,
        traffic_manager_port=tm_port or (port + 600),
        enable_pedestrian=ped,
        enable_traffic=bool(enable_traffic),
        n_traffic=int(n_traffic),
        seed=int(seed),
        red_moving_thresh_ms=float(red_moving_thresh_ms),
    )
    if cost_weights is not None:
        env.cost_weights = cost_weights
    return env

def _agent_map() -> Dict[str, Tuple[str, str, str]]:
    """
    Returns algo -> (module, AgentClass, HyperClass).
    Adjust names if your project uses different class/module names.
    """
    return {
        "lagu": ("safe_rl.algorithms.lagu", "LagUAgent", "LagUHyper"),
        "lag":  ("safe_rl.algorithms.lag",  "LagAgent",  "LagHyper"),
        "td3":  ("safe_rl.algorithms.td3",  "TD3Agent",  "TD3Hyper"),
    }

def make_agent(algo: str, env: CarlaCMDPEnv):
    amap = _agent_map()
    if algo not in amap:
        raise ValueError("Unknown --algo %r (try one of: %s)" % (algo, ", ".join(sorted(amap.keys()))))
    mod_name, agent_cls_name, hyper_cls_name = amap[algo]

    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        raise RuntimeError("Failed to import module %r: %s" % (mod_name, e))

    try:
        AgentCls = getattr(mod, agent_cls_name)
        HyperCls = getattr(mod, hyper_cls_name)
    except AttributeError as e:
        raise RuntimeError("Expected classes %s / %s not found in %s" % (agent_cls_name, hyper_cls_name, mod_name))

    cfg = HyperCls()
    act_dim = int(env.action_space.shape[0])
    vec_dim = int(env.observation_space["vector"].shape[0])

    # Some agents may ignore some of these kwargs; they should be accepted.
    agent = AgentCls(
        obs_shape=(0,),
        act_dim=act_dim,
        max_action=1.0,
        cfg=cfg,
        vec_dim=vec_dim,
    )
    return agent

def _expected_vec_dim(agent) -> int:
    """
    Infer the actor's expected vector length from the first Linear layer.
    in_dim = 1 (speed) + vec_dim + MAX_ACTORS*ACTOR_FEATURES
    """
    try:
        in_dim = int(agent.actor.head[0].in_features)
        return max(2, in_dim - 1 - (MAX_ACTORS * ACTOR_FEATURES))
    except Exception:
        # Fallback to a sane default
        return 4

def _fit_vec_to_actor(agent, vec_np: np.ndarray) -> np.ndarray:
    want = _expected_vec_dim(agent)
    out = np.zeros(want, dtype=np.float32)
    if vec_np is not None and len(vec_np) > 0:
        out[:min(len(vec_np), want)] = vec_np[:min(len(vec_np), want)]
    return out

def _safe_act(agent, speed, vec, actors_flat, use_guard: bool = True):
    """
    Call agent.act with best-effort args. Prefer deterministic, safe-guarded call if supported.
    """
    try:
        return agent.act(speed, vec, actors_flat, explore=False, safe_guard=use_guard)
    except TypeError:
        try:
            return agent.act(speed, vec, actors_flat, explore=False)
        except TypeError:
            return agent.act(speed, vec, actors_flat)

def run_one_episode(env: CarlaCMDPEnv, agent, prog_scale: float, speed_bonus: float,
                    use_guard: bool,
                    kick_seconds: float,
                    kick_accel: float,
                    kick_speed_thresh: float) -> Dict[str, Any]:
    obs, _ = env.reset()
    speed = float(obs["speed"][0])
    vec   = _fit_vec_to_actor(agent, obs["vector"])
    actors_flat = obs["actors"].reshape(-1)

    ep_cost = 0.0
    ep_ret  = 0.0
    t0 = time.time()
    no_move_streak = 0

    while True:
        a = _safe_act(agent, speed, vec, actors_flat, use_guard=use_guard)
        # --- eval-time anti-stall kick (mirrors training's stuck-kick idea) ---
        now = time.time()
        if (now - t0) < kick_seconds or no_move_streak >= 10:
            # ensure at least some forward pressure
            try:
                a[1] = max(a[1], kick_accel)
            except Exception:
                pass
        obs2, _, terminated, truncated, info = env.step(a)

        completion = float(info.get("route_completion", 0.0))
        speed2     = float(obs2["speed"][0])
        vec2       = _fit_vec_to_actor(agent, obs2["vector"])

        cte_norm = float(vec2[0]) if len(vec2) > 0 else 0.0
        he_norm  = float(vec2[1]) if len(vec2) > 1 else 0.0
        rew = (completion / 100.0) * prog_scale \
              + speed2 * speed_bonus \
              - 0.5 * abs(cte_norm) \
              - 0.25 * abs(he_norm)

        ep_cost += float(info.get("cost", 0.0))
        ep_ret  += rew

        # update no-move streak (use *current* speed2)
        if speed2 < kick_speed_thresh:
            no_move_streak += 1
        else:
            no_move_streak = 0

        if terminated or truncated:
            dt = time.time() - t0
            km = float(info.get("km_travelled_est", 0.0))
            km = max(1e-6, km)
            return {
                "route_completion": completion,
                "episode_time_s": dt,
                "score": ep_ret,
                "cost": ep_cost,
                "km_travelled": km,
                "cost_per_km": ep_cost / km,
                "collisions_total": int(info.get("collisions_total", 0)),
                "lane_marks_total": int(info.get("lane_marks_total", 0)),
                "red_ticks_total": int(info.get("red_ticks_total", 0)),
                "collisions_per_km": float(info.get("collisions_total", 0)) / km,
                "lane_marks_per_km": float(info.get("lane_marks_total", 0)) / km,
                "red_ticks_per_km": float(info.get("red_ticks_total", 0)) / km,
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "collision_fail": int(bool(info.get("collision_fail", False))),
                "stagnation": int(bool(info.get("stagnation", False))),
                "timeout_s": float(info.get("timeout_s", 0.0)),
            }

        speed = speed2
        vec   = vec2
        actors_flat = obs2["actors"].reshape(-1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["lagu","lag","td3"], default="lagu")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--routes", required=True)
    p.add_argument("--scenarios", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--carla-host", default="localhost")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=0)
    p.add_argument("--route-index", type=int, default=0)
    p.add_argument("--out-csv", default="policy_eval_results.csv")
    p.add_argument("--label", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prog-scale", type=float, default=10.0)
    p.add_argument("--speed-bonus", type=float, default=0.3)
    p.add_argument("--enable-traffic", type=int, default=0)
    p.add_argument("--n-traffic", type=int, default=8)
    p.add_argument("--red-moving-thresh-ms", type=float, default=1.4)
    p.add_argument("--disable-guard", action="store_true",
                   help="Disable evaluation-time uncertainty brake for sanity checks.")
    # --- NEW: gentle eval-time kick options ---
    p.add_argument("--eval-kick-seconds", type=float, default=1.0,
                   help="Duration after episode start to enforce a small forward accel.")
    p.add_argument("--eval-kick-accel", type=float, default=0.30,
                   help="Minimum accel value during the kick window (clamped into [-1,1]).")
    p.add_argument("--eval-kick-speed-thresh", type=float, default=0.05,
                   help="Speed-norm threshold below which we keep nudging forward.")
    p.add_argument("--cost-weights", default="",
                   help="Override as 'collisions,off_lane,red' (e.g. '1.0,0.5,0.2'); empty=env default.")
    args = p.parse_args()

    cw = None
    if args.cost_weights.strip():
        c0, c1, c2 = [float(x) for x in args.cost_weights.split(",")]
        cw = dict(collisions=c0, off_lane=c1, red=c2)

    env = make_env(
        host=args.carla_host,
        port=args.carla_port,
        routes_xml=args.routes,
        scenarios_json=args.scenarios,
        route_index=args.route_index,
        tm_port=args.tm_port or (args.carla_port + 600),
        red_moving_thresh_ms=args.red_moving_thresh_ms,
        enable_traffic=bool(args.enable_traffic),
        n_traffic=int(args.n_traffic),
        seed=int(args.seed),
        cost_weights=cw,
    )

    agent = make_agent(args.algo, env)
    # Try to load checkpoint (cpu is fine for eval)
    if hasattr(agent, "load_models"):
        agent.load_models(args.ckpt, map_location="cpu")
    elif hasattr(agent, "load"):
        agent.load(args.ckpt, map_location="cpu")
    else:
        raise RuntimeError("Agent does not expose a load() or load_models() method.")

    results: List[Dict[str, Any]] = []
    use_guard = not args.disable_guard
    for ep in range(args.episodes):
        res = run_one_episode(
            env, agent,
            args.prog_scale, args.speed_bonus,
            use_guard=use_guard,
            kick_seconds=float(args.eval_kick_seconds),
            kick_accel=float(args.eval_kick_accel),
            kick_speed_thresh=float(args.eval_kick_speed_thresh),
        )
        res["episode"] = ep
        res["label"] = args.label
        res["algo"] = args.algo
        res["ckpt"] = os.path.basename(args.ckpt)
        # keep both basename (for legacy tools) and full paths (for exact replay)
        res["routes"] = os.path.basename(args.routes)
        res["routes_path"] = args.routes
        res["scenarios"] = os.path.basename(args.scenarios)
        res["scenarios_path"] = args.scenarios
        # exact environment knobs for reproduction
        res["route_index"] = int(args.route_index)
        res["enable_traffic"] = int(args.enable_traffic)
        res["n_traffic"] = int(args.n_traffic)
        res["red_moving_thresh_ms"] = float(args.red_moving_thresh_ms)
        res["seed"] = int(args.seed)
        results.append(res)
        print("[eval] algo=%s ep=%02d completion=%.1f%% cost=%.2f time=%.1fs"
              % (args.algo, ep, res["route_completion"], res["cost"], res["episode_time_s"]))

    out = pathlib.Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label","algo","ckpt","routes","scenarios","episode",
        "route_completion","episode_time_s","score","cost","km_travelled",
        "cost_per_km","collisions_total","lane_marks_total","red_ticks_total",
        "collisions_per_km","lane_marks_per_km","red_ticks_per_km",
        "terminated","truncated","collision_fail","stagnation","timeout_s",
        # new columns for exact replay
        "routes_path","scenarios_path","route_index","enable_traffic",
        "n_traffic","red_moving_thresh_ms","seed"
    ]
    write_header = (not out.exists())
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    try: env.close()
    except Exception: pass

if __name__ == "__main__":
    main()
