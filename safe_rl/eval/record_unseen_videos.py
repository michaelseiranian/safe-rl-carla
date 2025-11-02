# safe-rl-carla\safe_rl\eval\record_unseen_videos.py
#!/usr/bin/env python3
"""
Re-run ONLY the unseen labels, for lagu/lag/td3, using the best checkpoint per (label,algo)
from eval_logs/best_by_row.csv, and record MP4 videos.

Camera modes:
  - topdown: a high, downward-facing sensor attached to ego (default)
  - chase:   behind/above the car, light pitch down
  - ego:     near-hood/driver POV

Usage (typical):
  python -u -m safe_rl.eval.record_unseen_videos \
    --best-csv eval_logs/best_by_row.csv \
    --outdir videos_seen \
    --camera topdown \
    --width 1280 --height 720 --fps 20 --episodes 1
"""

import argparse, os, sys, time, glob, re
from typing import Dict, Tuple, Optional, Union, TYPE_CHECKING, cast
# TypedDict is only in typing on py>=3.8; support py3.7 via typing_extensions.
try:
    from typing import TypedDict  # type: ignore[attr-defined]  # py>=3.8
except Exception:  # pragma: no cover
    try:
        from typing_extensions import TypedDict  # type: ignore
    except Exception:  # last-resort runtime fallback (static type checkers can still understand the above)
        class TypedDict(dict):  # type: ignore
            pass
import pandas as pd
import numpy as np

# Import your env + helpers
from safe_rl.env.carla_cmdp_env import CarlaCMDPEnv
from safe_rl.eval.run_policy_eval import make_env as make_eval_env, make_agent, _fit_vec_to_actor, _safe_act

import carla
import imageio
from threading import Event

try:
    import imageio_ffmpeg  # ensures ffmpeg binary is available without trying to download
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_EXE = None

try:
    import cv2  # fallback writer
except Exception:
    cv2 = None

if TYPE_CHECKING:
    # imageio doesn't expose a top-level Writer type; the stub lives here.
    from imageio.core.format import FormatWriter  # type: ignore
else:
    FormatWriter = object  # fallback placeholder for type hints

ALGOS = ("lagu","lag","td3")

class EpisodeVideoResult(TypedDict):
    route_completion: float
    episode_time_s: float
    frames: int
    cost: float

def find_route_file(basename: str) -> Optional[str]:
    """Search curricula/** for a route XML matching the basename (first match)."""
    for root_glob in ("curricula/**/*.xml",):
        for p in glob.glob(root_glob, recursive=True):
            if os.path.basename(p) == basename:
                return p
    return None

def find_route_file_exact(basename: str) -> Optional[str]:
    """Find a unique match; raise if ambiguous so we don't silently pick the wrong map."""
    matches = [p for p in glob.glob("curricula/**/*.xml", recursive=True)
               if os.path.basename(p) == basename]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple route XML matches for {basename}: {matches}")
    return None

def find_ckpt_file(basename: str) -> Optional[str]:
    """Resolve checkpoint by basename under checkpoints/"""
    p = os.path.join("checkpoints", basename)
    return p if os.path.exists(p) else None

def attach_camera(world: carla.World, ego: carla.Actor, mode: str, w: int, h: int, fps: int) -> Tuple[carla.Sensor, Optional["FormatWriter"]]:
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(w))
    cam_bp.set_attribute("image_size_y", str(h))
    cam_bp.set_attribute("fov", "50" if mode == "topdown" else "90")
    cam_bp.set_attribute("sensor_tick", f"{1.0/float(fps):.6f}")
    # Tag so the env's purge skips us; also reduce post effects for stable exposure
    try:
        cam_bp.set_attribute("role_name", "recorder_cam")
        cam_bp.set_attribute("enable_postprocess_effects", "false")
    except Exception:
        pass

    if mode == "topdown":
        # Slightly lower altitude to avoid white sky glare on some builds
        rel = carla.Transform(carla.Location(x=0.0, y=0.0, z=35.0),
                              carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
    elif mode == "chase":
        rel = carla.Transform(carla.Location(x=-7.5, y=0.0, z=2.8),
                              carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0))
    else:  # ego
        rel = carla.Transform(carla.Location(x=0.7, y=0.0, z=1.5),
                              carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))

    cam = world.spawn_actor(cam_bp, rel, attach_to=ego)
    # We'll stream frames to a writer incrementally
    writer: Optional["FormatWriter"] = None  # late-bind in run loop
    return cam, writer

def image_to_numpy(img: carla.Image, w: int, h: int) -> np.ndarray:
    # CARLA packs BGRA uint8
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3][:, :, ::-1]  # BGR->RGB

def run_one_episode_and_record(env: CarlaCMDPEnv,
                               agent,
                               out_mp4: str,
                               camera_mode: str,
                               w: int,
                               h: int,
                               fps: int,
                               speed_bonus: float = 0.3,
                               prog_scale: float = 10.0,
                               eval_kick_seconds: float = 1.0,
                               eval_kick_accel: float = 0.30,
                               eval_kick_speed_thresh: float = 0.05,
                               use_guard: bool = True,
                               start_frame_timeout_s: float = 10.0,
                               stall_timeout_s: float = 20.0) -> EpisodeVideoResult:
    # Reset and get ego + world
    obs, _ = env.reset()
    world = env._world
    ego = env._ego

    cam, writer = attach_camera(world, ego, camera_mode, w, h, fps)
    frames_written: int = 0
    first_frame_evt = Event()

    # Give the renderer a slightly longer warm-up; the previous 0.2s is too tight on clusters
    # for _ in range(5):
    #     env._tick_world_safe(1.0)
    # IMPORTANT: do not advance sim time before the first action; camera will stream as we step.

    # Prepare writer (prefer imageio-ffmpeg; fallback to OpenCV if unavailable)
    writer_err: Optional[Exception] = None
    if _FFMPEG_EXE is not None:
        try:
            writer = imageio.get_writer(
                out_mp4,
                fps=fps,
                codec="libx264",
                quality=8,
                format="FFMPEG",
            )
        except Exception as e:
            writer_err = e
            writer = None
    if writer is None and cv2 is not None:
        # Fallback: MJPG in AVI (widely supported; no external ffmpeg needed)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw = cv2.VideoWriter(out_mp4 if out_mp4.lower().endswith(".avi") else out_mp4.replace(".mp4",".avi"),
                             fourcc, float(fps), (int(w), int(h)))
        if not vw.isOpened():
            raise RuntimeError(f"Failed to open fallback OpenCV writer. Prior ffmpeg error: {writer_err}")
        # Wrap OpenCV writer to match the .append_data/.close interface
        class _CV2Writer:
            def append_data(self, frame: np.ndarray):
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            def close(self):
                vw.release()
        writer = _CV2Writer()  # type: ignore[assignment]
    elif writer is None:
        raise RuntimeError(f"No video writer available: imageio-ffmpeg={_FFMPEG_EXE is not None}, opencv={cv2 is not None}. Error: {writer_err}")

    def _on_img(img: carla.Image):
        nonlocal frames_written
        try:
            frame = image_to_numpy(img, w, h)
            writer.append_data(frame)
            frames_written += 1
            if frames_written == 1:
                first_frame_evt.set()
        except Exception as e:
            # swallow sporadic failures to keep sim going
            sys.stderr.write(f"[record] frame write failed: {e}\n")

    cam.listen(_on_img)

    speed = float(obs["speed"][0])
    vec = _fit_vec_to_actor(agent, obs["vector"])
    actors_flat = obs["actors"].reshape(-1)

    ep_cost = 0.0
    ep_ret  = 0.0
    t0 = time.time()
    no_move_streak = 0

    # Ensure the camera stream actually started; avoid getting “stuck” before loop work
    # if not first_frame_evt.wait(timeout=float(start_frame_timeout_s)):
    #     raise TimeoutError(f"No camera frame within {start_frame_timeout_s:.1f}s (mode={camera_mode}).")
    # Do not block waiting for first frame; stepping the sim will produce frames.

    result: Optional[EpisodeVideoResult] = None
    try:
        last_progress_time = time.time()
        while True:
            a = _safe_act(agent, speed, vec, actors_flat, use_guard=use_guard)
            # gentle kick (like your eval)
            now = time.time()
            if (now - t0) < eval_kick_seconds or no_move_streak >= 10:
                a[1] = max(a[1], eval_kick_accel)

            obs2, _, terminated, truncated, info = env.step(a)
            completion = float(info.get("route_completion", 0.0))
            speed2 = float(obs2["speed"][0])
            vec2 = _fit_vec_to_actor(agent, obs2["vector"])
            cte = float(vec2[0]) if len(vec2) > 0 else 0.0
            he  = float(vec2[1]) if len(vec2) > 1 else 0.0

            # match your lightweight eval scoring
            rew = (completion/100.0)*prog_scale + speed2*speed_bonus - 0.5*abs(cte) - 0.25*abs(he)
            ep_ret += rew
            ep_cost += float(info.get("cost", 0.0))

            no_move_streak = (no_move_streak + 1) if speed2 < eval_kick_speed_thresh else 0

            if terminated or truncated:
                dt = time.time() - t0
                # Build a strongly-typed result (no Union/Optional)
                res: EpisodeVideoResult = {
                    "route_completion": float(completion),
                    "episode_time_s": float(dt),
                    "frames": int(frames_written),
                    "cost": float(ep_cost),
                }
                result = res
                break

            # ---- watchdog: break & raise if we appear stalled for too long ----
            if frames_written > 0 or completion > 0.01:
                last_progress_time = time.time()
            if (time.time() - last_progress_time) > float(stall_timeout_s):
                raise TimeoutError(f"Stalled for >{stall_timeout_s:.1f}s (no frames/progress).")

            speed = speed2; vec = vec2
            actors_flat = obs2["actors"].reshape(-1)

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cam.destroy()
        except Exception:
            pass
        try: writer.close()
        except Exception: pass

    # After cleanup, return the (now guaranteed) result
    # (cast silences type checkers that can't see the loop always sets it)
    assert result is not None, "Recording ended without producing a result."
    return cast(EpisodeVideoResult, result)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-csv", default="eval_logs/best_by_row.csv")
    ap.add_argument("--outdir", default="videos_seen")
    ap.add_argument("--camera", choices=["topdown","chase","ego"], default="topdown")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--disable-guard", action="store_true",
                    help="Match eval when run with --disable-guard (uncertainty brake off).")
    args = ap.parse_args()

    df = pd.read_csv(args.best_csv)
    need_cols = {"label","algo","ckpt","routes","scenarios"}
    missing = need_cols - set(df.columns)
    if missing:
        raise SystemExit(f"{args.best_csv} missing columns: {missing}")

    # seen only + 3 algos
    sel = df[df["label"].astype(str).str.endswith("_seen") & df["algo"].astype(str).isin(ALGOS)]
    if not len(sel):
        raise SystemExit("No seen rows for lagu/lag/td3 in best_by_row.csv")

    os.makedirs(args.outdir, exist_ok=True)

    for ri, row in sel.iterrows():
        label = str(row["label"])
        algo  = str(row["algo"])
        ckpt_base = str(row["ckpt"])
        routes_base = str(row["routes"])
        scen_base = str(row["scenarios"])

        ckpt = find_ckpt_file(ckpt_base)
        if not ckpt:
            print(f"[skip] checkpoint not found: {ckpt_base}")
            continue

        # Prefer exact paths saved by eval; fall back to unique basename resolution
        routes_xml = str(row.get("routes_path", "")).strip()
        if not routes_xml or not os.path.exists(routes_xml):
            routes_xml = find_route_file_exact(routes_base) or find_route_file(routes_base)
        if not routes_xml:
            print(f"[skip] route XML not found: {routes_base}")
            continue

        scenarios_json = str(row.get("scenarios_path", "")).strip()
        if not scenarios_json or not os.path.exists(scenarios_json):
            # only "none.json" in your manifests; still support basename lookup
            scen_guess = find_route_file(scen_base)
            scenarios_json = scen_guess if scen_guess else os.path.join("curricula","simple","none.json")

        # Use the exact knobs captured in eval (fallbacks keep old behavior)
        ridx = int(row.get("route_index", 0) or 0)
        enable_tm = bool(int(row.get("enable_traffic", 1 if "_traffic" in label else 0) or 0))
        n_tm = int(row.get("n_traffic", 8 if ("_traffic" in label and "T06" in label) else (6 if "_traffic" in label else 0)) or 0)
        red_thresh = float(row.get("red_moving_thresh_ms", 3.0 if enable_tm else 1.4) or (3.0 if enable_tm else 1.4))
        seed = int(row.get("seed", 42) or 42)

        print(f"\n=== {label} :: {algo} :: {ckpt_base} (row {ri}) ===")
        print(f"routes: {routes_xml}")
        print(f"scenarios: {scenarios_json}")


        env = make_eval_env(
            host="localhost",
            port=int(os.environ.get("CARLA_PORT", "2000")),
            routes_xml=routes_xml,
            scenarios_json=scenarios_json,
            route_index=ridx,
            tm_port=int(os.environ.get("CARLA_TM_PORT", str(2000+600))),
            red_moving_thresh_ms=red_thresh,
            enable_traffic=bool(enable_tm),
            n_traffic=n_tm,
            seed=seed,
            cost_weights=None
        )
        agent = make_agent(algo, env)
        # load checkpoint
        if hasattr(agent, "load_models"):
            agent.load_models(ckpt, map_location="cpu")
        elif hasattr(agent, "load"):
            agent.load(ckpt, map_location="cpu")
        else:
            raise RuntimeError("Agent missing load/load_models")

        # out path
        safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
        run_dir = os.path.join(args.outdir, safe_label)
        os.makedirs(run_dir, exist_ok=True)
        mp4 = os.path.join(run_dir, f"{algo}_{os.path.splitext(ckpt_base)[0]}.mp4")

        # One retry on transient stalls/timeouts
        attempts = 0
        while True:
            attempts += 1
            try:
                res: EpisodeVideoResult = run_one_episode_and_record(
                    env, agent, mp4, args.camera, args.width, args.height, args.fps,
                    use_guard=not args.disable_guard
                )
                break
            except TimeoutError as e:
                print(f"[warn] {e} — attempt {attempts}/2")
                if attempts >= 2:
                    raise
                # Soft refresh of episode after a short pause
                try:
                    env.close()
                except Exception:
                    pass
                time.sleep(1.0)
                env = make_eval_env(
                    host="localhost",
                    port=int(os.environ.get("CARLA_PORT", "2000")),
                    routes_xml=routes_xml,
                    scenarios_json=scenarios_json,
                    route_index=ridx,
                    tm_port=int(os.environ.get("CARLA_TM_PORT", str(2000+600))),
                    red_moving_thresh_ms=red_thresh,
                    enable_traffic=bool(enable_tm),
                    n_traffic=n_tm,
                    seed=seed,
                    cost_weights=None
                )
                agent = make_agent(algo, env)
                if hasattr(agent, "load_models"):
                    agent.load_models(ckpt, map_location="cpu")
                elif hasattr(agent, "load"):
                    agent.load(ckpt, map_location="cpu")
        print(f"[video] saved {mp4} | completion={res['route_completion']:.1f}% "
              f"cost={res['cost']:.2f} frames={res['frames']}")

        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
