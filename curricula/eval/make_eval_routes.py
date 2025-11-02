# safe-rl-carla\curricula\eval\make_eval_routes.py
#!/usr/bin/env python3
"""
Generate evaluation routes & a pedestrian JSON for unseen towns (CARLA 0.9.10.1):
  - Town04_eval_straight.xml
  - Town06_eval_curve.xml
  - Town05_eval_turn.xml
  - ped_cross_Town03.json (for use on a seen route in an unseen scenario)

Usage:
  python curricula/eval/make_eval_routes.py --port 2000
"""
import argparse
import pathlib
import xml.etree.ElementTree as ET
import carla
import json

# Define the output directory relative to the script's location
OUTDIR = pathlib.Path(__file__).parent

def _wp_to_elem(parent, tf: carla.Transform):
    ET.SubElement(parent, "waypoint",
                  x=f"{tf.location.x:.3f}",
                  y=f"{tf.location.y:.3f}",
                  z="0.0", pitch="0.0", roll="0.0",
                  yaw=f"{tf.rotation.yaw:.3f}")

def _write_route(fname: pathlib.Path, town: str, waypoints: list):
    routes = ET.Element("routes")
    route  = ET.SubElement(routes, "route", id="0", town=town)
    for tf in waypoints:
        _wp_to_elem(route, tf)
    fname.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(routes).write(str(fname), encoding="UTF-8", xml_declaration=True)
    print(f"[make_eval_routes] Wrote {fname}")

def _abs_yaw_diff(a_deg, b_deg):
    diff = (b_deg - a_deg + 180) % 360 - 180
    return abs(diff)

def _yaw_delta(a_deg, b_deg):
    """Signed smallest-angle difference b - a in degrees."""
    return (b_deg - a_deg + 180) % 360 - 180


def gen_town04_straight(client):
    """Find a straight stretch in Town04."""
    world = client.load_world("Town04")
    m = world.get_map()
    for sp in m.get_spawn_points():
        start_wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        path = [start_wp]
        for _ in range(25): # Look ahead ~125m
            next_wps = path[-1].next(5.0)
            if not next_wps: break
            path.append(next_wps[0])

        if len(path) < 15: continue # Too short

        total_yaw_change = _abs_yaw_diff(path[0].transform.rotation.yaw, path[-1].transform.rotation.yaw)
        dist = path[0].transform.location.distance(path[-1].transform.location)

        if total_yaw_change < 5.0 and dist > 80.0:
            _write_route(OUTDIR / "Town04_eval_straight.xml", "Town04", [path[0].transform, path[-1].transform])
            return
    raise RuntimeError("Could not find a suitable straight route in Town04.")

def gen_town06_curve(client):
    """Find a significant curve in Town06."""
    world = client.load_world("Town06")
    m = world.get_map()
    for sp in m.get_spawn_points():
        start_wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        path = [start_wp]
        for _ in range(50): # Look ahead ~250m
            next_wps = path[-1].next(5.0)
            if not next_wps: break
            path.append(next_wps[0])

        if len(path) < 20: continue

        total_yaw_change = _abs_yaw_diff(path[0].transform.rotation.yaw, path[-1].transform.rotation.yaw)

        if total_yaw_change > 60.0:
            _write_route(OUTDIR / "Town06_eval_curve.xml", "Town06", [path[0].transform, path[-1].transform])
            return
    raise RuntimeError("Could not find a suitable curve in Town06.")

def gen_town05_turn(client):
    """Find a right turn in Town05 (robust junction search)."""
    world = client.load_world("Town05")
    m = world.get_map()

    # Try every spawn as an entry point
    for sp in m.get_spawn_points():
        start_wp = m.get_waypoint(
            sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if start_wp is None:
            continue

        # 1) March forward until we *enter* a junction
        wp = start_wp
        for _ in range(80):  # up to ~160 m @ 2 m steps
            nxt = wp.next(2.0)
            if not nxt:
                break
            wp = nxt[0]
            if wp.is_junction:
                break

        if not wp.is_junction:
            # This spawn didn't reach a junction, try the next spawn
            continue

        # 2) Inside the junction: look for outgoing connectors that turn right
        candidates = []
        for conn in wp.next(3.0):  # small fanout step to expose connectors
            dyaw = _yaw_delta(wp.transform.rotation.yaw, conn.transform.rotation.yaw)
            # Filter for “right-ish” (avoid U-turns and very shallow bends)
            if -150.0 < dyaw < -30.0:
                candidates.append((abs(dyaw), conn))

        if candidates:
            # Prefer the strongest right turn (largest |dyaw|)
            candidates.sort(reverse=True)
            right_wp = candidates[0][1]
            end_wps = right_wp.next(40.0)  # ~40 m after the turn
            if end_wps:
                _write_route(
                    OUTDIR / "Town05_eval_turn.xml",
                    "Town05",
                    [start_wp.transform, end_wps[0].transform],
                )
                return

        # 3) Fallback: detect a right turn by cumulative yaw change along a forward trace
        path = [start_wp]
        for _ in range(80):  # ~160 m @ 2 m
            nxt = path[-1].next(2.0)
            if not nxt:
                break
            path.append(nxt[0])

        # Look for a window where yaw cumulatively decreases (right turn)
        yaws = [p.transform.rotation.yaw for p in path]
        for i in range(5, len(path) - 10):
            # cumulative change over a ~20 m window
            delta = _yaw_delta(yaws[i - 5], yaws[i + 5])
            if delta < -45.0:  # meaningful right turn
                end_idx = min(i + 20, len(path) - 1)
                _write_route(
                    OUTDIR / "Town05_eval_turn.xml",
                    "Town05",
                    [path[i - 5].transform, path[end_idx].transform],
                )
                return

    raise RuntimeError("Could not find a suitable turn in Town05 (after robust search).")

def gen_ped_json_for_town03(client):
    """Create a pedestrian scenario file for Town03 for evaluation purposes."""
    world = client.load_world("Town03")
    m = world.get_map()
    # Pick a predictable location on a straight section for the crossing
    sp = m.get_spawn_points()[0]
    wp = m.get_waypoint(sp.location).next(60.0)[0]

    ped_json = {
        "available_scenarios": [{
            "Town03": [{
                "scenario_type": "PedestrianCrossing",
                "available_event_configurations": [{
                    "transform": {
                        "x": round(wp.transform.location.x, 3),
                        "y": round(wp.transform.location.y, 3),
                        "z": 0.0,
                        "yaw": round(wp.transform.rotation.yaw, 3)
                    },
                    "other_config": {"crossing_distance": 8.0, "walker_speed": 1.4}
                }]
            }]
        }]
    }
    out_path = OUTDIR / "ped_cross_Town03.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ped_json, f, indent=2)
    print(f"[make_eval_routes] Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate CARLA routes for evaluation.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA simulator RPC port.")
    args = parser.parse_args()

    client = carla.Client("localhost", args.port)
    client.set_timeout(30.0)

    gen_town04_straight(client)
    gen_town06_curve(client)
    gen_town05_turn(client)
    gen_ped_json_for_town03(client)

    print("\n[make_eval_routes] All evaluation routes and scenarios created successfully.")

if __name__ == "__main__":
    main()