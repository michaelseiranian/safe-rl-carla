# safe-rl-carla\curricula\medium\make_routes_medium.py
"""
Generate a few **medium‑difficulty** CARLA routes that bridge the gap between
simple (straight / single turn) and your long shuffled routes.

What these routes look like
---------------------------
1) Town03_s_bend_medium.xml
   • An S‑bend: gentle left then right (or vice versa) within ~200–300 m.
   • Many waypoints (5 m spacing) are saved so the route polyline follows the road.

2) Town02_turn_then_bend.xml
   • A junction right/left turn immediately followed by a gentle bend and a straight.
   • Waypoints recorded along the post‑turn segment for ~150–220 m.

3) Town05_two_curves_short.xml
   • Two gentle curves with the same turning direction separated by a short straight.
   • ~250–350 m total length, good for speed control + curvature handling.

All routes are written into:  curricula/medium/
Each file contains *many* waypoints (id="0") so your env’s polyline closely
tracks the lane centre. You can reference them from the curriculum
manifest just like your existing simple routes.

Usage
-----
python curricula/medium/make_routes_medium.py --port 2000

Notes
-----
• We use only lane‑centre waypoints (Driving lanes) → no sidewalks/buildings.
• Defensive scanning with fallbacks so a valid segment is found even if map builds differ.
• Distances/yaw thresholds are tuned for CARLA 0.9.x towns; tweak constants at top if needed.
"""

import argparse
import pathlib
import xml.etree.ElementTree as ET
from typing import List, Optional

import carla

# -------------------------------
# Tunables / heuristics
# -------------------------------
STEP_M          = 5.0     # waypoint step (m)
MAX_STEPS_S     = 80      # S-bend search budget (~400 m)
MAX_STEPS_POST  = 60      # post-turn follow budget (~300 m)
MIN_BEND_DEG    = 28.0    # each bend in S-bend must exceed this
POST_TURN_MIN_M = 150.0
POST_TURN_MAX_M = 220.0
TWO_CURVES_MIN  = 250.0
TWO_CURVES_MAX  = 360.0
CURVE_DEG_MIN   = 18.0    # min single-step accumulated curve to qualify as a curve


# -------------------------------
# Helpers
# -------------------------------

def _unwrap_deg(prev: float, curr: float) -> float:
    """Return signed smallest diff (curr - prev) in degrees in [-180, 180]."""
    d = (curr - prev + 540.0) % 360.0 - 180.0
    return d


def _wp_to_elem(parent: ET.Element, tf: carla.Transform):
    ET.SubElement(
        parent,
        "waypoint",
        x=f"{tf.location.x:.6f}",
        y=f"{tf.location.y:.6f}",
        z=f"{tf.location.z:.6f}",
        yaw=f"{tf.rotation.yaw:.6f}",
        pitch="0.0",
        roll="0.0",
    )


def _write_route_xml(path: pathlib.Path, town: str, tfs: List[carla.Transform]):
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route", id="0", town=town)
    for tf in tfs:
        _wp_to_elem(route, tf)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(routes).write(str(path), encoding="UTF-8", xml_declaration=True)


def _follow_next(wp: carla.Waypoint, steps: int) -> List[carla.Waypoint]:
    """Follow lane forward, picking first next() each time (LANEFOLLOW)."""
    seq = [wp]
    cur = wp
    for _ in range(steps):
        nxts = cur.next(STEP_M)
        if not nxts:
            break
        cur = nxts[0]
        seq.append(cur)
    return seq


# -------------------------------
# Finders for specific patterns
# -------------------------------

def find_s_bend(m: carla.Map) -> Optional[List[carla.Waypoint]]:
    """Return a waypoint sequence that exhibits an S-bend.

    Strategy: walk forward from each spawn; detect a turn exceeding
    MIN_BEND_DEG, then an opposite-direction turn also exceeding MIN_BEND_DEG.
    Save waypoints along the full segment covering both bends.
    """
    for spawn in m.get_spawn_points():
        start_wp = m.get_waypoint(spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        seq = [start_wp]
        cur = start_wp
        phase = 0
        accum = 0.0
        prev_yaw = cur.transform.rotation.yaw
        turn_dir = 0  # +1 left, -1 right

        for _ in range(MAX_STEPS_S):
            nxts = cur.next(STEP_M)
            if not nxts:
                break
            nxt = nxts[0]
            dyaw = _unwrap_deg(prev_yaw, nxt.transform.rotation.yaw)
            prev_yaw = nxt.transform.rotation.yaw
            seq.append(nxt)

            if phase == 0:
                accum += dyaw
                if abs(accum) >= MIN_BEND_DEG:
                    phase = 1
                    turn_dir = 1 if accum > 0 else -1
                    accum = 0.0
            else:
                if turn_dir * dyaw < 0:  # opposite direction contribution only
                    accum += dyaw
                    if abs(accum) >= MIN_BEND_DEG:
                        # include a few extra steps after finishing second bend
                        tail = _follow_next(nxt, 6)
                        seq.extend(tail[1:])
                        return seq
            cur = nxt
    return None


def find_turn_then_bend(m: carla.Map) -> Optional[List[carla.Waypoint]]:
    """
    Junction turn (30–120 deg) followed by ≥150 m with at least a gentle bend.
    Broader search window and ‘walk to junction’ logic.
    """
    def walk_to_junction(wp: carla.Waypoint, max_steps=20):
        cur = wp
        for _ in range(max_steps):
            if cur.is_junction:
                return cur
            nxts = cur.next(5.0)
            if not nxts: break
            cur = nxts[0]
        return None

    for spawn in m.get_spawn_points():
        start_wp = m.get_waypoint(spawn.location, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)

        j_wp = walk_to_junction(start_wp, max_steps=30)
        if j_wp is None:
            # step a little further and try again once
            probe = _follow_next(start_wp, 8)[-1]
            j_wp = walk_to_junction(probe, max_steps=20)
        if j_wp is None:
            continue

        # try all outgoing branches from the junction
        branches = j_wp.next(2.0)
        for b in branches:
            dy = _unwrap_deg(j_wp.transform.rotation.yaw, b.transform.rotation.yaw)
            if not (30.0 <= abs(dy) <= 120.0):
                continue  # not a proper turn

            seq = [start_wp, j_wp, b]
            cur = b
            prev_yaw = cur.transform.rotation.yaw
            dist = 0.0
            accum_curve = 0.0
            while dist < POST_TURN_MAX_M:
                nxts = cur.next(STEP_M)
                if not nxts: break
                nxt = nxts[0]
                dyaw = _unwrap_deg(prev_yaw, nxt.transform.rotation.yaw)
                prev_yaw = nxt.transform.rotation.yaw
                accum_curve += abs(dyaw)
                seq.append(nxt)
                dist += STEP_M
                cur = nxt
                if dist >= POST_TURN_MIN_M and accum_curve >= CURVE_DEG_MIN:
                    return seq
    return None

def force_turn_then_bend(m: carla.Map) -> List[carla.Waypoint]:
    """
    Deterministic Town02 fallback: step ~15 m to the junction, take the first
    right/left branch that turns 30–120°, then follow ~180 m.
    Mirrors your earlier 'simple' Town02 logic but records dense waypoints.
    """
    spawn = m.get_spawn_points()[3] if len(m.get_spawn_points()) >= 4 else m.get_spawn_points()[0]
    start_wp = m.get_waypoint(spawn.location, project_to_road=True,
                              lane_type=carla.LaneType.Driving)

    # reach junction centre
    mid = start_wp
    for _ in range(4):  # ~20 m
        nxts = mid.next(5.0)
        if not nxts: break
        mid = nxts[0]

    # pick a turning branch
    turn = None
    for c in mid.next(1.0):
        dy = _unwrap_deg(mid.transform.rotation.yaw, c.transform.rotation.yaw)
        if 30.0 <= abs(dy) <= 120.0:
            turn = c
            break
    if turn is None:
        # as a last resort, just keep lane and follow anyway
        turn = mid

    # follow for ~200 m
    seq = [start_wp, mid, turn]
    seq.extend(_follow_next(turn, int(200.0 / STEP_M))[1:])
    return seq


def find_two_curves_same_dir(m: carla.Map) -> Optional[List[carla.Waypoint]]:
    """Find a segment with two gentle curves of the same sign separated by short straight.

    Good for sustained steering with speed control. We accumulate signed dyaw and
    look for: curve (>=CURVE_DEG_MIN) → near‑straight → same‑sign curve.
    """
    for spawn in m.get_spawn_points():
        start_wp = m.get_waypoint(spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        seq = [start_wp]
        cur = start_wp
        prev_yaw = cur.transform.rotation.yaw
        phase = 0
        accum = 0.0
        sign = 0
        dist = 0.0
        while dist < TWO_CURVES_MAX:
            nxts = cur.next(STEP_M)
            if not nxts:
                break
            nxt = nxts[0]
            dyaw = _unwrap_deg(prev_yaw, nxt.transform.rotation.yaw)
            prev_yaw = nxt.transform.rotation.yaw
            seq.append(nxt)
            dist += STEP_M

            if phase == 0:
                accum += dyaw
                if abs(accum) >= CURVE_DEG_MIN:
                    phase = 1
                    sign = 1 if accum > 0 else -1
                    accum = 0.0
            elif phase == 1:
                # wait for near‑straight (|dyaw| small over ~20 m)
                accum += abs(dyaw)
                if accum >= 6.0:  # ~6° accumulated ~ straightish section consumed
                    phase = 2
                    accum = 0.0
            else:  # phase 2: second curve with same sign
                accum += dyaw
                if sign * accum >= CURVE_DEG_MIN and dist >= TWO_CURVES_MIN:
                    # add a short tail and return
                    tail = _follow_next(nxt, 6)
                    seq.extend(tail[1:])
                    return seq
            cur = nxt
    return None


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    args = p.parse_args()

    outdir = pathlib.Path("curricula/medium")
    outdir.mkdir(parents=True, exist_ok=True)

    client = carla.Client("localhost", args.port)
    client.set_timeout(30.0)

    # 1) Town03 S‑bend
    world = client.load_world("Town03")
    m = world.get_map()
    seq = find_s_bend(m)
    if not seq:
        raise RuntimeError("Failed to find S‑bend in Town03 within search budget.")
    _write_route_xml(outdir / "Town03_s_bend_medium.xml", "Town03", [wp.transform for wp in seq])

    # 2) Town02 turn then bend
    world = client.load_world("Town02")
    m = world.get_map()
    seq = find_turn_then_bend(m)
    if not seq:
        print("[medium] heuristic turn-then-bend not found; using deterministic fallback on Town02.")
        seq = force_turn_then_bend(m)
    _write_route_xml(outdir / "Town02_turn_then_bend.xml", "Town02", [wp.transform for wp in seq])


    # 3) Town05 two curves same direction
    world = client.load_world("Town05")
    m = world.get_map()
    seq = find_two_curves_same_dir(m)
    if not seq:
        raise RuntimeError("Failed to find two‑curves segment in Town05 within search budget.")
    _write_route_xml(outdir / "Town05_two_curves_short.xml", "Town05", [wp.transform for wp in seq])

    print("Medium routes written to:")
    for fn in ("Town03_s_bend_medium.xml", "Town02_turn_then_bend.xml", "Town05_two_curves_short.xml"):
        print("  -", (outdir / fn).resolve())


if __name__ == "__main__":
    main()
