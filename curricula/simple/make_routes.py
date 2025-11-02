# safe-rl-carla\curricula\simple\make_routes.py
"""
Write three minimal route XMLs:
  Town01_straight.xml
  Town03_curve.xml
  Town02_tjunction.xml
They are all id="0" so CarlaCMDPEnv will always load the single route.
Also, generate a pedestrian crossing JSON for Town01.
This is used by the curriculum to spawn a pedestrian crossing scenario.
"""

import argparse, pathlib, xml.etree.ElementTree as ET, carla, math, random

def wp_to_elem(parent, tf: carla.Transform):
    ET.SubElement(parent, "waypoint",
                  x=f"{tf.location.x:.3f}",
                  y=f"{tf.location.y:.3f}",
                  z="0.0",
                  pitch="0.0", roll="0.0",
                  yaw=f"{tf.rotation.yaw:.3f}")

def write_route(fname, town, wp0, wp1):
    routes = ET.Element("routes")
    route  = ET.SubElement(routes, "route", id="0", town=town)
    for tf in (wp0, wp1):
        wp_to_elem(route, tf)
    ET.ElementTree(routes).write(fname, encoding="UTF-8", xml_declaration=True)

def main(port):
    outdir = pathlib.Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)

    client = carla.Client("localhost", port)
    client.set_timeout(30.0)

    # ---------- Town01 straight -----------------------------------
    world = client.load_world("Town01")
    m     = world.get_map()
    start = m.get_spawn_points()[0]                # pick first spawn

    start_wp = m.get_waypoint(start.location, project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    # advance along the current lane centre until the accumulated
    # yaw change exceeds a small threshold (here: 5°); in practice
    # Town01 stays dead‑straight for ~95 m, then bends—so we stop earlier
    travelled, goal_wp = 0.0, start_wp
    MAX_AHEAD, STRAIGHT_YAW = 120.0, 5.0          # m, deg
    while travelled < MAX_AHEAD:
        nxt = goal_wp.next(5.0)[0]                # 5‑m increments
        dyaw = abs(((nxt.transform.rotation.yaw -
                     start.rotation.yaw + 540) % 360) - 180)
        if dyaw > STRAIGHT_YAW:
            break
        travelled += goal_wp.transform.location.distance(nxt.transform.location)
        goal_wp = nxt
    goal     = goal_wp.transform
    write_route(outdir/"Town01_straight.xml", "Town01", start, goal)


    # --------------------------------------------------------------
    #   NEW: generate a matching pedestrian‑crossing JSON
    # --------------------------------------------------------------
    mid_wp   = start_wp.next(75.0)[0]              # ~ middle of the segment
    ped_json = {
        "available_scenarios": [{
            "Town01": [{
                "scenario_type": "PedestrianCrossing",
                "available_event_configurations": [{
                    "transform": {
                        "x":  round(mid_wp.transform.location.x, 3),
                        "y":  round(mid_wp.transform.location.y, 3),
                        "z":  0.0,
                        "yaw": round(mid_wp.transform.rotation.yaw, 3)
                    },
                    "other_config": {
                        "crossing_distance": 8.0,
                        "walker_speed":      1.4
                    }
                }]
            }]
        }]
    }
    import json, io
    with io.open(outdir/"ped_cross_Town01.json", "w", encoding="utf‑8") as f:
        json.dump(ped_json, f, indent=2)

    # ---------- Town03 gentle 90° curve ---------------------------
    # MINIMAL FIX: instead of hard-coding spawn index 5 and risking a long straight,
    # scan spawn points until we find one that produces a ≥60° bend within a short budget.
    world = client.load_world("Town03")
    m     = world.get_map()

    found_pair = None
    # First pass: target ≥60° within ~200–300 m
    for step_budget in (40, 60):              # 40*5m=200m, 60*5m=300m
        for spawn in m.get_spawn_points():
            start = spawn
            start_wp = m.get_waypoint(spawn.location, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
            goal_wp  = start_wp
            for _ in range(step_budget):
                nxt = goal_wp.next(5.0)[0]
                dyaw = abs((nxt.transform.rotation.yaw - start.rotation.yaw + 540) % 360 - 180)
                goal_wp = nxt
                if dyaw > 60:                 # found a decent bend
                    found_pair = (start, goal_wp.transform)  # include first over-threshold point
                    break
            if found_pair:
                break
        if found_pair:
            break

    # Fallback: accept ≥45° if ≥60° not found (layouts can vary)
    if not found_pair:
        for step_budget in (60, 80):
            for spawn in m.get_spawn_points():
                start = spawn
                start_wp = m.get_waypoint(spawn.location, project_to_road=True,
                                          lane_type=carla.LaneType.Driving)
                goal_wp  = start_wp
                for _ in range(step_budget):
                    nxt = goal_wp.next(5.0)[0]
                    dyaw = abs((nxt.transform.rotation.yaw - start.rotation.yaw + 540) % 360 - 180)
                    goal_wp = nxt
                    if dyaw > 45:
                        found_pair = (start, goal_wp.transform)
                        break
                if found_pair:
                    break
            if found_pair:
                break

    if not found_pair:
        raise RuntimeError("Town03_curve: could not locate a curve within search budget.")

    write_route(outdir/"Town03_curve.xml", "Town03", found_pair[0], found_pair[1])

    # ---------- Town02 T‑junction (right turn) --------------------
    world = client.load_world("Town02")
    m     = world.get_map()
    spawn = m.get_spawn_points()[3]                # just before the junction
    start = spawn

    start_wp = m.get_waypoint(spawn.location, project_to_road=True,
                              lane_type=carla.LaneType.Driving)

    # advance ~15 m to the junction centre
    mid_wp = start_wp.next(15.0)[0]

    # pick the first right‑turn branch from the junction
    right_wp = None
    for conn in mid_wp.next(1.0):
        yaw_diff = ((conn.transform.rotation.yaw -
                     mid_wp.transform.rotation.yaw + 540) % 360 - 180)
        if -120 < yaw_diff < -30:                  # ≈ 30°–120° right turn
            right_wp = conn
            break

    goal_wp = (right_wp or mid_wp).next(40.0)[0]   # follow right lane 40 m
    goal    = goal_wp.transform
    write_route(outdir/"Town02_tjunction.xml", "Town02", start, goal)

    print("XML files written.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2000)
    args = p.parse_args()
    main(args.port)
