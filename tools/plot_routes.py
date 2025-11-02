import argparse, time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import carla

# ---------------- XML ----------------
def parse_route_xml(path):
    root = ET.parse(path).getroot()
    route = root.find(".//route")
    if route is None or "town" not in route.attrib:
        raise RuntimeError(f"Missing <route town=...> in {path}")
    pts = [(float(w.get("x")), float(w.get("y"))) for w in root.findall(".//waypoint")]
    if len(pts) < 2:
        raise RuntimeError(f"Need >=2 waypoints in {path}")
    return route.attrib["town"], pts

# ---------------- CARLA utils ----------------
def ensure_world_on_town(client: "carla.Client", town: str) -> "carla.World":
    client.set_timeout(30.0)
    world = client.get_world()
    have = world.get_map().name.split("/")[-1].lower()
    if have != town.lower():
        world = client.load_world(town)
        time.sleep(0.5)
        for _ in range(3):
            try: world.tick(5.0)
            except Exception: pass
    return world

def densify_grp(world: "carla.World", pts, step=2.5):
    """Prefer GlobalRoutePlanner path across junctions; fallback to lane-follow."""
    amap = world.get_map()
    xs, ys = [], []

    # Try GRP
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlannerDAO
        except ImportError:
            from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        dao = GlobalRoutePlannerDAO(amap, sampling_resolution=step)
        grp = GlobalRoutePlanner(dao); grp.setup()

        dense_xy = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            a = carla.Location(x=x0, y=y0, z=0.0)
            b = carla.Location(x=x1, y=y1, z=0.0)
            seg = grp.trace_route(a, b)  # list[(Waypoint, RoadOption)]
            if not seg: continue
            for i, (wp, _ro) in enumerate(seg):
                loc = wp.transform.location
                if dense_xy and (abs(dense_xy[-1][0]-loc.x) + abs(dense_xy[-1][1]-loc.y)) < 1e-3:
                    continue
                dense_xy.append((loc.x, loc.y))
        if dense_xy:
            xs, ys = zip(*dense_xy)
            return np.array(xs, np.float32), np.array(ys, np.float32)
    except Exception:
        pass  # fall back below if GRP unavailable

    # Fallback: lane-follow between consecutive points
    prev_wp = amap.get_waypoint(carla.Location(x=pts[0][0], y=pts[0][1], z=0.0),
                                project_to_road=True, lane_type=carla.LaneType.Driving)
    xs = [prev_wp.transform.location.x]; ys = [prev_wp.transform.location.y]
    for x, y in pts[1:]:
        dst = amap.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                                project_to_road=True, lane_type=carla.LaneType.Driving)
        cur, guard = prev_wp, 0
        while cur.transform.location.distance(dst.transform.location) > step and guard < 5000:
            nxts = cur.next(step)
            if not nxts: break
            cur = nxts[0]
            xs.append(cur.transform.location.x); ys.append(cur.transform.location.y)
            guard += 1
        xs.append(dst.transform.location.x); ys.append(dst.transform.location.y)
        prev_wp = dst
    return np.array(xs, np.float32), np.array(ys, np.float32)

def draw_topology(world, alpha=0.25, lw=0.35):
    for a, b in world.get_map().get_topology():
        ax, ay = a.transform.location.x, a.transform.location.y
        bx, by = b.transform.location.x, b.transform.location.y
        plt.plot([ax,bx], [ay,by], linewidth=lw, alpha=alpha, zorder=0)

# ---------------- RGB underlay (single-shot orthophoto) ----------------
def capture_topdown_rgb(world, center_xy, bbox_w, bbox_h, out_png,
                        fov_deg=90.0, img_w=1920, img_h=1920):
    """
    Place an RGB camera at nadir (pitch -90) above center_xy so the ground
    coverage >= bbox (with a small margin). Save the frame and return scale.
    """
    margin = 1.10  # pad 10%
    want_w = bbox_w * margin
    want_h = bbox_h * margin

    # Horizontal coverage from FOV: cover_w = 2 * H * tan(fov/2)
    import math
    H_w = want_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    # Vertical coverage scales by aspect ratio
    cover_h_for_Hw = want_w * (img_h / img_w)
    H_h = cover_h_for_Hw / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    H = max(60.0, H_w, H_h)  # keep a sane minimum height

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(img_w)))
    bp.set_attribute("image_size_y", str(int(img_h)))
    bp.set_attribute("fov", str(float(fov_deg)))
    bp.set_attribute("sensor_tick", "0.0")

    cx, cy = center_xy
    cam_tf = carla.Transform(
        carla.Location(x=cx, y=cy, z=H),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    )
    cam = world.spawn_actor(bp, cam_tf)

    arr_holder = {"img": None}
    def _cb(image):
        # BGRA uint8
        arr_holder["img"] = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((img_h, img_w, 4))[..., :3]
    cam.listen(_cb)

    # tick a couple frames
    for _ in range(5):
        try: world.tick(2.0)
        except Exception: pass
        if arr_holder["img"] is not None:
            break

    img = arr_holder["img"]
    cam.stop()
    try: cam.destroy()
    except Exception: pass
    if img is None:
        raise RuntimeError("Failed to capture RGB frame.")

    from PIL import Image
    Image.fromarray(img[:, :, ::-1]).save(out_png)  # convert BGR->RGB for PIL

    # meters-per-pixel scales for mapping route to pixels
    cover_w = 2.0 * H * math.tan(math.radians(fov_deg) / 2.0)
    cover_h = cover_w * (img_h / img_w)
    mpp_x = cover_w / img_w
    mpp_y = cover_h / img_h
    return (cx, cy, mpp_x, mpp_y)

def world_to_pixels(xs, ys, cx, cy, mpp_x, mpp_y, img_w, img_h):
    # yaw=0, pitch=-90, roll=0
    # image right  <- world +Y
    # image down   <- world -X
    px = (ys - cy) / mpp_x + (img_w / 2.0)     # use world Y
    py = -(xs - cx) / mpp_y + (img_h / 2.0)    # use -world X
    return px, py


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no-topology", action="store_true")
    ap.add_argument("--rgb", action="store_true",
                    help="Capture a photorealistic top-down image and draw the route on it.")
    ap.add_argument("--rgb-fov", type=float, default=90.0)
    ap.add_argument("--rgb-w", type=int, default=1920)
    ap.add_argument("--rgb-h", type=int, default=1920)
    args = ap.parse_args()

    town, pts = parse_route_xml(args.routes)
    client = carla.Client(args.host, int(args.port))
    world = ensure_world_on_town(client, town)

    xs, ys = densify_grp(world, pts, step=float(args.step))

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    if args.rgb:
        # RGB underlay path
        cx, cy = float(xs.mean()), float(ys.mean())
        bbox_w = float(xs.max() - xs.min()); bbox_h = float(ys.max() - ys.min())
        cx, cy, mpp_x, mpp_y = capture_topdown_rgb(
            world, (cx, cy), bbox_w, bbox_h, str(outp),
            fov_deg=float(args.rgb_fov), img_w=int(args.rgb_w), img_h=int(args.rgb_h)
        )
        # overlay polyline
        px, py = world_to_pixels(xs, ys, cx, cy, mpp_x, mpp_y, args.rgb_w, args.rgb_h)
        import PIL.Image, PIL.ImageDraw
        im = PIL.Image.open(str(outp)).convert("RGB")
        dr = PIL.ImageDraw.Draw(im)
        poly = list(zip(px.tolist(), py.tolist()))
        dr.line(poly, width=6, fill=(20, 200, 20))

        # Optional: Add markers at waypoints for debugging
        # for i in range(0, len(px), max(1, len(px)//10)):
        #     dr.ellipse([(px[i]-5, py[i]-5), (px[i]+5, py[i]+5)], fill=(255, 0, 0))

        im.save(str(outp))
        print("wrote", outp)
        return

    # Topology + route (matplotlib)
    plt.figure(figsize=(8, 8))
    if not args.no_topology:
        draw_topology(world, alpha=0.25, lw=0.35)
    plt.plot(xs, ys, linewidth=3.0, label="route", zorder=5)
    plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title(f"{town} route"); plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(str(outp), dpi=int(args.dpi))
    print("wrote", outp)

if __name__ == "__main__":
    main()