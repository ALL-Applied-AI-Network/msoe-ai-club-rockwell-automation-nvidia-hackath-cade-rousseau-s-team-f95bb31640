"""Render synthetic/manifest.csv with Isaac Sim. Writes synthetic/ (video_id 100), replacing fake_render.py.

    /isaac-sim/python.sh twin/render.py --calibrate    # writes twin/material_lut.json
    /isaac-sim/python.sh twin/render.py --limit 12     # first 12 vials; drop --limit for all

Runs in Isaac Sim's own Python (numpy and cv2, no changeover_guard). See BREV.md.
The liquid material's color isn't the pixel color, so --calibrate renders gray liquids, measures
calibration.json's box, and rendering inverts that curve per channel (lut.py).
Every frame is the finished vial (time code 686) with the liquid scaled to the fill level.
ponytail: no dispensing stream during in_fill frames, and every tank_level frame is the same real still.
"""
import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from isaacsim import SimulationApp

sys.path.insert(0, str(Path(__file__).parent))
from lut import diffuse_for
from writer import write_root

ROOT = Path(__file__).parents[1]
STAGE = ROOT / "data" / "csi_digital_twin_v1" / "stations" / "fill_station" / "fill_station.usda"
CALIBRATION = ROOT / "twin" / "calibration.json"
LUT = ROOT / "twin" / "material_lut.json"
MANIFEST = ROOT / "synthetic" / "manifest.csv"
TANK_STILL = ROOT / "synthetic" / "tank_still.jpg"   # a real tank_level frame, from fake_render.py
OUT = ROOT / "synthetic"

P = "/FillStation/FillStation_LocalRoot/"
CAMERA = P + "FillCam/FillCam"
LIQUID = P + "Fill_liquid"
SHADER = "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
LIGHTS = [P + f"{n}/{n}" for n in ("Fill_overhead_panel", "V2_base_reflection", "Fill_front_softbox",
                                   "Fill_right_reflection", "V2_guard_bounce")] + ["/FillStation/env_light"]

W, H = 640, 480
FRAME = 686                                  # finished vial
TWIN_ML = 70.0                               # the animation plays recipe 7
LIQUID_BOTTOM = 0.116                        # liquid mesh z (inches) at the vial floor; scaling keeps it there
VIDEO_ID = 100
OPACITY = 0.9                                # liquid opacity; high enough to reach the darkest real vials
FILL_STEPS = (0.33, 0.66, 1.0)               # in_fill frames per vial, as fractions of total mL
SWATCHES = np.linspace(0, 1, 11) ** 2        # gray diffuse levels, dense near black where real vials sit

# omni and pxr can only be imported once the app is running
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": W, "height": H})

import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdShade


class Twin:
    """The fill station open in Isaac Sim, with an RGB annotator on FillCam. All edits go to the session layer."""

    def __init__(self, subframes: int):
        self.subframes = subframes
        stage = self._open_stage()
        stage.SetEditTarget(stage.GetSessionLayer())     # overrides stay out of the USD files
        omni.timeline.get_timeline_interface().set_current_time(FRAME / stage.GetTimeCodesPerSecond())
        kit.update()

        liquid = stage.GetPrimAtPath(LIQUID)
        # the Xform's visibility is animated and RTX ignores an override of it; the mesh's isn't
        self.visibility = UsdGeom.Imageable(stage.GetPrimAtPath(LIQUID + "/Fill_liquid")).GetVisibilityAttr()
        self.translate, _, self.scale = UsdGeom.Xformable(liquid).GetOrderedXformOps()
        shader = UsdShade.Shader(stage.GetPrimAtPath(SHADER))
        self.diffuse, self.opacity = shader.GetInput("diffuseColor"), shader.GetInput("opacity")
        self.lights = {}
        for path in LIGHTS:
            attr = UsdLux.LightAPI(stage.GetPrimAtPath(path)).GetIntensityAttr()
            self.lights[attr] = attr.Get()

        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([rep.create.render_product(CAMERA, (W, H))])
        rep.orchestrator.set_capture_on_play(False)
        kit.update()
        self._resync_camera(stage)

    @staticmethod
    def _open_stage() -> Usd.Stage:
        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(STAGE)):
            raise RuntimeError(f"could not open stage: {STAGE}")
        while ctx.get_stage_loading_status()[2] > 0:
            kit.update()
        for _ in range(10):
            kit.update()
        return ctx.get_stage()

    @staticmethod
    def _resync_camera(stage: Usd.Stage) -> None:
        """FillCam sits under the 0.0254 inch scale, so RTX misreads its clipping range on first sync.

        Changing and restoring the value forces a correct re-sync, as tools/render_example.py does.
        """
        clip = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA)).GetClippingRangeAttr()
        near_far = clip.Get()
        clip.Set((near_far[0] * 0.5, near_far[1]))
        kit.update()
        clip.Set(near_far)
        kit.update()

    def set_liquid(self, ml: float, diffuse: list[float]) -> None:
        """Show ml of liquid (0 hides it) with the given diffuse color."""
        # the xform ops carry time samples, so their overrides are time samples at FRAME
        s = ml / TWIN_ML if ml else 1.0
        self.visibility.Set("inherited" if ml else "invisible")
        self.scale.Set((1.0, 1.0, s), FRAME)
        self.translate.Set((0.0, 0.0, LIQUID_BOTTOM * (1 - s)), FRAME)
        self.diffuse.Set(tuple(float(c) for c in diffuse))
        self.opacity.Set(OPACITY)

    def set_lights(self, light: float | None = None) -> None:
        """light in [0, 1] scales all lights by 0.85-1.15 (as fake_render.py), each ±10%; None restores them."""
        rng = np.random.default_rng(int((light or 0) * 1e3))
        for attr, base in self.lights.items():
            scale = 1.0 if light is None else (0.85 + 0.3 * light) * rng.uniform(0.9, 1.1)
            attr.Set(float(base * scale))

    def capture(self) -> np.ndarray:
        """Render one frame and return it as an RGB uint8 array."""
        kit.update()
        for _ in range(2):                   # two steps, as tools/render_example.py does
            rep.orchestrator.step(rt_subframes=self.subframes, delta_time=0.0)
        im = np.asarray(self.rgb.get_data())
        if im.ndim != 3 or im.shape[:2] != (H, W):
            raise RuntimeError(f"RGB capture returned shape {im.shape}; check GPU/RTX startup errors")
        return im[..., :3].copy()


def calibrate(twin: Twin, cal: dict) -> None:
    """Render gray swatches into twin/material_lut.json, then check it against the real recipes' colors."""
    x0, y0, x1, y1 = cal["box"]
    box_of = lambda im: np.median(im[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    out = OUT / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    twin.set_lights()

    box = []
    for v in SWATCHES:
        twin.set_liquid(TWIN_ML, [v, v, v])
        im = twin.capture()
        cv2.imwrite(str(out / f"swatch_{v:.3f}.png"), im[..., ::-1])
        box.append(box_of(im))
        print(f"diffuse {v:.3f} -> box {box[-1].round(1)}", flush=True)
    lut = {"opacity": OPACITY, "diffuse": SWATCHES.round(4).tolist(), "box_rgb": np.round(box, 1).tolist()}
    LUT.write_text(json.dumps(lut, indent=1))
    print(f"wrote {LUT.relative_to(ROOT)}; box range {np.min(box, 0).round()} - {np.max(box, 0).round()}")

    for rid, target in cal["target_rgb"].items():
        twin.set_liquid(TWIN_ML, diffuse_for(lut, target))
        im = twin.capture()
        cv2.imwrite(str(out / f"recipe_{rid}.png"), im[..., ::-1])
        got = box_of(im)
        print(f"recipe {rid:>2}  real {np.round(target)}  twin {got.round()}  max err {np.abs(got - target).max():.0f}")


def render(twin: Twin, limit: int | None) -> None:
    """Render the first limit vials of the manifest (all if None): 3 in_fill frames and 1 idle frame each."""
    lut = json.loads(LUT.read_text())
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))[:limit]
    tank = cv2.imread(str(TANK_STILL))
    if tank is None:
        raise FileNotFoundError(TANK_STILL)

    frames, jobs, recipes = [], [], {}       # jobs: (mL, diffuse, light) per frame
    idle = {"in_fill": 0, "vial": None, "recipe_id": None}
    rng = random.Random(0)
    for r in rows:
        vial, rid = int(r["vial"]), int(r["recipe_id"])
        recipes[vial] = {t: float(r[f"{t.lower()}_ml"]) for t in ("T100", "T200", "T400") if r[f"{t.lower()}_ml"]}
        total = sum(recipes[vial].values())
        diffuse = diffuse_for(lut, [float(r[c]) for c in "rgb"])
        for f in FILL_STEPS:
            frames.append({"in_fill": 1, "vial": vial, "recipe_id": rid})
            jobs.append((f * total, diffuse, float(r["light"])))
        frames.append(idle)
        jobs.append((0, diffuse, rng.random()))

    t0 = time.time()
    for i, ((vf, tl), (ml, diffuse, light)) in enumerate(zip(write_root(OUT, VIDEO_ID, frames, recipes), jobs)):
        twin.set_liquid(ml, diffuse)
        twin.set_lights(light)
        cv2.imwrite(str(vf), twin.capture()[..., ::-1])
        cv2.imwrite(str(tl), tank)
        if i % 20 == 0 or i == len(jobs) - 1:
            print(f"{i + 1}/{len(jobs)} frames, {(time.time() - t0) / (i + 1):.1f} s/frame", flush=True)
    print(f"wrote {len(rows)} vials, {len(frames)} frames to synthetic/ (video_id {VIDEO_ID})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="fit twin/material_lut.json instead of rendering")
    ap.add_argument("--limit", type=int, default=None, help="max vials")
    ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality vs speed)")
    a = ap.parse_args()

    try:
        twin = Twin(a.subframes)
        calibrate(twin, json.loads(CALIBRATION.read_text())) if a.calibrate else render(twin, a.limit)
    except BaseException:
        traceback.print_exc()
        kit.close(exit_code=1)               # Kit's fast shutdown would otherwise drop the failure status
        raise
    kit.close()


if __name__ == "__main__":
    main()
