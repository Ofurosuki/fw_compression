# fog_viz — Keio fog-chamber LiDAR visualization

Standalone visualizers for the Keio fog-chamber full-waveform LiDAR dataset. Self-contained;
no model / no downstream repo needed. Kept in this subdir to keep the main repo clean.

## Dataset (read-only, owned by `ikeda`)
- Root: `/data3/user/ikeda/fog/keio_chamber`
- `annotation_v1/<scene>/*_annotation.b2` — dense voxel labels, `(400, 512, 700)` int8
- `b2/**/*.b2` — raw waveforms, `(400, 512, 700)` float32 (intensity 0–~150)
- 5 annotated scenes: `place1`, `place2_only_lidar`, `scene1`, `scene2`, `scene3`
- `.b2` = **Blosc2** array. Read with: `blosc2.load_array(path)` → returns the ndarray directly.
  (shape/dtype is in the SChunk vlmeta `__pack_tensor__`.)

### Label enum (same as ghost dataset + FOG)
`AnnotationLabel: UNKNOWN=0, OBJECT=1, GLASS=2, GHOST=3, FOG=4`

| enum | label | color (RGB) |
|---|---|---|
| 0 | unknown | gray (128,128,128) |
| 1 | object | green (0,255,0) |
| 2 | glass | blue (0,0,255) |
| 3 | ghost | red (255,0,0) |
| 4 | fog | yellow (255,255,0) |

Class presence per scene (FOG is labeled ONLY in scene3; the others use object/glass/ghost):
- place1 / place2_only_lidar / scene1 / scene2 → {object, glass, ghost}
- scene3 → {object, fog}

### Clear (no-fog) reference frames `-gt`
Each scene has a clear capture (mean of ~50 clear frames), pixel-aligned to the fog frame:
`scene1→scene1-gt`, `scene2→scene2_gt`, `scene3→scene3-gt`, `place1→b2/place1/gt`,
`place2_only_lidar→b2/place2/gt`. Only `scene1` has explicit `-fog` raw dirs
(`scene1-2-1fog`, `scene1-3-1fog`).

## Environment
```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/yoshida/fw_compression
# everything runs via: uv run python fog_viz/<script>.py ...
```
Deps already in this project's `.venv`: `blosc2`, `rerun` (0.33), `matplotlib`, `numpy`.
Outputs go to `fog_viz/outputs/`.

## Fixed coordinate transform (voxel → 3D point) — APPROVED, do not change
SRL geometry, "centered" mode on the raw (400,512,700) grid:
```
az  = (px_x/400 - 0.5) * 90deg      # px_x in [0,400)  (axis 0, width)
alt = (px_y/512 - 0.5) * 26deg      # px_y in [0,512)  (axis 1, height)
d   = bin * 0.15  [m]               # bin in [0,700)   (axis 2, range)
x =  d*cos(az)*cos(alt);  y = -d*sin(az)*cos(alt);  z = -d*sin(alt)
```
NOTE: the ghost-fwl repo's own constants (`pointcloud_wrapper.py`: az=(x*47-4512)/100,
dist=bin*1e-7*c/2 ≈ 15 m/bin) are calibrated for their PROCESSED grid, NOT this raw grid —
they give ±10 km here. The above retune was approved by the user.

---

## 1. `viz_waveforms.py` — per-class waveform PNGs
Random pixels per scene; waveform line with class-colored bands at labeled bins. One PNG per
scene (rows = classes present).
```bash
uv run python fog_viz/viz_waveforms.py                 # all 5 scenes, 2 figs each
uv run python fog_viz/viz_waveforms.py --figs_per_scene 1 --cols 4 --seed 0
```
Out: `fog_viz/outputs/<scene>_fig<N>.png`

## 2. `vis_pcd_rerun.py` — GT point cloud (rerun .rrd + 2D-projection PNG)
Lifts the dense annotation to 3D (no waveform plot). Saves a `.rrd` (open locally with
`rerun file.rrd`) AND top/side projection PNGs for a quick scale/orientation check.
```bash
uv run python fog_viz/vis_pcd_rerun.py --scene scene2 --frames 1
uv run python fog_viz/vis_pcd_rerun.py --scene scene3 --with_raw   # also raw strongest cloud
```
Key flags: `--scene`, `--frames`, `--start`, `--with_raw`, `--show_unknown`, `--no_png`,
`--serve` (host web viewer instead of saving). Geometry knobs exist (`--bin_to_m`, `--fov_h`,
`--fov_v`, `--geom ghost|centered`) but defaults are the approved transform.
Out: `fog_viz/outputs/pcd_<scene>.rrd`, `pcd_<scene>_proj.png`

## 3. `vis_pcd_waveform.py` — point cloud + per-pixel waveform (MAIN viewer)
3D point cloud + a per-pixel full-waveform plot with **clear(gt) vs fog overlay** and
**annotation markers on the waveform**. Layout: `[ 3D view | waveform plot + info ]`.
```bash
uv run python fog_viz/vis_pcd_waveform.py --scene scene1 --per_class 200
# all scenes, one frame each:
for s in place1 place2_only_lidar scene1 scene2 scene3; do
  uv run python fog_viz/vis_pcd_waveform.py --scene $s --per_class 200; done
```
Flags: `--scene`, `--frame` (index into annotated frames, fog frames first), `--per_class`
(pixels sampled per class), `--seed`, `--no_overlay` (drop clear overlay), `--serve`.
Out: `fog_viz/outputs/pcd_wf_<scene>.rrd`

### How to use the .rrd viewer
```bash
rerun fog_viz/outputs/pcd_wf_scene1.rrd
```
- **Left**: 3D point cloud colored by class; a white point = the currently inspected pixel.
- **Right top**: waveform plot — **cyan = clear(gt)**, **gray = fog**, x = range bin, up =
  intensity. **Dots = annotation** at labeled bins, colored by class
  (green=object, blue=glass, red=ghost, yellow=fog).
- **Right bottom**: pixel info (coords, 3D pos, peak heights, labeled bins per class).
- **Bottom `pixel` timeline**: SCRUB it to step through sampled pixels (ordered by pixel
  position). The white 3D point + waveform + info update in sync.

## rerun gotchas (why it's built this way)
- A **BarChart/waveform renders only inside a view** (BarChartView / Spatial2DView). The
  selection panel shows raw component values, NOT a chart. So waveforms are drawn as
  `LineStrips2D` in a `Spatial2DView`.
- **No click→plot callback** in the standalone `.rrd` / `serve_web_viewer`. Viewer selection
  callbacks (`rerun.notebook.Viewer.on_event` → `EntitySelectionItem.instance_id/position`)
  exist ONLY via the `rerun-sdk[notebook]` widget in Jupyter. Hence the `pixel`-timeline scrub
  drives fixed plot entities instead of reacting to clicks.
- The 3D point cloud is from the **fog/annotated** frame (the clear `-gt` frames have no
  annotation, so they appear only as the cyan waveform overlay, not as points).
