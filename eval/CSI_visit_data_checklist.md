# CSI visit: data audit and collection checklist

Date: October 23, 2026. Project: Changeover Guard.

## What was checked

Inspected the ZIP directory and all CSV/JSON annotation files in `C:\Users\palaciosn\Downloads\vial_fill-20261023T185918Z-1-001.zip` (182,620,499 bytes), the supplied background notes, and the repository's loader, calibration code, calibration output, and evaluation notes. Visually inspected one extracted still. This is a file/annotation audit, not a full video decode or a measured carryover analysis.

## Verified inventory

- Five MP4 files: vial-fill videos 1–5. No tank-level camera, cap-seal camera, dataset README, or root `videos.csv` in this ZIP.
- 1,735 JPEG files; 1,733 indexed images. Indexed counts for videos 1–5: 558, 78, 95, 157, 845. Mean still sampling interval is approximately 2.01 seconds in each video.
- Every indexed image exists; no duplicate index filenames.
- Two unindexed JPEGs have `(1)` filename suffixes. Their content was not compared for duplication.
- Run 5 has 845 frame-label rows, all matching its image index, with no duplicate label filenames.
- All 30 vials have in-fill image labels (10–24 labeled stills per vial).
- Recipe CSV/JSON describe 30 vials: recipes 5–14, three consecutive vials each. CSV component volumes sum correctly to the stated totals.
- Observed recipe-block order: 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14. This provides nine adjacent transitions, each once; initial preceding recipe is unknown. It does not provide repeated trials of the same transition.
- `data/` is absent in this checkout. The ZIP is downloaded but not installed in the layout required by the dashboard loader. That loader requires both vial-fill and tank-level indexes and joins on identical timestamps.

## Have versus need

| Data | Current evidence | Visit action |
|---|---|---|
| Recipe IDs, nominal component/total volumes, dispense order | Present for Run 5 | Reuse; record any changes to settings |
| Vial sequence, first/second/third position | Derivable for Run 5 | Extend to first five vials for repeated transitions |
| Vial imagery and video | Five videos, indexed stills | Record experiments with stable camera settings |
| Fill intervals and frame ranges | Present in Run 5 annotations | Verify timing against actual fill events |
| Tank footage and cross-camera synchronization | Absent from this ZIP | Ask for existing tank-level package before recollecting |
| Clean color references | Provisional RGB values in `twin/calibration.json` | Collect independently verified clean samples, fixed lighting, color/gray reference |
| Known actual mixture fractions / carryover | No measurement fields or ground truth found | Prepare measured reference mixtures with CSI staff |
| Actual delivered volume | Nominal recipe amounts only | Measure tare and filled mass; record density if converting to mL |
| Flush volume, duration, procedure and residual fluid | No explicit annotations found | Log approved procedure, measured volume/time and observations |
| Operator pass/fail and acceptance criteria | No annotations found | Label each vial and record the reason and criterion |
| PLC recipe commands, pump/valve events and alarms | No event log found | Request timestamped export and camera clock offset |
| Repeatability of a particular transition | Each adjacent pair occurs once | Repeat one A → B transition; include B → A if time allows |

Absence above means absent from the inspected ZIP/repository, not proof that CSI has never recorded it. Some events might be visible in video, but visibility is not a measured label.

## Important calibration caveats

`twin/calibrate.py` explicitly reports that still-image labels run 12–22 seconds ahead of the pour and therefore samples video directly. This timing offset was found in code; it was not independently measured during this audit. Ask CSI to verify the video/still/PLC time mapping before using still labels as physical-event truth.

The calibration code treats every third vial as clean. That is an assumption, not independently verified cleanliness. Existing RGB references are useful starting points, but do not establish physical carryover percentages. Different target fill volumes also change liquid depth and appearance. The single inspected still shows reflections, a holder, and a burned-in timestamp, so choose and verify the liquid measurement region carefully.

## Prioritized collection list to bring today

- [ ] **Ask for existing missing files first:** tank-level images/index/videos, dataset README, `videos.csv`, any PLC/flush logs, camera settings, and explanation of annotation timing.
- [ ] **Collect known-ratio reference mixtures:** clean B and measured amounts of A in B, prepared with CSI-approved fluids. Suggested starting fractions: 0%, 1%, 2%, 5%, 10%; record actual volumes, fraction definition, preparation uncertainty, total volume and mixing method. Use matched vial geometry/fill depth and repeat photos. These are calibration examples, not assumed acceptance limits.
- [ ] **Collect verified clean baselines:** at least five vials per chosen recipe if feasible; record cleaning/reset confirmation, fixed illumination/exposure/white balance, camera position, color/gray reference and an empty vial.
- [ ] **Record a controlled A → B transition:** preserve first five B vials in order, physical vial IDs, normal flush procedure, exact fill times and images. Repeat the same transition at least twice more if time permits; add B → A to assess direction.
- [ ] **Measure output:** tare, filled mass, scale resolution, fluid density/temperature if available, measured flush volume and duration, and any independent residual-fluid measurement CSI can provide.
- [ ] **Get quality ground truth:** operator pass/fail/unknown per vial, reason, acceptable color/volume tolerances and whether those are established criteria or provisional judgments.
- [ ] **Record mechanism and costs:** ask whether fluids share a manifold/nozzle/tube, where residual fluid could remain, whether cleaning occurs between recipe blocks, and normal changeover time/waste. Record operator answers separately from measurements.

For 60–90 minutes, prioritize missing-file handoff, clean references and known-ratio mixtures, then one measured transition with first five vials and operator labels. Repeated transitions are the next priority. Follow CSI's approved operating and cleaning procedures.

Recipe 13 (T200 50 mL + T400 20 mL) and recipe 6 (T100 30 mL + T200 10 mL) are verified definitions, but this pair is not an observed adjacent transition in Run 5. They also differ in total volume (70 versus 40 mL). Ask CSI to select a visibly distinct pair and control or record fill depth; do not assume this pair is optimal.

## Short request for CSI staff

“We already have vial-fill videos and the Run 5 recipe/timing labels. We need physical ground truth: verified clean and known-ratio reference samples, actual fill measurements, flush records, and operator quality judgments. Could we preserve the first five vials of one approved recipe transition, repeat it if possible, and synchronize the camera with recipe/fill events? We also need the tank-camera package and clarification of the existing still/video timing offset.”

## Recording rules

Use `CSI_vial_collection.csv` for one row per physical vial/reference sample. Match IDs on containers, photos and event records. Leave unknown numeric values blank; zero means a real zero. Mark values as measured, commanded, derived or unknown in notes. Record run-level setup, camera settings, lighting, fluid identity, clock offset, cleaning procedure and instruments in a separate run note keyed by `run_id`. A color difference alone is not proof of carryover, and no detectable difference is a valid result within measurement sensitivity.
