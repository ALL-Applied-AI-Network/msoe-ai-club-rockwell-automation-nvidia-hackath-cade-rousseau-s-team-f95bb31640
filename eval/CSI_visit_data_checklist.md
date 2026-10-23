# CSI visit: data audit and collection checklist

Date: October 23, 2026. Project: Changeover Guard.

## What was checked

Updated after inspecting `C:\Users\palaciosn\Downloads\Rockwell-NVIDIA-Fall-2026-Hackathon-20261023T211153Z-1-001.zip` (1,154,508,184 bytes), including both dataset/twin READMEs, camera JSONs, image indexes, video manifest, and inventories/readmes of both nested ZIPs. Earlier checks covered all Run 5 annotation files, repository model/calibration code, and one vial-fill still. This is a file/annotation/documentation audit, not a full video decode, visual review of every frame, or measured carryover analysis. Findings below supersede the earlier vial-fill-only audit.

## Verified inventory

- All three cameras are supplied: vial-fill, tank-level and cap/seal, each with five MP4 videos. Dataset README and `videos.csv` are present.
- 4,650 JPEG files: 1,735 vial-fill, 1,734 tank-level and 1,181 cap/seal. There are 4,647 indexed images: 1,733, 1,733 and 1,181 respectively. The three extra images are unindexed; their content was not compared for duplication.
- Vial-fill and tank-level indexed counts for videos 1–5 are both 558, 78, 95, 157, 845. Every timestamp matches exactly between these two cameras, satisfying the current loader's join requirement. Mean vial-fill still sampling interval is approximately 2.01 seconds.
- Every indexed image exists across all cameras; no duplicate indexed filenames or timestamps within a camera.
- Cap/seal has 5 indexed images assigned to video 1, one unassigned image, and 78/95/157/845 for videos 2–5. Video 1 has zero exact timestamp matches to vial-fill; videos 2–5 match fully. This limits still-based three-camera analysis of video 1, but its cap/seal MP4 is present.
- Run 5 has 845 frame-label rows, all matching its image index, with no duplicate label filenames.
- All 30 vials have in-fill image labels (10–24 labeled stills per vial).
- Recipe CSV/JSON describe 30 vials: recipes 5–14, three consecutive vials each. CSV component volumes sum correctly to the stated totals.
- Observed recipe-block order: 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14. This provides nine adjacent transitions, each once; initial preceding recipe is unknown. It does not provide repeated trials of the same transition.
- Manifest documents 64 vials total: 19/3/6/6/30 in runs 1–5. Only the final 30 have recipe labels. The other 34 are available for additional labeling, rather than requiring new collection merely to increase vial count.
- `data/` is absent in this checkout. The complete package is downloaded but not installed in the dashboard's expected layout. It now supplies both camera indexes required by the loader.
- Digital twin includes fill/tray stations, facility/track geometry, camera calibration JSONs, render tools and previews. The nested twin ZIP is another packaged copy, not additional independent experimental trials. The FANUC starter is a separate cube grasping/stacking task and supplies no vial-fluid measurements.

## Have versus need

| Data | Current evidence | Visit action |
|---|---|---|
| Recipe IDs, nominal component/total volumes, dispense order | Present for Run 5 | Reuse; record any changes to settings |
| Vial sequence, first/second/third position | Derivable for Run 5 | Extend to first five vials for repeated transitions |
| Vial imagery and video | Fifteen videos across three cameras, indexed stills | Reuse existing footage; record targeted experiments with stable camera settings |
| Fill intervals and frame ranges | Present in Run 5 annotations | Verify timing against actual fill events |
| Tank footage and cross-camera synchronization | Present; all vial/tank indexed timestamps match | Reuse; verify physical event alignment against video |
| Clean color references | Provisional RGB values in `twin/calibration.json` | Collect independently verified clean samples, fixed lighting, color/gray reference |
| Known actual mixture fractions / carryover | No measurement fields or ground truth found | Prepare measured reference mixtures with CSI staff |
| Actual delivered volume | Nominal recipe amounts only | Measure tare and filled mass; record density if converting to mL |
| Flush volume, duration, procedure and residual fluid | No explicit annotations found | Log approved procedure, measured volume/time and observations |
| Operator pass/fail and acceptance criteria | No annotations found | Label each vial and record the reason and criterion |
| PLC recipe commands, pump/valve events and alarms | No event log found | Request timestamped export and camera clock offset |
| Repeatability of a particular transition | Each adjacent pair occurs once | Repeat one A → B transition; include B → A if time allows |

Absence above means absent from the supplied annotation/documentation fields, not proof that CSI has never recorded it. Some events might be visible in video, but visibility is not a measured label. The README calls tank volumes “mL dispensed”; it does not document an independent weighing/volume measurement method or uncertainty, so do not treat them as verified actual delivery measurements without clarification.

## Important calibration caveats

`twin/calibrate.py` explicitly reports that still-image labels run 12–22 seconds ahead of the pour and therefore samples video directly. This timing offset was found in code; it was not independently measured during this audit. Ask CSI to verify the video/still/PLC time mapping before using still labels as physical-event truth.

The calibration code treats every third vial as clean. That is an assumption, not independently verified cleanliness. Existing RGB references are useful starting points, but do not establish physical carryover percentages. Different target fill volumes also change liquid depth and appearance. The single inspected still shows reflections, a holder, and a burned-in timestamp, so choose and verify the liquid measurement region carefully.

## Prioritized collection list to bring today

- [ ] **Ask for missing metadata first:** any PLC/flush logs, camera exposure/white-balance settings, recipe identities for videos 1–4, measurement provenance of recipe volumes, and explanation of annotation timing. Tank-level files, dataset README and `videos.csv` are already supplied; do not request them again.
- [ ] **Collect known-ratio reference mixtures:** clean B and measured amounts of A in B, prepared with CSI-approved fluids. Suggested starting fractions: 0%, 1%, 2%, 5%, 10%; record actual volumes, fraction definition, preparation uncertainty, total volume and mixing method. Use matched vial geometry/fill depth and repeat photos. These are calibration examples, not assumed acceptance limits.
- [ ] **Collect verified clean baselines:** at least five vials per chosen recipe if feasible; record cleaning/reset confirmation, fixed illumination/exposure/white balance, camera position, color/gray reference and an empty vial.
- [ ] **Record a controlled A → B transition:** preserve first five B vials in order, physical vial IDs, normal flush procedure, exact fill times and images. Repeat the same transition at least twice more if time permits; add B → A to assess direction.
- [ ] **Measure output:** tare, filled mass, scale resolution, fluid density/temperature if available, measured flush volume and duration, and any independent residual-fluid measurement CSI can provide.
- [ ] **Get quality ground truth:** operator pass/fail/unknown per vial, reason, acceptable color/volume tolerances and whether those are established criteria or provisional judgments.
- [ ] **Record mechanism and costs:** ask whether fluids share a manifold/nozzle/tube, where residual fluid could remain, whether cleaning occurs between recipe blocks, and normal changeover time/waste. Record operator answers separately from measurements.

For 60–90 minutes, prioritize metadata clarification, clean references and known-ratio mixtures, then one measured transition with first five vials and operator labels. Repeated transitions are the next priority. Follow CSI's approved operating and cleaning procedures.

Recipe 13 (T200 50 mL + T400 20 mL) and recipe 6 (T100 30 mL + T200 10 mL) are verified definitions, but this pair is not an observed adjacent transition in Run 5. They also differ in total volume (70 versus 40 mL). Ask CSI to select a visibly distinct pair and control or record fill depth; do not assume this pair is optimal.

## Short request for CSI staff

“We have all three camera streams, Run 5 recipe/timing labels, and the digital twin. We need physical ground truth: verified clean and known-ratio reference samples, actual fill measurements, flush records, and operator quality judgments. Could we preserve the first five vials of one approved recipe transition, repeat it if possible, and synchronize the camera with recipe/fill events? Can you also clarify recipe identities for videos 1–4, whether listed volumes were measured or commanded, and the existing still/video timing mapping?”

## Model-specific holes and how to close them

| Model | What supplied data supports | Remaining hole and best next action |
|---|---|---|
| Color/composition predictor | Ten known nominal mixtures, three vials each; tank and fill imagery; existing provisional RGB fit | No pure-fluid/independently clean reference set or known carryover fractions. Collect references at matched fill depths; test lighting/exposure sensitivity. Recipes 5 and 8 have identical nominal fractions but different total volume (30/60 mL), useful for checking fill-depth effects. |
| Carryover estimator | Nine distinct adjacent recipe transitions, three vials per block | Previous recipe, flushing and actual residual fraction are not independently measured; no repeated same-pair trials. Collect known-ratio mixtures plus repeated measured transitions. First-vial appearance alone cannot identify the cause. |
| Flush optimizer | Video may expose observable timing and actions after manual review | No explicit flush intervention/outcome/cost table. Collect approved flush procedure, volume, duration, waste and measured post-flush quality; without this, optimization remains assumption-driven. |
| Recipe scheduler | Recipe definitions and one observed block ordering | Nine of 90 possible directed non-self pairs among ten recipes are observed once, with no measured pair-specific changeover cost. Start by testing a few contrasting pairs in both directions, then expand coverage; label scheduling scores as estimates until validated. |
| Vial volume/level estimator | 640×480 images, nominal volume range 30–80 mL and fill-camera geometry | No real measured fill-height-to-volume calibration, tare/mass truth, meniscus masks or measurement uncertainty. Collect several known volumes in the actual vial and camera view. |
| Tank reader | Complete synchronized tank imagery; T100 clear, T200 orange, T400 blue | No tank bounding boxes, level/color ground truth, tank dimensions/level-to-volume calibration or tank-camera JSON in the supplied twin camera files. Annotate ROIs locally; ask CSI for tank geometry and measured levels. T300 was intermittent/out of service, so T300 operation is outside current validated coverage. |
| Visual defect / quality detector | Existing normal-operation footage, including cap/seal | No supplied defect classes, per-vial verdicts, masks or acceptance thresholds. Annotate existing visible events locally; collect operator-labeled examples and approved reference deviations. Timestamp alignment alone does not establish the same vial's identity downstream at cap/seal; obtain transit/identity mapping if using that view. |
| Synthetic-data generator | Calibrated FillCam/CapSealCam geometry, materials, labels and a 686-frame animation | Twin README explicitly says no physics; blue-then-clear animation and approximate liquid/glass do not simulate residual mixing or flushing. Fit appearance to measured references and add parameterized scenarios; synthetic carryover labels are chosen inputs, not experimentally established truth. |

## Work possible now without another CSI experiment

1. Extract the complete dataset into the expected local `data/` layout without committing media. Preserve twin relative paths separately.
2. Review videos 1–4 and annotate vial boundaries; recover recipe IDs only with supporting operator/log evidence. Do not infer ground-truth composition solely from color.
3. Verify Run 5 annotation timing against actual fills and create a corrected mapping if needed. The README describes synchronized captures, while repository code reports a still-to-pour discrepancy; cross-camera index equality does not resolve that discrepancy.
4. Label tank/vial regions, occlusions, glare, completed-fill frames and visible quality issues. Keep physical truth separate from image-derived estimates.
5. Compare first/second/third vial appearance at equivalent completed-fill phases. Treat differences as exploratory observations, not measured contamination.
6. Evaluate by holding out whole vials, runs or transition trials. Do not randomly split adjacent frames of the same vial between training and test. Only one recipe-labeled run exists, limiting independent real-world validation; collect another labeled session if feasible.

## Recording rules

Use `CSI_vial_collection.csv` for one row per physical vial/reference sample. Match IDs on containers, photos and event records. Leave unknown numeric values blank; zero means a real zero. Mark values as measured, commanded, derived or unknown in notes. Record run-level setup, camera settings, lighting, fluid identity, clock offset, cleaning procedure and instruments in a separate run note keyed by `run_id`. A color difference alone is not proof of carryover, and no detectable difference is a valid result within measurement sensitivity.
