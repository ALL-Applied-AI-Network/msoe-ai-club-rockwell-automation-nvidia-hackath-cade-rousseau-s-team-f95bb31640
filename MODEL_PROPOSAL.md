# Tank initialization and recipe color model

## Part 1: implemented initialization

Open **Tank Initialization** (the default operator UI tab), select **Tank Top View**, review the four liquid regions, and save the baseline. The expected reference view has T_100, T_200, T_300 and T_400 from left to right. Tank identity comes from this camera layout, not from the liquid color. Select a tank and click a clean liquid patch to move its sampling box; numeric bounds also support keyboard adjustment and different camera layouts. Re-measure after edits and confirm the tank IDs before saving.

Detection searches for liquid components inside four camera-specific windows. It is geometry-assisted segmentation, not a trained general object detector. A different crop, camera pose, clear liquid, or poor visibility can require manual regions. A grayscale background can be mistaken for clear liquid in a manual region, so operator review is required. Detection alone does not certify tank identity or color accuracy.

The output labels are `T_100 = (L*, a*, b*)` through `T_400 = (L*, a*, b*)`. Values use CIELAB D65 / 2° observer with an assumed sRGB input, rather than OpenCV's packed 8-bit Lab encoding. Glare and black hardware pixels are excluded; a robust color median limits small reflections. Spatial Lab spread warns of an uneven region, but is not a calibrated confidence interval. Lock exposure/white balance and use a neutral/color reference to obtain comparable measurements across sessions. Tank depth, lighting and optical path mean these baseline values are **camera-relative estimates**, not dye concentrations or calibrated vial colors.

Confirmed snapshots and their normalized source images are saved under ignored `outputs/baselines/<uuid>/`. The UI remembers the snapshot ID in this browser and supports JSON download. Previewing or editing never activates a baseline. Immutable snapshots allow a future model to associate every prediction with its exact baseline. Images are limited to opaque PNG/JPEG, 8 MiB and 24 million pixels.

API: `POST /api/baselines/analyze` accepts `image_base64`, `source_name` and optional `regions` (all four normalized `[left, top, right, bottom]` boxes). `POST /api/baselines` additionally requires `confirmed: true` and reviewed regions. `GET /api/baselines/<id>` and `/image` restore the saved values and image. T_300 is measured but flagged `recipe_enabled: false` to respect the station's existing out-of-service configuration. Existing contracts keep T100/T200/T400 names; translate underscores only at the future integration boundary.

## Part 2: recommended model (proposal, not trained or connected yet)

Use a **calibrated optical-density mixing model with a recipe-level state-space carryover model**. It is interpretable and can be fitted with less data than a neural network. A small regularized regression correction can capture remaining nonlinearities. Start with this baseline before deciding whether a larger learned model is warranted.

### Clean recipe color

Inputs are tank baseline colors, current per-tank quantities `q_i` in mL, and a baseline ID. Let `V = sum(q_i)` and `f_i = q_i/V`. Reject negative or nonfinite quantities, zero total volume, unavailable tanks, and recipes outside validated volume limits. Equal fractions should predict the same clean color at a fixed observation path, even if total throughput changes.

Do not average Lab triples. Recover linear sRGB from the saved RGB (or Lab via XYZ), normalize against a measured blank/reference, and form an optical-density proxy:

```
A_i = -log(clip(linear_rgb_i / linear_rgb_blank, epsilon, 1))
A_clean = sum_i f_i * K_i
rgb_clean = camera_response(blank_vial_linear_rgb * exp(-A_clean))
Lab_clean = sRGB_to_CIELAB(rgb_clean)
```

`K_i` are effective nonnegative three-channel absorption vectors for the **vial observation path**, fitted from dilution series; top-view `A_i` supplies an initialization/covariate, not an assumed exact coefficient. A Beer–Lambert approximation is appropriate for transparent, nonreacting dyes. Broadband RGB optical density is only a proxy for spectral absorbance. Scattering, opacity, chemical interactions and different optical paths can violate it. Test these effects; if necessary fit a regularized interaction term in optical density (`f_i*f_j`) using held-out mixtures. More complex/spectral models should be justified by validation error.

### Recipe run state, not previous vial state

Represent each continuous production run as `{run_id, recipe_id, composition_ml, total_processed_ml, baseline_id, flush_ml}`. A run may fill many vials. Advance the **previous recipe** only at an explicit recipe/run change. A vial index never changes recipe identity. Repeating the same recipe retains the physical state and should not introduce a spurious color discontinuity. A known cleaning/flush updates residue; an unknown history must be reported as unknown, not assumed clean.

Keep a liquid holdup state and, if supported by data, a per-dye adsorption state. One starting parameterization for the transition from previous run `p` to current run `r` is:

```
loading_p = 1 - exp(-V_p / V_load)
H_p = H_max * loading_p * exp(-F / V_flush)        # effective residual mL
m_p,i = M_i * f_p,i * (1 - exp(-q_p,i / Q_i)) * exp(-F / F_i)
alpha = H_p / (V_r + H_p)                         # retained liquid fraction
A_effective = (1-alpha)*A_clean_r + alpha*A_previous_state
              + sum_i (m_p,i / (V_r + H_p)) * K_i
```

`V_p` is **actual total volume processed during the previous recipe run**, not just the previous vial's fill. `q_p,i = f_p,i * V_p`. `F` is flush volume. `H_max`, `V_load`, `V_flush`, `M_i`, `Q_i` and `F_i` must be fitted with nonnegative constraints; `m_p,i` is an effective equivalent mL of dye released in the observation interval. Start with holdup only and add adsorption only if the data identifies it. There is no basis for guessing these constants from one photograph. Total processed volume matters through saturating loading; it should not imply that unlimited liquid remains trapped.

The equation above describes a simplified well-mixed first collected volume `V_r` with added effective residue. Validate the station's displacement/volume balance before adopting it. For fixed metered output where residue displaces new liquid, fit the fraction against that fixed output volume instead. To predict later samples, evolve holdup and adsorption with cumulative **current-recipe throughput** (e.g. exponential washout), then integrate over each sample's collection interval. Preserve the residual state across transitions for multi-recipe memory. The previous recipe is its dominant driver; it is not necessarily a complete description of the residual mixture.

### Impurity index and direction of transition

Return these separately:

1. `residual_fraction`: estimated retained previous-fluid fraction, with uncertainty.
2. `delta_e_00`: CIEDE2000 between predicted contaminated and clean recipe Lab.
3. `impurity_index = 100 * (1 - exp(-delta_e_00 / tau))`: bounded **color-impact score**, where positive `tau` is calibrated to the accepted color tolerance.

The index is not a measured contaminant percentage or chemical purity. The full residue spectrum/color, previous quantities and current dilution determine the color shift. A dark-to-light transition can therefore receive a much larger score than the reversed transition at equal residual volume. Do not use only a symmetric distance between the two clean recipe colors: that cannot capture this directionality. Do not force every dark-to-light pair to be worse; hue, saturation, dose and nonlinear perception also matter. Unknown carryover history produces an unknown index or an explicit uncertainty range.

### Data and validation needed

- Measure a blank/reference, pure tanks and single-dye dilution series under fixed tank and vial lighting. Sample mixtures covering the valid proportion simplex and volume range. Keep actual measured dispensed volumes where possible.
- Collect **ordered recipe transitions** including both directions for dark/light pairs, several previous-run total volumes, several current volumes, no-carryover controls and several flush volumes. Record repeated observations over cumulative throughput; label run boundaries explicitly.
- Fit clean mixing first, freeze/regularize it while fitting residue, and compare against volume-weighted RGB/Lab and no-carryover baselines. Add adsorption or regression complexity only when it improves held-out predictions.
- Split by entire runs and collection sessions (and hold out recipe transitions), never random adjacent vials. Report median/95th-percentile CIEDE2000, error by transition direction, false-negative rate at the team's color tolerance, and uncertainty coverage. Never infer chemical residue accuracy from color accuracy alone.
- Validate zero-residue recovery, same-recipe continuity, zero flush vs increasing flush, increasing loading saturation, dilution with current volume, numerical stability near black, and inactive-tank rejection. If data cannot separate holdup and adsorption, report that identifiability limit and keep the simpler model.

No learned coefficients, recipe predictions or impurity scores are fabricated in the current implementation. The initialization API is available now; this proposal defines the next model implementation and calibration work.
