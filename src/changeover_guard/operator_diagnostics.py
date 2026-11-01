"""Evidence-aware retrospective operator attribution, without pre-fill subtraction.

Color signatures and recovery trends support hypotheses, not chemical causal
proof. Missing references abstain. Requested mL never stand in for measured fill.
Follow-up observations used for attribution are disclosed as retrospective.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from skimage.color import deltaE_ciede2000

from changeover_guard.diagnostic_baselines import (
    ML,
    RGB,
    empirical_target,
    trusted_rows,
    truth,
)
from changeover_guard.vial_color import color_errors, lab_values

STATE_LABELS = {
    "measurement_sensor_reject": "Measurement / sensor reject",
    "expected_changeover_carryover": "Changeover carryover suspected",
    "persistent_tank_or_recipe_fault": "Persistent tank / recipe / fill fault suspected",
    "in_spec_clean_production": "In-spec / clean production",
    "evidence_pending": "Evidence pending",
}

PREFILL_SUGGESTION = (
    "Inspect this vial's glass, position and reflections. If available, review an image "
    "of the same vial before dispensing, or collect a pre-fill image on a repeat. "
    "Use it as a visual diagnostic; do not subtract it from the measured color."
)


def trajectory_evidence(observed, current, previous, threshold=2.2) -> dict:
    """Distance to a Lab color segment; alpha is not a chemical mixing fraction."""
    a, b, x = lab_values(current), lab_values(previous), lab_values(observed)
    separation = float(deltaE_ciede2000(a, b))
    if separation < 3 * threshold:
        return {
            "trajectory_supported": False,
            "trajectory_residual_delta_e": None,
            "prior_color_weight": None,
            "reason": "Previous and target colors are insufficiently separated.",
        }

    def error(alpha):
        return float(deltaE_ciede2000(x, a + alpha * (b - a)))

    fit = minimize_scalar(error, bounds=(0, 1), method="bounded")
    alpha = float(fit.x)
    return {
        "trajectory_supported": bool(error(alpha) < threshold and alpha > 0.05),
        "trajectory_residual_delta_e": error(alpha),
        "prior_color_weight": alpha,
        "reason": "Lab color-segment agreement is exploratory, not proof of fluid identity.",
    }


def recovery_trends(measurements, scores, method, threshold=2.2) -> dict:
    """Require three contiguous usable observations, beginning at the first vial."""
    reference = scores[scores.method.eq(method)][["vial_key", "delta_e"]]
    sequence = measurements.merge(
        reference, on="vial_key", how="left", validate="one_to_one"
    )
    trends = {}
    for run_id, run in sequence.groupby("run_id", sort=False):
        run = run.sort_values("vial")
        prefix = []
        for _, row in run.iterrows():
            if not trusted_rows(pd.DataFrame([row])).iloc[0] or pd.isna(row.delta_e):
                break
            prefix.append(float(row.delta_e))
        decreasing = False
        persistent = False
        if len(prefix) >= 3 and prefix[0] >= threshold:
            steps = np.diff(prefix)
            decreasing = (
                prefix[-1] <= 0.8 * prefix[0]
                and np.mean(steps <= 0.1 * threshold) >= 2 / 3
            )
            persistent = (
                all(value >= threshold for value in prefix)
                and prefix[-1] >= 0.8 * prefix[0]
            )
        trends[run_id] = {
            "run_decay_supported": bool(decreasing),
            "run_persistent_deviation": bool(persistent),
            "run_late_rebound": bool(
                len(prefix) >= 3 and prefix[-1] > min(prefix[1:]) + threshold / 3
            ),
            "trend_usable_prefix_vials": len(prefix),
            "trend_first_delta_e": prefix[0] if prefix else None,
            "trend_last_delta_e": prefix[-1] if prefix else None,
            "trend_scope": "retrospective; later vials in the run may be used",
        }
    return trends


def _verified_fill(row, tolerance_ml):
    measured = pd.to_numeric(
        pd.Series([row.get("measured_fill_ml", np.nan)]), errors="coerce"
    ).iloc[0]
    verified = truth(pd.Series([row.get("fill_calibration_verified", False)])).iloc[0]
    if not verified or not np.isfinite(measured):
        return None, "not_measured"
    error = float(measured - row.requested_ml)
    return error, "abnormal" if abs(
        error
    ) > tolerance_ml else "within_measured_tolerance"


def diagnose_vials(
    measurements,
    benchmark,
    *,
    camera_proven=False,
    clean_reference_validated=False,
    expected_clearance_ml=None,
    fill_tolerance_ml=5.0,
) -> pd.DataFrame:
    """Every input identity gets a report; unsupported evidence stays pending.

    expected_clearance_ml must come from separate validated experiments; it is
    never derived from this run's tail. Static overhead tank colors alone cannot
    identify wrong concentration, levels or valves.
    """
    method = benchmark["selection"]["primary_method"]
    threshold = float(benchmark["selection"]["threshold"])
    if expected_clearance_ml is not None and (
        not np.isfinite(expected_clearance_ml) or expected_clearance_ml <= 0
    ):
        raise ValueError(
            "expected_clearance_ml must be a positive independently validated volume"
        )
    if not np.isfinite(fill_tolerance_ml) or fill_tolerance_ml <= 0:
        raise ValueError("fill_tolerance_ml must be finite and positive")
    scores = benchmark["scores"]
    selected = (
        scores[scores.method.eq(method)] if method is not None else scores.iloc[:0]
    )
    selected = selected.set_index("vial_key")
    trends = recovery_trends(measurements, scores, method, threshold) if method else {}
    tails = benchmark["tails"]
    previous_runs = {}
    for _, session in measurements.sort_values("vial").groupby(
        "session_id", sort=False
    ):
        previous = None
        for run_id, run in session.groupby("run_id", sort=False):
            previous_runs[run_id] = previous
            previous = run.iloc[0]

    records = []
    for _, vial in measurements.sort_values(["session_id", "vial"]).iterrows():
        score = selected.loc[vial.vial_key] if vial.vial_key in selected.index else None
        de = np.nan if score is None else score.delta_e
        previous = previous_runs[vial.run_id]
        trend = trends.get(vial.run_id, {})
        fill_error, fill_status = _verified_fill(vial, fill_tolerance_ml)
        state, assessment, evidence = "evidence_pending", "unassessable", []
        production = "PENDING"
        trajectory = {
            "trajectory_supported": False,
            "trajectory_residual_delta_e": None,
            "prior_color_weight": None,
        }
        suggestion = ""
        action = "Review measurement and reference evidence before making a production decision."
        explanation = "No independent clean reference is available; chemical mixture cannot be verified."
        previous_rgb = None
        if score is not None and all(
            f"previous_target_{c}" in score.index for c in RGB
        ):
            predicted_previous = score[[f"previous_target_{c}" for c in RGB]].to_numpy(
                float
            )
            prior_domain = score.get("previous_composition_in_training_hull", True)
            if (
                np.isfinite(predicted_previous).all()
                and pd.notna(prior_domain)
                and bool(prior_domain)
            ):
                previous_rgb = predicted_previous
        elif previous is not None:
            previous_rgb, _ = empirical_target(
                tails,
                int(previous.recipe_id),
                previous[ML].to_numpy(float),
                vial.run_id,
            )
        trusted = bool(trusted_rows(pd.DataFrame([vial])).iloc[0])
        if not trusted:
            state = "measurement_sensor_reject"
            flags = str(vial.get("quality_flags", "") or "")
            if not truth(pd.Series([vial.source_clock_verified])).iloc[0]:
                flags += ";source_clock_unverified"
            if not np.isfinite(vial[RGB].to_numpy(float)).all():
                flags += ";color_not_observed"
            explanation = f"Cannot verify chemical mixture: {flags.strip(';') or 'measurement quality gate failed'}."
            if "stale" in flags:
                age = vial.get("source_timestamp_age_s", np.nan)
                if np.isfinite(age):
                    explanation += f" Camera frame is stale by {age:.2f} seconds."
            action = "Inspect the camera, exposure, vial position and clock alignment; reacquire a trustworthy frame."
            evidence.append("measurement_gate_failed")
            suggestion = PREFILL_SUGGESTION
        elif fill_status == "abnormal":
            state = "persistent_tank_or_recipe_fault"
            assessment = "measured_fill_problem"
            evidence.append("independently_calibrated_fill_measurement")
            explanation = f"Measured fill differs from the request by {fill_error:+.2f} mL. Check dispense actuation and the fill sensor; color alone cannot identify the cause."
            action = "Verify delivered volume, tank supply, valves and requested recipe under the approved procedure."
            suggestion = PREFILL_SUGGESTION
        elif score is None or not np.isfinite(de):
            evidence.append("reference_missing")
            action = "Collect clean same-camera controls for this recipe; do not substitute an unmapped tank color."
            suggestion = PREFILL_SUGGESTION
        elif score.reference_scope == "tank_camera_unmapped":
            assessment = "cross_camera_candidate_only"
            explanation = (
                f"Candidate tank-to-vial difference is {de:.2f} ΔE00, but camera transfer "
                "and optical path length are uncalibrated. This difference cannot identify a chemical fault."
            )
            evidence.append("camera_transfer_unvalidated")
            action = "Validate paired tank/vial clean controls before attributing this color difference to the process."
            if de >= threshold:
                suggestion = PREFILL_SUGGESTION
        elif (
            score.reference_scope == "csi_train_only_camera_transfer"
            and pd.notna(score.get("composition_in_training_hull"))
            and not bool(score.composition_in_training_hull)
        ):
            assessment = "outside_model_domain"
            evidence.append("recipe_extrapolates_beyond_training_compositions")
            explanation = (
                f"Held-out color error is {de:.2f} ΔE00, but this recipe lies outside "
                "the training composition hull. Retain the error as an extrapolation test; "
                "model extrapolation cannot identify carryover or a process fault."
            )
            action = "Collect additional CSI training compositions around this recipe before causal interpretation."
            if de >= threshold:
                suggestion = PREFILL_SUGGESTION
        else:
            assessment = (
                "within_reference_exploratory"
                if de < threshold
                else "outside_reference_exploratory"
            )
            evidence.extend(
                [f"delta_e={de:.3f}", f"threshold={threshold}", score.reference_scope]
            )
            if previous_rgb is not None:
                trajectory = trajectory_evidence(
                    vial[RGB].to_numpy(float),
                    score[[f"target_{c}" for c in RGB]].to_numpy(float),
                    previous_rgb,
                    threshold,
                )
            verified_tank_alert = (
                truth(pd.Series([vial.get("tank_state_verified", False)])).iloc[0]
                and truth(pd.Series([vial.get("tank_state_alert", False)])).iloc[0]
            )
            wrong_tank = None
            # Only in-camera pure-tank signatures can support this hypothesis.
            if de >= threshold:
                for tail in tails:
                    composition = np.asarray(tail["composition"])
                    nonzero = np.flatnonzero(composition > 0)
                    if tail["run_id"] == vial.run_id or len(nonzero) != 1:
                        continue
                    tank_index = nonzero[0]
                    if vial[ML].iloc[tank_index] > 0:
                        continue
                    distance = float(
                        color_errors(vial[RGB].to_numpy(float), tail["rgb"])
                    )
                    if distance < threshold and de - distance > threshold:
                        # Matching the previous recipe is also explained by carryover.
                        if previous is not None and previous[ML].iloc[tank_index] > 0:
                            continue
                        wrong_tank = ML[tank_index][:4].upper()
            past_expected = (
                expected_clearance_ml is not None
                and truth(pd.Series([vial.volume_usable])).iloc[0]
                and vial.cumulative_ml >= expected_clearance_ml
            )
            if verified_tank_alert or wrong_tank is not None:
                state = "persistent_tank_or_recipe_fault"
                evidence.append(
                    "verified_tank_alert"
                    if verified_tank_alert
                    else f"unrequested_in_camera_signature={wrong_tank}"
                )
                explanation = (
                    (
                        "Verified tank-state alert accompanies this inspection."
                        if verified_tank_alert
                        else f"Color is closer to an unrequested {wrong_tank} in-camera signature than to the recipe reference."
                    )
                    + " A tank, recipe or valve problem is a hypothesis, not a confirmed root cause."
                )
                action = "Inspect tank identity, available levels, valve actuation and requested parameters."
            elif de < threshold:
                explanation = f"Observed color is {de:.2f} ΔE00 from the reference (< {threshold:g})."
                if (
                    camera_proven
                    and clean_reference_validated
                    and score.reference_scope == "independent_clean_controls"
                ):
                    state, production = "in_spec_clean_production", "PASS"
                else:
                    explanation += " This is an exploratory match; camera repeatability or clean-reference validation is incomplete."
                    action = "Retain as an exploratory color match; complete camera and clean-control validation before release."
            elif (
                previous is not None
                and trend.get("run_decay_supported", False)
                and trajectory["trajectory_supported"]
            ):
                state = "expected_changeover_carryover"
                explanation = (
                    f"Vial differs by {de:.2f} ΔE00 from the recipe reference. Its color lies toward previous recipe "
                    f"{int(previous.recipe_id)}, and the run's subsequent usable vials decrease toward the target. "
                    "This supports shared-path carryover as a hypothesis."
                )
                evidence.extend(["prior_recipe_color_trajectory", "observed_run_decay"])
                action = "Monitor the recovery curve; confirmation uses later observations and does not establish chemical purity."
            elif trend.get("run_persistent_deviation", False) and (
                vial.vial_in_run >= 3 or past_expected
            ):
                state = "persistent_tank_or_recipe_fault"
                explanation = (
                    f"Deviation has not decreased across {trend['trend_usable_prefix_vials']} consecutive usable vials. "
                    f"Requested throughput so far is {vial.cumulative_ml:g} mL. "
                    + (
                        "An independently established clearance volume has been exceeded. "
                        if past_expected
                        else "Expected clearance volume is not established, so persistent carryover remains possible. "
                    )
                    + "Check tank state, recipe parameters and valve actuation; root cause is unconfirmed."
                )
                evidence.append("persistent_observed_color_deviation")
                action = "Investigate tank supply, dispensing and the reference; do not infer a required flush volume from these data."
            else:
                explanation = (
                    f"Observed color differs by {de:.2f} ΔE00, but the available prior-recipe signature, "
                    "recovery trend or independent baseline does not support a specific cause."
                )
                if trend.get("run_decay_supported"):
                    explanation += (
                        f" Run error decreases from {trend['trend_first_delta_e']:.2f} to "
                        f"{trend['trend_last_delta_e']:.2f} ΔE00 across "
                        f"{trend['trend_usable_prefix_vials']} consecutive usable vials."
                    )
                if trajectory["trajectory_residual_delta_e"] is not None:
                    explanation += (
                        f" Closest previous-to-current Lab trajectory residual is "
                        f"{trajectory['trajectory_residual_delta_e']:.2f} ΔE00; "
                        "trajectory agreement is required to attribute carryover."
                    )
                elif previous is not None:
                    explanation += (
                        " No identifiable previous-recipe endpoint is available."
                    )
                if trend.get("run_late_rebound"):
                    explanation += " Later vials rebound away from the target; inspect vial appearance and process state."
            if de >= threshold:
                suggestion = PREFILL_SUGGESTION
        records.append(
            {
                "session_id": vial.session_id,
                "vial_key": vial.vial_key,
                "vial": int(vial.vial),
                "recipe_id": int(vial.recipe_id),
                "run_id": vial.run_id,
                "previous_recipe_id": None
                if previous is None
                else int(previous.recipe_id),
                "vial_in_run": int(vial.vial_in_run),
                "diagnostic_state": state,
                "diagnostic_label": STATE_LABELS[state],
                "color_assessment": assessment,
                "primary_method": method,
                "delta_e": de,
                "threshold": threshold,
                "reference_scope": None if score is None else score.reference_scope,
                "composition_in_training_hull": None
                if score is None
                else score.get("composition_in_training_hull"),
                "previous_composition_in_training_hull": None
                if score is None
                else score.get("previous_composition_in_training_hull"),
                "reference_runs": "[]" if score is None else score.reference_runs,
                "trusted_color": trusted,
                "quality_flags": vial.get("quality_flags", ""),
                "requested_ml": float(vial.requested_ml),
                "cumulative_requested_ml": float(vial.cumulative_ml),
                "volume_source": vial.volume_source,
                "source_image": vial.get("source_image", None),
                "source_sha256": vial.get("source_sha256", None),
                "source_timestamp_age_s": vial.get("source_timestamp_age_s", None),
                "fill_status": fill_status,
                "measured_fill_error_ml": fill_error,
                "production_status": production,
                "root_cause_confirmed": False,
                "explanation": explanation,
                "recommended_action": action,
                "prefill_inspection_suggestion": suggestion,
                "evidence": ";".join(evidence),
                **trajectory,
                **trend,
            }
        )
    report = pd.DataFrame(records)
    report["inspection_index"] = np.arange(1, len(report) + 1)
    return report
